import copy
import os
import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from monotonic_align import mask_from_lens
from munch import Munch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .dataset import build_dataloader
from .losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, WavLMLoss
from .models import (
    build_ASR_model_shape,
    build_F0_model_shape,
    build_model,
    load_ASR_model,
    load_checkpoint,
    load_F0_model,
)
from .modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from .modules.slmadv import SLMAdversarialLoss
from .pretrained.plbert.util import build_plbert_shape, load_plbert
from .utils import (
    get_data_path_list,
    get_image,
    length_to_mask,
    log_norm,
    maximum_path,
    recursive_munch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_and_cat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concatenate two tensors along dim 0, zero-padding variable-length dims.

    Different validation batches may have different max sequence lengths (e.g.
    texts padded to different lengths). torch.cat requires all non-batch dims
    to match, so we pad the shorter tensor before concatenating.
    """
    if a.shape[1:] == b.shape[1:]:
        return torch.cat([a, b], dim=0)
    orig_a, orig_b = a.shape, b.shape
    # F.pad spec is built last-dim-first
    pad_b = []
    pad_a = []
    for d in reversed(range(1, a.dim())):
        pad_b += [0, max(0, orig_a[d] - orig_b[d])]
        pad_a += [0, max(0, orig_b[d] - orig_a[d])]
    if any(p > 0 for p in pad_b):
        b = F.pad(b, pad_b)
    if any(p > 0 for p in pad_a):
        a = F.pad(a, pad_a)
    return torch.cat([a, b], dim=0)


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------


class StyleTTS2DataModule(L.LightningDataModule):
    def __init__(self, config, load_for_everyvoice=False):
        super().__init__()

        self.load_for_everyvoice = load_for_everyvoice

        if load_for_everyvoice:
            self._ev_text_config = config["ev_config"].text
            self._pretrained_symbols = config["ev_config"].pretrained.pretrained_symbols
            self._preprocessed_dir = str(config["ev_config"].preprocessing.save_dir)
            self._output_sampling_rate = config[
                "ev_config"
            ].preprocessing.audio.output_sampling_rate
            self.config = config
        else:
            self._ev_text_config = None
            self._pretrained_symbols = None
            self._preprocessed_dir = None
            self._output_sampling_rate = None
            self.config = config

        self.train_list = None
        self.val_list = None

    def setup(self, stage=None):
        dp = self.config["data_params"]
        if self.load_for_everyvoice:
            from everyvoice.utils import generic_psv_filelist_reader

            self.train_list = generic_psv_filelist_reader(dp["train_data"])
            self.val_list = generic_psv_filelist_reader(dp["val_data"])
        else:
            self.train_list, self.val_list = get_data_path_list(
                dp["train_data"], dp["val_data"]
            )

    def train_dataloader(self):
        dp = self.config["data_params"]
        return build_dataloader(
            self.train_list,
            dp["root_path"],
            self.config,
            preprocessed_dir=self._preprocessed_dir,
            output_sampling_rate=self._output_sampling_rate,
            ev_text_config=self._ev_text_config,
            pretrained_symbols=self._pretrained_symbols,
            OOD_data=dp["OOD_data"],
            min_length=dp["min_length"],
            batch_size=self.config.get("batch_size", 16),
            num_workers=2,
        )

    def val_dataloader(self):
        dp = self.config["data_params"]
        return build_dataloader(
            self.val_list,
            dp["root_path"],
            self.config,
            preprocessed_dir=self._preprocessed_dir,
            output_sampling_rate=self._output_sampling_rate,
            ev_text_config=self._ev_text_config,
            pretrained_symbols=self._pretrained_symbols,
            OOD_data=dp["OOD_data"],
            min_length=dp["min_length"],
            batch_size=self.config.get("batch_size", 16),
            validation=True,
            num_workers=0,
        )


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class StyleTTS2Module(L.LightningModule):
    """Unified LightningModule for first-stage, second-stage, and fine-tuning.

    Args:
        config: parsed YAML config dict.
        mode: one of ``'first'``, ``'second'``, or ``'finetune'``.
    """

    _VERSION: str = "1.0"

    def __init__(self, config: dict | None = None, mode: str = "first"):
        super().__init__()

        assert mode in ("first", "second", "finetune"), f"Unknown mode: {mode}"
        self.automatic_optimization = False
        self.config = config
        self.mode = mode

        # If loading from a checkpoint, we have to first load an empty config, and then initialize in on_load_checkpoint
        if self.config is not None:
            self.initialize_from_config(self.config)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def initialize_from_config(self, config, load_pretrained_weights=True):
        # Core hyper-parameters
        self.sr = config["preprocess_params"].get("sr", 24000)
        self.hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
        self.max_len = config.get("max_len", 200)
        model_params = recursive_munch(config["model_params"])
        self.model_params = model_params
        self.multispeaker = model_params.multispeaker
        loss_params = Munch(config["loss_params"])
        self.loss_params = loss_params

        # Epoch thresholds (all relative to the start of this training run)
        self.TMA_epoch = loss_params.TMA_epoch
        self.diff_epoch = getattr(loss_params, "diff_epoch", 0)
        self.joint_epoch = getattr(loss_params, "joint_epoch", 0)

        # Build pretrained backbones then the full model.
        # When loading from a checkpoint, skip downloading pretrained weights —
        # load_state_dict will overwrite them from the checkpoint anyway.
        if load_pretrained_weights:
            text_aligner = load_ASR_model(config["pretrained_asr"])
            pitch_extractor = load_F0_model(config["pretrained_f0"])
            plbert = load_plbert(config["pretrained_plbert"])
        else:
            text_aligner = build_ASR_model_shape(config["pretrained_asr"])
            pitch_extractor = build_F0_model_shape()
            plbert = build_plbert_shape(config["pretrained_plbert"])
        # TODO: model_params passes an incorrect value for n_symbols in the text embedding
        nets = build_model(model_params, text_aligner, pitch_extractor, plbert)
        # Register every sub-network as a direct attribute so Lightning / DDP
        # tracks parameters and state correctly.
        for key, module in nets.items():
            setattr(self, key, module)
        self._net_keys = list(nets.keys())
        self.n_down = text_aligner.n_down

        # Loss modules (registered as attributes → DDP-tracked)
        self.stft_loss = MultiResolutionSTFTLoss(sample_rate=self.sr)
        self.gl = GeneratorLoss(self.mpd, self.msd)
        self.dl = DiscriminatorLoss(self.mpd, self.msd)
        self.wl = WavLMLoss(
            model_params.slm.model, self.wd, self.sr, model_params.slm.sr
        )

        # Diffusion sampler — plain Python object (no parameters)
        self._sampler = DiffusionSampler(
            self.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

        # SLM adversarial helper (no nn.Module, avoids circular reference)
        slmadv_cfg = config.get("slmadv_params", {})
        self._slmadv_params = Munch(slmadv_cfg) if slmadv_cfg else Munch()
        self._slmadv: SLMAdversarialLoss | None = None  # built in setup()

        # Optimizer key ordering (index in the list returned by configure_optimizers)
        self._opt_key_to_idx: dict[str, int] = {}

        # Running std used for diffusion sigma estimation
        self._running_std: list[float] = []

    def setup(self, stage=None):
        if stage == "predict":
            return

        p = self._slmadv_params
        self._slmadv = SLMAdversarialLoss(
            self,
            self.wl,
            self._sampler,
            getattr(p, "min_len", 400),
            getattr(p, "max_len", 500),
            batch_percentage=getattr(p, "batch_percentage", 0.5),
            skip_update=getattr(p, "iter", 10),
            sig=getattr(p, "sig", 1.5),
        )

        # Stage 2 / finetune: load Stage 1 weights (weights only, not optimizer state)
        if self.mode in ("second", "finetune") and not self.config.get(
            "pretrained_model", ""
        ):
            first_stage_path = self.config.get("first_stage_path", "")
            if os.path.isfile(first_stage_path):
                _ignore = {
                    "bert",
                    "bert_encoder",
                    "predictor",
                    "predictor_encoder",
                    "msd",
                    "mpd",
                    "wd",
                    "diffusion",
                }
                state = torch.load(
                    first_stage_path, map_location="cpu", weights_only=True
                )
                if "state_dict" in state:
                    # Lightning checkpoint: extract per-submodule state dicts
                    for key in self._net_keys:
                        if key in _ignore:
                            continue
                        prefix = key + "."
                        sub_sd = {
                            k[len(prefix) :]: v
                            for k, v in state["state_dict"].items()
                            if k.startswith(prefix)
                        }
                        if sub_sd:
                            getattr(self, key).load_state_dict(sub_sd, strict=False)
                            print(f"{key} loaded")
                else:
                    # Legacy format (original train_first.py checkpoints)
                    load_checkpoint(
                        {k: getattr(self, k) for k in self._net_keys},
                        None,
                        first_stage_path,
                        load_only_params=True,
                        ignore_modules=list(_ignore),
                    )
                self.predictor_encoder = copy.deepcopy(self.style_encoder)
                # load_checkpoint (legacy path) calls .eval() on all networks;
                # reset to train mode so trainable RNN layers (predictor etc.)
                # can backpropagate correctly.
                self.train()
            else:
                raise FileNotFoundError(
                    f"first_stage_path not found: {first_stage_path}. "
                    "Set pretrained_model in config to load a Stage 2 checkpoint directly."
                )

        # Freeze parameters never trained in this mode.
        # DDP only allreduces params with requires_grad=True, so freezing unused
        # networks eliminates the "unused parameters" DDP error without the
        # overhead of find_unused_parameters=True.
        if self.mode == "first":
            for key in (
                "bert",
                "bert_encoder",
                "predictor",
                "predictor_encoder",
                "diffusion",
                "wd",
                "pitch_extractor",
            ):
                getattr(self, key).requires_grad_(False)
        elif self.mode == "second":
            # These are always called inside torch.no_grad() in Stage 2
            for net in (self.text_aligner, self.text_encoder, self.pitch_extractor):
                net.requires_grad_(False)

    def on_load_checkpoint(self, checkpoint):
        """Deserialize the checkpoint hyperparameters."""
        checkpoint = self.check_and_upgrade_checkpoint(checkpoint)

        hp = checkpoint.get("hyper_parameters", {})
        raw_config = hp["config"]

        # New checkpoints store ev_config as a JSON-safe dict (via
        # model_checkpoint_dump).  Old checkpoints stored the pydantic object
        # directly.  Reconstruct when necessary so the rest of the code always
        # gets a StyleTTS2Config instance.
        if isinstance(raw_config.get("ev_config"), dict):
            from .ev_config import StyleTTS2Config

            raw_config = dict(raw_config)
            raw_config["ev_config"] = StyleTTS2Config.model_validate(
                raw_config["ev_config"]
            )

        self.config = raw_config
        self.mode = hp.get("mode", self.mode)

        self.initialize_from_config(self.config, load_pretrained_weights=False)

        # Older checkpoints may have WavLM weights (wl.wavlm.*).  Drop them
        # before Lightning applies the state dict — they are unused at synthesis
        # time and are re-loaded lazily from the local cache when needed during
        # training.
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("wl.wavlm.")
        }

    @staticmethod
    def _config_to_checkpoint_safe(config: dict) -> dict:
        """Recursively convert a native config dict to a JSON-safe representation.

        Pydantic models are serialized via model_dump(mode='json'), which
        converts Enums to their values and Paths to strings.  The result is then
        recursed to catch any remaining non-primitive types.  Plain Path and Enum
        values that appear outside pydantic models are also converted.  The result
        contains only built-in Python types, so torch.load(weights_only=True) can
        deserialise it.
        """
        import enum
        from pathlib import Path

        if hasattr(config, "model_dump"):
            return StyleTTS2Module._config_to_checkpoint_safe(
                config.model_dump(mode="json")
            )
        if isinstance(config, dict):
            return {
                k: StyleTTS2Module._config_to_checkpoint_safe(v)
                for k, v in config.items()
            }
        if isinstance(config, (list, tuple)):
            return [StyleTTS2Module._config_to_checkpoint_safe(x) for x in config]
        if isinstance(config, Path):
            return str(config)
        if isinstance(config, enum.Enum):
            return config.value
        return config

    def on_save_checkpoint(self, checkpoint):
        hp = checkpoint.setdefault("hyper_parameters", {})
        hp["config"] = self._config_to_checkpoint_safe(self.config)
        hp["mode"] = self.mode
        checkpoint["model_info"] = {
            "name": self.__class__.__name__,
            "version": self._VERSION,
        }
        # WavLM weights are fixed pretrained weights that never change during
        # training.  Strip them from the checkpoint to avoid bloating every
        # saved checkpoint with ~300 MB of weights that are always re-loaded
        # from the local cache on demand anyway.
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("wl.wavlm.")
        }

    def check_and_upgrade_checkpoint(self, checkpoint):
        """
        Check model's compatibility and possibly upgrade.
        """
        from packaging.version import Version

        model_info = checkpoint.get(
            "model_info",
            {
                "name": self.__class__.__name__,
                "version": "1.0",
            },
        )

        ckpt_model_type = model_info.get("name", "MISSING_TYPE")
        if ckpt_model_type != self.__class__.__name__:
            raise TypeError(
                f"""Wrong model type ({ckpt_model_type}), we are expecting a '{self.__class__.__name__}' model"""
            )

        ckpt_version = Version(model_info.get("version", "0.0"))
        if ckpt_version > Version(self._VERSION):
            raise ValueError(
                "Your model was created with a newer version of EveryVoice, please update your software."
            )
        # Successively convert model checkpoints to newer version.
        if ckpt_version < Version("1.0"):
            # Upgrading from 0.0 to 1.0 requires no changes; future versions might require changes
            checkpoint["model_info"]["version"] = "1.0"

        return checkpoint

    def configure_optimizers(self):
        opt_cfg = Munch(self.config["optimizer_params"])
        # OneCycleLR with pct_start=0, div_factor=1, final_div_factor=1 is a
        # flat LR schedule — epochs/steps_per_epoch values don't affect the LR.
        epochs = self.trainer.max_epochs
        steps = max(self.trainer.estimated_stepping_batches, 1)
        steps_per_epoch = max(steps // max(epochs, 1), 1)

        optimizers, schedulers = [], []
        for key in self._net_keys:
            trainable = [p for p in getattr(self, key).parameters() if p.requires_grad]
            if not trainable:
                continue  # frozen in setup() — no optimizer needed

            self._opt_key_to_idx[key] = len(optimizers)

            lr = float(opt_cfg.lr)
            betas = (0.0, 0.99)
            wd = 1e-4
            max_lr = float(opt_cfg.lr)

            if key == "bert":
                lr = float(opt_cfg.bert_lr)
                betas = (0.9, 0.99)
                wd = 0.01
                max_lr = float(opt_cfg.bert_lr) * 2
            elif key in ("decoder", "style_encoder") and self.mode != "first":
                lr = float(opt_cfg.ft_lr)
                max_lr = float(opt_cfg.ft_lr) * 2

            opt = AdamW(trainable, lr=lr, weight_decay=wd, betas=betas, eps=1e-9)
            sched = OneCycleLR(
                opt,
                max_lr=max_lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.0,
                div_factor=1,
                final_div_factor=1,
            )

            optimizers.append(opt)
            schedulers.append(
                {
                    "scheduler": sched,
                    "interval": "step",
                    "frequency": 1,
                    "name": f"lr/{key}",
                }
            )

        return optimizers, schedulers

    def on_train_epoch_end(self):
        # Persist running sigma_data estimate back to config YAML after each epoch
        if (
            self._running_std
            and self.model_params.diffusion.dist.estimate_sigma_data
            and self.trainer.is_global_zero
        ):
            self.config["model_params"]["diffusion"]["dist"]["sigma_data"] = float(
                np.mean(self._running_std)
            )
            self._running_std.clear()

    _MAX_VAL_AUDIO = 7

    def on_validation_epoch_start(self):
        self._val_batch = None

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero or self._val_batch is None:
            return

        tb = self.logger.experiment
        epoch = self.current_epoch
        b = self._val_batch

        with torch.no_grad():
            if self.mode == "first":
                tb.add_figure(
                    "eval/attn", get_image(b["s2s_attn"][0].numpy().squeeze()), epoch
                )
                for bib in range(min(len(b["asr"]), self._MAX_VAL_AUDIO)):
                    mel_length = int(b["mel_input_length"][bib].item())
                    gt = b["mels"][bib, :, :mel_length].unsqueeze(0).to(self.device)
                    en = (
                        b["asr"][bib, :, : mel_length // 2].unsqueeze(0).to(self.device)
                    )

                    F0_real, _, _ = self.pitch_extractor(gt.unsqueeze(1))
                    s = self.style_encoder(gt.unsqueeze(1))
                    real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                    y_rec = self.decoder(en, F0_real, real_norm, s)

                    name = b["basenames"][bib]
                    tb.add_audio(
                        f"eval/{name}",
                        y_rec.cpu().numpy().squeeze(),
                        epoch,
                        sample_rate=self.sr,
                    )
                    if epoch == 0:
                        tb.add_audio(
                            f"gt/{name}",
                            b["waves"][bib].squeeze(),
                            epoch,
                            sample_rate=self.sr,
                        )

            else:
                if epoch < self.joint_epoch and self.mode != "finetune":
                    for bib in range(min(len(b["asr"]), self._MAX_VAL_AUDIO)):
                        mel_length = int(b["mel_input_length"][bib].item())
                        gt = b["mels"][bib, :, :mel_length].unsqueeze(0).to(self.device)
                        en = (
                            b["asr"][bib, :, : mel_length // 2]
                            .unsqueeze(0)
                            .to(self.device)
                        )

                        F0_real, _, _ = self.pitch_extractor(gt.unsqueeze(1))
                        s = self.style_encoder(gt.unsqueeze(1))
                        real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                        y_rec = self.decoder(en, F0_real, real_norm, s)

                        name = b["basenames"][bib]
                        tb.add_audio(
                            f"eval/{name}",
                            y_rec.cpu().numpy().squeeze(),
                            epoch,
                            sample_rate=self.sr,
                        )

                        s_dur = self.predictor_encoder(gt.unsqueeze(1))
                        p_en = (
                            b["p"][bib, :, : mel_length // 2]
                            .unsqueeze(0)
                            .to(self.device)
                        )
                        F0_fake, N_fake = self.predictor.F0Ntrain(p_en, s_dur)
                        y_pred = self.decoder(en, F0_fake, N_fake, s)
                        tb.add_audio(
                            f"pred/{name}",
                            y_pred.cpu().numpy().squeeze(),
                            epoch,
                            sample_rate=self.sr,
                        )

                        if epoch == 0:
                            tb.add_audio(
                                f"gt/{name}",
                                b["waves"][bib].squeeze(),
                                epoch,
                                sample_rate=self.sr,
                            )
                else:
                    # Diffusion-sampled TTS from text
                    texts = b["texts"].to(self.device)
                    input_lengths = b["input_lengths"].to(self.device)
                    text_mask = b["text_mask"].to(self.device)
                    bert_dur = b["bert_dur"].to(self.device)
                    d_en = b["d_en"].to(self.device)

                    ref_s = None
                    if self.multispeaker and b["ref_mels"] is not None:
                        ref_mels = b["ref_mels"].to(self.device)
                        style_enc = self.style_encoder(ref_mels.unsqueeze(1))
                        predictor_enc = self.predictor_encoder(ref_mels.unsqueeze(1))
                        ref_s = torch.cat([style_enc, predictor_enc], dim=1)

                    t_en = self.text_encoder(texts, input_lengths, text_mask)

                    for bib in range(min(d_en.size(0), self._MAX_VAL_AUDIO)):
                        noise = torch.randn((1, 256), device=self.device).unsqueeze(1)
                        sampler_kwargs = dict(
                            noise=noise,
                            embedding=bert_dur[bib].unsqueeze(0),
                            embedding_scale=1,
                            num_steps=5,
                        )
                        if self.multispeaker and ref_s is not None:
                            sampler_kwargs["features"] = ref_s[bib].unsqueeze(0)
                        s_pred = self._sampler(**sampler_kwargs).squeeze(1)

                        s = s_pred[:, 128:]
                        ref = s_pred[:, :128]

                        il = input_lengths[bib].unsqueeze(0)
                        tm = text_mask[bib, : input_lengths[bib]].unsqueeze(0)
                        d = self.predictor.text_encoder(
                            d_en[bib, :, : input_lengths[bib]].unsqueeze(0), s, il, tm
                        )
                        x, _ = self.predictor.lstm(d)
                        duration = torch.sigmoid(self.predictor.duration_proj(x)).sum(
                            axis=-1
                        )
                        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
                        pred_dur[-1] += 5

                        pred_aln = torch.zeros(
                            input_lengths[bib],
                            int(pred_dur.sum().item()),
                            device=self.device,
                        )
                        c = 0
                        for i in range(pred_aln.size(0)):
                            pred_aln[i, c : c + int(pred_dur[i].item())] = 1
                            c += int(pred_dur[i].item())

                        en = d.transpose(-1, -2) @ pred_aln.unsqueeze(0)
                        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
                        out = self.decoder(
                            t_en[bib, :, : input_lengths[bib]].unsqueeze(0)
                            @ pred_aln.unsqueeze(0),
                            F0_pred,
                            N_pred,
                            ref.squeeze().unsqueeze(0),
                        )

                        name = b["basenames"][bib]
                        tb.add_audio(
                            f"pred/{name}",
                            out.cpu().numpy().squeeze(),
                            epoch,
                            sample_rate=self.sr,
                        )
                        if epoch == 0:
                            tb.add_audio(
                                f"gt/{name}",
                                b["waves"][bib].squeeze(),
                                epoch,
                                sample_rate=self.sr,
                            )

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------

    def _opt(self, key: str):
        return self.optimizers()[self._opt_key_to_idx[key]]

    def _step(self, *keys: str):
        for key in keys:
            self._opt(key).step()

    def _zero_grad_all(self):
        for opt in self.optimizers():
            opt.zero_grad()

    # ------------------------------------------------------------------
    # Shared clip-extraction helper
    # ------------------------------------------------------------------

    def _get_clips(
        self, asr, mels, mel_input_length, waves, mel_len, p=None, mel_len_st=None
    ):
        """Randomly crop aligned segments from a batch.

        Returns a list: [en, gt, wav, (p_en,) (st,)]
        """
        device = self.device
        en, gt, wav, p_en, st = [], [], [], [], []
        for bib in range(len(mel_input_length)):
            half_len = int(mel_input_length[bib].item() / 2)
            rs = np.random.randint(0, half_len - mel_len)
            en.append(asr[bib, :, rs : rs + mel_len])
            gt.append(mels[bib, :, rs * 2 : (rs + mel_len) * 2])
            y = waves[bib][
                rs * 2 * self.hop_length : (rs + mel_len) * 2 * self.hop_length
            ]
            wav.append(torch.from_numpy(y).to(device))
            if p is not None:
                p_en.append(p[bib, :, rs : rs + mel_len])
            if mel_len_st is not None:
                rs_st = np.random.randint(0, half_len - mel_len_st)
                st.append(mels[bib, :, rs_st * 2 : (rs_st + mel_len_st) * 2])

        out = [
            torch.stack(en),
            torch.stack(gt).detach(),
            torch.stack(wav).float().detach(),
        ]
        if p is not None:
            out.append(torch.stack(p_en))
        if mel_len_st is not None:
            out.append(torch.stack(st).detach())
        return out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if self.mode == "first":
            self._train_first(batch, batch_idx)
        else:
            self._train_second(batch, batch_idx)

    # --- Stage 1 --------------------------------------------------------

    def _train_first(self, batch, batch_idx):
        epoch = self.current_epoch
        device = self.device
        waves = batch[0]
        texts, input_lengths, _, _, mels, mel_input_length, _ = [
            b.to(device) for b in batch[1:-1]
        ]

        with torch.no_grad():
            mask = length_to_mask(mel_input_length // (2**self.n_down)).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

        ppgs, s2s_pred, s2s_attn = self.text_aligner(mels, mask, texts)
        s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)

        with torch.no_grad():
            attn_mask = (
                (~mask)
                .unsqueeze(-1)
                .expand(*mask.shape, text_mask.shape[-1])
                .float()
                .transpose(-1, -2)
            )
            attn_mask = attn_mask * (
                (~text_mask)
                .unsqueeze(-1)
                .expand(*text_mask.shape, mask.shape[-1])
                .float()
            )
            attn_mask = attn_mask < 1
        s2s_attn.masked_fill_(attn_mask, 0.0)

        with torch.no_grad():
            mask_ST = mask_from_lens(
                s2s_attn, input_lengths, mel_input_length // (2**self.n_down)
            )
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        t_en = self.text_encoder(texts, input_lengths, text_mask)
        asr = t_en @ (s2s_attn if random.getrandbits(1) else s2s_attn_mono)

        mel_len = min(int(mel_input_length.min().item() / 2 - 1), self.max_len // 2)
        mel_len_st = int(mel_input_length.min().item() / 2 - 1)
        clips = self._get_clips(
            asr, mels, mel_input_length, waves, mel_len, mel_len_st=mel_len_st
        )
        en, gt, wav, st = clips

        if gt.shape[-1] < 80:
            return

        with torch.no_grad():
            real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
            F0_real, _, _ = self.pitch_extractor(gt.unsqueeze(1))

        s = self.style_encoder(
            st.unsqueeze(1) if self.multispeaker else gt.unsqueeze(1)
        )
        y_rec = self.decoder(en, F0_real, real_norm, s)

        # -- Discriminator update -----------------------------------------
        # Always backward through msd/mpd so DDP can allreduce their grads;
        # only apply the step after TMA_epoch when discriminator training begins.
        self._zero_grad_all()
        d_loss = self.dl(wav.unsqueeze(1).float(), y_rec.detach()).mean()
        self.manual_backward(d_loss)
        if epoch >= self.TMA_epoch:
            self._step("msd", "mpd")

        # -- Generator update ---------------------------------------------
        self._zero_grad_all()
        loss_mel = self.stft_loss(y_rec.squeeze(), wav)
        # Always compute every term so text_aligner stays in the DDP graph
        # regardless of the random s2s_attn / s2s_attn_mono choice above.
        loss_s2s = sum(
            F.cross_entropy(pred[:l], text[:l])
            for pred, text, l in zip(s2s_pred, texts, input_lengths)
        ) / texts.size(0)
        loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
        loss_gen_all = self.gl(wav.unsqueeze(1).float(), y_rec).mean()
        loss_slm = self.wl(wav, y_rec).mean()

        if epoch >= self.TMA_epoch:
            g_loss = (
                self.loss_params.lambda_mel * loss_mel
                + self.loss_params.lambda_mono * loss_mono
                + self.loss_params.lambda_s2s * loss_s2s
                + self.loss_params.lambda_gen * loss_gen_all
                + self.loss_params.lambda_slm * loss_slm
            )
        else:
            # Zero-scale the extra terms: gradients reach text_aligner (via
            # s2s_pred / s2s_attn) and msd/mpd (via loss_gen_all) so DDP can
            # sync them, but the values don't affect the actual parameter update.
            g_loss = self.loss_params.lambda_mel * loss_mel + 0.0 * (
                loss_s2s + loss_mono + loss_gen_all + loss_slm
            )

        self.manual_backward(g_loss)
        self._step("text_encoder", "style_encoder", "decoder")
        if epoch >= self.TMA_epoch:
            self._step("text_aligner")

        self.log_dict(
            {
                "train/mel": loss_mel,
                "train/gen": (
                    loss_gen_all
                    if isinstance(loss_gen_all, torch.Tensor)
                    else torch.tensor(loss_gen_all)
                ),
                "train/disc": d_loss,
                "train/mono": (
                    loss_mono
                    if isinstance(loss_mono, torch.Tensor)
                    else torch.tensor(loss_mono)
                ),
                "train/s2s": (
                    loss_s2s
                    if isinstance(loss_s2s, torch.Tensor)
                    else torch.tensor(loss_s2s)
                ),
                "train/slm": (
                    loss_slm
                    if isinstance(loss_slm, torch.Tensor)
                    else torch.tensor(loss_slm)
                ),
            },
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )

    # --- Stage 2 / Finetune ---------------------------------------------

    def _train_second(self, batch, batch_idx):  # noqa: C901
        epoch = self.current_epoch
        device = self.device
        is_ft = self.mode == "finetune"

        waves = batch[0]
        (
            texts,
            input_lengths,
            ref_texts,
            ref_lengths,
            mels,
            mel_input_length,
            ref_mels,
        ) = [b.to(device) for b in batch[1:-1]]

        # -- Frozen inference pass ---------------------------------------
        with torch.no_grad():
            mask = length_to_mask(mel_input_length // (2**self.n_down)).to(device)
            text_mask = length_to_mask(input_lengths).to(device)
            try:
                _, s2s_pred, s2s_attn = self.text_aligner(mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
            except Exception:
                return

            mask_ST = mask_from_lens(
                s2s_attn, input_lengths, mel_input_length // (2**self.n_down)
            )
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            t_en = self.text_encoder(texts, input_lengths, text_mask)
            if is_ft:
                asr = t_en @ (s2s_attn if random.getrandbits(1) else s2s_attn_mono)
            else:
                asr = t_en @ s2s_attn_mono
            d_gt = s2s_attn_mono.sum(axis=-1).detach()

            ref = None
            if self.multispeaker:
                style_enc = self.style_encoder(ref_mels.unsqueeze(1))
                predictor_enc = self.predictor_encoder(ref_mels.unsqueeze(1))
                ref = torch.cat([style_enc, predictor_enc], dim=1)

        # Per-utterance style (adaptive avgpool prevents batching)
        ss, gs = [], []
        for bib in range(len(mel_input_length)):
            mel = mels[bib, :, : mel_input_length[bib]]
            ss.append(self.predictor_encoder(mel.unsqueeze(0).unsqueeze(1)))
            gs.append(self.style_encoder(mel.unsqueeze(0).unsqueeze(1)))
        s_dur = torch.stack(ss).squeeze()
        gs = torch.stack(gs).squeeze()
        s_trg = torch.cat([gs, s_dur], dim=-1).detach()

        bert_dur = self.bert(texts, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        # -- Diffusion training ------------------------------------------
        if epoch >= self.diff_epoch:
            num_steps = np.random.randint(3, 5)
            if self.model_params.diffusion.dist.estimate_sigma_data:
                sigma = s_trg.std(axis=-1).mean().item()
                self.diffusion.diffusion.sigma_data = sigma
                self._running_std.append(sigma)

            if self.multispeaker:
                s_preds = self._sampler(
                    noise=torch.randn_like(s_trg).unsqueeze(1),
                    embedding=bert_dur,
                    embedding_scale=1,
                    features=ref,
                    embedding_mask_proba=0.1,
                    num_steps=num_steps,
                ).squeeze(1)
                loss_diff = self.diffusion(
                    s_trg.unsqueeze(1), embedding=bert_dur, features=ref
                ).mean()
            else:
                s_preds = self._sampler(
                    noise=torch.randn_like(s_trg).unsqueeze(1),
                    embedding=bert_dur,
                    embedding_scale=1,
                    embedding_mask_proba=0.1,
                    num_steps=num_steps,
                ).squeeze(1)
                loss_diff = self.diffusion.diffusion(
                    s_trg.unsqueeze(1), embedding=bert_dur
                ).mean()
            loss_sty = F.l1_loss(s_preds, s_trg.detach())
        else:
            # Zero-weight forward keeps diffusion in DDP sync graph before
            # diff_epoch without running the expensive sampler.
            if self.multispeaker:
                loss_diff = (
                    0.0
                    * self.diffusion.diffusion(
                        s_trg.unsqueeze(1), embedding=bert_dur, features=ref
                    ).mean()
                )
            else:
                loss_diff = (
                    0.0
                    * self.diffusion.diffusion(
                        s_trg.unsqueeze(1), embedding=bert_dur
                    ).mean()
                )
            loss_sty = 0.0

        d, p = self.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)

        # -- Clip extraction ---------------------------------------------
        mel_len = min(int(mel_input_length.min().item() / 2 - 1), self.max_len // 2)
        mel_len_st = int(mel_input_length.min().item() / 2 - 1)
        clips = self._get_clips(
            asr, mels, mel_input_length, waves, mel_len, p=p, mel_len_st=mel_len_st
        )
        en, gt, wav_clips, p_en, st = clips

        if gt.size(-1) < 80:
            return

        s_dur_clip = self.predictor_encoder(
            st.unsqueeze(1) if self.multispeaker else gt.unsqueeze(1)
        )
        s = self.style_encoder(
            st.unsqueeze(1) if self.multispeaker else gt.unsqueeze(1)
        )

        with torch.no_grad():
            F0_real, _, F0 = self.pitch_extractor(gt.unsqueeze(1))
            N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
            y_rec_gt = wav_clips.unsqueeze(1)
            y_rec_gt_pred = self.decoder(en, F0_real, N_real, s)
            # After joint_epoch use the ground-truth recording; before that use
            # the reconstruction (decoder is still being tuned).
            wav_for_disc = (
                y_rec_gt if (epoch >= self.joint_epoch or is_ft) else y_rec_gt_pred
            )

        F0_fake, N_fake = self.predictor.F0Ntrain(p_en, s_dur_clip)
        y_rec = self.decoder(en, F0_fake, N_fake, s)

        loss_F0_rec = F.smooth_l1_loss(F0_real, F0_fake) / 10
        loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

        # -- Discriminator update -----------------------------------------
        # Always backward through msd/mpd so DDP can allreduce their grads;
        # only apply the step once discriminator training is active.
        start_ds = epoch >= self.diff_epoch or is_ft
        self._zero_grad_all()
        d_loss = self.dl(wav_for_disc.detach(), y_rec.detach()).mean()
        self.manual_backward(d_loss)
        if start_ds:
            self._step("msd", "mpd")

        # -- Duration / CE losses ----------------------------------------
        loss_ce = loss_dur = 0.0
        for _pred, _text, _length in zip(d, d_gt, input_lengths):
            _pred = _pred[:_length, :]
            _text = _text[:_length].long()
            _trg = torch.zeros_like(_pred)
            for pp in range(_trg.shape[0]):
                _trg[pp, : _text[pp]] = 1
            _dur_pred = torch.sigmoid(_pred).sum(axis=1)
            loss_dur = loss_dur + F.l1_loss(
                _dur_pred[1 : _length - 1], _text[1 : _length - 1]
            )
            loss_ce = loss_ce + F.binary_cross_entropy_with_logits(
                _pred.flatten(), _trg.flatten()
            )
        loss_ce = loss_ce / texts.size(0)
        loss_dur = loss_dur / texts.size(0)

        loss_mel = self.stft_loss(y_rec, wav_for_disc)
        loss_gen_all = self.gl(wav_for_disc, y_rec).mean() if start_ds else 0.0
        loss_lm = self.wl(wav_for_disc.detach().squeeze(), y_rec.squeeze()).mean()

        # TMA losses — finetune only
        if is_ft:
            loss_s2s = sum(
                F.cross_entropy(pred[:l], text[:l])
                for pred, text, l in zip(s2s_pred, texts, input_lengths)
            ) / texts.size(0)
            loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
        else:
            loss_s2s = loss_mono = 0.0

        lp = self.loss_params
        g_loss = (
            lp.lambda_mel * loss_mel
            + lp.lambda_F0 * loss_F0_rec
            + lp.lambda_ce * loss_ce
            + lp.lambda_norm * loss_norm_rec
            + lp.lambda_dur * loss_dur
            + lp.lambda_gen * loss_gen_all
            + lp.lambda_slm * loss_lm
            + lp.lambda_sty * loss_sty
            + lp.lambda_diff * loss_diff
        )
        if is_ft:
            g_loss = g_loss + lp.lambda_mono * loss_mono + lp.lambda_s2s * loss_s2s

        # -- Generator update --------------------------------------------
        self._zero_grad_all()
        self.manual_backward(g_loss)
        self._step("bert_encoder", "bert", "predictor", "predictor_encoder")
        if epoch >= self.diff_epoch:
            self._step("diffusion")
        if epoch >= self.joint_epoch or is_ft:
            self._step("style_encoder", "decoder")
        if is_ft:
            self._step("text_encoder", "text_aligner")

        # -- SLM adversarial pass (joint epoch / finetune) ---------------
        d_loss_slm = loss_gen_lm = 0.0
        if (epoch >= self.joint_epoch or is_ft) and self._slmadv is not None:
            use_ind = np.random.rand() < 0.5
            rt = texts if use_ind else ref_texts
            rl = input_lengths if use_ind else ref_lengths

            slm_out = self._slmadv.forward(
                batch_idx,
                y_rec_gt,
                y_rec_gt_pred,
                waves,
                mel_input_length,
                rt,
                rl,
                use_ind,
                s_trg.detach(),
                ref if self.multispeaker else None,
            )
            if slm_out is not None:
                d_loss_slm, loss_gen_lm, _ = slm_out

                # Scale gradients before stepping (prevents SLM from
                # dominating the predictor / diffusion updates)
                self._zero_grad_all()
                self.manual_backward(loss_gen_lm)

                p_norm = (
                    sum(
                        param.grad.detach().norm(2).item() ** 2
                        for param in self.predictor.parameters()
                        if param.grad is not None and param.requires_grad
                    )
                    ** 0.5
                )

                thresh = getattr(self._slmadv_params, "thresh", 5)
                scale = getattr(self._slmadv_params, "scale", 0.01)

                if p_norm > thresh:
                    inv = 1.0 / p_norm
                    for key in self._net_keys:
                        for param in getattr(self, key).parameters():
                            if param.grad is not None:
                                param.grad.mul_(inv)

                for param in (
                    *self.predictor.duration_proj.parameters(),
                    *self.predictor.lstm.parameters(),
                    *self.diffusion.parameters(),
                ):
                    if param.grad is not None:
                        param.grad.mul_(scale)

                self._step("bert_encoder", "bert", "predictor", "diffusion")

                if d_loss_slm != 0:
                    self._zero_grad_all()
                    self.manual_backward(d_loss_slm, retain_graph=True)
                    self._step("wd")

        self.log_dict(
            {
                "train/mel": loss_mel,
                "train/disc": d_loss,
                "train/dur": (
                    loss_dur
                    if isinstance(loss_dur, torch.Tensor)
                    else torch.tensor(float(loss_dur))
                ),
                "train/ce": (
                    loss_ce
                    if isinstance(loss_ce, torch.Tensor)
                    else torch.tensor(float(loss_ce))
                ),
                "train/norm": loss_norm_rec,
                "train/F0": loss_F0_rec,
                "train/slm": loss_lm,
                "train/gen": (
                    loss_gen_all
                    if isinstance(loss_gen_all, torch.Tensor)
                    else torch.tensor(float(loss_gen_all))
                ),
                "train/sty": (
                    loss_sty
                    if isinstance(loss_sty, torch.Tensor)
                    else torch.tensor(float(loss_sty))
                ),
                "train/diff": (
                    loss_diff
                    if isinstance(loss_diff, torch.Tensor)
                    else torch.tensor(float(loss_diff))
                ),
            },
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_reference(self, ref_mel: "torch.Tensor") -> "torch.Tensor":
        """Compute a combined style+predictor encoding from a normalised mel.

        ``ref_mel`` should be shape ``[1, n_mels, T]`` and already on
        ``self.device``.  Returns ``ref_s`` of shape ``[1, 256]``.
        """
        style_enc = self.style_encoder(ref_mel.unsqueeze(1))
        predictor_enc = self.predictor_encoder(ref_mel.unsqueeze(1))
        return torch.cat([style_enc, predictor_enc], dim=1)

    @torch.no_grad()
    def _synthesize_text(
        self,
        tokens: "torch.Tensor",
        input_lengths: "torch.Tensor",
        ref_mel: "torch.Tensor | None" = None,
        diffusion_steps: int = 5,
        embedding_scale: float = 1.0,
        acoustic_blend: float = 0.3,
        prosody_blend: float = 0.7,
        ref_s: "torch.Tensor | None" = None,
    ):
        """Run a single text→waveform forward pass.

        All tensors must already be on ``self.device``.
        Exactly one of ``ref_mel`` or ``ref_s`` must be supplied.
        Returns a float32 numpy waveform (shape ``[T]``).
        """
        if ref_s is None:
            assert ref_mel is not None, "Either ref_mel or ref_s must be provided"
            ref_s = self._encode_reference(ref_mel)

        text_mask = length_to_mask(input_lengths).to(self.device)

        bert_dur = self.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        t_en = self.text_encoder(tokens, input_lengths, text_mask)

        noise = torch.randn((1, 256), device=self.device).unsqueeze(1)
        s_pred = self._sampler(
            noise=noise,
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            num_steps=diffusion_steps,
            features=ref_s,
        ).squeeze(1)

        ref = acoustic_blend * s_pred[:, :128] + (1 - acoustic_blend) * ref_s[:, :128]
        s = prosody_blend * s_pred[:, 128:] + (1 - prosody_blend) * ref_s[:, 128:]

        T = int(input_lengths[0].item())
        tm = text_mask[0, :T].unsqueeze(0)
        d = self.predictor.text_encoder(
            d_en[0, :, :T].unsqueeze(0), s, input_lengths, tm
        )
        x, _ = self.predictor.lstm(d)
        duration = torch.sigmoid(self.predictor.duration_proj(x)).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        if pred_dur.ndim == 0:
            pred_dur = pred_dur.unsqueeze(0)
        pred_dur[-1] += 5

        pred_aln = torch.zeros(T, int(pred_dur.sum().item()), device=self.device)
        c = 0
        for i in range(T):
            pred_aln[i, c : c + int(pred_dur[i].item())] = 1
            c += int(pred_dur[i].item())

        en = d.transpose(-1, -2) @ pred_aln.unsqueeze(0)
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        out = self.decoder(
            t_en[0, :, :T].unsqueeze(0) @ pred_aln.unsqueeze(0),
            F0_pred,
            N_pred,
            ref.squeeze().unsqueeze(0),
        )
        return out.cpu().numpy().squeeze()

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        """Lightning predict step for batch synthesis.

        Expects ``batch`` to be a dict with keys: ``raw_text``, ``basename``,
        ``speaker``, ``language``, ``reference_path``, and optional synthesis
        control params (``diffusion_steps``, ``embedding_scale``,
        ``acoustic_blend``, ``prosody_blend``).
        """
        from .text_utils import TextCleaner
        from .utils import _load_reference_mel

        device = self.device
        raw_text = batch["raw_text"]

        text_cleaner = TextCleaner()
        tokens = torch.LongTensor(text_cleaner(raw_text)).unsqueeze(0).to(device)
        if tokens.numel() == 0:
            return None

        input_lengths = torch.LongTensor([tokens.size(1)]).to(device)

        if not hasattr(self, "_mel_transform"):
            from .utils import make_mel_transform

            self._mel_transform = make_mel_transform(self.config).to(device)

        ref_mel = _load_reference_mel(
            batch["reference_path"], self.sr, self._mel_transform
        ).to(device)

        wav = self._synthesize_text(
            tokens,
            input_lengths,
            ref_mel=ref_mel,
            diffusion_steps=batch.get("diffusion_steps", 5),
            embedding_scale=batch.get("embedding_scale", 1.0),
            acoustic_blend=batch.get("acoustic_blend", 0.3),
            prosody_blend=batch.get("prosody_blend", 0.7),
        )

        return {
            "wav": wav,
            "sample_rate": self.sr,
            "basename": batch["basename"],
            "speaker": batch["speaker"],
            "language": batch["language"],
            "raw_text": raw_text,
            "duration_seconds": len(wav) / self.sr,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        if self.mode == "first":
            self._validate_first(batch, batch_idx)
        else:
            self._validate_second(batch, batch_idx)

    def _validate_first(self, batch, batch_idx):
        device = self.device
        waves = batch[0]
        texts, input_lengths, _, _, mels, mel_input_length, _ = [
            b.to(device) for b in batch[1:-1]
        ]
        paths = batch[-1]

        mask = length_to_mask(mel_input_length // (2**self.n_down)).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        ppgs, s2s_pred, s2s_attn = self.text_aligner(mels, mask, texts)
        s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)

        attn_mask = (
            (~mask)
            .unsqueeze(-1)
            .expand(*mask.shape, text_mask.shape[-1])
            .float()
            .transpose(-1, -2)
        )
        attn_mask = attn_mask * (
            (~text_mask).unsqueeze(-1).expand(*text_mask.shape, mask.shape[-1]).float()
        )
        s2s_attn.masked_fill_(attn_mask < 1, 0.0)

        t_en = self.text_encoder(texts, input_lengths, text_mask)
        asr = t_en @ s2s_attn

        mel_len = min(int(mel_input_length.min().item() / 2 - 1), self.max_len // 2)
        clips = self._get_clips(asr, mels, mel_input_length, waves, mel_len)
        en, gt, wav = clips

        F0_real, _, _ = self.pitch_extractor(gt.unsqueeze(1))
        s = self.style_encoder(gt.unsqueeze(1))
        real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
        y_rec = self.decoder(en, F0_real, real_norm, s)
        loss_mel = self.stft_loss(y_rec.squeeze(), wav)

        self.log(
            "val/mel",
            loss_mel,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if self.trainer.is_global_zero:
            new = dict(
                s2s_attn=s2s_attn.detach().cpu(),
                asr=asr.detach().cpu(),
                mels=mels.detach().cpu(),
                mel_input_length=mel_input_length.detach().cpu(),
                waves=waves,
                texts=texts.detach().cpu(),
                input_lengths=input_lengths.detach().cpu(),
                basenames=[Path(p).stem for p in paths],
            )
            if self._val_batch is None:
                self._val_batch = new
            elif len(self._val_batch["asr"]) < self._MAX_VAL_AUDIO:
                n_have = len(self._val_batch["asr"])
                n_take = min(len(new["asr"]), self._MAX_VAL_AUDIO - n_have)
                for key in (
                    "s2s_attn",
                    "asr",
                    "mels",
                    "mel_input_length",
                    "texts",
                    "input_lengths",
                ):
                    self._val_batch[key] = _pad_and_cat(
                        self._val_batch[key], new[key][:n_take]
                    )
                self._val_batch["waves"] = (
                    self._val_batch["waves"] + new["waves"][:n_take]
                )
                self._val_batch["basenames"] = (
                    self._val_batch["basenames"] + new["basenames"][:n_take]
                )

    def _validate_second(self, batch, batch_idx):
        device = self.device
        waves = batch[0]
        texts, input_lengths, _, _, mels, mel_input_length, ref_mels = [
            b.to(device) for b in batch[1:-1]
        ]
        paths = batch[-1]

        mask = length_to_mask(mel_input_length // (2**self.n_down)).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        _, _, s2s_attn = self.text_aligner(mels, mask, texts)
        s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
        mask_ST = mask_from_lens(
            s2s_attn, input_lengths, mel_input_length // (2**self.n_down)
        )
        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        t_en = self.text_encoder(texts, input_lengths, text_mask)
        asr = t_en @ s2s_attn_mono
        d_gt = s2s_attn_mono.sum(axis=-1).detach()

        ss, gs = [], []
        for bib in range(len(mel_input_length)):
            mel = mels[bib, :, : mel_input_length[bib]]
            ss.append(self.predictor_encoder(mel.unsqueeze(0).unsqueeze(1)))
            gs.append(self.style_encoder(mel.unsqueeze(0).unsqueeze(1)))
        s_dur = torch.stack(ss).squeeze()

        bert_dur = self.bert(texts, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        d, p = self.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)

        mel_len = int(mel_input_length.min().item() / 2 - 1)
        clips = self._get_clips(asr, mels, mel_input_length, waves, mel_len, p=p)
        en, gt, wav, p_en = clips

        s = self.predictor_encoder(gt.unsqueeze(1))
        F0_fake, N_fake = self.predictor.F0Ntrain(p_en, s)

        loss_dur = torch.tensor(0.0, device=device)
        for _pred, _text, _length in zip(d, d_gt, input_lengths):
            _pred = _pred[:_length, :]
            _text = _text[:_length].long()
            _trg = torch.zeros_like(_pred)
            for bib in range(_trg.shape[0]):
                _trg[bib, : _text[bib]] = 1
            _dur_pred = torch.sigmoid(_pred).sum(axis=1)
            loss_dur = loss_dur + F.l1_loss(
                _dur_pred[1 : _length - 1], _text[1 : _length - 1]
            )
        loss_dur = loss_dur / texts.size(0)

        s = self.style_encoder(gt.unsqueeze(1))
        y_rec = self.decoder(en, F0_fake, N_fake, s)
        loss_mel = self.stft_loss(y_rec.squeeze(), wav)
        F0_real, _, _ = self.pitch_extractor(gt.unsqueeze(1))
        loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

        self.log_dict(
            {
                "val/mel": loss_mel,
                "val/dur": loss_dur,
                "val/F0": loss_F0,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if self.trainer.is_global_zero:
            new = dict(
                asr=asr.detach().cpu(),
                mels=mels.detach().cpu(),
                mel_input_length=mel_input_length.detach().cpu(),
                waves=waves,
                p=p.detach().cpu(),
                d_en=d_en.detach().cpu(),
                bert_dur=bert_dur.detach().cpu(),
                texts=texts.detach().cpu(),
                input_lengths=input_lengths.detach().cpu(),
                text_mask=text_mask.detach().cpu(),
                ref_mels=ref_mels.detach().cpu() if self.multispeaker else None,
                basenames=[Path(p).stem for p in paths],
            )
            if self._val_batch is None:
                self._val_batch = new
            elif len(self._val_batch["asr"]) < self._MAX_VAL_AUDIO:
                n_have = len(self._val_batch["asr"])
                n_take = min(len(new["asr"]), self._MAX_VAL_AUDIO - n_have)
                for key in (
                    "asr",
                    "mels",
                    "mel_input_length",
                    "p",
                    "d_en",
                    "bert_dur",
                    "texts",
                    "input_lengths",
                    "text_mask",
                ):
                    self._val_batch[key] = _pad_and_cat(
                        self._val_batch[key], new[key][:n_take]
                    )
                self._val_batch["waves"] = (
                    self._val_batch["waves"] + new["waves"][:n_take]
                )
                self._val_batch["basenames"] = (
                    self._val_batch["basenames"] + new["basenames"][:n_take]
                )
                if self.multispeaker and self._val_batch["ref_mels"] is not None:
                    self._val_batch["ref_mels"] = _pad_and_cat(
                        self._val_batch["ref_mels"], new["ref_mels"][:n_take]
                    )
