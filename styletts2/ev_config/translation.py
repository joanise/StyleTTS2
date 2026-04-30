"""Translate a StyleTTS2Config into the native config dict that StyleTTS2's
Lightning module and model builder expect.

This is the single place where EveryVoice field names are mapped to
StyleTTS2's internal YAML keys. Keep all key-name gymnastics here so
the rest of the codebase can work purely with StyleTTS2Config objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from everyvoice.text.text_processor import TextProcessor

if TYPE_CHECKING:
    from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.ev_config import (
        StyleTTS2Config,
    )


def to_native_config(config: StyleTTS2Config) -> dict:
    """Return a native StyleTTS2 config dict derived from a StyleTTS2Config.

    Derived quantities:
    - ``n_mels``  ← preprocessing.audio.n_mels
    - ``n_token`` ← len(TextProcessor(text).symbols)

    These are intentionally absent from StyleTTS2ModelConfig so there is a
    single source of truth for each value.
    """
    audio = config.preprocessing.audio
    tr = config.training
    m = config.model
    pre = config.pretrained

    decoder_dict: dict = {
        "type": "istftnet" if m.decoder.istft_layer else "hifigan",
        "resblock_kernel_sizes": m.decoder.resblock_kernel_sizes,
        "upsample_rates": m.decoder.upsample_rates,
        "upsample_initial_channel": m.decoder.upsample_initial_channel,
        "resblock_dilation_sizes": m.decoder.resblock_dilation_sizes,
        "upsample_kernel_sizes": m.decoder.upsample_kernel_sizes,
    }
    if m.decoder.istft_layer:
        decoder_dict["gen_istft_n_fft"] = m.decoder.gen_istft_n_fft
        decoder_dict["gen_istft_hop_size"] = m.decoder.gen_istft_hop_size

    return {
        # --- top-level training bookkeeping ---
        "log_dir": str(Path(tr.logger.save_dir) / tr.logger.name / tr.logger.version),
        "first_stage_path": tr.first_stage_path,
        "save_freq": tr.ckpt_epochs if tr.ckpt_epochs is not None else 1,
        "log_interval": 10,
        "epochs_1st": tr.epochs_1st,
        "epochs_2nd": tr.epochs_2nd,
        "batch_size": tr.batch_size,
        "max_len": tr.max_len,
        "pretrained_model": str(tr.finetune_checkpoint) if tr.finetune_checkpoint else "",
        "second_stage_load_pretrained": tr.second_stage_load_pretrained,
        "load_only_params": tr.load_only_params,
        # --- pretrained backbone paths ---
        "F0_path": str(pre.f0_path),
        "ASR_config": str(pre.asr_config),
        "ASR_path": str(pre.asr_path),
        "PLBERT_dir": str(pre.plbert_dir),
        # --- data ---
        "data_params": {
            "train_data": str(tr.training_filelist),
            "val_data": str(tr.validation_filelist),
            "root_path": str(tr.root_path),
            "OOD_data": str(tr.ood_data),
            "min_length": tr.min_length,
            "target_text_representation": m.target_text_representation_level.value,
        },
        # --- audio preprocessing ---
        "preprocess_params": {
            "sr": audio.output_sampling_rate,
            "spect_params": {
                "n_fft": audio.n_fft,
                "win_length": audio.fft_window_size,
                "hop_length": audio.fft_hop_size,
            },
        },
        # --- model architecture ---
        "model_params": {
            "multispeaker": m.multispeaker,
            "dim_in": m.dim_in,
            "hidden_dim": m.hidden_dim,
            "max_conv_dim": m.max_conv_dim,
            "n_layer": m.n_layer,
            # Derived from preprocessing and text configs:
            "n_mels": audio.n_mels,
            "n_token": len(TextProcessor(config.text).symbols),
            "max_dur": m.max_dur,
            "style_dim": m.style_dim,
            "dropout": m.dropout,
            "decoder": decoder_dict,
            "slm": {
                "model": m.slm.model,
                "sr": m.slm.sr,
                "hidden": m.slm.hidden,
                "nlayers": m.slm.nlayers,
                "initial_channel": m.slm.initial_channel,
            },
            "diffusion": {
                "embedding_mask_proba": m.diffusion.embedding_mask_proba,
                "transformer": {
                    "num_layers": m.diffusion.transformer.num_layers,
                    "num_heads": m.diffusion.transformer.num_heads,
                    "head_features": m.diffusion.transformer.head_features,
                    "multiplier": m.diffusion.transformer.multiplier,
                },
                "dist": {
                    "sigma_data": m.diffusion.dist.sigma_data,
                    "estimate_sigma_data": m.diffusion.dist.estimate_sigma_data,
                    "mean": m.diffusion.dist.mean,
                    "std": m.diffusion.dist.std,
                },
            },
        },
        # --- loss weights ---
        "loss_params": {
            "lambda_mel": tr.losses.lambda_mel,
            "lambda_gen": tr.losses.lambda_gen,
            "lambda_slm": tr.losses.lambda_slm,
            "lambda_mono": tr.losses.lambda_mono,
            "lambda_s2s": tr.losses.lambda_s2s,
            "TMA_epoch": tr.losses.tma_epoch,
            "lambda_F0": tr.losses.lambda_f0,
            "lambda_norm": tr.losses.lambda_norm,
            "lambda_dur": tr.losses.lambda_dur,
            "lambda_ce": tr.losses.lambda_ce,
            "lambda_sty": tr.losses.lambda_sty,
            "lambda_diff": tr.losses.lambda_diff,
            "diff_epoch": tr.losses.diff_epoch,
            "joint_epoch": tr.losses.joint_epoch,
        },
        # --- optimizers ---
        "optimizer_params": {
            "lr": tr.optimizer.lr,
            "bert_lr": tr.optimizer.bert_lr,
            "ft_lr": tr.optimizer.ft_lr,
        },
        # --- SLM adversarial training ---
        "slmadv_params": {
            "min_len": tr.slmadv.min_len,
            "max_len": tr.slmadv.max_len,
            "batch_percentage": tr.slmadv.batch_percentage,
            "iter": tr.slmadv.iter,
            "thresh": tr.slmadv.thresh,
            "scale": tr.slmadv.scale,
            "sig": tr.slmadv.sig,
        },
    }
