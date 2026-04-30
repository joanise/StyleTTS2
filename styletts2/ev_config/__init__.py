from pathlib import Path
from typing import Annotated, Any, Optional, Union

from everyvoice.config.preprocessing_config import PreprocessingConfig
from everyvoice.config.shared_types import (
    BaseModelWithContact,
    BaseTrainingConfig,
    ConfigModel,
    init_context,
)
from everyvoice.config.text_config import TextConfig
from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
from everyvoice.config.utils import PossiblyRelativePath, load_partials
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import (
    HiFiGANModelConfig,
)
from everyvoice.utils import load_config_from_json_or_yaml_path
from pydantic import Field, FilePath, ValidationInfo, model_validator

LATEST_VERSION: str = "0.1"


def _default_pretrained_symbols() -> list[str]:
    """Return the symbol list bundled with the pretrained StyleTTS2 text encoder."""
    from styletts2.text_utils import (
        symbols,
    )
    return list(symbols)


# ---------------------------------------------------------------------------
# Pretrained backbone paths
# ---------------------------------------------------------------------------


class StyleTTS2PretrainedConfig(ConfigModel):
    """Paths to the frozen pretrained models bundled with StyleTTS2."""

    f0_path: PossiblyRelativePath = Field(
        default=Path("styletts2/pretrained/jdc/bst.t7"),
        description="Path to the JDC F0 extractor checkpoint.",
    )
    asr_config: PossiblyRelativePath = Field(
        default=Path("styletts2/pretrained/asr/config.yml"),
        description="Path to the ASR model config.",
    )
    asr_path: PossiblyRelativePath = Field(
        default=Path("styletts2/pretrained/asr/epoch_00080.pth"),
        description="Path to the ASR model checkpoint.",
    )
    plbert_dir: PossiblyRelativePath = Field(
        default=Path("styletts2/pretrained/plbert"),
        description="Directory containing the PLBERT checkpoint and config.",
    )
    pretrained_symbols: list[str] = Field(
        default_factory=lambda: _default_pretrained_symbols(),
        description=(
            "Ordered symbol list that matches the pretrained text-encoder embedding table. "
            "The index of each symbol here is its embedding-table row. "
            "Override only if you are using a custom pretrained text encoder."
        ),
    )


# ---------------------------------------------------------------------------
# Decoder (reuses HiFiGANModelConfig, extended for iSTFTNet)
# ---------------------------------------------------------------------------


class StyleTTS2DecoderConfig(HiFiGANModelConfig):
    """HiFiGAN/iSTFTNet decoder config for StyleTTS2.

    Inherits all HiFiGAN fields. ``istft_layer=True`` selects iSTFTNet;
    ``istft_layer=False`` selects plain HiFiGAN. The extra iSTFT fields
    are only used when ``istft_layer=True``.

    Note: ``activation_function``, ``msd_layers``, and ``mpd_layers`` are
    inherited but not forwarded to StyleTTS2 — discriminators are configured
    separately inside the StyleTTS2 model builder.
    """

    # iSTFTNet defaults (base.yml single-speaker)
    # These values are already defined in HiFiGANModelConfig, but they need different defaults.
    upsample_rates: list[int] = Field(default=[10, 6])
    upsample_kernel_sizes: list[int] = Field(default=[20, 12])
    istft_layer: bool = Field(
        default=True,
    )
    # TODO: Are these necessary?
    gen_istft_n_fft: int = Field(
        default=20,
        description="FFT size for the iSTFTNet generator. Only used when istft_layer=True.",
    )
    gen_istft_hop_size: int = Field(
        default=5,
        description="Hop size for the iSTFTNet generator. Only used when istft_layer=True.",
    )


# ---------------------------------------------------------------------------
# SLM (speech language model)
# ---------------------------------------------------------------------------


class StyleTTS2SLMConfig(ConfigModel):
    model: str = Field(
        default="microsoft/wavlm-base-plus",
        description="HuggingFace model ID for the speech language model.",
    )
    sr: int = Field(default=16000, description="Sampling rate expected by the SLM.")
    hidden: int = Field(default=768, description="Hidden size of the SLM.")
    nlayers: int = Field(default=13, description="Number of layers in the SLM.")
    initial_channel: int = Field(
        default=64, description="Initial channels of the SLM discriminator head."
    )


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------


class StyleTTS2DiffusionTransformerConfig(ConfigModel):
    num_layers: int = 3
    num_heads: int = 8
    head_features: int = 64
    multiplier: int = 2


class StyleTTS2DiffusionDistConfig(ConfigModel):
    sigma_data: float = Field(
        default=0.2,
        description="Placeholder when estimate_sigma_data=False.",
    )
    estimate_sigma_data: bool = Field(
        default=True,
        description="Estimate sigma_data from each batch instead of using the fixed value.",
    )
    mean: float = -3.0
    std: float = 1.0


class StyleTTS2DiffusionConfig(ConfigModel):
    embedding_mask_proba: float = 0.1
    transformer: StyleTTS2DiffusionTransformerConfig = Field(
        default_factory=StyleTTS2DiffusionTransformerConfig
    )
    dist: StyleTTS2DiffusionDistConfig = Field(
        default_factory=StyleTTS2DiffusionDistConfig
    )


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


class StyleTTS2ModelConfig(ConfigModel):
    """Architecture hyperparameters for StyleTTS2.

    Note: ``n_mels`` and ``n_token`` are intentionally absent — they are
    derived at runtime from ``preprocessing.audio.n_mels`` and
    ``len(TextProcessor(text).symbols)`` respectively in the translation layer.
    """

    multispeaker: bool = Field(
        default=False,
        description="Enable multi-speaker conditioning.",
    )
    target_text_representation_level: TargetTrainingTextRepresentationLevel = Field(
        default=TargetTrainingTextRepresentationLevel.characters,
        description=(
            "Whether to train on 'characters' or 'ipa_phones'. "
            "Must match the representation used during everyvoice preprocess."
        ),
    )
    dim_in: int = Field(default=64, description="Input channel dimension.")
    hidden_dim: int = Field(default=512, description="Main hidden dimension.")
    max_conv_dim: int = Field(default=512, description="Maximum convolutional dimension.")
    n_layer: int = Field(default=3, description="Number of style encoder layers.")
    max_dur: int = Field(
        default=50, description="Maximum duration (frames) for a single phoneme."
    )
    style_dim: int = Field(default=128, description="Style vector dimension.")
    dropout: float = Field(default=0.2)
    decoder: StyleTTS2DecoderConfig = Field(default_factory=StyleTTS2DecoderConfig)
    slm: StyleTTS2SLMConfig = Field(default_factory=StyleTTS2SLMConfig)
    diffusion: StyleTTS2DiffusionConfig = Field(default_factory=StyleTTS2DiffusionConfig)


# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------


class StyleTTS2LossConfig(ConfigModel):
    lambda_mel: float = Field(default=5.0, description="Mel reconstruction loss weight.")
    lambda_gen: float = Field(default=1.0, description="Generator (GAN) loss weight.")
    lambda_slm: float = Field(default=1.0, description="SLM feature-matching loss weight.")
    lambda_mono: float = Field(
        default=1.0, description="Monotonic alignment loss weight (stage 1 TMA)."
    )
    lambda_s2s: float = Field(
        default=1.0, description="Sequence-to-sequence loss weight (stage 1 TMA)."
    )
    tma_epoch: int = Field(
        default=50, description="Epoch at which TMA training begins (stage 1)."
    )
    lambda_f0: float = Field(
        default=1.0, description="F0 reconstruction loss weight (stage 2)."
    )
    lambda_norm: float = Field(
        default=1.0, description="Norm reconstruction loss weight (stage 2)."
    )
    lambda_dur: float = Field(default=1.0, description="Duration loss weight (stage 2).")
    lambda_ce: float = Field(
        default=20.0,
        description="Duration predictor cross-entropy loss weight (stage 2).",
    )
    lambda_sty: float = Field(
        default=1.0, description="Style reconstruction loss weight (stage 2)."
    )
    lambda_diff: float = Field(
        default=1.0, description="Score-matching (diffusion) loss weight (stage 2)."
    )
    diff_epoch: int = Field(
        default=20, description="Epoch at which style diffusion training begins (stage 2)."
    )
    joint_epoch: int = Field(
        default=50,
        description="Epoch at which joint (decoder + style encoder) training begins (stage 2).",
    )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class StyleTTS2OptimizerConfig(ConfigModel):
    lr: float = Field(default=1e-4, description="General learning rate.")
    bert_lr: float = Field(default=1e-5, description="Learning rate for PLBERT.")
    ft_lr: float = Field(
        default=1e-5,
        description="Learning rate for acoustic modules during stage 2 / fine-tuning.",
    )


# ---------------------------------------------------------------------------
# SLM adversarial training
# ---------------------------------------------------------------------------


class StyleTTS2SLMAdvConfig(ConfigModel):
    min_len: int = Field(default=100, description="Minimum sample length for SLM adversarial loss.")
    max_len: int = Field(default=500, description="Maximum sample length for SLM adversarial loss.")
    batch_percentage: float = Field(
        default=0.5,
        description="Fraction of the batch to use for SLM adversarial loss (to save memory).",
    )
    iter: int = Field(
        default=10,
        description="Update the SLM discriminator every this many generator steps.",
    )
    thresh: float = Field(
        default=5.0,
        description="Gradient norm threshold above which gradients are scaled.",
    )
    scale: float = Field(
        default=0.01,
        description="Gradient scaling factor applied to predictor/diffusion gradients from the SLM discriminator.",
    )
    sig: float = Field(
        default=1.5, description="Sigma for differentiable duration modelling."
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class StyleTTS2TrainingConfig(BaseTrainingConfig):
    """Training hyperparameters for StyleTTS2.

    Inherits shared fields from BaseTrainingConfig: batch_size,
    save_top_k_ckpts, ckpt_steps/ckpt_epochs, val_check_interval,
    max_epochs/max_steps, finetune_checkpoint (→ pretrained_model),
    training_filelist (→ data_params.train_data),
    validation_filelist (→ data_params.val_data), logger, and data workers.
    """

    # TODO: Do we even need stage 1 and stage 2 to be separated from the training routine now that multi-gpu training is possible?
    epochs_1st: int = Field(
        default=200, description="Number of epochs for stage 1 (pre-training)."
    )
    epochs_2nd: int = Field(
        default=100, description="Number of epochs for stage 2 (joint training)."
    )
    max_len: int = Field(
        default=400,
        description="Maximum clip length in mel frames used during training.",
    )
    first_stage_path: str = Field(
        default="first_stage.pth",
        description="Filename (relative to log_dir) where the stage 1 checkpoint is saved.",
    )
    second_stage_load_pretrained: bool = Field(
        default=True,
        description="If True, load the stage 1 checkpoint when starting stage 2 training.",
    )
    load_only_params: bool = Field(
        default=False,
        description="If True, load only model parameters from a checkpoint (skip optimizer state and epoch number).",
    )
    root_path: PossiblyRelativePath = Field(
        default=Path("."),
        description="Root directory that audio file paths in the filelist are relative to.",
    )
    ood_data: PossiblyRelativePath = Field(
        default=Path("data/OOD_texts.txt"),
        description="Path to out-of-distribution texts used for validation audio generation.",
    )
    min_length: int = Field(
        default=50,
        description="Minimum text length (characters) when sampling OOD validation texts.",
    )
    optimizer: StyleTTS2OptimizerConfig = Field(
        default_factory=StyleTTS2OptimizerConfig
    )
    losses: StyleTTS2LossConfig = Field(default_factory=StyleTTS2LossConfig)
    slmadv: StyleTTS2SLMAdvConfig = Field(default_factory=StyleTTS2SLMAdvConfig)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class StyleTTS2Config(BaseModelWithContact):
    VERSION: Annotated[str, Field(init_var=False)] = LATEST_VERSION

    model: StyleTTS2ModelConfig = Field(
        default_factory=StyleTTS2ModelConfig,
        description="Model architecture configuration.",
    )
    path_to_model_config_file: Optional[FilePath] = Field(
        default=None, description="Path to an external model configuration file."
    )

    training: StyleTTS2TrainingConfig = Field(
        default_factory=StyleTTS2TrainingConfig,
        description="Training hyperparameter configuration.",
    )
    path_to_training_config_file: Optional[FilePath] = Field(
        default=None, description="Path to an external training configuration file."
    )

    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Preprocessing configuration (audio settings, dataset paths).",
    )
    path_to_preprocessing_config_file: Optional[FilePath] = Field(
        default=None, description="Path to an external preprocessing configuration file."
    )

    text: TextConfig = Field(
        default_factory=TextConfig,
        description="Text processing configuration (symbols, G2P, cleaners).",
    )
    path_to_text_config_file: Optional[FilePath] = Field(
        default=None, description="Path to an external text configuration file."
    )

    pretrained: StyleTTS2PretrainedConfig = Field(
        default_factory=StyleTTS2PretrainedConfig,
        description="Paths to frozen pretrained backbone models.",
    )

    @model_validator(mode="after")
    def check_symbols_in_pretrained(self) -> "StyleTTS2Config":
        from everyvoice.text.text_processor import TextProcessor

        tp = TextProcessor(self.text)
        pretrained_set = set(self.pretrained.pretrained_symbols)
        # Only validate user-declared content symbols (letters, IPA phones, etc.).
        # Internal punctuation tokens (<EXCL>, <COMMA>, …) are handled by the
        # EVStyleTTS2TextEncoder remapping layer and don't need to be in the
        # pretrained table.  <SIL> is dropped silently at encode time.
        _skip = {tp._pad_symbol, "<SIL>"}
        _skip |= set(tp.punctuation_internal_hash.values())
        ev_symbols = [
            s for s in tp.config.symbols.all_except_punctuation if s not in _skip
        ]
        missing = [s for s in ev_symbols if s not in pretrained_set]
        if missing:
            raise ValueError(
                f"The following symbols declared in your TextConfig are not present in "
                f"the pretrained StyleTTS2 text-encoder symbol table: {missing}. "
                f"Either remove them from your TextConfig or use a custom pretrained_symbols list."
            )
        return self

    @model_validator(mode="before")  # type: ignore
    def load_partials(self: dict[Any, Any], info: ValidationInfo):  # type: ignore[misc]
        config_path = (
            info.context.get("config_path", None) if info.context is not None else None
        )
        return load_partials(
            self,
            ("model", "training", "preprocessing", "text"),
            config_path=config_path,
        )

    @model_validator(mode="before")
    @classmethod
    def check_and_upgrade_checkpoint(cls, data: Any) -> Any:
        from packaging.version import Version

        ckpt_version = Version(data.get("VERSION", "0.0"))
        if ckpt_version > Version(LATEST_VERSION):
            raise ValueError(
                "Your config was created with a newer version of EveryVoice. "
                "Please update your software."
            )
        return data

    @staticmethod
    def load_config_from_path(path: Path) -> "StyleTTS2Config":
        config = load_config_from_json_or_yaml_path(path)
        with init_context({"config_path": path}):
            config = StyleTTS2Config(**config)
        return config
