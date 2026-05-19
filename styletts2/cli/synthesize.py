"""Core synthesis helpers for StyleTTS2.

Shared by the `everyvoice synthesize text-to-wav` CLI command
and the `everyvoice demo text-to-wav` Gradio app.
"""

from __future__ import annotations

from pathlib import Path

import typer
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import (
    SynthesizeOutputFormats,
)
from loguru import logger


def load_styletts2_model(model_path: Path, device):
    """Load a StyleTTS2 Lightning module and mel transform from a checkpoint."""
    import torch

    from .lightning import StyleTTS2Module
    from .utils import make_mel_transform

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    hp = state.get("hyper_parameters", {})
    native_config = hp["config"]
    mode = hp.get("mode", "second")

    module = StyleTTS2Module(native_config, mode=mode)
    module.check_and_upgrade_checkpoint(state)
    module.load_state_dict(state["state_dict"])
    module.eval()
    module.to(device)

    mel_transform = make_mel_transform(native_config).to(device)
    return module, mel_transform


def load_reference_style(
    module,
    mel_transform,
    reference_path: Path,
    device,
):
    """Load a reference audio file and return a pre-computed style encoding.

    Runs ``_load_reference_mel`` then ``module._encode_reference``, returning
    ``ref_s`` of shape ``[1, 256]`` on ``device``.  Call this at startup to
    avoid re-computing on every synthesis request.
    """
    import torch

    from .utils import (
        _load_reference_mel,
    )

    with torch.no_grad():
        ref_mel = _load_reference_mel(reference_path, module.sr, mel_transform).to(
            device
        )
        return module._encode_reference(ref_mel)


def synthesize_one(
    module,
    mel_transform,
    text: str,
    device,
    reference_path: Path,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
    acoustic_blend: float = 0.3,
    prosody_blend: float = 0.7,
):
    """Synthesize a single utterance and return a float32 numpy waveform.

    Works only with stage-2 (or finetune) checkpoints that include the
    diffusion sampler.  Stage-1 checkpoints will raise an AttributeError
    because ``module._sampler`` does not exist.
    """
    import torch

    from .text_utils import (
        TextCleaner,
    )
    from .utils import (
        _load_reference_mel,
    )

    with torch.no_grad():
        text_cleaner = TextCleaner()
        tokens = torch.LongTensor(text_cleaner(text)).unsqueeze(0).to(device)
        if tokens.numel() == 0:
            raise ValueError(f"Text produced no tokens: {text!r}")

        input_lengths = torch.LongTensor([tokens.size(1)]).to(device)
        ref_mel = _load_reference_mel(reference_path, module.sr, mel_transform).to(
            device
        )

        return module._synthesize_text(
            tokens,
            input_lengths,
            ref_mel=ref_mel,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            acoustic_blend=acoustic_blend,
            prosody_blend=prosody_blend,
        )


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command(
    name="text-to-wav",
    short_help="Synthesize audio from text using a trained StyleTTS2 model",
)
def synthesize(
    model_path: Path = typer.Argument(
        ...,
        help="Path to a trained StyleTTS2 checkpoint (.ckpt).",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    reference: Path = typer.Option(
        ...,
        "--reference",
        "-r",
        help="Reference audio file used to extract speaker style.",
        exists=True,
    ),
    text: list[str] = typer.Option(
        ...,
        "--text",
        "-t",
        help="Text string(s) to synthesize. Repeat the flag for multiple utterances.",
    ),
    output_dir: Path = typer.Option(
        Path("synthesis_output"),
        "--output-dir",
        "-o",
        help="Directory where synthesized files will be written.",
    ),
    output_type: list[SynthesizeOutputFormats] = typer.Option(
        [SynthesizeOutputFormats.wav],
        "--output-type",
        help="Output format(s) to produce.",
    ),
    accelerator: str = typer.Option(
        "auto",
        "--accelerator",
        help="Lightning accelerator: 'cpu', 'gpu', or 'auto'.",
    ),
    speaker: str = typer.Option(
        "default",
        "--speaker",
        "-s",
        help="Speaker label written into output filenames.",
    ),
    language: str = typer.Option(
        "und",
        "--language",
        "-l",
        help="Language tag written into output filenames.",
    ),
    diffusion_steps: int = typer.Option(
        5,
        "--diffusion-steps",
        help="Number of diffusion sampling steps (higher = slower but smoother).",
    ),
    embedding_scale: float = typer.Option(
        1.0,
        "--embedding-scale",
        help="Classifier-free guidance scale for the diffusion sampler.",
    ),
    acoustic_blend: float = typer.Option(
        0.3,
        "--acoustic-blend",
        help="Blend weight for acoustic style (0 = pure reference, 1 = pure diffusion).",
    ),
    prosody_blend: float = typer.Option(
        0.7,
        "--prosody-blend",
        help="Blend weight for prosody style (0 = pure reference, 1 = pure diffusion).",
    ),
):
    """Synthesize audio from text using a trained StyleTTS2 model.

    Example:

    **everyvoice synthesize text-to-wav logs_and_checkpoints/.../last.ckpt \\
        --reference path/to/reference.wav \\
        --text "Hello world" --text "How are you?"**
    """
    import lightning as L
    import torch
    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.utils import (
        truncate_basename,
    )
    from everyvoice.utils import slugify

    from .utils_heavy import (
        StyleTTS2SynthesisDataModule,
        get_styletts2_synthesis_output_callbacks,
    )

    device = torch.device(
        "cuda"
        if (
            accelerator == "gpu"
            or (accelerator == "auto" and torch.cuda.is_available())
        )
        else "cpu"
    )

    logger.info(f"Loading StyleTTS2 model from {model_path}")
    module, mel_transform = load_styletts2_model(model_path, device)
    module._mel_transform = mel_transform

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    global_step = int(state.get("global_step", 0))

    entries = [
        {
            "raw_text": t,
            "basename": truncate_basename(slugify(t)),
            "speaker": speaker,
            "language": language,
            "reference_path": str(reference),
            "diffusion_steps": diffusion_steps,
            "embedding_scale": embedding_scale,
            "acoustic_blend": acoustic_blend,
            "prosody_blend": prosody_blend,
        }
        for t in text
    ]

    callbacks = get_styletts2_synthesis_output_callbacks(
        output_type, output_dir, global_step, module.sr
    )
    if not callbacks:
        logger.warning("No output format requested; nothing to do.")
        return

    datamodule = StyleTTS2SynthesisDataModule(entries)

    trainer = L.Trainer(
        accelerator=accelerator,
        callbacks=list(callbacks.values()),
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    trainer.predict(module, datamodule=datamodule)

    logger.info(f"Synthesis complete. Output saved to {output_dir}")
