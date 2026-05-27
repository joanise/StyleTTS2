from pathlib import Path

import typer


def fetch_pretrained(
    config_file: Path = typer.Argument(
        ...,
        help="Path to your EveryVoice StyleTTS2 configuration file.",
        exists=True,
        dir_okay=False,
        file_okay=True,
    ),
):
    """Download StyleTTS2 pretrained model weights from HuggingFace.

    Run this command on a node with internet access before submitting a GPU
    training job.  The files are stored in the HuggingFace hub cache and are
    reused automatically by ``everyvoice train text-to-wav``.

    Example:

    **everyvoice fetch-pretrained text-to-wav config/e2e-text-to-wav-config.yaml**
    """
    from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.ev_config import (
        StyleTTS2Config,
    )
    from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.ev_config.translation import (
        to_native_config,
    )
    from huggingface_hub import hf_hub_download

    ev_config = StyleTTS2Config.load_config_from_path(config_file)
    config = to_native_config(ev_config)

    def _fetch(repo_id: str, filename: str, local_override=None):
        if local_override is not None:
            typer.echo(f"  Using local file: {local_override}")
            return
        typer.echo(f"  {filename}")
        hf_hub_download(repo_id, filename=filename)

    f0_cfg = config["pretrained_f0"]
    typer.echo(f"Fetching F0 extractor from '{f0_cfg['repo_id']}':")
    _fetch(f0_cfg["repo_id"], f0_cfg["filename"], f0_cfg.get("local_path"))

    asr_cfg = config["pretrained_asr"]
    typer.echo(f"Fetching ASR aligner from '{asr_cfg['repo_id']}':")
    _fetch(asr_cfg["repo_id"], asr_cfg["config_filename"], asr_cfg.get("local_config"))
    _fetch(
        asr_cfg["repo_id"],
        asr_cfg["checkpoint_filename"],
        asr_cfg.get("local_checkpoint"),
    )

    plbert_cfg = config["pretrained_plbert"]
    typer.echo(f"Fetching PLBERT from '{plbert_cfg['repo_id']}':")
    _fetch(
        plbert_cfg["repo_id"],
        plbert_cfg["config_filename"],
        plbert_cfg.get("local_config"),
    )
    _fetch(
        plbert_cfg["repo_id"],
        plbert_cfg["checkpoint_filename"],
        plbert_cfg.get("local_checkpoint"),
    )

    slm_model = config["model_params"]["slm"]["model"]
    typer.echo(f"Fetching WavLM SLM from '{slm_model}':")
    typer.echo(f"  {slm_model}")
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    # Use snapshot_download to a flat local directory rather than relying on
    # AutoModel.from_pretrained's standard HF cache.  The other pretrained
    # models (F0, ASR, PLBERT) are fetched with hf_hub_download for a specific
    # filename, so they bypass transformers entirely.  WavLM is the only model
    # loaded via AutoModel.from_pretrained, which always runs transformers'
    # format-detection logic: try model.safetensors first, get 404, then do a
    # multi-request PR-discovery search.  Even when the files are cached, this
    # discovery runs again because the weights land in the HF cache under the PR
    # commit hash while from_pretrained resolves from the main commit hash, so
    # they never match.  Downloading to local_dir bypasses the revision-keyed
    # cache entirely: training calls from_pretrained("<path>") on a plain
    # directory, which skips all online revision resolution.
    _slm_local_dir = (
        Path(HF_HUB_CACHE) / "everyvoice-wavlm" / slm_model.replace("/", "--")
    )
    snapshot_download(repo_id=slm_model, local_dir=str(_slm_local_dir))

    typer.echo("Done. All pretrained models are cached and ready for training.")
