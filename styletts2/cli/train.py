import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from everyvoice.base_cli.interfaces import train_base_command_interface
from everyvoice.utils import spinner
from merge_args import merge_args


class Mode(str, Enum):
    first = "first"
    second = "second"
    finetune = "finetune"


@merge_args(train_base_command_interface)
def train(
    mode: Annotated[
        Mode,
        typer.Option(
            "-m",
            "--mode",
            help="Training mode: 'first' (acoustic pre-training with TMA), 'second' (joint diffusion+adversarial), or 'finetune'.",
        ),
    ] = Mode.first,
    precision: Annotated[
        str,
        typer.Option(
            help="Floating-point precision passed to Lightning Trainer (e.g. '32', '16-mixed', 'bf16-mixed').",
        ),
    ] = "32",
    **kwargs,
):
    """Train a StyleTTS2 end-to-end TTS model."""
    with spinner():
        from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.ev_config import (
            StyleTTS2Config,
        )
        from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.ev_config.translation import (
            to_native_config,
        )
        from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.lightning import (
            StyleTTS2DataModule,
            StyleTTS2Module,
        )
        from everyvoice.utils import update_config_from_cli_args

        import lightning as L
        from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
        from lightning.pytorch.loggers import TensorBoardLogger
        from lightning.pytorch.strategies import DDPStrategy

    config_file: Path = kwargs["config_file"]
    config_args: list[str] = kwargs.get("config_args", [])

    ev_config = StyleTTS2Config.load_config_from_path(config_file)
    ev_config = update_config_from_cli_args(config_args, ev_config)

    config = to_native_config(ev_config)

    tr = ev_config.training
    max_epochs = (
        tr.epochs_1st
        if mode == Mode.first
        else tr.epochs_2nd
        if mode == Mode.second
        else tr.max_epochs
    )

    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(str(config_file), os.path.join(log_dir, config_file.name))

    tb_logger = TensorBoardLogger(save_dir=log_dir, name="tensorboard", version="")

    checkpoint_cb = ModelCheckpoint(
        dirpath=log_dir,
        filename=f"epoch_{mode.value[0]}_" + "{epoch:05d}",
        every_n_epochs=config.get("save_freq", 2),
        save_top_k=-1,
        save_last=True,
        monitor="val/mel",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    devices = kwargs.get("devices", "auto")
    strategy = kwargs.get("strategy", "ddp")

    # GAN training uses separate discriminator/generator backward passes,
    # so find_unused_parameters=True is required for DDP correctness.
    try:
        n_devices = int(devices)
        multi_gpu = n_devices > 1
    except (TypeError, ValueError):
        multi_gpu = devices not in ("auto", "1", 1)

    if strategy == "ddp" or (multi_gpu and strategy == "auto"):
        resolved_strategy = DDPStrategy(find_unused_parameters=True)
    else:
        resolved_strategy = strategy

    trainer = L.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        num_nodes=kwargs.get("nodes", 1),
        accelerator=kwargs.get("accelerator", "auto"),
        strategy=resolved_strategy,
        precision=precision,
        logger=tb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=config.get("log_interval", 10),
        enable_progress_bar=True,
    )

    datamodule = StyleTTS2DataModule(config)
    model = StyleTTS2Module(config, mode=mode.value)

    resume_ckpt = (
        str(tr.finetune_checkpoint)
        if tr.finetune_checkpoint and os.path.exists(tr.finetune_checkpoint)
        else None
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
