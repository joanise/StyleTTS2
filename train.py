import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from styletts2.lightning import StyleTTS2DataModule, StyleTTS2Module


class Mode(str, Enum):
    first = 'first'
    second = 'second'
    finetune = 'finetune'


def main(
    config_path: Annotated[Path, typer.Option('-p', '--config',
        help='Path to YAML config file.')] = Path('configs/base.yml'),
    mode: Annotated[Mode, typer.Option('-m', '--mode',
        help='Training mode.')] = Mode.first,
    resume: Annotated[Optional[Path], typer.Option(
        help='Path to a Lightning checkpoint to resume from.')] = None,
    devices: Annotated[int, typer.Option(
        help='Number of GPUs to use.')] = 1,
    strategy: Annotated[str, typer.Option(
        help='Lightning strategy (auto, ddp, deepspeed, etc.).')] = 'auto',
    precision: Annotated[str, typer.Option(
        help='Floating-point precision (32, 16-mixed, bf16-mixed).')] = '32',
):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    epochs_key = 'epochs_1st' if mode == Mode.first else ('epochs_2nd' if mode == Mode.second else 'epochs')
    max_epochs = config.get(epochs_key, 200)

    datamodule = StyleTTS2DataModule(config)
    model = StyleTTS2Module(config, mode=mode.value)

    tb_logger = TensorBoardLogger(save_dir=log_dir, name='tensorboard', version='')

    checkpoint_cb = ModelCheckpoint(
        dirpath=log_dir,
        filename=f'epoch_{mode.value[0]}_{"{epoch:05d}"}',
        every_n_epochs=config.get('save_freq', 2),
        save_top_k=-1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # GAN training uses separate discriminator and generator backward passes per
    # step, so different subsets of trainable parameters participate in each
    # backward. find_unused_parameters=True is required for DDP correctness here.
    # The freezing done in setup() already removes the truly-never-trained networks
    # from DDP's parameter list, keeping the graph traversal cheap.
    strategy_str = strategy
    if strategy_str == 'ddp' or (devices > 1 and strategy_str == 'auto'):
        resolved_strategy = DDPStrategy(find_unused_parameters=True)
    else:
        resolved_strategy = strategy_str

    trainer = L.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        strategy=resolved_strategy,
        precision=precision,
        logger=tb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=config.get('log_interval', 10),
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume)


if __name__ == '__main__':
    typer.run(main)
