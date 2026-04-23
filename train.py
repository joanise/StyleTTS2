import os
import shutil

import click
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from styletts2.lightning import StyleTTS2DataModule, StyleTTS2Module


@click.command()
@click.option('-p', '--config_path', default='configs/base.yml', type=str,
              help='Path to YAML config file.')
@click.option('-m', '--mode', default='first', type=click.Choice(['first', 'second', 'finetune']),
              help='Training mode.')
@click.option('--resume', default=None, type=str,
              help='Path to a Lightning checkpoint to resume from.')
@click.option('--devices', default=1, type=int,
              help='Number of GPUs to use.')
@click.option('--strategy', default='auto', type=str,
              help='Lightning strategy (auto, ddp, deepspeed, etc.).')
@click.option('--precision', default='32', type=str,
              help='Floating-point precision (32, 16-mixed, bf16-mixed).')
def main(config_path, mode, resume, devices, strategy, precision):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    epochs_key = 'epochs_1st' if mode == 'first' else ('epochs_2nd' if mode == 'second' else 'epochs')
    max_epochs = config.get(epochs_key, 200)

    datamodule = StyleTTS2DataModule(config)
    model = StyleTTS2Module(config, mode=mode)

    tb_logger = TensorBoardLogger(save_dir=log_dir, name='tensorboard', version='')

    checkpoint_cb = ModelCheckpoint(
        dirpath=log_dir,
        filename=f'epoch_{mode[0]}_{"{epoch:05d}"}',
        every_n_epochs=config.get('save_freq', 2),
        save_top_k=-1,          # keep all checkpoints
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=tb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=config.get('log_interval', 10),
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume)


if __name__ == '__main__':
    main()
