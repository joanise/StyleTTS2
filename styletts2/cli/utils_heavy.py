"""Heavy synthesis helpers for StyleTTS2 — imported lazily to keep CLI startup fast."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import lightning as L
import torch
import torchaudio
from everyvoice.base_cli.prediction_writing_callback import (
    BasePredictionWritingCallback,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import (
    SynthesizeOutputFormats,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.utils import (
    truncate_basename,
)
from everyvoice.utils import slugify
from loguru import logger
from torch.utils.data import DataLoader, Dataset


def _synthesis_collate_fn(batch):
    assert len(batch) == 1, "StyleTTS2 synthesis requires batch_size=1"
    return batch[0]


class StyleTTS2SynthesisDataset(Dataset):
    def __init__(self, entries: list[dict]):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


class StyleTTS2SynthesisDataModule(L.LightningDataModule):
    def __init__(self, entries: list[dict]):
        super().__init__()
        self.entries = entries

    def predict_dataloader(self):
        return DataLoader(
            StyleTTS2SynthesisDataset(self.entries),
            batch_size=1,
            collate_fn=_synthesis_collate_fn,
            shuffle=False,
            num_workers=0,
        )


class StyleTTS2PredictionWritingWavCallback(BasePredictionWritingCallback):
    def __init__(self, save_dir: Path, global_step: int):
        super().__init__(
            save_dir=save_dir / "wav",
            file_extension="pred.wav",
            global_step=global_step,
            include_global_step_in_filename=True,
        )
        self.last_file_written: Optional[str] = None

    def on_predict_batch_end(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        _trainer,
        _pl_module,
        outputs,
        batch,
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ):
        if outputs is None:
            return
        wav_tensor = torch.from_numpy(outputs["wav"]).unsqueeze(0)
        basename = truncate_basename(slugify(outputs["raw_text"]))
        filename = self.get_filename(basename, outputs["speaker"], outputs["language"])
        torchaudio.save(
            filename,
            wav_tensor,
            outputs["sample_rate"],
            format="wav",
            encoding="PCM_S",
            bits_per_sample=16,
        )
        self.last_file_written = filename
        logger.info(f"Saved WAV: {filename}")


def get_styletts2_synthesis_output_callbacks(
    output_type: Sequence[SynthesizeOutputFormats],
    output_dir: Path,
    global_step: int,
    sample_rate: int,
) -> dict[SynthesizeOutputFormats, BasePredictionWritingCallback]:
    """Build the set of synthesis callbacks for the requested output formats. Only supports wav for now."""
    callbacks: dict[SynthesizeOutputFormats, BasePredictionWritingCallback] = {}
    if SynthesizeOutputFormats.wav in output_type:
        callbacks[SynthesizeOutputFormats.wav] = StyleTTS2PredictionWritingWavCallback(
            output_dir, global_step
        )
    return callbacks
