from enum import Enum

import typer
from everyvoice.base_cli.interfaces import preprocess_base_command_interface
from everyvoice.utils import spinner
from merge_args import merge_args


class PreprocessCategories(str, Enum):
    audio = "audio"
    text = "text"


@merge_args(preprocess_base_command_interface)
def preprocess(
    steps: list[PreprocessCategories] = typer.Option(
        [cat.value for cat in PreprocessCategories],
        "-s",
        "--steps",
        help="Which preprocessing steps to run. If none are provided, text and audio processing steps are performed.",
    ),
    **kwargs,
):
    """Preprocess audio and text data for StyleTTS2 training."""
    with spinner():
        from everyvoice.base_cli.helpers import preprocess_base_command

        from everyvoice.model.e2e.StyleTTS2_lightning.styletts2.ev_config import (
            StyleTTS2Config,
        )

    preprocess_base_command(
        model_config=StyleTTS2Config,
        steps=[step.name for step in steps],
        **kwargs,
    )
