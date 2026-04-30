import typer
from everyvoice.wizard import TEXT_TO_WAV_CONFIG_FILENAME_PREFIX

from .preprocess import preprocess as app_preprocess
from .train import train as app_train

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
    help="A StyleTTS2 end-to-end text-to-speech model configured via EveryVoice.",
)

app.command(
    name="preprocess",
    short_help="Preprocess your data",
    help=f"""Preprocess your data for StyleTTS2 training. For example:

    **everyvoice preprocess e2e config/{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml**
    """,
)(app_preprocess)

app.command(
    name="train",
    short_help="Train your StyleTTS2 model",
    help=f"""Train a StyleTTS2 end-to-end model. For example:

    **everyvoice train e2e config/{TEXT_TO_WAV_CONFIG_FILENAME_PREFIX}.yaml --mode first**
    """,
)(app_train)
