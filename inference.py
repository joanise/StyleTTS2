import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import soundfile as sf
import torch
import torchaudio
import typer
import yaml

from styletts2.lightning import StyleTTS2Module
from styletts2.text_utils import TextCleaner
from styletts2.utils import MEL_MEAN, MEL_STD, length_to_mask, make_mel_transform

try:
    from phonemizer.backend import EspeakBackend
    _HAS_PHONEMIZER = True
except ImportError:
    _HAS_PHONEMIZER = False

_text_cleaner = TextCleaner()


class Mode(str, Enum):
    first = 'first'
    second = 'second'
    finetune = 'finetune'


def _phonemize(text, language):
    if not _HAS_PHONEMIZER:
        raise RuntimeError(
            "phonemizer is not installed. Run: uv pip install phonemizer && "
            "brew install espeak  # macOS  |  apt-get install espeak-ng  # Linux"
        )
    backend = EspeakBackend(language, preserve_punctuation=True, with_stress=True)
    result = backend.phonemize([text])
    return result[0] if result else ''


def _load_reference_mel(path, target_sr, mel_transform):
    wave, sr = torchaudio.load(path)
    wave = wave.mean(0)
    if sr != target_sr:
        wave = torchaudio.functional.resample(wave, sr, target_sr)
    wave = wave.to(next(mel_transform.buffers()).device)
    mel = mel_transform(wave)
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - MEL_MEAN) / MEL_STD
    return mel  # [1, n_mels, T]


def load_model(config_path, checkpoint_path, mode, device):
    config = yaml.safe_load(open(config_path))
    module = StyleTTS2Module(config, mode=mode)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    module.load_state_dict(state['state_dict'])
    module.eval()
    module.to(device)
    mel_transform = make_mel_transform(config).to(device)
    return module, mel_transform


@torch.no_grad()
def synthesize(module, mel_transform, text, device, reference_path,
               diffusion_steps=5, embedding_scale=1.0,
               acoustic_blend=0.3, prosody_blend=0.7):
    tokens = torch.LongTensor(_text_cleaner(text)).unsqueeze(0).to(device)
    if tokens.numel() == 0:
        raise ValueError(f"Text produced no tokens: {text!r}")

    input_lengths = torch.LongTensor([tokens.size(1)]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)

    bert_dur = module.bert(tokens, attention_mask=(~text_mask).int())
    d_en = module.bert_encoder(bert_dur).transpose(-1, -2)
    t_en = module.text_encoder(tokens, input_lengths, text_mask)

    ref_mel = _load_reference_mel(reference_path, module.sr, mel_transform).to(device)
    ref_ss = module.style_encoder(ref_mel.unsqueeze(1))
    ref_sp = module.predictor_encoder(ref_mel.unsqueeze(1))
    ref_s = torch.cat([ref_ss, ref_sp], dim=1)

    noise = torch.randn((1, 256), device=device).unsqueeze(1)
    s_pred = module._sampler(
        noise=noise,
        embedding=bert_dur,
        embedding_scale=embedding_scale,
        num_steps=diffusion_steps,
        features=ref_s,
    ).squeeze(1)

    ref = acoustic_blend * s_pred[:, :128] + (1 - acoustic_blend) * ref_s[:, :128]
    s = prosody_blend * s_pred[:, 128:] + (1 - prosody_blend) * ref_s[:, 128:]

    T = input_lengths[0].item()
    tm = text_mask[0, :T].unsqueeze(0)
    d = module.predictor.text_encoder(d_en[0, :, :T].unsqueeze(0), s, input_lengths, tm)
    x, _ = module.predictor.lstm(d)
    duration = torch.sigmoid(module.predictor.duration_proj(x)).sum(axis=-1)
    pred_dur = torch.round(duration.squeeze()).clamp(min=1)
    if pred_dur.ndim == 0:
        pred_dur = pred_dur.unsqueeze(0)
    pred_dur[-1] += 5

    pred_aln = torch.zeros(T, int(pred_dur.sum().item()), device=device)
    c = 0
    for i in range(T):
        pred_aln[i, c:c + int(pred_dur[i].item())] = 1
        c += int(pred_dur[i].item())

    en = d.transpose(-1, -2) @ pred_aln.unsqueeze(0)
    F0_pred, N_pred = module.predictor.F0Ntrain(en, s)
    out = module.decoder(
        t_en[0, :, :T].unsqueeze(0) @ pred_aln.unsqueeze(0),
        F0_pred, N_pred, ref.squeeze().unsqueeze(0),
    )
    return out.cpu().numpy().squeeze()


def main(
    config_path: Annotated[Path, typer.Option('-c', '--config',
        exists=True, help='YAML config used for training.')],
    checkpoint: Annotated[Path, typer.Option('-k', '--checkpoint',
        exists=True, help='Lightning .ckpt checkpoint file.')],
    reference: Annotated[Path, typer.Option('-r', '--reference',
        exists=True, help='Reference audio for speaker style.')],
    text: Annotated[Optional[str], typer.Option('-t', '--text',
        help='Text to synthesize. Required if --input-file is not given.')] = None,
    input_file: Annotated[Optional[Path], typer.Option('-f', '--input-file',
        exists=True, help='PSV file with filename|text columns.')] = None,
    output_dir: Annotated[Path, typer.Option('-o', '--output-dir',
        help='Directory to write output WAV files.')] = Path('.'),
    do_phonemize: Annotated[bool, typer.Option('--phonemize',
        help='Run espeak phonemization on input text before synthesis.')] = False,
    language: Annotated[str, typer.Option(
        help='espeak language code, used with --phonemize.')] = 'en-us',
    mode: Annotated[Mode, typer.Option(
        help='Training mode the checkpoint was produced by.')] = Mode.second,
    device: Annotated[Optional[str], typer.Option(
        help='Device (cuda, cpu). Auto-selects cuda if available.')] = None,
    diffusion_steps: Annotated[int, typer.Option(
        help='Number of diffusion sampling steps.')] = 5,
    embedding_scale: Annotated[float, typer.Option(
        help='Classifier-free guidance scale for diffusion.')] = 1.0,
    acoustic_blend: Annotated[float, typer.Option(
        help='Blend weight for the acoustic style embedding (controls voice timbre and quality). '
             '1.0 uses the diffusion sample only; 0.0 uses the reference audio encoding directly. '
             'Lower values may reduce artefacts.')] = 0.3,
    prosody_blend: Annotated[float, typer.Option(
        help='Blend weight for the prosody style embedding (controls pitch and duration). '
             '1.0 uses the diffusion sample only; 0.0 uses the reference audio encoding directly. '
             'Lower values may stabilise F0 and duration prediction, while higher values may condition better on the input text.')] = 0.7,
):
    if text is None and input_file is None:
        typer.echo("Error: provide either --text or --input-file.", err=True)
        raise typer.Exit(code=1)
    if text is not None and input_file is not None:
        typer.echo("Error: --text and --input-file are mutually exclusive.", err=True)
        raise typer.Exit(code=1)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(output_dir, exist_ok=True)

    typer.echo(f'Loading model from {checkpoint} …')
    module, mel_transform = load_model(config_path, checkpoint, mode=mode.value, device=device)

    if input_file:
        rows = []
        with open(input_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) < 2:
                    typer.echo(f'Skipping malformed line: {line!r}', err=True)
                    continue
                rows.append((parts[0], parts[1]))
    else:
        rows = [('output', text)]

    for stem, raw_text in rows:
        if do_phonemize:
            raw_text = _phonemize(raw_text, language)
        if not raw_text.strip():
            typer.echo(f'Skipping empty text for {stem!r}', err=True)
            continue

        audio = synthesize(module, mel_transform, raw_text, device,
                           reference_path=reference,
                           diffusion_steps=diffusion_steps,
                           embedding_scale=embedding_scale,
                           acoustic_blend=acoustic_blend,
                           prosody_blend=prosody_blend)

        out_path = os.path.join(output_dir, Path(stem).stem + '.wav')
        sf.write(out_path, audio, module.sr)
        typer.echo(f'Wrote {out_path}')


if __name__ == '__main__':
    typer.run(main)
