import os
import sys
from pathlib import Path

import click
import soundfile as sf
import torch
import torchaudio
import yaml

from styletts2.lightning import StyleTTS2Module
from styletts2.text_utils import TextCleaner
from styletts2.utils import length_to_mask

try:
    from phonemizer.backend import EspeakBackend
    _HAS_PHONEMIZER = True
except ImportError:
    _HAS_PHONEMIZER = False

_text_cleaner = TextCleaner()
_MEL_MEAN, _MEL_STD = -4, 4


def _make_mel_transform(config):
    pp = config['preprocess_params']
    sp = pp.get('spect_params', {})
    mp = pp.get('mel_params', {})
    return torchaudio.transforms.MelSpectrogram(
        n_mels=mp.get('n_mels', 80),
        n_fft=sp.get('n_fft', 2048),
        win_length=sp.get('win_length', 1200),
        hop_length=sp.get('hop_length', 300),
    )


def _phonemize(text, language):
    if not _HAS_PHONEMIZER:
        raise RuntimeError(
            "phonemizer is not installed. Run: uv pip install phonemizer && "
            "brew install espeak  # macOS  |  apt-get install espeak-ng  # Linux"
        )
    backend = EspeakBackend(language, preserve_punctuation=True, with_stress=True)
    result = backend.phonemize([text])
    return result[0] if result else ''


def _load_reference_mel(path, target_sr, to_mel):
    wave, sr = torchaudio.load(path)
    wave = wave.mean(0)
    if sr != target_sr:
        wave = torchaudio.functional.resample(wave, sr, target_sr)
    mel = to_mel(wave)
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - _MEL_MEAN) / _MEL_STD
    return mel  # [1, n_mels, T]


def load_model(config_path, checkpoint_path, mode, device):
    config = yaml.safe_load(open(config_path))
    module = StyleTTS2Module(config, mode=mode)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    module.load_state_dict(state['state_dict'])
    module.eval()
    module.to(device)
    to_mel = _make_mel_transform(config).to(device)
    return module, to_mel


@torch.no_grad()
def synthesize(module, to_mel, text, device, reference_path=None,
               diffusion_steps=5, embedding_scale=1.0):
    tokens = torch.LongTensor(_text_cleaner(text)).unsqueeze(0).to(device)
    if tokens.numel() == 0:
        raise ValueError(f"Text produced no tokens: {text!r}")

    input_lengths = torch.LongTensor([tokens.size(1)]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)

    bert_dur = module.bert(tokens, attention_mask=(~text_mask).int())
    d_en = module.bert_encoder(bert_dur).transpose(-1, -2)
    t_en = module.text_encoder(tokens, input_lengths, text_mask)

    ref_s = None
    if reference_path is not None:
        ref_mel = _load_reference_mel(reference_path, module.sr, to_mel).to(device)
        ref_ss = module.style_encoder(ref_mel.unsqueeze(1))
        ref_sp = module.predictor_encoder(ref_mel.unsqueeze(1))
        ref_s = torch.cat([ref_ss, ref_sp], dim=1)

    noise = torch.randn((1, 256), device=device).unsqueeze(1)
    sampler_kwargs = dict(
        noise=noise,
        embedding=bert_dur,
        embedding_scale=embedding_scale,
        num_steps=diffusion_steps,
    )
    if ref_s is not None:
        sampler_kwargs['features'] = ref_s
    s_pred = module._sampler(**sampler_kwargs).squeeze(1)

    s = s_pred[:, 128:]
    ref = s_pred[:, :128]

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


@click.command()
@click.option('-c', '--config', 'config_path', required=True, type=click.Path(exists=True),
              help='YAML config used for training.')
@click.option('-k', '--checkpoint', required=True, type=click.Path(exists=True),
              help='Lightning .ckpt checkpoint file.')
@click.option('-t', '--text', default=None,
              help='Text to synthesize. Required if --input-file is not given.')
@click.option('-f', '--input-file', 'input_file', default=None, type=click.Path(exists=True),
              help='PSV file with filename|text columns (one utterance per line).')
@click.option('-r', '--reference', default=None, type=click.Path(exists=True),
              help='Reference audio for speaker style (multispeaker models).')
@click.option('-o', '--output-dir', 'output_dir', default='.', show_default=True,
              help='Directory to write output WAV files.')
@click.option('--phonemize', 'do_phonemize', is_flag=True, default=False,
              help='Run espeak phonemization on input text before synthesis.')
@click.option('--language', default='en-us', show_default=True,
              help='espeak language code, used with --phonemize.')
@click.option('--mode', default='second', show_default=True,
              type=click.Choice(['first', 'second', 'finetune']),
              help='Training mode the checkpoint was produced by.')
@click.option('--device', default=None,
              help='Device (cuda, cpu). Auto-selects cuda if available.')
@click.option('--diffusion-steps', 'diffusion_steps', default=5, show_default=True, type=int,
              help='Number of diffusion sampling steps.')
@click.option('--embedding-scale', 'embedding_scale', default=1.0, show_default=True, type=float,
              help='Classifier-free guidance scale for diffusion.')
def main(config_path, checkpoint, text, input_file, reference, output_dir,
         do_phonemize, language, mode, device, diffusion_steps, embedding_scale):
    if text is None and input_file is None:
        raise click.UsageError("Provide either --text or --input-file.")
    if text is not None and input_file is not None:
        raise click.UsageError("--text and --input-file are mutually exclusive.")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading model from {checkpoint} …')
    module, to_mel = load_model(config_path, checkpoint, mode=mode, device=device)

    if input_file:
        rows = []
        with open(input_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) < 2:
                    print(f'Skipping malformed line: {line!r}', file=sys.stderr)
                    continue
                rows.append((parts[0], parts[1]))
    else:
        rows = [('output', text)]

    for stem, raw_text in rows:
        if do_phonemize:
            raw_text = _phonemize(raw_text, language)
        if not raw_text.strip():
            print(f'Skipping empty text for {stem!r}', file=sys.stderr)
            continue

        audio = synthesize(module, to_mel, raw_text, device,
                           reference_path=reference,
                           diffusion_steps=diffusion_steps,
                           embedding_scale=embedding_scale)

        out_path = os.path.join(output_dir, Path(stem).stem + '.wav')
        sf.write(out_path, audio, module.sr)
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
