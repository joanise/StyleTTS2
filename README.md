# StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models

### Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani

> In this paper, we present StyleTTS 2, a text-to-speech (TTS) model that leverages style diffusion and adversarial training with large speech language models (SLMs) to achieve human-level TTS synthesis. StyleTTS 2 differs from its predecessor by modeling styles as a latent random variable through diffusion models to generate the most suitable style for the text without requiring reference speech, achieving efficient latent diffusion while benefiting from the diverse speech synthesis offered by diffusion models. Furthermore, we employ large pre-trained SLMs, such as WavLM, as discriminators with our novel differentiable duration modeling for end-to-end training, resulting in improved speech naturalness. StyleTTS 2 surpasses human recordings on the single-speaker LJSpeech dataset and matches it on the multispeaker VCTK dataset as judged by native English speakers. Moreover, when trained on the LibriTTS dataset, our model outperforms previous publicly available models for zero-shot speaker adaptation. This work achieves the first human-level TTS synthesis on both single and multispeaker datasets, showcasing the potential of style diffusion and adversarial training with large SLMs.

Paper: [https://arxiv.org/abs/2306.07691](https://arxiv.org/abs/2306.07691)

Audio samples: [https://styletts2.github.io/](https://styletts2.github.io/)

Online demo: [Hugging Face](https://huggingface.co/spaces/styletts2/styletts2) (thank [@fakerybakery](https://github.com/fakerybakery) for the wonderful online demo)

[![Discord](https://img.shields.io/discord/1197679063150637117?logo=discord&logoColor=white&label=Join%20our%20Community)](https://discord.gg/ha8sxdG2K4)

## Installation

Requires Python >= 3.9 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
uv sync
```

**CUDA / PyTorch index**: by default `uv sync` installs the CPU build of PyTorch. To use a CUDA build, uncomment the `[[tool.uv.index]]` block in `pyproject.toml` and set the correct CUDA version before running `uv sync`.

**Phonemizer** (required for inference notebooks):
```bash
uv pip install phonemizer
sudo apt-get install espeak-ng   # Linux
brew install espeak               # macOS
```

**WavLM discriminator**: Stage 2 training downloads `microsoft/wavlm-base-plus` from HuggingFace on first run. To avoid authentication warnings set either:
```bash
export HF_TOKEN=<your_token>
# or, to use a cached copy with no network calls:
export TRANSFORMERS_OFFLINE=1
```

## Data preparation

Place your data under `data/`. The list files must use the format:

```
path/to/audio.wav|phoneme transcription|speaker_id
```

See `data/val_list.txt` for an example. Speaker labels are required for multi-speaker models. For LJSpeech (single-speaker) use `0` for all entries.

For LibriTTS, combine `train-clean-100` and `train-clean-360` into a single `train-clean-460` folder before generating the list files.

Out-of-distribution texts for SLM adversarial training go in `data/OOD_texts.txt`, one entry per line in `text|anything` format.

## Training

Training uses a single entry point with three sequential modes.

### Stage 1 — acoustic pre-training

```bash
python train.py --mode first --config configs/base.yml
```

For multi-GPU training:
```bash
python train.py --mode first --config configs/base.yml --devices 4 --strategy ddp
```

Mixed precision (recommended for Stage 1 only with batch size >= 16):
```bash
python train.py --mode first --config configs/base.yml --devices 4 --strategy ddp --precision 16-mixed
```

### Stage 2 — joint training with SLM adversarial loss

```bash
python train.py --mode second --config configs/base.yml --devices 4 --strategy ddp
```

For LibriTTS multi-speaker:
```bash
python train.py --mode second --config configs/libritts.yml --devices 4 --strategy ddp
```

### Fine-tuning

```bash
python train.py --mode finetune --config configs/finetune.yml --devices 4 --strategy ddp
```

### Resuming from a checkpoint

Pass `--resume` with the path to a Lightning `.ckpt` file:
```bash
python train.py --mode second --config configs/base.yml --resume logs/styletts2/version_0/checkpoints/epoch=10-step=50000.ckpt
```

Checkpoints and TensorBoard logs are saved under `log_dir` as configured in the config file.

### Important config options

In `configs/base.yml` (and the other config files):

- `OOD_data`: path to out-of-distribution texts for SLM adversarial training
- `min_length`: minimum OOD text length; ensures synthesized speech has a minimum duration
- `max_len`: maximum training audio length in frames (hop size 300 → ~0.0125 s/frame); reduce if you hit OOM
- `multispeaker`: set `true` for multi-speaker models (changes the denoiser architecture)
- `batch_percentage`: fraction of the batch used for SLM adversarial steps; reduce if you hit OOM during Stage 2
- `joint_epoch`: epoch at which SLM adversarial training begins; set higher than `epochs` to skip it

### Pre-trained modules

Three pretrained components live under `styletts2/pretrained/`:

- **`asr/`** — text aligner pretrained on English (LibriTTS), Japanese (JVS), and Chinese (AiShell). Works for most languages without fine-tuning. Retrain with [yl4579/AuxiliaryASR](https://github.com/yl4579/AuxiliaryASR).
- **`jdc/`** — pitch extractor pretrained on English (LibriTTS). F0 is language-independent so it generalises well; retrain for singing with [yl4579/PitchExtractor](https://github.com/yl4579/PitchExtractor).
- **`plbert/`** — [PL-BERT](https://arxiv.org/abs/2301.08810) pretrained on English Wikipedia. For other languages use [yl4579/PL-BERT](https://github.com/yl4579/PL-BERT) or the [multilingual PL-BERT](https://huggingface.co/papercup-ai/multilingual-pl-bert) (14 languages).

### Common training issues

- **Loss becomes NaN**: For Stage 1, avoid mixed precision unless batch size >= 16. For Stage 2, try a lower batch size; 16 is recommended.
- **Out of memory**: Lower `batch_size` or `max_len` in the config; reduce `batch_percentage` for OOM during Stage 2 SLM adversarial steps.
- **Non-English data**: Use a PL-BERT pretrained for your language. The [multilingual PL-BERT](https://huggingface.co/papercup-ai/multilingual-pl-bert) supports 14 languages.

## Inference

See [Demo/Inference_LJSpeech.ipynb](Demo/Inference_LJSpeech.ipynb) (single-speaker) and [Demo/Inference_LibriTTS.ipynb](Demo/Inference_LibriTTS.ipynb) (multi-speaker). For LibriTTS, download [reference_audio.zip](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/reference_audio.zip) and unzip it under `Demo/` before running.

Pre-trained models:
- LJSpeech (24 kHz): [https://huggingface.co/yl4579/StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main)
- LibriTTS: [https://huggingface.co/yl4579/StyleTTS2-LibriTTS](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main)

### Common inference issues

- **High-pitched background noise**: Caused by float precision differences on older GPUs. Use a newer GPU or run inference on CPU. See [#13](https://github.com/yl4579/StyleTTS2/issues/13).
- **Pre-trained model license**: The license terms below apply when using pre-trained models with speakers not in the training set.

## References
- [archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)
- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [nii-yamagishilab/project-NN-Pytorch-scripts/project/01-nsf](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf)

## License

Code: MIT License

Pre-Trained Models: Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.
