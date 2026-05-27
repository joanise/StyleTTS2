from collections import OrderedDict

import torch
import yaml
from transformers import AlbertConfig, AlbertModel


class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state


def _resolve(repo_id: str, filename: str, local_path=None) -> str:
    if local_path is not None:
        return str(local_path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename=filename)


def build_plbert_shape(config: dict):
    """Construct PLBERT from its config file only, without loading pretrained weights."""
    config_path = _resolve(
        config["repo_id"], config["config_filename"], config.get("local_config")
    )
    plbert_config = yaml.safe_load(open(config_path))
    albert_base_configuration = AlbertConfig(**plbert_config["model_params"])
    return CustomAlbert(albert_base_configuration)


def load_plbert(config: dict):
    config_path = _resolve(
        config["repo_id"], config["config_filename"], config.get("local_config")
    )
    ckpt_path = _resolve(
        config["repo_id"], config["checkpoint_filename"], config.get("local_checkpoint")
    )

    plbert_config = yaml.safe_load(open(config_path))
    albert_base_configuration = AlbertConfig(**plbert_config["model_params"])
    bert = CustomAlbert(albert_base_configuration)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["net"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        if name.startswith("encoder."):
            name = name[8:]  # remove `encoder.`
            new_state_dict[name] = v
    new_state_dict.pop("embeddings.position_ids", None)
    bert.load_state_dict(new_state_dict, strict=False)

    return bert
