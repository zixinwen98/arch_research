from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Dict

from . import gpt2, gpt2_rope, gpt_ttt


@dataclass
class ModelSpec:
    name: str
    config_cls: type
    model_cls: type
    forward_adapter: Callable[..., Any]


def _apply_overrides(config_obj, overrides: Dict[str, Any]):
    allowed = {field.name for field in fields(config_obj)}
    for key, value in overrides.items():
        if key not in allowed:
            raise ValueError(f"Unknown config override for {type(config_obj).__name__}: {key}")
        setattr(config_obj, key, value)
    return config_obj


def _forward_gpt(model, batch, runtime_cfg):
    return model(
        batch.input_ids,
        attn_mask=batch.attention_mask,
        position_ids=batch.position_ids,
    )


def _forward_ttt(model, batch, runtime_cfg):
    return model(
        batch.input_ids,
        attn_mask_1d=batch.attention_mask,
        position_ids=batch.position_ids,
        ttt_config=batch.metadata.get("ttt_config"),
        use_muon=runtime_cfg.muon,
    )


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "gpt2": ModelSpec("gpt2", gpt2.Config, gpt2.GPT, _forward_gpt),
    "gpt2_rope": ModelSpec("gpt2_rope", gpt2_rope.Config, gpt2_rope.MinimalGPT2, _forward_gpt),
    "gpt_ttt": ModelSpec("gpt_ttt", gpt_ttt.Config, gpt_ttt.GPT_TTT, _forward_ttt),
}


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]


def build_model(name: str, vocab_size: int, overrides: Dict[str, Any]):
    spec = get_model_spec(name)
    config = spec.config_cls(vocab_size=vocab_size)
    config = _apply_overrides(config, overrides)
    return spec.model_cls(config), config


def forward_model(name: str, model, batch, runtime_cfg):
    spec = get_model_spec(name)
    return spec.forward_adapter(model, batch, runtime_cfg)
