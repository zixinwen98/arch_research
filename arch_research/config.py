from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    if "," in value:
        return [_parse_scalar(v.strip()) for v in value.split(",") if v.strip()]
    return value


def parse_kv_overrides(items: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected key=value override, got: {item}")
        key, raw = item.split("=", 1)
        out[key] = _parse_scalar(raw)
    return out


@dataclass
class ModelSpecConfig:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpecConfig:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    steps: int = 20
    batch_size: int = 8
    lr: float = 3e-4
    seed: int = 0
    device: str = "cpu"
    window: int = 128
    eval_batches: int = 4
    muon: bool = False
    two_phase: bool = False

