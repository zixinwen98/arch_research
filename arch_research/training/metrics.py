from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def loss_and_ppl(loss: torch.Tensor, prefix: str) -> Dict[str, float]:
    loss_value = float(loss.item())
    ppl = math.exp(min(loss_value, 20.0))
    return {
        f"{prefix}/loss": loss_value,
        f"{prefix}/ppl": float(ppl),
    }

