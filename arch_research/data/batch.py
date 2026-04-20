from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class TaskBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    loss_mask: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: str | torch.device) -> "TaskBatch":
        loss_mask = None if self.loss_mask is None else self.loss_mask.to(device)
        metadata: Dict[str, Any] = {}
        for key, value in self.metadata.items():
            if isinstance(value, torch.Tensor):
                metadata[key] = value.to(device)
            elif isinstance(value, dict):
                metadata[key] = {
                    inner_key: inner_value.to(device) if isinstance(inner_value, torch.Tensor) else inner_value
                    for inner_key, inner_value in value.items()
                }
            else:
                metadata[key] = value
        return TaskBatch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            attention_mask=self.attention_mask.to(device),
            position_ids=self.position_ids.to(device),
            loss_mask=loss_mask,
            metadata=metadata,
        )

