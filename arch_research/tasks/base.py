from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from arch_research.data.batch import TaskBatch


class BaseTask(ABC):
    name: str

    def __init__(self, **task_args: Any):
        self.task_args = task_args

    @abstractmethod
    def build_tokenizer(self, rng):
        raise NotImplementedError

    @abstractmethod
    def build_batch(self, rng, tokenizer, batch_size: int, split: str, runtime_cfg) -> TaskBatch:
        raise NotImplementedError

    @abstractmethod
    def compute_eval_metrics(self, batch: TaskBatch, logits, tokenizer) -> Dict[str, float]:
        raise NotImplementedError

