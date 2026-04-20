from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from arch_research.config import ModelSpecConfig, TaskSpecConfig, TrainConfig
from arch_research.models import build_model, forward_model
from arch_research.tasks import build_task
from arch_research.training.metrics import compute_loss, loss_and_ppl


@dataclass
class RuntimeContext:
    model_name: str
    task_name: str
    tokenizer: object
    model: torch.nn.Module
    model_config: object
    task: object


class Trainer:
    def __init__(self, model_cfg: ModelSpecConfig, task_cfg: TaskSpecConfig, train_cfg: TrainConfig):
        self.model_cfg = model_cfg
        self.task_cfg = task_cfg
        self.train_cfg = train_cfg
        self.rng = random.Random(train_cfg.seed)
        torch.manual_seed(train_cfg.seed)
        self.task = build_task(task_cfg.name, **task_cfg.args)
        self.tokenizer = self.task.build_tokenizer(self.rng)
        self.model, self.model_config = build_model(model_cfg.name, self.tokenizer.vocab_size, model_cfg.args)
        self.model = self.model.to(train_cfg.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_cfg.lr)

    def make_runtime_context(self) -> RuntimeContext:
        return RuntimeContext(
            model_name=self.model_cfg.name,
            task_name=self.task_cfg.name,
            tokenizer=self.tokenizer,
            model=self.model,
            model_config=self.model_config,
            task=self.task,
        )

    def run_step(self, split: str = "train") -> Dict[str, float]:
        batch = self.task.build_batch(self.rng, self.tokenizer, self.train_cfg.batch_size, split, self.train_cfg).to(self.train_cfg.device)
        self.model.train(split == "train")
        with torch.set_grad_enabled(split == "train"):
            logits = forward_model(self.model_cfg.name, self.model, batch, self.train_cfg)
            loss = compute_loss(logits, batch.labels)
            metrics = loss_and_ppl(loss, "train" if split == "train" else "val")
            if split == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return metrics


class Evaluator:
    def __init__(self, runtime: RuntimeContext, task_cfg: TaskSpecConfig, train_cfg: TrainConfig):
        self.runtime = runtime
        self.task_cfg = task_cfg
        self.train_cfg = train_cfg
        self.rng = random.Random(train_cfg.seed + 17)

    def run(self, num_batches: int) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        self.runtime.model.eval()
        for _ in range(num_batches):
            batch = self.runtime.task.build_batch(
                self.rng,
                self.runtime.tokenizer,
                self.train_cfg.batch_size,
                "eval",
                self.train_cfg,
            ).to(self.train_cfg.device)
            with torch.no_grad():
                logits = forward_model(self.runtime.model_name, self.runtime.model, batch, self.train_cfg)
                loss = compute_loss(logits, batch.labels)
                metrics = loss_and_ppl(loss, "eval")
                metrics.update(self.runtime.task.compute_eval_metrics(batch, logits, self.runtime.tokenizer))
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        return {key: totals[key] / counts[key] for key in totals}
