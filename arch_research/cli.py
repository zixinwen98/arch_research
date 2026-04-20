from __future__ import annotations

import argparse
from typing import Sequence

from arch_research.config import ModelSpecConfig, TaskSpecConfig, TrainConfig, parse_kv_overrides
from arch_research.training import Evaluator, Trainer


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default="gpt_ttt")
    parser.add_argument("--task", default="capo")
    parser.add_argument("--model-arg", action="append", default=[], help="Repeatable key=value model override")
    parser.add_argument("--task-arg", action="append", default=[], help="Repeatable key=value task override")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--muon", action="store_true")
    parser.add_argument("--two-phase", action="store_true")
    return parser


def trainer_from_args(ns) -> Trainer:
    model_cfg = ModelSpecConfig(name=ns.model, args=parse_kv_overrides(ns.model_arg))
    task_args = parse_kv_overrides(ns.task_arg)
    task_args.setdefault("window", ns.window)
    task_cfg = TaskSpecConfig(name=ns.task, args=task_args)
    train_cfg = TrainConfig(
        steps=ns.steps,
        batch_size=ns.batch_size,
        lr=ns.lr,
        seed=ns.seed,
        device=ns.device,
        window=ns.window,
        eval_batches=ns.eval_batches,
        muon=ns.muon,
        two_phase=ns.two_phase,
    )
    return Trainer(model_cfg, task_cfg, train_cfg)


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = build_common_parser("Generic autoregressive trainer")
    ns = parser.parse_args(argv)
    trainer = trainer_from_args(ns)
    print(f"task={ns.task} model={ns.model} vocab_size={trainer.tokenizer.vocab_size}")
    for step in range(ns.steps):
        metrics = trainer.run_step("train")
        parts = [f"step {step + 1}/{ns.steps}"] + [f"{k} {v:.4f}" for k, v in sorted(metrics.items())]
        print(" | ".join(parts))
    val_metrics = trainer.run_step("val")
    print("validation | " + " | ".join(f"{k} {v:.4f}" for k, v in sorted(val_metrics.items())))
    eval_metrics = Evaluator(trainer.make_runtime_context(), trainer.task_cfg, trainer.train_cfg).run(ns.eval_batches)
    print("evaluation | " + " | ".join(f"{k} {v:.4f}" for k, v in sorted(eval_metrics.items())))
    return 0


def eval_main(argv: Sequence[str] | None = None) -> int:
    parser = build_common_parser("Generic evaluator")
    ns = parser.parse_args(argv)
    trainer = trainer_from_args(ns)
    eval_metrics = Evaluator(trainer.make_runtime_context(), trainer.task_cfg, trainer.train_cfg).run(ns.eval_batches)
    print("evaluation | " + " | ".join(f"{k} {v:.4f}" for k, v in sorted(eval_metrics.items())))
    return 0
