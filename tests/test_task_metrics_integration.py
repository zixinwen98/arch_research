import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from arch_research.config import ModelSpecConfig, TaskSpecConfig, TrainConfig
from arch_research.training import Evaluator, Trainer


TASK_CONFIGS = {
    "capo": {"tokenizer": "ws", "window": 32},
    "depo": {"variant": "depo1", "N": 8, "K": 3, "ctx": 128},
    "mano": {"L": 4, "ctx": 64},
    "brevo": {"variant": "brevo2", "N": 8, "ctx": 128, "schedule": "uniform"},
    "lano": {"variant": "cfg3j", "ctx": 64},
}


def test_task_specific_eval_metrics_exist():
    for task_name, task_args in TASK_CONFIGS.items():
        trainer = Trainer(
            ModelSpecConfig(
                name="gpt2",
                args={"n_layers": 1, "n_heads": 2, "d_model": 32, "d_mlp": 64, "max_seq_len": 128},
            ),
            TaskSpecConfig(name=task_name, args=task_args),
            TrainConfig(steps=1, batch_size=2, device="cpu", eval_batches=1),
        )
        eval_metrics = Evaluator(trainer.make_runtime_context(), trainer.task_cfg, trainer.train_cfg).run(1)
        assert "eval/loss" in eval_metrics
        assert "eval/ppl" in eval_metrics
        assert len([key for key in eval_metrics if key not in {"eval/loss", "eval/ppl"}]) >= 1
