from __future__ import annotations

from typing import Dict

from arch_research.tasks.capo_task import CapoTask
from arch_research.tasks.synthetic_tasks import BrevoTask, DepoTask, LanoTask, ManoTask


TASK_REGISTRY: Dict[str, type] = {
    "capo": CapoTask,
    "depo": DepoTask,
    "mano": ManoTask,
    "brevo": BrevoTask,
    "lano": LanoTask,
}


def get_task(name: str):
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown task: {name}")
    return TASK_REGISTRY[name]


def build_task(name: str, **task_args):
    task_cls = get_task(name)
    return task_cls(**task_args)

