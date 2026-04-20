#!/usr/bin/env python3
from __future__ import annotations

from typing import Sequence

from arch_research.cli import train_main


def main(argv: Sequence[str] | None = None) -> int:
    base_args = [
        "--task",
        "capo",
        "--model",
        "gpt_ttt",
        "--task-arg",
        "tokenizer=ws",
        "--model-arg",
        "n_layers=6",
        "--model-arg",
        "n_heads=8",
        "--model-arg",
        "d_head=32",
        "--model-arg",
        "d_model=256",
        "--model-arg",
        "d_mlp=256",
        "--model-arg",
        "d_ttt=0",
        "--model-arg",
        "sliding_window=512",
    ]
    return train_main(base_args + list(argv or ()))


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
