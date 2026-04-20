#!/usr/bin/env python3
import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from arch_research.cli import train_main


if __name__ == "__main__":
    preset_args = [
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
    raise SystemExit(train_main(preset_args + sys.argv[1:]))
