#!/usr/bin/env python3
import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from arch_research.cli import train_main


if __name__ == "__main__":
    raise SystemExit(train_main(sys.argv[1:]))

