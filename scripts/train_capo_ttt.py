#!/usr/bin/env python3
import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tttlab.training.capo_ttt import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
