#!/usr/bin/env python3
"""Start and own a local target plus the unchanged DSpark reference evaluator."""

import sys
from pathlib import Path

# The controller remains usable before importing/installing the training stack.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from speculators_dsv4.eval_launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
