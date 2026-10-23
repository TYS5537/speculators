#!/usr/bin/env python3
"""Command-line entrypoint for LEAGLE's single-chain acceptance evaluator."""

from pathlib import Path

from speculators_eval.leagle import main

if __name__ == "__main__":
    main(entrypoint=Path(__file__))
