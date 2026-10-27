"""EAGLE3 entrypoint sharing LEAGLE's speculative verification and metrics."""

from pathlib import Path

from speculators_eval.leagle import main

if __name__ == "__main__":
    main(entrypoint=Path(__file__))
