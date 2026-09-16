"""Audit a local converted checkpoint without loading the target or importing torch."""

import argparse
import json

from speculators_dsv4.contract import inspect_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", help="Local, already-dequantized DSV4-Flash BF16 directory"
    )
    args = parser.parse_args()
    report = inspect_checkpoint(args.model, require_bf16=True)
    report.pop("config")
    print(json.dumps(report, indent=2))
    print(
        "Header audit passed. Numerical conversion and NPU runtime are NOT validated."
    )


if __name__ == "__main__":
    main()
