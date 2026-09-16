"""Audit DSV4 checkpoint headers and training-side floating-point IO weights.

Quantized backbone tensors are permitted. This read-only audit does not prove
that the chosen target backend or Ascend hardware supports their quantization.
"""

import argparse
import json

from speculators_dsv4.contract import inspect_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local DSV4-Flash checkpoint directory")
    args = parser.parse_args()
    report = inspect_checkpoint(args.model)
    report.pop("config")
    print(json.dumps(report, indent=2))
    print(
        "Header and training IO audit passed. Target quantization support, "
        "NPU runtime and teacher-logit parity are NOT validated."
    )


if __name__ == "__main__":
    main()
