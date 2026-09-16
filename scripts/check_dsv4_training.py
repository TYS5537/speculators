#!/usr/bin/env python3
"""Run one NPU smoke phase through the production train.py implementation."""

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speculators_dsv4.training_smoke import (
    SmokeSettings,
    make_smoke_trainer,
    recipe_digest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-phase", choices=("fresh", "resume"), required=True)
    parser.add_argument("--smoke-report-dir", type=Path, required=True)
    parser.add_argument("--train-batches", type=int, default=2)
    parser.add_argument("--val-batches", type=int, default=1)
    parser.add_argument(
        "train_args", nargs=argparse.REMAINDER, help="-- train.py arguments"
    )
    options = parser.parse_args()
    arguments = options.train_args
    if not arguments or arguments[0] != "--":
        parser.error("pass production train.py arguments after --")
    arguments = arguments[1:]
    settings = SmokeSettings(
        options.smoke_phase,
        options.smoke_report_dir,
        options.train_batches,
        options.val_batches,
        recipe_digest(arguments),
    )
    # Explicitly register the NPU backend; never label a CPU run A3 acceptance.
    try:
        import torch_npu  # noqa: PLC0415, F401
    except ImportError:
        parser.error("A3 smoke requires torch_npu in the training environment")
    # Import the production implementation, not a copied training loop.
    path = Path(__file__).with_name("train.py")
    spec = importlib.util.spec_from_file_location("_dsv4_smoke_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch  # noqa: PLC0415

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        parser.error(
            "A3 smoke requires an available NPU; CPU unit tests are not A3 acceptance"
        )
    module.Trainer = make_smoke_trainer(module.Trainer, settings)
    sys.argv = [str(path), *arguments]
    args = module.parse_args()
    if args.target_hidden_state_format != "deepseek_v4_mean_hc_head":
        parser.error("this acceptance command is only for the DSV4 HS format")
    if args.dry_run or args.from_pretrained:
        parser.error("smoke requires fresh init followed by automatic save-path resume")
    # The normal 12 workers x 4 prefetch batches would generate far more HS
    # requests than the few consumed smoke batches. Keep dataset/sampler intact.
    args.num_workers = 0
    checkpoint_dir = Path(args.save_path)
    if (
        settings.phase == "fresh"
        and checkpoint_dir.exists()
        and any(checkpoint_dir.iterdir())
    ):
        parser.error(
            "fresh smoke needs a new/empty save-path, not an existing experiment"
        )
    module.main(args)


if __name__ == "__main__":
    main()
