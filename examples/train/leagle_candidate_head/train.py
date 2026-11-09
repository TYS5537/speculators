#!/usr/bin/env python3
"""Train only a LEAGLE candidate head from an existing draft checkpoint.

Edit the settings below, then run this file with Python (it launches torchrun).
Start the existing LEAGLE server.sh separately, with the same OUTPUT_DIR and
target model/layers as the source checkpoint. Relative paths use the repo root.
--dry-run prints the command without loading models or creating run artifacts.
--resume restores training settings from OUTPUT_DIR/checkpoints/run.yaml.
"""

import argparse
import os
import re
import shlex
import sys
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

# ==================== Edit these settings ====================
DRAFT_MODEL = ""  # Required: your existing LEAGLE checkpoint directory.
OUTPUT_DIR = "./output/leagle_candidate_r640"
TRAIN_NPUS = "2,3,4,5,6,7"  # Target server uses 0,1.
VLLM_ENDPOINT = "http://127.0.0.1:8003/v1"

# Keys are the actual training CLI flags. Every setting can also be overridden
# on this script's command line, e.g. --lr 0.0002 --candidate-head-rank 640.
# On --resume, saved training settings take precedence over this entire block.
TRAIN_SETTINGS = {
    "verifier-name-or-path": "../../Qwen3-4B",
    "data-path": "../../datasets/open_perfectblend_qwen3_4b_700k",
    "candidate-head-init": "svd",  # New head only; existing factors are retained.
    "candidate-head-rank": 640,
    "candidate-head-top-k": 512,  # Recall metric and default inference top-k.
    "candidate-head-loss-weight": 1.0,
    "candidate-head-chunk-size": 32,
    "candidate-head-train-stride": 1,
    "epochs": 5,
    "total-seq-len": 3072,
    "train-data-ratio": 0.9,
    "seed": 42,
    "optimizer": "adamw",
    "lr": 6e-4,
    "weight-decay": 0.0,
    "adam-beta1": 0.9,
    "adam-beta2": 0.999,
    "max-grad-norm": 1.0,
    "scheduler-type": "cosine",
    "scheduler-warmup-ratio": 0.04,
    "ttt-steps": 7,
    "ttt-step-loss-decay": 1.0,
    "draft-attn-impl": "sdpa",
    "hidden-states-dtype": "bfloat16",
    "noise-std": 0.0,
    "fsdp-shard": True,
    "activation-checkpointing": False,  # Frozen backbone needs no backward pass.
    "logger": "tensorboard",
    "log-freq": 10,
    "checkpoint-freq": 1,
    "leagle-safe-optimizations": None,  # None: existing defaults; True/False: on/off.
    "leagle-numerical-optimizations": None,
}

ENV_DEFAULTS = {
    "OMP_PROC_BIND": "false",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VE_OMP_NUM_THREADS": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "TASK_QUEUE_ENABLE": "2",
    "ACLNN_CACHE_LIMIT": "100000",
    "NPU_ASD_ENABLE": "0",
    "ASCEND_LAUNCH_BLOCKING": "0",
    "GLOO_SOCKET_IFNAME": "lo",
}
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[3]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-pretrained", default=DRAFT_MODEL)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--train-npus", default=TRAIN_NPUS)
    parser.add_argument("--vllm-endpoint", default=VLLM_ENDPOINT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    for name, value in TRAIN_SETTINGS.items():
        options = {"default": value}
        if value is None or isinstance(value, bool):
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = type(value)
        if name == "candidate-head-init":
            options["choices"] = ("svd", "random")
        parser.add_argument(f"--{name}", **options)
    return parser


def repo_path(value):
    return (REPO_ROOT / Path(value).expanduser()).resolve()


def training_arguments(args):
    output = repo_path(args.output_dir)
    paths = [
        "--save-path",
        str(output / "checkpoints"),
        "--log-dir",
        str(output / "tensorboard"),
        "--hidden-states-path",
        str(output / "hidden_states"),
        "--vllm-endpoint",
        args.vllm_endpoint,
    ]
    if args.resume:
        return [
            "--config",
            str(output / "checkpoints/run.yaml"),
            "--resume-from-checkpoint",
            *paths,
        ]
    flags = [
        "--training-recipe",
        "upstream",
        "--speculator-type",
        "leagle",
        "--from-pretrained",
        args.from_pretrained,
        "--candidate-head",
        "--candidate-head-only",
        "--candidate-rerank",
        "--loss-fn",
        "kl_div",
        "--loss-implementation",
        "legacy",
        "--hidden-states-backend",
        "file",
        "--on-missing",
        "generate",
        "--on-generate",
        "delete",
        "--no-resume-from-checkpoint",
        "--run-name",
        output.name,
        *paths,
    ]
    for name in TRAIN_SETTINGS:
        value = getattr(args, name.replace("-", "_"))
        if value is None:
            continue
        if isinstance(value, bool):
            flags.append(f"--{'' if value else 'no-'}{name}")
        else:
            flags.extend((f"--{name}", str(value)))
    return flags


def validate_launch(args):
    if "LOCAL_RANK" in os.environ:
        raise ValueError("Run this launcher with python, not inside torchrun.")
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", args.train_npus):
        raise ValueError("--train-npus must contain comma-separated device IDs.")
    devices = [int(value) for value in args.train_npus.split(",")]
    if len(devices) != len(set(devices)):
        raise ValueError("--train-npus cannot repeat a device ID.")
    output = repo_path(args.output_dir)
    if not args.resume:
        if not args.from_pretrained.strip():
            raise ValueError("Set DRAFT_MODEL or pass --from-pretrained CHECKPOINT.")
        source = repo_path(args.from_pretrained)
        if source.is_dir() and (
            source == output or source in output.parents or output in source.parents
        ):
            raise ValueError("Choose an output directory separate from the source.")
    if args.dry_run:
        return
    validate_run_files(args, output)
    url = args.vllm_endpoint.rstrip("/") + "/models"
    try:
        with build_opener(ProxyHandler({})).open(url, timeout=5):
            pass
    except OSError as exc:
        raise ValueError(
            f"Target unavailable at {url}; start server.sh first."
        ) from exc


def validate_run_files(args, output):
    """Check fresh-run and resume artifacts before contacting the target server."""
    if args.resume:
        import yaml  # noqa: PLC0415 -- Needed only for an actual resume.

        checkpoint = output / "checkpoints"
        if not (checkpoint / "run.yaml").is_file() or not any(
            checkpoint.glob("*/training_state.json")
        ):
            raise ValueError("--resume requires run.yaml and saved training progress.")
        with (checkpoint / "run.yaml").open(encoding="utf-8") as stream:
            saved = yaml.safe_load(stream)
        if isinstance(saved, dict):
            saved = saved.get("train", saved)
        if (
            not isinstance(saved, dict)
            or saved.get("speculator_type") != "leagle"
            or not isinstance(saved.get("draft"), dict)
            or not saved.get("draft", {}).get("candidate_head_only")
        ):
            raise ValueError("--resume requires a LEAGLE candidate-only training run.")
    else:
        if (output / "checkpoints").exists():
            raise ValueError(
                "Output already has checkpoints; use --resume or a new path."
            )
        for value in (args.verifier_name_or_path, args.data_path, args.from_pretrained):
            if not repo_path(value).is_dir():
                raise ValueError(
                    f"Local model/checkpoint/data directory not found: {value}"
                )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_launch(args)
    except ValueError as exc:
        parser.error(str(exc))
    env = {**ENV_DEFAULTS, **os.environ, "ASCEND_RT_VISIBLE_DEVICES": args.train_npus}
    bypass = ["localhost", "127.0.0.1", env.get("NO_PROXY", env.get("no_proxy", ""))]
    env["NO_PROXY"] = env["no_proxy"] = ",".join(value for value in bypass if value)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), str(REPO_ROOT / "hs_connectors/src"), str(REPO_ROOT)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(args.train_npus.split(","))),
        str(REPO_ROOT / "scripts/train.py"),
        *training_arguments(args),
    ]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    (repo_path(args.output_dir) / "hidden_states").mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)
    os.execvpe(command[0], command, env)  # noqa: S606 -- Explicit argv, no shell.


if __name__ == "__main__":
    main()
