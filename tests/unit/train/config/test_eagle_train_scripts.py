"""Exercise the actual two launch scripts through their model-free dry run."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from speculators.train.config import TrainConfig

ROOT = Path(__file__).resolve().parents[4]
BASH = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"


def launch_args(algorithm, **overrides):
    if not Path(BASH).is_file():
        pytest.skip("bash unavailable")
    result = subprocess.run(  # noqa: S603 -- Repository script in non-executing mode.
        [BASH, str(ROOT / f"examples/train/{algorithm}_qwen3_4b_700k/train.sh")],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1", **overrides},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    return command[command.index("speculators.train") + 1 :]


@pytest.mark.parametrize("algorithm", ["eagle3", "leagle"])
def test_script_memory_retention_and_loss_overrides(algorithm):
    flags = launch_args(
        algorithm,
        MODEL="model",
        NUM_LAYERS="2",
        NUM_LOOPS="3",
        LR="0.00003",
        MUON_LR="0.00004",
        LOSS_FN="tv",
    )
    args = TrainConfig.resolve(flags).flatten()
    assert args["fsdp_shard"]
    assert args["activation_checkpointing"]
    assert not args["save_best"]
    assert args["checkpoint_freq"] == 1
    assert args["epochs"] == 5
    assert args["lr"] == 0.00003
    assert args["muon_lr"] == 0.00004
    assert args["loss_fn"] == "tv"
    assert args["num_layers"] == 2
    assert args["num_loops"] == (3 if algorithm == "leagle" else 1)
    assert args["draft_attn_impl"] == "sdpa"


@pytest.mark.parametrize("algorithm", ["eagle3", "leagle"])
def test_resume_uses_saved_recipe_and_can_disable_best_only(algorithm, tmp_path):
    original = TrainConfig.resolve(
        launch_args(
            algorithm, MODEL="model", NUM_LAYERS="2", NUM_LOOPS="3", LR="0.00003"
        )
    )
    path = tmp_path / "checkpoints"
    path.mkdir()
    (path / "run.yaml").write_text(original.dump_yaml())
    flags = launch_args(algorithm, RESUME="1", OUTPUT_DIR=tmp_path.as_posix())
    args = TrainConfig.resolve(flags).flatten()
    assert not args["no_resume_from_checkpoint"]
    assert not args["save_best"]
    assert args["lr"] == 0.00003  # Original recipe, not the script's default.
    assert args["num_layers"] == 2
    assert args["num_loops"] == (3 if algorithm == "leagle" else 1)


def test_memory_flags_can_be_disabled():
    args = TrainConfig.resolve(
        launch_args(
            "leagle", MODEL="model", FSDP_SHARD="0", ACTIVATION_CHECKPOINTING="0"
        )
    ).flatten()
    assert not args["fsdp_shard"]
    assert not args["activation_checkpointing"]
