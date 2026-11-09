"""Validate the standalone Python launcher's stage and resume contracts."""

import importlib.util
import os
import shlex
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from speculators.train.config import TrainConfig

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "examples/train/leagle_candidate_head/train.py"


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    spec = importlib.util.spec_from_file_location("candidate_launcher", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_dry_run_resolves_stage_two_from_any_working_directory(tmp_path):
    output = tmp_path / "new output"
    result = subprocess.run(  # noqa: S603 -- Model-free dry run of a repo script.
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--from-pretrained",
            "draft with spaces",
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "LOCAL_RANK"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    entry = str(ROOT / "scripts/train.py")
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert command[command.index("--nproc_per_node") + 1] == "6"
    argv = command[command.index(entry) + 1 :]
    cfg = TrainConfig.resolve(argv)
    assert cfg.speculator_type == "leagle"
    assert cfg.draft.from_pretrained == "draft with spaces"
    assert cfg.draft.candidate_head_only
    assert cfg.draft.candidate_head
    assert cfg.draft.candidate_rerank
    assert cfg.draft.candidate_head_init == "svd"
    assert cfg.draft.candidate_head_rank == 640
    assert cfg.draft.candidate_head_top_k == 512
    assert cfg.trainer.fsdp_shard
    assert not cfg.trainer.activation_checkpointing
    assert cfg.draft.draft_attn_impl == "sdpa"
    assert cfg.optimizer.lr == 6e-4
    assert cfg.data.noise_std == 0
    assert not output.exists()
    for flag in (
        "--num-layers",
        "--num-loops",
        "--draft-arch",
        "--draft-vocab-size",
        "--target-layer-ids",
        "--full-attention-indices",
        "--token-freq-path",
    ):
        assert flag not in argv


def test_resume_keeps_saved_training_settings(launcher, tmp_path):
    parser = launcher.build_parser()
    original = parser.parse_args(
        [
            "--from-pretrained",
            "original",
            "--output-dir",
            str(tmp_path),
            "--candidate-head-init",
            "random",
            "--candidate-head-rank",
            "320",
            "--lr",
            "0.0002",
            "--no-fsdp-shard",
            "--leagle-safe-optimizations",
        ]
    )
    cfg = TrainConfig.resolve(launcher.training_arguments(original))
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "run.yaml").write_text(cfg.dump_yaml(), encoding="utf-8")
    progress = checkpoints / "checkpoint_1"
    progress.mkdir()
    (progress / "training_state.json").write_text("{}", encoding="utf-8")
    args = parser.parse_args(
        [
            "--resume",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--candidate-head-rank",
            "64",
            "--lr",
            "0.1",
            "--vllm-endpoint",
            "http://localhost:9999/v1",
        ]
    )
    launcher.validate_launch(args)
    launcher.validate_run_files(args, tmp_path)
    restored = TrainConfig.resolve(launcher.training_arguments(args))
    assert restored.draft.candidate_head_only
    assert restored.draft.from_pretrained == "original"
    assert restored.draft.candidate_head_rank == 320
    assert restored.draft.candidate_head_init == "random"
    assert restored.optimizer.lr == 0.0002
    assert not restored.trainer.fsdp_shard
    assert restored.trainer.leagle_safe_optimizations
    assert not restored.trainer.no_resume_from_checkpoint
    assert restored.generation.vllm_endpoint == "http://localhost:9999/v1"


def test_resume_rejects_a_backbone_training_run(launcher, tmp_path):
    checkpoint = tmp_path / "checkpoints/checkpoint_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "training_state.json").write_text("{}", encoding="utf-8")
    (checkpoint.parent / "run.yaml").write_text(
        "train:\n  speculator_type: leagle\n  draft:\n    candidate_head_only: false\n",
        encoding="utf-8",
    )
    args = launcher.build_parser().parse_args(
        ["--resume", "--output-dir", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="candidate-only training run"):
        launcher.validate_run_files(args, tmp_path)


def test_dispatch_uses_selected_devices_and_repo_python_paths(
    launcher, tmp_path, monkeypatch
):
    directories = [tmp_path / name for name in ("draft", "target", "data")]
    for directory in directories:
        directory.mkdir()
    output = tmp_path / "new run"
    calls = []

    class TargetReady:
        def open(self, url, timeout):
            assert url == "http://127.0.0.1:8003/v1/models"
            assert timeout > 0
            return nullcontext()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(launcher, "build_opener", lambda _: TargetReady())
    monkeypatch.setattr(launcher.os, "execvpe", lambda *args: calls.append(args))
    launcher.main(
        [
            "--from-pretrained",
            str(directories[0]),
            "--verifier-name-or-path",
            str(directories[1]),
            "--data-path",
            str(directories[2]),
            "--output-dir",
            str(output),
            "--train-npus",
            "4,5",
        ]
    )
    executable, command, env = calls[0]
    assert executable == sys.executable
    assert command[command.index("--nproc_per_node") + 1] == "2"
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"
    assert env["PYTHONPATH"].split(os.pathsep)[:3] == [
        str(ROOT / "src"),
        str(ROOT / "hs_connectors/src"),
        str(ROOT),
    ]
    assert Path.cwd() == ROOT
    assert (output / "hidden_states").is_dir()
    assert not (output / "checkpoints").exists()


@pytest.mark.parametrize("relation", ["same", "inside", "ancestor"])
def test_source_checkpoint_is_protected_before_any_writes(launcher, tmp_path, relation):
    source = tmp_path / "source"
    source.mkdir()
    output = {"same": source, "inside": source / "run", "ancestor": tmp_path}[relation]
    args = launcher.build_parser().parse_args(
        [
            "--from-pretrained",
            str(source),
            "--output-dir",
            str(output),
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="separate"):
        launcher.validate_launch(args)
    assert not list(source.iterdir())


@pytest.mark.parametrize("devices", ["2,2", "2,02", "2,,3", "-1"])
def test_invalid_device_selection_fails_before_launch(launcher, devices):
    args = launcher.build_parser().parse_args(
        [
            "--from-pretrained",
            "draft",
            "--train-npus",
            devices,
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="device IDs|repeat"):
        launcher.validate_launch(args)


def test_missing_source_and_nested_torchrun_fail_before_launch(launcher, monkeypatch):
    args = launcher.build_parser().parse_args(["--dry-run"])
    with pytest.raises(ValueError, match="DRAFT_MODEL"):
        launcher.validate_launch(args)
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(ValueError, match="not inside torchrun"):
        launcher.validate_launch(args)
