"""Capture the real eval script's argv without starting an evaluator or model."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"


def script_command(**overrides):
    if not Path(BASH).is_file():
        pytest.skip("bash unavailable")
    env = {
        **os.environ,
        "CANDIDATE_HEAD": "auto",
        "CANDIDATE_RERANK": "auto",
        "CANDIDATE_SVD_RANK": "",
        "CANDIDATE_TOP_K": "",
        **overrides,
    }
    return subprocess.run(  # noqa: S603 -- exec is replaced with an argv printer.
        [
            BASH,
            "-c",
            'exec() { printf "\\0ARGV\\0"; printf "%s\\0" "$@"; }; '
            'script_path="$1"; shift; source "$script_path"',
            "_",
            (ROOT / "examples/evaluate/leagle_qwen3_4b_700k.sh").as_posix(),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(("rank", "top_k"), [("", ""), ("256", "128"), ("192", "64")])
def test_svd_and_rescore_sizes_reach_eval_arguments(rank, top_k):
    result = script_command(CANDIDATE_SVD_RANK=rank, CANDIDATE_TOP_K=top_k)
    assert result.returncode == 0, result.stderr
    args = result.stdout.split("\0ARGV\0")[1].rstrip("\0").split("\0")
    for flag, expected in (
        ("--candidate-svd-rank", rank),
        ("--candidate-top-k", top_k),
    ):
        if expected:
            assert args[args.index(flag) + 1] == expected
        else:
            assert flag not in args


@pytest.mark.parametrize(
    "settings",
    [
        {"CANDIDATE_SVD_RANK": "0"},
        {"CANDIDATE_SVD_RANK": "abc"},
        {"CANDIDATE_TOP_K": "-1"},
        {"CANDIDATE_SVD_RANK": "256", "CANDIDATE_HEAD": "0"},
    ],
)
def test_invalid_svd_script_options_fail_without_launching(settings):
    result = script_command(**settings)
    assert result.returncode == 2
    assert "\0ARGV\0" not in result.stdout
