"""Pinned local identities, content hashes and model-free worker inheritance."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from speculators_eval import provenance


def fixture_args(tmp_path):
    for label in ("target", "4"):
        directory = tmp_path / label
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(label.encode())
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "speculators_model_type": "leagle",
                    "num_loops": 3,
                    "transformer_layer_config": {"num_hidden_layers": 2},
                }
            )
        )
    (tmp_path / "4" / "training_state.json").write_text(
        json.dumps(
            {
                "epoch": 4,
                "global_step": 100,
                "local_step": 0,
            }
        )
    )
    return argparse.Namespace(
        verifier_model=str(tmp_path / "target"),
        draft_model=str(tmp_path / "4"),
        output_dir=tmp_path / "results",
        target_backend="hf",
        spec_tokens=7,
        checkpoint_manifest=None,
    )


def test_weights_epoch_and_architecture_are_recorded(tmp_path):
    args = fixture_args(tmp_path)
    identities = provenance.prepare_checkpoints(args)
    provenance.write_eval_provenance(args)
    draft = identities["draft"]
    assert draft["resolved_path"] == str((tmp_path / "4").resolve())
    assert draft["epoch_zero_based"] == 4
    assert draft["global_step"] == 100
    assert (
        draft["files"]["model.safetensors"]["sha256"]
        == hashlib.sha256(b"4").hexdigest()
    )
    metadata = json.loads((args.output_dir / "eval_config.json").read_text())
    assert metadata["physical_layers"] == 2
    assert metadata["num_loops"] == 3
    assert metadata["effective_depth"] == 6
    assert metadata["spec_tokens"] == 7
    for name in (
        "checkpoint_sha256.txt",
        "drafter_checkpoint_sha256.txt",
        "eval_command.txt",
        "speculators.patch",
    ):
        assert (args.output_dir / name).is_file()


def test_workers_inherit_exact_hashes_without_rehashing(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    parent = provenance.prepare_checkpoints(args)
    worker = copy.deepcopy(args)
    worker.output_dir = tmp_path / "worker"
    monkeypatch.setattr(
        provenance, "_fingerprint", lambda _p: pytest.fail("worker rehashed weights")
    )
    assert provenance.prepare_checkpoints(worker) == parent
    worker.draft_model = str(tmp_path / "target")
    with pytest.raises(ValueError, match="differs from pinned"):
        provenance.prepare_checkpoints(worker)


def test_mutation_after_pinning_is_detected(tmp_path):
    args = fixture_args(tmp_path)
    provenance.prepare_checkpoints(args)
    (Path(args.draft_model) / "model.safetensors").write_bytes(b"replaced")
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        provenance.validate_checkpoint_files(args)


def test_external_vocab_mapping_is_also_fingerprinted(tmp_path):
    args = fixture_args(tmp_path)
    args.d2t_path = tmp_path / "mapping.pt"
    args.d2t_path.write_bytes(b"mapping")
    provenance.prepare_checkpoints(args)
    args.d2t_path.write_bytes(b"new mapping")
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        provenance.validate_checkpoint_files(args)


def test_retargeting_best_pointer_cannot_switch_eval_epoch(tmp_path):
    args = fixture_args(tmp_path)
    alias = tmp_path / "checkpoint_best"
    try:
        alias.symlink_to(tmp_path / "4", target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit directory symlinks")
    args.draft_model = str(alias)
    provenance.prepare_checkpoints(args)
    alias.unlink()
    alias.symlink_to(tmp_path / "target", target_is_directory=True)
    assert args.draft_model == str((tmp_path / "4").resolve())
    provenance.validate_checkpoint_files(args)


def test_parent_pinning_does_not_import_torch(tmp_path):
    args = fixture_args(tmp_path)
    code = """
import argparse, sys
from pathlib import Path
from speculators_eval.provenance import prepare_checkpoints, write_eval_provenance
args = argparse.Namespace(verifier_model=sys.argv[1], draft_model=sys.argv[2],
                          output_dir=Path(sys.argv[3]), target_backend="hf")
prepare_checkpoints(args)
write_eval_provenance(args)
assert "torch" not in sys.modules
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    result = subprocess.run(  # noqa: S603 -- Fixed test code and fixture paths.
        [
            sys.executable,
            "-c",
            code,
            args.verifier_model,
            args.draft_model,
            str(args.output_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
