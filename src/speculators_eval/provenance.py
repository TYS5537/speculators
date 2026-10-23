"""Pin and fingerprint eval checkpoints before loading or dispatching workers.

This module deliberately imports no Torch/Transformers model runtime.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _helpers():
    # Importing speculators.__init__ initializes the model runtime. Load its
    # standalone stdlib helper file directly before NPU worker isolation.
    package = importlib.util.find_spec("speculators")
    path = Path(package.origin).with_name("provenance.py")
    spec = importlib.util.spec_from_file_location("_eval_provenance_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _signature(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ino]


def _fingerprint(path):
    before = _signature(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if before != _signature(path):
        raise RuntimeError(f"Checkpoint changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "stat": before}


def _resolve_checkpoint(value):
    path = Path(value).expanduser()
    if not path.is_dir():
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        path = Path(snapshot_download(str(value)))
    return path.resolve(strict=True)


def _checkpoint_identity(value):
    path = _resolve_checkpoint(value)
    weights = sorted(path.glob("*.safetensors")) or sorted(
        path.glob("pytorch_model*.bin")
    )
    if not weights:
        raise ValueError(f"No model weights found in {path}")
    # Config, generation/tokenizer settings and optional Python code also affect
    # the experiment. Optimizer states are not inference weights.
    files = sorted(
        set(
            weights
            + [
                p
                for p in path.iterdir()
                if p.is_file()
                and (
                    p.suffix in {".json", ".model", ".tiktoken", ".py", ".npy"}
                    or p.name in {"merges.txt", "vocab.txt", "d2t.pt", "t2d.pt"}
                )
            ]
        )
    )
    fingerprints = {p.name: _fingerprint(p) for p in files}
    content_id = hashlib.sha256(
        json.dumps(
            {name: value["sha256"] for name, value in fingerprints.items()},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    state_path = path / "training_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    epoch = state.get("epoch", int(path.name) if path.name.isdecimal() else None)
    return {
        "requested_path": str(value),
        "resolved_path": str(path),
        "checkpoint_sha256": content_id,
        "files": fingerprints,
        "epoch_zero_based": epoch,
        "global_step": state.get("global_step"),
        "local_step": state.get("local_step"),
    }


def prepare_checkpoints(args):
    """Resolve checkpoint_best once; all workers receive the exact same paths."""
    inherited = getattr(args, "checkpoint_manifest", None)
    if inherited:
        manifest = json.loads(Path(inherited).read_text(encoding="utf-8"))
        for role, field in (("target", "verifier_model"), ("draft", "draft_model")):
            identity = manifest[role]
            if (
                "resolved_path" in identity
                and str(Path(getattr(args, field)).resolve())
                != identity["resolved_path"]
            ):
                raise ValueError(
                    f"Worker {role} differs from pinned checkpoint manifest"
                )
    else:
        manifest = {"draft": _checkpoint_identity(args.draft_model)}
        manifest["target"] = (
            _checkpoint_identity(args.verifier_model)
            if getattr(args, "target_backend", "hf") == "hf"
            else {
                "requested_path": args.verifier_model,
                "identity_source": "eval_backend.json",
            }
        )
    for role, field in (("target", "verifier_model"), ("draft", "draft_model")):
        if "resolved_path" in manifest[role]:
            setattr(args, field, manifest[role]["resolved_path"])
    _pin_vocab_mappings(args, manifest, inherited)
    args.checkpoint_identity = manifest
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "eval_checkpoints.json"
    _helpers().atomic_write(destination, json.dumps(manifest, indent=2) + "\n")
    args.checkpoint_manifest = destination.resolve()
    validate_checkpoint_files(args)
    return manifest


def _pin_vocab_mappings(args, manifest, inherited):
    for field in ("d2t_path", "t2d_path"):
        value = getattr(args, field, None)
        if value is None:
            continue
        path = Path(value).resolve(strict=True)
        if inherited:
            if manifest.get(field, {}).get("resolved_file") != str(path):
                raise ValueError(
                    f"Worker {field} differs from pinned vocabulary mapping"
                )
        else:
            manifest[field] = {
                "resolved_path": str(path.parent),
                "resolved_file": str(path),
                "files": {path.name: _fingerprint(path)},
            }
        setattr(args, field, path)


def validate_checkpoint_files(args):
    """Fail if files disappeared/replaced after pinning, including during a load."""
    for identity in args.checkpoint_identity.values():
        if "resolved_path" not in identity:
            continue
        root = Path(identity["resolved_path"])
        for name, expected in identity["files"].items():
            path = root / name
            if not path.is_file() or _signature(path) != expected["stat"]:
                raise RuntimeError(
                    f"Pinned checkpoint changed during evaluation: {path}"
                )


def write_eval_provenance(args, metadata=None):
    helpers = _helpers()
    repo = helpers.find_package_repo("speculators")
    command = "\n".join(
        [
            f"# Timestamp: {datetime.now(timezone.utc).isoformat()}",
            f"# Git SHA: {helpers.git_sha(repo)}",
            f"# Python: {sys.executable}",
            *helpers.package_versions(),
            shlex.join([sys.executable, *sys.argv]),
        ]
    )
    helpers.atomic_write(args.output_dir / "eval_command.txt", command + "\n")
    helpers.atomic_write(
        args.output_dir / "speculators.patch", helpers.git_diff(repo) + "\n"
    )
    manifest = args.checkpoint_identity
    config_path = Path(args.draft_model) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    layers = config.get("transformer_layer_config", {}).get("num_hidden_layers")
    loops = config.get("num_loops", 1)
    budget = getattr(args, "spec_tokens", None)
    if budget is None and config.get("block_size") is not None:
        anchor = getattr(args, "sample_from_anchor", None)
        sample_from_anchor = (
            config.get("sample_from_anchor", True)
            if anchor is None
            else anchor == "true"
        )
        budget = config["block_size"] - (0 if sample_from_anchor else 1)
    record = {
        "algorithm": config.get("speculators_model_type"),
        "draft_config": config,
        "checkpoints": manifest,
        "physical_layers": layers,
        "num_loops": loops,
        "effective_depth": layers * loops if layers is not None else None,
        "spec_tokens": budget,
        "protocol": "offline-single-chain-v1",
        "acceptance_length": "1 + accepted_draft_tokens / proposals",
        "position_accept_rates": "accepted_prefix_count / proposed_count",
        "position_accept_rates_per_round": "accepted_prefix_count / proposals",
        "tail_policy": (
            "reserve one target token; count zero-draft tail; truncate at accepted EOS"
        ),
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key != "checkpoint_identity"
        },
        **(metadata or {}),
    }
    helpers.atomic_write(
        args.output_dir / "eval_config.json",
        json.dumps(record, default=str, indent=2) + "\n",
    )
    for role, name in (
        ("target", "checkpoint_sha256.txt"),
        ("draft", "drafter_checkpoint_sha256.txt"),
    ):
        identity = manifest[role]
        if "files" in identity:
            hashes = [
                f"# Resolved path: {identity['resolved_path']}",
                f"# Checkpoint SHA256: {identity['checkpoint_sha256']}",
            ]
            hashes.extend(
                f"{item['sha256']}  {filename}"
                for filename, item in identity["files"].items()
            )
            helpers.atomic_write(args.output_dir / name, "\n".join(hashes) + "\n")
