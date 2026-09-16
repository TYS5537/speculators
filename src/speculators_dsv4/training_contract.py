"""Fail-closed training identities, independent of torch and transformers.

The signature is the existing checkpoint/header fingerprint, not a full tensor
payload hash. A contract records a verified exporter manifest; it does not claim
that teacher probabilities or accelerator kernels have passed numerical checks.
"""

import json
from pathlib import Path

from speculators_dsv4 import HS_FORMAT
from speculators_dsv4.contract import MANIFEST, make_manifest, validate_layers

_CONTRACT_FIELDS = {
    "schema_version",
    "format",
    "model_path",
    "checkpoint_signature",
    "auxiliary_hs_ids",
    "teacher_hs_id",
    "hidden_size",
    "target_dtype",
    "runtime_quantization",
}


def validate_training_contract(value):
    """Return an independent JSON-compatible copy of the fixed v1 identity."""
    if not isinstance(value, dict) or set(value) != _CONTRACT_FIELDS:
        raise ValueError("Invalid DSV4 target_training_contract fields/schema.")
    fixed = {
        "schema_version": 1,
        "format": HS_FORMAT,
        "teacher_hs_id": 43,
        "hidden_size": 4096,
        "target_dtype": "bfloat16",
    }
    for name, expected in fixed.items():
        if type(value[name]) is not type(expected) or value[name] != expected:
            raise ValueError(f"Invalid DSV4 training contract {name}.")
    for name in ("model_path", "checkpoint_signature"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"Invalid DSV4 training contract {name}.")
    if not isinstance(value["auxiliary_hs_ids"], list):
        raise ValueError("Invalid DSV4 training contract auxiliary_hs_ids.")
    validate_layers(value["auxiliary_hs_ids"])
    runtime = value["runtime_quantization"]
    if not isinstance(runtime, dict) or set(runtime) != {"method"}:
        raise ValueError("Invalid DSV4 training contract runtime_quantization.")
    method = runtime["method"]
    if method is not None and (not isinstance(method, str) or not method.strip()):
        raise ValueError("Invalid DSV4 runtime quantization method.")
    return json.loads(json.dumps(value))


def read_training_contract(directory, report, layers):
    """Bind to the exporter's actual quantization choice, including auto/None."""
    path = Path(directory) / MANIFEST
    if not path.is_file():
        raise ValueError(
            f"Missing {path}; start the DSV4 HS server first. "
            "DSV4 --dry-run also requires its verified HS manifest."
        )
    actual = json.loads(path.read_text(encoding="utf-8"))
    if "runtime_quantization" not in actual:
        raise ValueError(
            f"DSV4 HS manifest {path} does not record runtime_quantization. "
            "Use the current exporter with a fresh HS directory; old activations "
            "cannot be relabeled as a verified training identity."
        )
    actual = validate_training_contract(actual)
    expected = make_manifest(report, layers)
    expected["runtime_quantization"] = actual["runtime_quantization"]
    if actual != expected:
        raise ValueError(f"DSV4 HS contract/target mismatch: {path}.")
    return actual


def validate_draft_contract(saved, expected, *, source):
    """Check saved identity and its redundant model-config fields before loading."""
    expected = validate_training_contract(expected)
    if not isinstance(saved, dict):
        raise ValueError(f"Invalid DSV4 draft configuration: {source}.")
    if saved.get("target_training_contract") is None:
        raise ValueError(
            f"DSV4 checkpoint {source} lacks target_training_contract; its target "
            "weights/runtime quantization identity cannot be proven. Refusing "
            "resume or --from-pretrained. Do not add the current manifest to old "
            "weights. Start a new run from --draft-config in a new save directory; "
            "legacy weight migration needs separately audited original provenance."
        )
    actual = validate_training_contract(saved["target_training_contract"])
    changed = [key for key in sorted(expected) if expected[key] != actual[key]]
    if changed:
        raise ValueError(
            f"DSV4 checkpoint identity mismatch at {source}: {', '.join(changed)}. "
            "A fresh HS directory does not authorize restoring another identity."
        )
    if saved.get("target_hidden_state_format") != actual["format"]:
        raise ValueError(f"DSV4 saved HS format disagrees with its contract: {source}.")
    if saved.get("aux_hidden_state_layer_ids") != actual["auxiliary_hs_ids"]:
        raise ValueError(f"DSV4 saved HS layers disagree with its contract: {source}.")
    target = saved.get("speculators_config", {}).get("verifier", {}).get("name_or_path")
    if (
        not isinstance(target, str)
        or not target
        or Path(target).resolve() != Path(actual["model_path"]).resolve()
    ):
        raise ValueError(
            f"DSV4 saved target path disagrees with its contract: {source}."
        )


def validate_resume_contract(model_config, checkpoint_path):
    """Validate the actual auto-resume config, not just its model state_dict."""
    expected = model_config.get("target_training_contract")
    # Validate the live config too; direct Trainer callers must not bypass binding.
    validate_draft_contract(model_config, expected, source="current draft config")
    if checkpoint_path is None:
        return
    path = Path(checkpoint_path) / "config.json"
    if not path.is_file():
        raise ValueError(f"Missing DSV4 resume configuration: {path}.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    validate_draft_contract(saved, expected, source=str(path))


def distributed_validation(check, distributed=None):
    """Every rank reaches error exchange before any rank enters weight collectives.

    Passing the torch.distributed module keeps this module usable in stdlib-only
    tooling/tests. Unlike a rank-zero-only guard, a local filesystem failure on
    any rank aborts all ranks before DDP/FSDP setup and optimizer restoration.
    """
    if distributed is None:
        return check()
    result = None
    error = None
    try:
        result = check()
    except Exception as exc:  # noqa: BLE001 -- synchronize errors before re-raising.
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * distributed.get_world_size()
    distributed.all_gather_object(errors, error)
    failures = [f"rank {rank}: {item}" for rank, item in enumerate(errors) if item]
    if failures:
        raise ValueError("DSV4 training preflight failed: " + "; ".join(failures))
    return result
