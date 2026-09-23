"""Evaluation-only checkpoint relocation; training/exporter identities stay strict.

Paths locate weights, while the existing checkpoint/header fingerprint identifies
them. That fingerprint includes shard mtimes, so copies must preserve timestamps.
It is not a full tensor-content hash.
"""

import json
from pathlib import Path

from speculators_dsv4 import HS_FORMAT
from speculators_dsv4.contract import MANIFEST, make_manifest, validate_layers
from speculators_dsv4.training_contract import validate_draft_contract


def validate_eval_manifest(actual, expected):
    """Ignore only the host-local model path and exporter-owned runtime override."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise ValueError("Invalid DSV4 evaluation manifest")
    for manifest in (actual, expected):
        if (
            not isinstance(manifest.get("model_path"), str)
            or not manifest["model_path"]
        ):
            raise ValueError("Invalid DSV4 evaluation manifest model_path")
    actual_identity = dict(actual)
    expected_identity = dict(expected)
    actual_identity.pop("model_path", None)
    expected_identity.pop("model_path", None)
    actual_identity.pop("runtime_quantization", None)
    if actual_identity != expected_identity:
        raise ValueError(
            "DSV4 HS contract/target mismatch: checkpoint signature and HS fields "
            "must match. Copies must preserve shard timestamps; do not edit the "
            "manifest to bypass identity checks."
        )


def read_eval_manifest(directory, expected):
    path = Path(directory) / MANIFEST
    if not path.is_file():
        raise ValueError(f"Missing {path}; start the DSV4 HS server first.")
    actual = json.loads(path.read_text(encoding="utf-8"))
    validate_eval_manifest(actual, expected)
    return actual


def validate_draft_target(saved, report):
    """Validate saved provenance before rebinding the borrowed-weight directory."""
    if saved.get("target_hidden_state_format") != HS_FORMAT:
        raise ValueError("Draft checkpoint must use the DSV4 hidden-state format")
    layers = saved.get("aux_hidden_state_layer_ids")
    if not isinstance(layers, list):
        raise ValueError("Draft checkpoint must specify its auxiliary HS layer IDs")
    validate_layers(layers)
    target_path = (
        saved.get("speculators_config", {}).get("verifier", {}).get("name_or_path")
    )
    if not isinstance(target_path, str) or not target_path:
        raise ValueError("Draft checkpoint must specify its verifier model path")
    contract = saved.get("target_training_contract")
    if contract is not None:
        # Check the original redundant config fields before replacing any path.
        validate_draft_contract(saved, contract, source="evaluation draft config")
        validate_eval_manifest(contract, make_manifest(report, layers))
    elif Path(target_path).resolve() != Path(report["model_path"]).resolve():
        raise ValueError(
            "Relocating a draft verifier requires target_training_contract to "
            "verify its saved checkpoint signature. This legacy draft has none; "
            "keep its original verifier path or audit its training provenance."
        )
    return layers


def bind_draft_verifier(config, report):
    """Change only the in-memory eval config, before loading any borrowed weights."""
    validate_draft_target(config.to_dict(), report)
    config.speculators_config.verifier.name_or_path = report["model_path"]
