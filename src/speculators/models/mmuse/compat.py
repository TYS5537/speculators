"""Migration of pre-MMuse config identities without renaming checkpoint tensors."""

import warnings
from copy import deepcopy
from typing import Any

MMUSE_ENABLED_FLAGS = (
    "dflash_context_residual",
    "dflash_block_position_embedding",
    "dflash_gated_layer_fusion",
    "dflash2_dynamic_conv",
    "dflash2_candidate_selector",
    "enable_correction_head",
    "correction_hidden_aux_loss",
    "correction_hidden_feedback",
    "correction_project_corrected_hidden",
    "correction_with_markov",
    "correction_lm_head_fusion",
    "correction_rollout_metrics",
    "correction_base_diagnostics",
)


def migrate_legacy_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Rename Muse and enhanced DSpark configs; leave baselines untouched.

    Returning the same dictionary denotes no migration. This function changes
    only model identity, not tensor names, architecture options or defaults.
    DFlash extensions are not automatically converted: MMuse's DSpark training
    objective is not interchangeable with the old DFlash training objective.
    """
    model_type = config.get("speculators_model_type")
    if model_type not in ("muse", "dflash", "dspark"):
        return config
    enhanced = any(config.get(field, False) for field in MMUSE_ENABLED_FLAGS) or (
        config.get("selector_correction_feedback", "static") != "static"
        or config.get("correction_output_mode", "hidden") != "hidden"
    )
    if not enhanced and model_type != "muse":
        return config
    if model_type == "dflash":
        raise ValueError(
            "This legacy DFlash checkpoint enables architecture extensions now "
            "owned by MMuse. Automatic migration would change its training/loss "
            "semantics. Use the pre-split implementation for this checkpoint, or "
            "start a fresh model with --speculator-type mmuse. Do not simply rename "
            "the checkpoint's model type."
        )
    migrated = dict(config)
    migrated["speculators_model_type"] = "mmuse"
    migrated["architectures"] = ["MMuseDraftModel"]
    if isinstance(config.get("speculators_config"), dict):
        migrated["speculators_config"] = dict(config["speculators_config"])
        migrated["speculators_config"]["algorithm"] = "mmuse"
    source = "a legacy Muse" if model_type == "muse" else "an enhanced legacy DSpark"
    warnings.warn(
        f"Loading {source} checkpoint as MMuse. Architecture "
        "options and state-dict tensor names are preserved; the next save will "
        "record speculators_model_type='mmuse'.",
        UserWarning,
        stacklevel=3,
    )
    return migrated


def normalize_mmuse_checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    """Standardize saved MMuse metadata without reconstructing model configs.

    Keep explicit decoder settings, extension flags, unknown fields and original
    version/provenance metadata verbatim. Rebuilding through Transformers here
    could rewrite defaults using the converter's installed library version.
    """
    if not isinstance(config, dict):
        raise ValueError("Expected a checkpoint config JSON object.")
    speculators_config = config.get("speculators_config")
    if not isinstance(speculators_config, dict) or not isinstance(
        config.get("transformer_layer_config"), dict
    ):
        raise ValueError("Expected a full Speculators checkpoint config.")
    if speculators_config.get("algorithm") not in {"mmuse", "muse", "dspark"}:
        raise ValueError("Expected a MMuse or legacy enhanced-DSpark algorithm.")

    normalized = deepcopy(migrate_legacy_model_config(config))
    if normalized.get("speculators_model_type") != "mmuse":
        raise ValueError("Baseline DSpark and other algorithms cannot become MMuse.")
    normalized["architectures"] = ["MMuseDraftModel"]
    normalized["speculators_config"]["algorithm"] = "mmuse"

    auto_map = normalized.get("auto_map")
    if isinstance(auto_map, dict):
        auto_map.pop("", None)
        if not auto_map:
            normalized.pop("auto_map")

    # This optional initialization setting was absent in older checkpoints.
    # None is the legacy default and does not change saved weights or inference.
    normalized.setdefault("markov_init_std", None)
    return normalized
