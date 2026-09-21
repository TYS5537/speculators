"""Migration of pre-Muse config identities without renaming checkpoint tensors."""

import warnings
from typing import Any

MUSE_ENABLED_FLAGS = (
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
    """Route enhanced DSpark configs to Muse; leave baseline configs untouched.

    Returning the same dictionary denotes no migration. This function changes
    only model identity, not tensor names, architecture options or defaults.
    DFlash extensions are not automatically converted: Muse's DSpark training
    objective is not interchangeable with the old DFlash training objective.
    """
    model_type = config.get("speculators_model_type")
    if model_type not in ("dflash", "dspark"):
        return config
    enhanced = any(config.get(field, False) for field in MUSE_ENABLED_FLAGS) or (
        config.get("selector_correction_feedback", "static") != "static"
        or config.get("correction_output_mode", "hidden") != "hidden"
    )
    if not enhanced:
        return config
    if model_type == "dflash":
        raise ValueError(
            "This legacy DFlash checkpoint enables architecture extensions now "
            "owned by Muse. Automatic migration would change its training/loss "
            "semantics. Use the pre-split implementation for this checkpoint, or "
            "start a fresh model with --speculator-type muse. Do not simply rename "
            "the checkpoint's model type."
        )
    migrated = dict(config)
    migrated["speculators_model_type"] = "muse"
    migrated["architectures"] = ["MuseSpeculator"]
    if isinstance(config.get("speculators_config"), dict):
        migrated["speculators_config"] = dict(config["speculators_config"])
        migrated["speculators_config"]["algorithm"] = "muse"
    warnings.warn(
        "Loading an enhanced legacy DSpark checkpoint as Muse. Architecture "
        "options and state-dict tensor names are preserved; the next save will "
        "record speculators_model_type='muse'.",
        UserWarning,
        stacklevel=3,
    )
    return migrated
