"""Training model-configuration compatibility and checkpoint override policy."""

import argparse
import logging

from transformers import PretrainedConfig

from speculators.models.muse.config import MUSE_OPTION_FIELDS, validate_muse_options

# CLI flags that synthesize the draft decoder shape. They conflict with both
# --from-pretrained and --draft-config, each of which fully defines the draft.
DECODER_SHAPING_FLAGS: dict[str, str] = {
    "num_layers": "--num-layers",
    "draft_arch": "--draft-arch",
    "draft_hidden_act": "--draft-hidden-act",
    "sliding_window": "--sliding-window",
    "full_attention_indices": "--full-attention-indices",
}

# Model-config options that would otherwise be silently ignored when a complete
# checkpoint is restored via --from-pretrained. Most change parameter shapes or
# learned proposal semantics and therefore must match the checkpoint. The small
# runtime-only subset below can safely change without adding/removing weights.
PRETRAINED_MODEL_CONFIG_FLAGS: dict[str, str] = {
    "target_hidden_state_format": "--target-hidden-state-format",
    "block_size": "--block-size",
    "sample_from_anchor": "--sample-from-anchor",
    "sliding_window_non_causal": "--sliding-window-non-causal",
    "dflash_context_residual": "--dflash-context-residual",
    "dflash_block_position_embedding": "--dflash-block-position-embedding",
    "dflash_gated_layer_fusion": "--dflash-gated-layer-fusion",
    "dflash2_dynamic_conv": "--dflash2-dynamic-conv",
    "dflash2_conv_kernel_size": "--dflash2-conv-kernel-size",
    "dflash2_conv_group_size": "--dflash2-conv-group-size",
    "dflash2_candidate_selector": "--dflash2-candidate-selector",
    "dflash2_selector_rank": "--dflash2-selector-rank",
    "dflash2_selector_top_k": "--dflash2-selector-top-k",
    "dflash2_selector_search_mode": (
        "--dflash2-selector-greedy/--dflash2-selector-global"
    ),
    "dflash2_selector_loss_weight": "--dflash2-selector-loss-weight",
    "markov_rank": "--markov-rank",
    "markov_head_type": "--markov-head-type",
    "enable_correction_head": "--enable-correction-head",
    "correction_output_mode": "--correction-output-mode",
    "correction_hidden_size": "--correction-hidden-size",
    "correction_rank": "--correction-rank",
    "correction_lm_head_fusion": "--correction-lm-head-fusion",
    "correction_num_layers": "--correction-num-layers",
    "correction_num_heads": "--correction-num-heads",
    "correction_gate_bias": "--correction-gate-bias",
    "correction_hidden_aux_loss": "--correction-hidden-aux-loss",
    "correction_hidden_aux_weight": "--correction-hidden-aux-weight",
    "correction_hidden_feedback": "--correction-hidden-feedback",
    "selector_correction_feedback": "--selector-correction-feedback",
    "correction_project_corrected_hidden": ("--correction-project-corrected-hidden"),
    "correction_with_markov": "--correction-with-markov",
    "correction_markov_gate_bias": "--correction-markov-gate-bias",
    "correction_rollout_metrics": "--correction-rollout-metrics",
    "correction_base_diagnostics": "--correction-base-diagnostics",
    "enable_confidence_head": "--enable-confidence-head",
    "confidence_head_with_markov": "--confidence-head-with-markov",
    "confidence_detach_features": "--confidence-detach-features",
}

PRETRAINED_RUNTIME_CONFIG_FIELDS = {
    "correction_lm_head_fusion",
    "correction_hidden_aux_weight",
    "correction_rollout_metrics",
    "correction_base_diagnostics",
    "confidence_detach_features",
    "dflash2_selector_loss_weight",
    "dflash2_selector_search_mode",
}

MUSE_MODEL_CONFIG_FIELDS = MUSE_OPTION_FIELDS


def plan_pretrained_config_overrides(
    args: argparse.Namespace,
    config: PretrainedConfig,
    provided: set[str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Collect inherited arguments and allowed overrides without mutating either."""
    incompatible: list[str] = []
    inherited_args: dict[str, object] = {}
    runtime_overrides: dict[str, object] = {}

    for dest, flag in PRETRAINED_MODEL_CONFIG_FLAGS.items():
        if not hasattr(config, dest) or not hasattr(args, dest):
            continue
        checkpoint_value = getattr(config, dest)
        cli_value = getattr(args, dest)
        if dest not in provided:
            # Trainer kwargs are built from args later. Inherit the saved value so
            # parser defaults cannot silently reset the checkpoint's policy.
            inherited_args[dest] = checkpoint_value
            continue
        if cli_value == checkpoint_value:
            continue
        if dest in PRETRAINED_RUNTIME_CONFIG_FIELDS:
            runtime_overrides[dest] = cli_value
        else:
            incompatible.append(
                f"{flag}={cli_value!r} (checkpoint: {checkpoint_value!r})"
            )

    if incompatible:
        raise ValueError(
            "--from-pretrained cannot change checkpoint architecture or learned "
            f"proposal semantics: {', '.join(incompatible)}. Remove the conflicting "
            "option(s), use matching values, or start a fresh model without "
            "--from-pretrained."
        )
    return inherited_args, runtime_overrides


def reconcile_pretrained_config_args(
    args: argparse.Namespace,
    config: PretrainedConfig,
    *,
    logger: logging.Logger,
) -> None:
    """Validate the complete checkpoint/CLI merge, then commit it to both objects."""
    provided = set(getattr(args, "_provided_model_config_dests", set()))
    muse_overrides = provided & MUSE_MODEL_CONFIG_FIELDS
    if muse_overrides and getattr(config, "speculators_model_type", None) != "muse":
        raise ValueError(
            "Correction, backbone enhancement and Selector options require a "
            "Muse checkpoint; --from-pretrained cannot add them to a baseline "
            "checkpoint. Start a fresh model with --speculator-type muse."
        )

    inherited_args, runtime_overrides = plan_pretrained_config_overrides(
        args, config, provided
    )

    # Validate the final saved-config/CLI combination before mutating either
    # object. Parser defaults are not checkpoint values, and an invalid runtime
    # override must not leave an otherwise reusable config partially changed.
    if getattr(config, "speculators_model_type", None) == "muse":
        candidate = {
            field: getattr(config, field)
            for field in PRETRAINED_MODEL_CONFIG_FLAGS
            if hasattr(config, field)
        }
        validate_muse_options({**candidate, **runtime_overrides})

    for dest, value in inherited_args.items():
        setattr(args, dest, value)
    for dest, value in runtime_overrides.items():
        setattr(config, dest, value)
    if runtime_overrides:
        logger.info(
            "Applied explicit runtime-only overrides to pretrained config: %s",
            ", ".join(
                f"{PRETRAINED_MODEL_CONFIG_FLAGS[dest]}={value!r}"
                for dest, value in runtime_overrides.items()
            ),
        )


def validate_draft_init_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    provided: set[str],
) -> None:
    """Enforce the draft-init contract.

    The draft model may be defined in exactly one way:

    * ``--from-pretrained`` -- load a complete speculator checkpoint (or a
      config-only directory); or
    * ``--draft-config`` -- load just the decoder config and build the rest of
      the speculator from the other CLI args; or
    * the decoder-shaping flags (``--num-layers`` etc.) -- synthesize everything.

    ``--from-pretrained`` takes precedence over all other model-definition
    options: it is mutually exclusive with ``--draft-config`` and with the
    decoder-shaping flags, since those values come from the checkpoint.
    ``--draft-config`` is likewise incompatible with the decoder-shaping flags.
    MTP from scratch (``--speculator-type mtp`` without ``--from-pretrained``)
    reuses the verifier's own decoder config, so ``--draft-config`` and the
    decoder-shaping flags do not apply and are rejected.

    ``provided`` is the set of decoder-shaping dests the user explicitly passed
    (see :func:`speculators.utils.argparse_utils.explicitly_provided_dests`); a flag
    passed at its default value still counts as a conflict.
    """
    shaping = [flag for dest, flag in DECODER_SHAPING_FLAGS.items() if dest in provided]
    if args.from_pretrained:
        conflicting = shaping + (["--draft-config"] if args.draft_config else [])
        if conflicting:
            parser.error(
                "--from-pretrained loads a complete draft model and takes precedence "
                "over all other model-definition options, so these conflict with it "
                f"(remove them): {', '.join(conflicting)}"
            )
        return
    if args.speculator_type == "mtp":
        # MTP-from-scratch reuses the verifier's own decoder config and extracts the
        # native MTP head weights; --draft-config and the decoder-shaping flags do not
        # apply, so reject them rather than silently ignoring them.
        conflicting = shaping + (["--draft-config"] if args.draft_config else [])
        if conflicting:
            parser.error(
                "--speculator-type mtp reuses the verifier's decoder config, so these "
                f"options do not apply (remove them): {', '.join(conflicting)}"
            )
        return
    if args.draft_config and shaping:
        parser.error(
            "--draft-config defines the draft decoder, so these flags conflict with "
            f"it (remove them): {', '.join(shaping)}"
        )
