"""Two optional master switches for LEAGLE training execution optimizations.

Masters take precedence over individual settings, including saved run settings.
None preserves the existing individual controls. The safe group means no extra
gradient differences were observed against its prerequisite path in CPU tests;
it is not a guarantee of bitwise equivalence on every accelerator.
"""

from collections.abc import Mapping

GROUP_DEFAULTS = {
    "leagle_safe_optimizations": None,
    "leagle_numerical_optimizations": None,
}

SAFE_OPTIONS = (
    "leagle_teacher_cache",
    "leagle_teacher_cache_supervised",
    "leagle_cache_frozen_head_weights",
    "leagle_cache_teacher_indices",
    "leagle_selective_checkpoint",
    "leagle_cache_attention_masks",
    "leagle_fsdp_keep_forward_params",
    "leagle_share_output_head",
    "leagle_ddp_gradient_as_bucket_view",
)
NUMERICAL_OPTIONS = (
    "leagle_reuse_training_kv",
    "leagle_chunked_loss",
    "leagle_sparse_attention",
    "leagle_candidate_grad_addmm",
    "leagle_masked_head",
    "leagle_shared_loss_normalization",
    "leagle_stream_main_loss",
    "leagle_align_candidate_chunks",
    "leagle_ddp_batched_grad_copy",
)
_CANDIDATE_OPTIONS = (
    "leagle_candidate_grad_addmm",
    "leagle_align_candidate_chunks",
)
_CHUNKED_OPTIONS = (
    "leagle_chunked_loss",
    "leagle_masked_head",
    "leagle_shared_loss_normalization",
    "leagle_stream_main_loss",
    "leagle_align_candidate_chunks",
    "leagle_teacher_cache",
)


def resolve_optimization_groups(options: Mapping[str, object]) -> dict[str, object]:
    """Return effective member settings without changing model/training recipes.

    Safe optimizations never introduce numerical prerequisites themselves. Turning
    off the numerical group also disables implicit chunking/sparse prerequisites,
    and selects native candidate backward recomputation when a candidate head exists.
    Calling again after checkpoint architecture resolution is intentional/idempotent.
    """
    safe, numerical = (options.get(name) for name in GROUP_DEFAULTS)
    if safe is None and numerical is None:
        return {}
    if any(
        value is not None and not isinstance(value, bool) for value in (safe, numerical)
    ):
        raise ValueError("LEAGLE optimization groups must be boolean or unset")
    if options.get("speculator_type", "leagle") != "leagle":
        raise ValueError("LEAGLE optimization groups require --speculator-type leagle")

    updates = {}
    candidate = bool(options.get("candidate_head", False))
    ddp = not bool(options.get("fsdp_shard", False))
    if numerical is not None:
        updates.update(dict.fromkeys(NUMERICAL_OPTIONS, numerical))
        for name in _CANDIDATE_OPTIONS:
            updates[name] = numerical and candidate
        updates["leagle_candidate_backward_recompute"] = candidate and not numerical
        updates["leagle_ddp_batched_grad_copy"] = numerical and ddp
        if not numerical:
            updates["leagle_ddp_bucket_cap_mb"] = None

    effective = {**options, **updates}
    chunked = numerical is not False and any(
        effective.get(name, False) for name in _CHUNKED_OPTIONS
    )
    sparse = bool(effective.get("leagle_sparse_attention", False))
    if safe is not None:
        updates.update(
            leagle_teacher_cache=safe and chunked,
            leagle_teacher_cache_supervised=safe and chunked,
            leagle_cache_frozen_head_weights=safe and chunked,
            leagle_cache_teacher_indices=safe and chunked,
            leagle_selective_checkpoint=(
                safe and sparse and bool(options.get("activation_checkpointing", False))
            ),
            leagle_cache_attention_masks=safe and sparse,
            leagle_fsdp_keep_forward_params=(
                safe and bool(options.get("fsdp_shard", False))
            ),
            leagle_share_output_head=safe,
            leagle_ddp_gradient_as_bucket_view=safe and ddp,
        )
    if numerical is False:
        # Teacher caching implicitly enables chunking in model.forward. The safe
        # master must not bring that numerical path back through a dependency.
        updates.update(
            leagle_teacher_cache=False,
            leagle_teacher_cache_supervised=False,
            leagle_cache_frozen_head_weights=False,
            leagle_cache_teacher_indices=False,
            leagle_selective_checkpoint=False,
            leagle_cache_attention_masks=False,
        )
    return updates


def apply_runtime_optimization_groups(config, args, model):
    """Resolve checkpoint capabilities and record the effective runtime settings."""
    if all(getattr(args, name, None) is None for name in GROUP_DEFAULTS):
        return
    candidate = bool(getattr(model.config, "candidate_head", False))
    updates = resolve_optimization_groups(
        {
            **vars(args),
            "speculator_type": model.config.speculators_model_type,
            "candidate_head": candidate,
        }
    )
    # get_trainer_kwargs resolves the masters too; it must see checkpoint-owned
    # capabilities rather than an unset CLI default. This records, never creates,
    # the already loaded auxiliary head.
    args.candidate_head = config.draft.candidate_head = candidate
    config._provenance["candidate_head"] = "checkpoint"  # noqa: SLF001
    for name, value in updates.items():
        setattr(args, name, value)
        setattr(config.trainer, name, value)
        config._provenance[name] = "group"  # noqa: SLF001 -- resolved run manifest.
