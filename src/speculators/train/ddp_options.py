"""Optional LEAGLE DDP controls, with explicit runtime capability reporting."""

import inspect
import logging

from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger("speculators")

DDP_FLAGS = (
    "leagle_ddp_gradient_as_bucket_view",
    "leagle_ddp_batched_grad_copy",
)
BUCKET_CAP = "leagle_ddp_bucket_cap_mb"


def validate_ddp_options(config):
    bucket = getattr(config, BUCKET_CAP, None)
    enabled = any(getattr(config, name, False) for name in DDP_FLAGS)
    if bucket is not None and (
        isinstance(bucket, bool) or not isinstance(bucket, int) or bucket <= 0
    ):
        raise ValueError("LEAGLE DDP bucket size must be a positive integer")
    if (enabled or bucket is not None) and getattr(config, "fsdp_shard", False):
        raise ValueError("LEAGLE DDP options require --no-fsdp-shard")


def resolve_ddp_options(config, *, constructor=None, distributed=True):
    """Return constructor kwargs and overrides needed to record actual behavior."""
    validate_ddp_options(config)
    constructor = DistributedDataParallel if constructor is None else constructor
    selected = {name: getattr(config, name, False) for name in DDP_FLAGS}
    bucket = getattr(config, BUCKET_CAP, None)
    if not distributed:
        updates = {name: False for name, value in selected.items() if value}
        if bucket is not None:
            updates[BUCKET_CAP] = None
        if updates:
            logger.info("LEAGLE DDP options skipped: distributed training is inactive.")
        return {}, updates
    options, updates = {}, {}
    for name, value in selected.items():
        if not value:
            continue
        argument = name.removeprefix("leagle_ddp_")
        if argument not in inspect.signature(constructor).parameters:
            logger.warning(
                "LEAGLE requested %s, but this PyTorch DDP does not support it; "
                "disabled in the effective configuration.",
                argument,
            )
            updates[name] = False
        else:
            options[argument] = True
    if bucket is not None:
        options["bucket_cap_mb"] = bucket
    return options, updates


def apply_runtime_ddp_options(config, args, *, distributed=True):
    """Persist capability fallbacks before writing run.yaml or building Trainer."""
    _, updates = resolve_ddp_options(args, distributed=distributed)
    for name, value in updates.items():
        setattr(args, name, value)
        setattr(config.trainer, name, value)
        config._provenance[name] = "runtime"  # noqa: SLF001 -- actual run behavior.
    return bool(updates)
