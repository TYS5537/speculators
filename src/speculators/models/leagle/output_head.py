"""Runtime-only sharing and forward-local autocast copies of output matrices."""

import torch
from torch.distributed.tensor import DTensor


def training_output_weights(model, *, cache_autocast=False):
    """Reuse frozen FP32 weight casts for one chunked forward/backward graph.

    Autocast does not cache frozen parameters. Repeated chunk projections would
    otherwise cast the entire vocabulary matrix every time, including checkpoint
    replay. Never change model parameters or cache across optimizer steps. Shared
    heads reuse one copy; already low-precision and trainable weights stay intact.
    """
    weights = (model.lm_head.weight, model.verifier_lm_head.weight)
    if not cache_autocast:
        return weights
    copies = {}
    result = []
    for weight in weights:
        device_type = weight.device.type
        if (
            weight.dtype == torch.float32
            and not weight.requires_grad
            and torch.is_autocast_enabled(device_type)
            and torch.get_autocast_dtype(device_type) in (torch.bfloat16, torch.float16)
        ):
            key = id(weight)
            if key not in copies:
                copies[key] = weight.to(dtype=torch.get_autocast_dtype(device_type))
        result.append(copies.get(id(weight), weight))
    return tuple(result)


def share_output_head(model):
    draft, teacher = model.lm_head.weight, model.verifier_lm_head.weight
    if draft.requires_grad or teacher.requires_grad:
        raise ValueError("Output-head sharing requires both weights to be frozen")
    if (
        draft.is_meta
        or teacher.is_meta
        or draft.shape != teacher.shape
        or draft.dtype != teacher.dtype
        or draft.device != teacher.device
    ):
        raise ValueError("Output-head sharing requires loaded, matching matrices")
    if not torch.equal(draft, teacher) or not torch.isfinite(draft).all():
        raise ValueError(
            "Output-head sharing requires exactly identical finite weights"
        )
    if draft is teacher:
        return teacher.detach().clone()
    reference = teacher.detach()
    if draft.untyped_storage().data_ptr() == teacher.untyped_storage().data_ptr():
        reference = reference.clone()
    model.verifier_lm_head.weight = draft
    # Hold the old teacher only until setup/resume validation finishes. It is not
    # registered on the model, put into the optimizer, or saved in a checkpoint.
    return reference


@torch.no_grad()
def validate_shared_output_head(model, reference):
    draft, teacher = model.lm_head.weight, model.verifier_lm_head.weight
    if draft is not teacher or draft.requires_grad or teacher.requires_grad:
        raise ValueError("Shared output-head identity or frozen state was lost")
    actual = draft.full_tensor() if isinstance(draft, DTensor) else draft
    if not torch.equal(
        actual.to(device=reference.device, dtype=reference.dtype), reference
    ):
        raise ValueError(
            "Checkpoint output head differs from the frozen teacher; "
            "disable output-head sharing"
        )
