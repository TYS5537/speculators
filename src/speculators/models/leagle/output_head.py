"""Runtime-only sharing of two identical frozen output matrices."""

import torch
from torch.distributed.tensor import DTensor


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
