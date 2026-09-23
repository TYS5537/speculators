"""Stateless tensor operations for teacher-forced parallel Correction.

The model owns mode selection, head calls and LM-head projections. These helpers
preserve active-slot alignment, dtype conversions and anchor restoration without
registering parameters, caching tensors or changing gradient boundaries.
"""

import torch


def build_parallel_logit_kwargs(
    targets: torch.Tensor,
    block_positions: torch.Tensor,
    active_positions: torch.Tensor,
    *,
    num_blocks: int,
    block_size: int,
    start_position: int,
    selector_previous_rank_features: torch.Tensor | None,
    selector_previous_logits_mask: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    """Use compact Selector feedback or shift dense teacher logits, without detach."""
    correction_kwargs: dict[str, torch.Tensor | None] = {}
    if selector_previous_logits_mask is not None:
        previous_target_logits = None
        previous_target_mask = selector_previous_logits_mask[:, start_position:]
        if selector_previous_rank_features is not None:
            correction_kwargs["previous_rank_features"] = (
                selector_previous_rank_features[:, start_position:]
            )
    else:
        target_blocks = targets.view(num_blocks, block_size, -1)
        previous_target_logits = target_blocks[:, :-1]
        if start_position == 0:
            previous_target_logits = torch.cat(
                [torch.zeros_like(target_blocks[:, :1]), previous_target_logits], dim=1
            )
            previous_target_mask = block_positions > 0
        else:
            previous_target_mask = torch.ones_like(active_positions, dtype=torch.bool)
    correction_kwargs["previous_logits"] = previous_target_logits
    correction_kwargs["previous_logits_mask"] = previous_target_mask
    return correction_kwargs


def add_parallel_hidden_residual(
    hidden_blocks: torch.Tensor,
    delta_hidden: torch.Tensor | None,
    *,
    start_position: int,
) -> torch.Tensor | None:
    """Add the hidden residual on an independent view, restoring any reserved anchor."""
    if delta_hidden is None:
        return None
    # Do not accept/reuse Correction's active_hidden view: sharing its autograd
    # node changes accumulation order even though forward values are identical.
    residual_hidden = (
        hidden_blocks if start_position == 0 else hidden_blocks[:, start_position:]
    )
    corrected_hidden = residual_hidden + delta_hidden.to(hidden_blocks.dtype)
    if start_position:
        corrected_hidden = torch.cat(
            [hidden_blocks[:, :start_position], corrected_hidden], dim=1
        )
    return corrected_hidden


def add_parallel_projected_residual(
    projected_logits: torch.Tensor,
    residual: torch.Tensor,
    *,
    num_blocks: int,
    start_position: int,
    mask_tokens_size: int,
) -> torch.Tensor:
    """Add a zero-padded logit residual after full-block corrected-hidden projection."""
    if start_position:
        residual = torch.cat(
            [
                residual.new_zeros(num_blocks, start_position, residual.shape[-1]),
                residual,
            ],
            dim=1,
        )
    return projected_logits + residual.reshape(1, mask_tokens_size, -1).to(
        projected_logits.dtype
    )


def add_parallel_base_residual(
    base_logits_blocks: torch.Tensor,
    residual: torch.Tensor,
    *,
    start_position: int,
    mask_tokens_size: int,
) -> torch.Tensor:
    """Add only on active slots; preserve the reserved anchor's original base logits."""
    logits_blocks = base_logits_blocks[:, start_position:] + residual.to(
        base_logits_blocks.dtype
    )
    if start_position:
        logits_blocks = torch.cat(
            [base_logits_blocks[:, :start_position], logits_blocks], dim=1
        )
    return logits_blocks.reshape(1, mask_tokens_size, -1)
