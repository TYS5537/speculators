"""Utility functions for DFlash draft model."""

import torch


def get_base_indices_for_anchored_blocks(
    anchor_positions: torch.Tensor,  # shape: [1, num_anchors]
    block_size: int,
) -> torch.Tensor:  # shape: [num_anchors*block_size]
    anchor_positions = anchor_positions.to(dtype=torch.long).view(-1)
    # dtype: long, shape: [num_anchors]

    offsets = torch.arange(block_size, device=anchor_positions.device, dtype=torch.long)
    idx = (
        anchor_positions[:, None] + offsets[None, :]
    )  # shape: [num_anchors, block_size]

    return idx.reshape(-1)


def select_anchors(
    loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
    num_anchors: int,
    block_size: int,
    *,
    document_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly select anchor positions from valid tokens in sequence.

    Args:
        loss_mask: Binary mask indicating valid positions [1, total_seq_len]
        n: Number of anchors to select per batch item
        block_size: Block size (last block_size positions excluded)
        document_ids: Optional packed document IDs; -1 denotes padding.

    Returns:
        tuple: (anchors, anchor_valid)
            - anchors: Selected anchor indices [num_anchors]
            - anchor_valid: Boolean mask for valid anchors [num_anchors]
    """
    if loss_mask.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"Expected [B, T], got {loss_mask.shape}")

    if block_size <= 0:
        raise ValueError(f"Expected block size > 0, got {block_size}")

    valid_mask = loss_mask.bool().clone()
    valid_mask[:, -block_size:] = False
    if document_ids is not None:
        if document_ids.shape != loss_mask.shape:
            raise ValueError("document_ids and loss_mask must have the same shape")
        # Both layouts need at least one token after the anchor in its document.
        # Keep short/partial blocks; their invalid tail is masked separately.
        has_successor = torch.zeros_like(valid_mask)
        has_successor[:, :-1] = (document_ids[:, :-1] >= 0) & (
            document_ids[:, :-1] == document_ids[:, 1:]
        )
        valid_mask &= has_successor

    valid_indices = torch.nonzero(valid_mask.squeeze(0), as_tuple=False).squeeze(
        -1
    )  # shape: [num_non_zero]

    device = loss_mask.device
    anchors = torch.zeros(num_anchors, dtype=torch.long, device=device)
    anchor_valid = torch.zeros(num_anchors, dtype=torch.bool, device=device)

    k = min(num_anchors, valid_indices.numel())

    # Constrain value of k for torch dynamo
    torch._check(k <= valid_indices.numel())  # noqa: SLF001
    torch._check(k >= 0)  # noqa: SLF001

    perm = torch.randperm(valid_indices.numel(), device=loss_mask.device)
    # Contiguous anchors let flex attention use dense (fast) blocks instead of
    # scattered all-partial (slow) ones; the order never affects the loss.
    anchors[:k] = torch.sort(torch.gather(valid_indices, 0, perm[:k])).values
    anchor_valid[:k] = True

    return anchors, anchor_valid
    # shape: [num_anchors], [num_anchors]


def build_anchored_loss_mask(
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    anchor_valid: torch.Tensor,
    block_size: int,
    *,
    sample_from_anchor: bool,
) -> torch.Tensor:
    """Mask padded blocks and targets beyond each anchor's packed document.

    Preserve the existing per-position supervision convention. In the next-token
    layout, also check position + 1: logits at a document's final token predict
    outside that document even though the logits themselves are still inside it.
    Partial blocks retain their valid prefix instead of discarding short samples.
    """
    indices = get_base_indices_for_anchored_blocks(anchor_positions, block_size)
    seq_len = loss_mask.shape[1]
    source_indices = indices.clamp(max=seq_len - 1)
    target_indices = indices + int(sample_from_anchor)
    anchor_docs = document_ids[:, anchor_positions.reshape(-1)].repeat_interleave(
        block_size, dim=1
    )
    valid = (
        anchor_valid.reshape(1, -1).repeat_interleave(block_size, dim=1)
        & (anchor_docs >= 0)
        & (indices.unsqueeze(0) < seq_len)
        & (target_indices.unsqueeze(0) < seq_len)
        & (document_ids[:, source_indices] == anchor_docs)
        & (document_ids[:, target_indices.clamp(max=seq_len - 1)] == anchor_docs)
    )
    if not sample_from_anchor:
        valid[:, ::block_size] = False
    return loss_mask[:, source_indices] * valid.to(loss_mask.dtype)
