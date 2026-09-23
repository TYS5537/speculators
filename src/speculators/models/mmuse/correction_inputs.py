"""Shape and presence checks for the causal Correction head.

These helpers do not cast, mask, encode or detach tensors. The head owns cache
length validation, dense-distribution encoding, rank-width checks and projections.
Keep embedding/cache, hidden-feedback and previous-logit checks in that order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def validate_correction_embeddings(
    previous_token_embeddings: Tensor,
    dflash_hidden: Tensor,
    block_positions: Tensor,
    *,
    current_token_embeddings: Tensor | None,
) -> None:
    """Check previous-token alignment, current-token shape, then positions."""
    prefix_shape = dflash_hidden.shape[:-1]
    if previous_token_embeddings.shape[:-1] != prefix_shape:
        raise ValueError("previous-token embeddings and DFlash hidden must align")
    if current_token_embeddings is not None and (
        current_token_embeddings.shape != dflash_hidden.shape
    ):
        raise ValueError(
            "current selector-token embeddings and DFlash hidden must align"
        )
    if block_positions.shape != prefix_shape:
        raise ValueError("block positions and DFlash hidden must align")


def validate_hidden_feedback_inputs(
    dflash_hidden: Tensor,
    previous_corrected_hidden: Tensor | None,
    previous_corrected_hidden_mask: Tensor | None,
    *,
    enabled: bool,
) -> None:
    """Check feedback availability, pairing, hidden shape, then mask shape."""
    if not enabled:
        if (
            previous_corrected_hidden is not None
            or previous_corrected_hidden_mask is not None
        ):
            raise ValueError(
                "previous corrected hidden is only valid when hidden feedback "
                "is enabled"
            )
    else:
        if previous_corrected_hidden is None or previous_corrected_hidden_mask is None:
            raise ValueError(
                "hidden-feedback Correction requires previous corrected hidden and mask"
            )
        if previous_corrected_hidden.shape != dflash_hidden.shape:
            raise ValueError("previous corrected hidden and DFlash hidden must align")
        if previous_corrected_hidden_mask.shape != dflash_hidden.shape[:-1]:
            raise ValueError(
                "previous corrected hidden mask and DFlash hidden must align"
            )


def validate_previous_logit_inputs(
    previous_logits: Tensor | None,
    previous_logits_mask: Tensor | None,
    previous_rank_features: Tensor | None,
    *,
    output_mode: str,
    prefix_shape: tuple[int, ...],
) -> None:
    """Check mode, mask presence, representation exclusivity, then mask shape."""
    has_dense_logits = previous_logits is not None
    has_compact_features = previous_rank_features is not None
    if output_mode != "logits":
        if has_dense_logits or has_compact_features or previous_logits_mask is not None:
            raise ValueError("previous logits require logit-residual Correction mode")
    else:
        if previous_logits_mask is None:
            raise ValueError(
                "logit-aware Correction requires previous features and mask"
            )
        if has_dense_logits == has_compact_features:
            raise ValueError(
                "Provide exactly one dense or compact previous-logit representation"
            )
        if previous_logits_mask.shape != prefix_shape:
            raise ValueError("previous logits mask and DFlash hidden must align")
