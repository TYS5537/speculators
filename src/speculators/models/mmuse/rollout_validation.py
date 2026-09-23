"""Shape-only Selector conditioning checks for Correction rollout.

These helpers neither transform tensors nor encode distributions. The caller
owns model/block checks and base-logit requirements; the Correction head still
owns feature-width validation. Keep check order and error messages stable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def validate_selector_token_ids(
    conditioning_current_ids: Tensor | None,
    conditioning_previous_ids: Tensor | None,
    *,
    expected_block_shape: tuple[int, int],
    online_selector: bool,
) -> None:
    """Check static-path conflicts, current shape, pairing, then previous shape."""
    if online_selector and conditioning_current_ids is not None:
        raise ValueError("Corrected Selector feedback cannot use a static path")
    if conditioning_current_ids is not None and (
        conditioning_current_ids.shape != expected_block_shape
    ):
        raise ValueError(
            "Expected conditioning_current_ids shape "
            f"{expected_block_shape}, got {tuple(conditioning_current_ids.shape)}"
        )
    if (conditioning_current_ids is None) != (conditioning_previous_ids is None):
        raise ValueError(
            "Selector current and previous token IDs must be provided together"
        )
    if conditioning_previous_ids is not None and (
        conditioning_previous_ids.shape != expected_block_shape
    ):
        raise ValueError(
            "Expected conditioning_previous_ids shape "
            f"{expected_block_shape}, got {tuple(conditioning_previous_ids.shape)}"
        )


def validate_selector_logit_features(
    conditioning_previous_rank_features: Tensor | None,
    conditioning_previous_logits_mask: Tensor | None,
    *,
    expected_block_shape: tuple[int, int],
) -> bool:
    """Check compact feature/mask pairing and alignment, returning their presence.

    Features need not have a static token path. Their rank width, dtype and device
    are deliberately not validated or converted at this boundary.
    """
    has_conditioning_features = conditioning_previous_rank_features is not None
    if has_conditioning_features != (conditioning_previous_logits_mask is not None):
        raise ValueError(
            "Selector compact logit features and mask must be provided together"
        )
    if has_conditioning_features:
        assert conditioning_previous_logits_mask is not None  # noqa: S101
        if conditioning_previous_logits_mask.shape != expected_block_shape:
            raise ValueError(
                "Selector previous-logit mask must align with block positions"
            )
        if conditioning_previous_rank_features is not None and (
            conditioning_previous_rank_features.shape[:-1] != expected_block_shape
        ):
            raise ValueError("Selector previous-rank features must align")
    return has_conditioning_features
