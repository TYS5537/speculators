"""Token alignment and sparse predecessor rows for static Selector conditioning.

The model owns validation, vocabulary mapping, proposal semantics and gradient
boundaries. These helpers keep the existing shifts and allocation order; compact
encoding and insertion of initial dense logits remain in the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from speculators.models.mmuse.runtime_types import (
    SelectorCorrectionInputs,
    SelectorPreviousCandidates,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def align_selector_token_ids(
    selected_draft_ids: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    *,
    block_size: int,
    start_position: int,
    draft_to_verifier: Callable[[torch.Tensor], torch.Tensor],
) -> SelectorCorrectionInputs:
    """Map current IDs and shift previous IDs, preserving reserved anchor slots."""
    current_ids = draft_to_verifier(selected_draft_ids)
    previous_ids = anchor_token_ids[:, None].expand(-1, block_size).clone().long()
    if start_position + 1 < block_size:
        previous_ids[:, start_position + 1 :] = draft_to_verifier(
            selected_draft_ids[:, start_position:-1]
        )
    return SelectorCorrectionInputs(
        current_token_ids=current_ids,
        previous_token_ids=previous_ids,
        previous_rank_features=None,
        previous_logits_mask=None,
    )


def shift_selector_candidates(
    candidate_ids: torch.Tensor,
    proposal_logits: torch.Tensor,
    *,
    sample_from_anchor: bool,
    initial_previous_logits: torch.Tensor | None,
) -> SelectorPreviousCandidates:
    """Shift detached proposal rows without encoding or validating dense logits.

    With a reserved anchor, the caller checks and encodes its initial dense logits
    after sparse encoding. Moving that check here would change failure ordering.
    """
    num_blocks, block_size, _ = candidate_ids.shape
    start_position = 0 if sample_from_anchor else 1
    previous_candidate_ids = torch.zeros_like(candidate_ids)
    previous_candidate_logits = torch.zeros_like(proposal_logits)
    sparse_mask = torch.zeros(
        num_blocks,
        block_size,
        dtype=torch.bool,
        device=candidate_ids.device,
    )
    if sample_from_anchor:
        if initial_previous_logits is not None:
            raise ValueError(
                "Selector initial logits are only valid when sample_from_anchor=False"
            )
        if block_size > 1:
            previous_candidate_ids[:, 1:] = candidate_ids[:, :-1]
            previous_candidate_logits[:, 1:] = proposal_logits[:, :-1]
            sparse_mask[:, 1:] = True
    elif start_position + 1 < block_size:
        previous_candidate_ids[:, start_position + 1 :] = candidate_ids[
            :, start_position:-1
        ]
        previous_candidate_logits[:, start_position + 1 :] = proposal_logits[
            :, start_position:-1
        ]
        sparse_mask[:, start_position + 1 :] = True
    return SelectorPreviousCandidates(
        candidate_ids=previous_candidate_ids,
        candidate_logits=previous_candidate_logits,
        mask=sparse_mask,
    )
