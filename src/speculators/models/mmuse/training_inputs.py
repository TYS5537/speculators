"""Teacher-forced block views and predecessor IDs, without validation or detach."""

import torch

from speculators.models.mmuse.runtime_types import TrainingBlocks


def prepare_training_blocks(
    input_ids: torch.Tensor,
    anchored_block_indices: torch.Tensor,
    hidden: torch.Tensor,
    base_logits: torch.Tensor | None,
    *,
    num_blocks: int,
    block_size: int,
    sample_from_anchor: bool,
) -> TrainingBlocks:
    """Keep verifier token indexing, predecessor alignment and view/error order.

    A sampled anchor predicts the next token, so its block tokens already name
    each slot's predecessor. A reserved anchor shifts predecessors within the
    block, repeating the anchor at the front. Positions are never renumbered.
    """
    block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block_size)
    if sample_from_anchor:
        previous_token_ids = block_tokens
    else:
        previous_token_ids = torch.cat(
            [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
        )
    hidden_blocks = hidden.view(num_blocks, block_size, -1)
    block_positions = torch.arange(block_size, device=hidden.device).expand(
        num_blocks, -1
    )
    base_logits_blocks = (
        None if base_logits is None else base_logits.view(num_blocks, block_size, -1)
    )
    return TrainingBlocks(
        token_ids=block_tokens,
        previous_token_ids=previous_token_ids,
        hidden=hidden_blocks,
        positions=block_positions,
        base_logits=base_logits_blocks,
    )
