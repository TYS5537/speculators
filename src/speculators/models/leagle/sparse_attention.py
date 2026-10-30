"""Experimental exact trajectory visibility using compact SDPA query chunks.

Each chunk reads its verified context prefix and only the draft rows belonging
to its queries. Other trajectories' masked-out KV blocks are never materialized.
"""

from dataclasses import dataclass, field
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS

from speculators.models.leagle.attention import trajectory_mask


class TrajectoryMaskCache:
    """Bounded masks for one forward/backward's immutable packed positions.

    Shared by all physical layers and loops, never across batches. Full masks
    that do not fit the budget are rebuilt normally, without evicting earlier
    entries that later checkpoint recomputations will need.
    """

    def __init__(self, budget_bytes):
        if budget_bytes < 1:
            raise ValueError("Attention mask cache budget must be positive")
        self.budget_bytes = budget_bytes
        self.used_bytes = 0
        self.masks = {}

    def get(self, layout, start, end):
        key = (layout.step, layout.window, start, end)
        if key in self.masks:
            return self.masks[key]
        mask = _chunk_mask(layout, start, end)
        size = mask.numel() * mask.element_size()
        if self.used_bytes + size <= self.budget_bytes:
            self.masks[key] = mask
            self.used_bytes += size
        return mask


@dataclass(frozen=True)
class TrajectoryAttention:
    document_ids: torch.Tensor
    position_ids: torch.Tensor
    step: int
    window: int | None
    chunk_size: int
    mask_cache: TrajectoryMaskCache | None = field(default=None, compare=False)


def _chunk_mask(layout, start, end):
    length, device = layout.document_ids.numel(), layout.document_ids.device
    query_ids = torch.arange(start, end, device=device)
    draft_ids = (
        torch.arange(1, layout.step + 2, device=device)[:, None] * length
        + query_ids[None, :]
    ).reshape(-1)
    key_ids = torch.cat((torch.arange(end, device=device), draft_ids))
    return trajectory_mask(
        layout.document_ids, layout.position_ids, layout.step, layout.window
    )(None, None, query_ids[:, None], key_ids[None, :])


def _attend_chunk(
    module, layout, start, end, query, context_k, context_v, draft_k, draft_v
):
    length = query.shape[-2]
    width = end - start
    keys, values = [], []
    for context, draft, output in (
        (context_k, draft_k, keys),
        (context_v, draft_v, values),
    ):
        own = draft.reshape(*draft.shape[:2], layout.step + 1, length, -1)
        own = own[..., start:end, :].reshape(*draft.shape[:2], -1, draft.shape[-1])
        output.extend((context[..., :end, :], own))
    mask = (
        _chunk_mask(layout, start, end)
        if layout.mask_cache is None
        else layout.mask_cache.get(layout, start, end)
    )
    output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](
        module,
        query[..., start:end, :],
        torch.cat(keys, dim=-2),
        torch.cat(values, dim=-2),
        mask[None, None, :, :],
        scaling=module.scaling,
        dropout=0.0,
    )
    return output.reshape(query.shape[0], width, query.shape[1], query.shape[-1])


def trajectory_attention(module, layout, query, context_k, context_v, draft_k, draft_v):
    if module.config._attn_implementation != "sdpa" or module.attention_dropout != 0:  # noqa: SLF001
        raise ValueError("Sparse trajectory attention requires SDPA and zero dropout")
    outputs = []
    for start in range(0, query.shape[-2], layout.chunk_size):
        function = partial(
            _attend_chunk,
            module,
            layout,
            start,
            min(start + layout.chunk_size, query.shape[-2]),
        )
        arguments = (query, context_k, context_v, draft_k, draft_v)
        # Rebuild compact KV concatenations in backward instead of retaining a
        # copy of the target prefix for every query chunk.
        output = (
            checkpoint(
                function, *arguments, use_reentrant=False, preserve_rng_state=True
            )
            if torch.is_grad_enabled()
            else function(*arguments)
        )
        outputs.append(output)
    return torch.cat(outputs, dim=1)
