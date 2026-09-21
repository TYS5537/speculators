"""Bounded-memory target projection for pruned-vocabulary draft training."""

import torch
from torch import nn


@torch.no_grad()
def project_target_distribution(
    hidden_states: torch.Tensor,
    head: nn.Linear,
    draft_token_ids: torch.Tensor,
    *,
    token_chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return subset logits, full-vocabulary log Z and the true greedy target.

    The greedy target is expressed in draft IDs, or -1 when it is not in the
    draft vocabulary. Only a token chunk's full logits are materialized at once.
    The existing frozen head remains a normal module (including under FSDP), not
    a detached weight cache. Conditional KD still uses the returned subset logits;
    acceptance probabilities must use the separately returned full normalizer.
    """
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if draft_token_ids.ndim != 1:
        raise ValueError("draft_token_ids must be one-dimensional")
    prefix_shape = hidden_states.shape[:-1]
    rows = hidden_states.reshape(-1, hidden_states.shape[-1])
    draft_token_ids = draft_token_ids.to(device=rows.device, dtype=torch.long)
    inverse = torch.full((head.out_features,), -1, device=rows.device, dtype=torch.long)
    inverse[draft_token_ids] = torch.arange(
        draft_token_ids.numel(), device=rows.device, dtype=torch.long
    )
    subset_chunks, normalizer_chunks, greedy_chunks = [], [], []
    for chunk in rows.split(token_chunk_size, dim=0):
        full_logits = head(chunk)
        subset_chunks.append(full_logits.index_select(-1, draft_token_ids))
        normalizer_chunks.append(torch.logsumexp(full_logits.float(), dim=-1))
        greedy_chunks.append(inverse[full_logits.argmax(dim=-1)])
    return (
        torch.cat(subset_chunks, dim=0).reshape(*prefix_shape, draft_token_ids.numel()),
        torch.cat(normalizer_chunks, dim=0).reshape(prefix_shape),
        torch.cat(greedy_chunks, dim=0).reshape(prefix_shape),
    )
