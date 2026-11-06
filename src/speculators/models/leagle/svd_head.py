"""Inference-only truncated-SVD retrieval, derived from the loaded dense head.

The right singular subspace comes from an FP64 CPU Gram matrix. Vocabulary
rows are streamed so neither a full FP64 weight copy nor the full left singular
matrix is materialized. No random initialization, training or disk cache is used.
"""

import torch
from torch import nn
from torch.nn import functional

from speculators.models.leagle.candidate_head import rescore_candidates

_VOCAB_CHUNK_SIZE = 2048


@torch.no_grad()
def svd_factors(weight, rank):
    """Return A=W Q and B=Q.T for the leading rank right singular vectors Q."""
    if weight.is_meta or weight.ndim != 2 or not weight.is_floating_point():  # noqa: PLR2004
        raise ValueError("SVD requires a loaded, real floating-point LM head matrix")
    if type(rank) is not int or not 1 <= rank <= min(weight.shape):
        raise ValueError("SVD rank must be in [1, min(hidden_size, draft_vocab_size)]")
    vocab_size, hidden_size = weight.shape
    # FP64 reduces the loss of precision from forming W.T W. Keep this on CPU
    # so Ascend does not need FP64/eigh kernels or a large decomposition workspace.
    gram = torch.zeros(hidden_size, hidden_size, device="cpu", dtype=torch.float64)
    for start in range(0, vocab_size, _VOCAB_CHUNK_SIZE):
        rows = (
            weight[start : start + _VOCAB_CHUNK_SIZE]
            .detach()
            .to(device="cpu", dtype=torch.float64)
        )
        if not torch.isfinite(rows).all():
            raise ValueError("SVD requires finite LM head weights")
        gram.addmm_(rows.T, rows)
    if not torch.isfinite(gram).all():
        raise ValueError("SVD Gram matrix overflowed; check LM head weights")
    _, vectors = torch.linalg.eigh(gram)
    basis = vectors[:, -rank:].flip(-1).contiguous()
    del gram, vectors, rows
    up = torch.empty(vocab_size, rank, device="cpu", dtype=weight.dtype)
    for start in range(0, vocab_size, _VOCAB_CHUNK_SIZE):
        rows = (
            weight[start : start + _VOCAB_CHUNK_SIZE]
            .detach()
            .to(device="cpu", dtype=torch.float64)
        )
        up[start : start + rows.shape[0]].copy_(rows @ basis)
    return up.to(weight.device), basis.T.to(device=weight.device, dtype=weight.dtype)


class SVDCandidateHead(nn.Module):
    """Ephemeral buffers only: no parameters or checkpoint/config additions."""

    def __init__(self, weight, rank):
        super().__init__()
        up, down = svd_factors(weight, rank)
        self.register_buffer("up_weight", up, persistent=False)
        self.register_buffer("down_weight", down, persistent=False)
        self.validate_weights()

    def validate_weights(self):
        for weight in (self.up_weight, self.down_weight):
            if weight.is_meta or not torch.isfinite(weight).all().item():
                raise ValueError("SVD candidate factors are missing or non-finite")

    def coarse_logits(self, hidden_states):
        return functional.linear(
            functional.linear(hidden_states, self.down_weight), self.up_weight
        )

    def forward(self, hidden_states, full_weight, top_k, *, sparse=False, rerank=True):
        if not rerank:
            if sparse:
                raise ValueError("Sparse candidates require candidate reranking")
            return self.coarse_logits(hidden_states)
        return rescore_candidates(
            hidden_states,
            self.coarse_logits(hidden_states),
            full_weight,
            top_k,
            sparse=sparse,
        )
