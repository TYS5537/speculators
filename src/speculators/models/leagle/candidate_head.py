"""Optional low-rank retrieval followed by exact, full-width candidate scoring.

The original LM head stays intact. Returned logits have the draft vocabulary's
shape, with -inf outside the retrieved set. They describe the *actual* proposal
distribution, including its restricted support, for rejection sampling.
"""

import torch
from torch import nn
from torch.nn import functional

from speculators.models.leagle.candidate_loss import CandidateLossAccumulator


class LowRankCandidateHead(nn.Module):
    def __init__(
        self, hidden_size: int, vocab_size: int, rank: int, initializer_range=0.02
    ):
        super().__init__()
        if not 1 <= rank <= min(hidden_size, vocab_size):
            raise ValueError(
                "Candidate rank must be in [1, min(hidden_size, vocab_size)]"
            )
        self.down_proj = nn.Linear(hidden_size, rank, bias=False)
        self.up_proj = nn.Linear(rank, vocab_size, bias=False)
        # Fresh training learns both factors. A checkpoint missing either factor is
        # rejected by the model's load hook rather than silently initialized here.
        for layer in (self.down_proj, self.up_proj):
            nn.init.normal_(layer.weight, std=initializer_range)
            layer._is_hf_initialized = True  # noqa: SLF001

    def validate_weights(self):
        for parameter in self.parameters():
            if parameter.is_meta or not torch.isfinite(parameter).all().item():
                raise ValueError("Candidate head weights are missing or non-finite")

    def coarse_logits(self, hidden_states):
        return self.up_proj(self.down_proj(hidden_states))

    def loss_accumulator(self):
        return CandidateLossAccumulator(self)

    def distillation_loss(self, hidden_states, full_weight, mask, *, top_k, chunk_size):
        """Return summed KL(W h || A B h), top-1 recall count and valid count.

        Both h and W are detached: this objective updates only A and B. Model
        training shares one accumulator across TTT and reuses its dense logits.
        """
        accumulator = self.loss_accumulator()
        _, recall, count = accumulator.add(
            hidden_states, full_weight, mask, top_k=top_k, chunk_size=chunk_size
        )
        return accumulator.loss(), recall, count

    def forward(self, hidden_states, full_weight, top_k: int):
        vocab_size = self.up_proj.out_features
        if not 1 <= top_k <= vocab_size:
            raise ValueError("Candidate top_k must be in [1, draft_vocab_size]")
        candidates = (
            self.coarse_logits(hidden_states).topk(top_k, dim=-1, sorted=False).indices
        )
        # Only M rows of W[V,H] take part in the full-width dot product.
        rows = functional.embedding(candidates, full_weight)
        exact = torch.matmul(rows, hidden_states.unsqueeze(-1)).squeeze(-1)
        logits = hidden_states.new_full(
            (*hidden_states.shape[:-1], vocab_size), -torch.inf
        )
        return logits.scatter_(-1, candidates, exact)
