"""Optional low-rank retrieval followed by exact, full-width candidate scoring.

The original LM head stays intact. Returned logits have the draft vocabulary's
shape, with -inf outside the retrieved set. They describe the *actual* proposal
distribution, including its restricted support, for rejection sampling.
"""

import torch
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint


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

    def _distill_chunk(self, hidden, full_weight, mask, top_k):
        coarse = self.coarse_logits(hidden)
        log_q = functional.log_softmax(coarse, dim=-1, dtype=torch.float32)
        with torch.no_grad():
            # Recompute the frozen teacher per chunk so checkpointing never holds
            # an additional [sequence_length, vocabulary] tensor across TTT steps.
            teacher = functional.linear(hidden, full_weight)
            log_p = functional.log_softmax(teacher, dim=-1, dtype=torch.float32)
            candidates = coarse.topk(top_k, dim=-1, sorted=False).indices
            recalled = (candidates == teacher.argmax(-1, keepdim=True)).any(-1)
            recall = (recalled & mask).sum(dtype=torch.float32)
        loss = (log_p.exp() * (log_p - log_q)).sum(-1)
        return (loss * mask).sum(), recall

    def distillation_loss(self, hidden_states, full_weight, mask, *, top_k, chunk_size):
        """Return summed KL(W h || A B h), top-1 recall count and valid count.

        Both h and W are detached: this objective updates only A and B. Recomputed
        chunks bound the additional FP32 vocabulary activations during backward.
        Masked chunks stay connected to both factors with zero gradients for DDP.
        """
        if chunk_size < 1 or not 1 <= top_k <= self.up_proj.out_features:
            raise ValueError("Invalid candidate loss chunk size or top_k")
        hidden = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
        weight = full_weight.detach()
        valid = mask.reshape(-1).bool()
        if hidden.shape[0] != valid.shape[0] or hidden.shape[0] == 0:
            raise ValueError(
                "Candidate loss needs nonempty, aligned hidden states and mask"
            )
        loss_sum = hidden.new_zeros((), dtype=torch.float32)
        recall_sum = loss_sum.clone()
        for start in range(0, hidden.shape[0], chunk_size):
            chunk_args = (
                hidden[start : start + chunk_size],
                weight,
                valid[start : start + chunk_size],
                top_k,
            )
            if torch.is_grad_enabled():
                loss, recall = checkpoint(
                    self._distill_chunk,
                    *chunk_args,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                loss, recall = self._distill_chunk(*chunk_args)
            loss_sum = loss_sum + loss
            recall_sum = recall_sum + recall
        return loss_sum, recall_sum, valid.sum(dtype=torch.float32)

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
