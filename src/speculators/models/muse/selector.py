"""Top-K candidate transition scoring for Muse's optional selector."""

import torch
from torch import nn
from torch.nn.functional import embedding


class DFlash2CandidateSelector(nn.Module):
    """Score a Top-K token using its predecessor and DFlash hidden state."""

    def __init__(
        self,
        *,
        hidden_size: int,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        rank: int,
        top_k: int,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"selector_rank must be > 0, got {rank}")
        if top_k <= 0 or top_k > draft_vocab_size:
            raise ValueError(
                "selector_top_k must be in [1, draft_vocab_size], "
                f"got {top_k} for vocab {draft_vocab_size}"
            )
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(verifier_vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(draft_vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)

    def reset_unary(self, initializer_range: float) -> None:
        """Initialize the selector to reproduce its unary Top-K logits."""
        with torch.no_grad():
            nn.init.normal_(
                self.predecessor_codebook,
                mean=0.0,
                std=initializer_range,
            )
            nn.init.normal_(
                self.successor_codebook,
                mean=0.0,
                std=initializer_range,
            )
            self.hidden_projection.weight.zero_()

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_ids.shape != unary_logits.shape:
            raise ValueError("Candidate IDs and unary logits must align")
        if candidate_ids.shape[:-1] != hidden_states.shape[:-1]:
            raise ValueError("Candidates and hidden states must align")
        if previous_token_ids.shape != hidden_states.shape[:-1]:
            raise ValueError("Previous-token IDs and hidden states must align")
        if candidate_ids.shape[-1] != self.top_k:
            raise ValueError(
                f"Expected selector_top_k={self.top_k}, got {candidate_ids.shape[-1]}"
            )

        predecessor = embedding(previous_token_ids.long(), self.predecessor_codebook)
        successor = embedding(candidate_ids.long(), self.successor_codebook)
        hidden = self.hidden_projection(hidden_states)
        transition = torch.einsum(
            "...r,...kr->...k",
            predecessor.to(hidden.dtype) * hidden,
            successor.to(hidden.dtype),
        )
        return unary_logits + transition.to(unary_logits.dtype)

    def score_lattice(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return DFlash2's ``K_previous x K_current`` edge lattice."""
        if candidate_ids.shape != unary_logits.shape:
            raise ValueError("Candidate IDs and unary logits must align")
        if candidate_ids.shape[:-1] != hidden_states.shape[:-1]:
            raise ValueError("Candidates and hidden states must align")
        if predecessor_ids.shape != candidate_ids.shape:
            raise ValueError("Predecessor and current candidate lattices must align")
        if candidate_ids.shape[-1] != self.top_k:
            raise ValueError(
                f"Expected selector_top_k={self.top_k}, got {candidate_ids.shape[-1]}"
            )

        predecessor = embedding(predecessor_ids.long(), self.predecessor_codebook)
        successor = embedding(candidate_ids.long(), self.successor_codebook)
        hidden = self.hidden_projection(hidden_states)
        transitions = torch.einsum(
            "...pr,...r,...cr->...pc",
            predecessor.to(hidden.dtype),
            hidden,
            successor.to(hidden.dtype),
        )
        return unary_logits.unsqueeze(-2) + transitions.to(unary_logits.dtype)
