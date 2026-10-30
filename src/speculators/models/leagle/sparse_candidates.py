"""Compact proposal distributions with explicit vocabulary IDs."""

from dataclasses import dataclass

import torch

from speculators.proposals.sampling import logits_to_probs, sample_from_probs


@dataclass(frozen=True)
class SparseCandidateLogits:
    indices: torch.Tensor
    values: torch.Tensor
    vocab_size: int

    def sample(self, temperature):
        if temperature == 0:
            # Dense argmax resolves ties by the lowest vocabulary ID, not top-k
            # retrieval order. Preserve that rule in the compact representation.
            best = self.values == self.values.amax(dim=-1, keepdim=True)
            return torch.where(best, self.indices, self.vocab_size).amin(-1)
        selected = sample_from_probs(logits_to_probs(self.values, temperature))
        return self.indices.gather(-1, selected.unsqueeze(-1)).squeeze(-1)

    @classmethod
    def concatenate(cls, parts):
        return cls(
            torch.cat([part.indices for part in parts], dim=1),
            torch.cat([part.values for part in parts], dim=1),
            parts[0].vocab_size,
        )
