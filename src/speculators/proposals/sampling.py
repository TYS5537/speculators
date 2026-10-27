"""Temperature sampling shared by autoregressive draft proposals and evaluation."""

import math

import torch


def validate_temperature(temperature: float) -> None:
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and >= 0")


def logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """FP32 sampling distribution; temperature zero is a greedy point mass."""
    validate_temperature(temperature)
    if temperature == 0:
        return torch.nn.functional.one_hot(
            logits.argmax(dim=-1), num_classes=logits.shape[-1]
        ).float()
    scores = logits.float()
    # Center before division so very small positive temperatures do not turn
    # finite positive logits into +inf and produce NaN softmax probabilities.
    scores = (scores - scores.amax(dim=-1, keepdim=True)) / temperature
    return torch.softmax(scores, dim=-1)


def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Sample the final dimension; multinomial also normalizes residual weights."""
    sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1)
    return sampled.reshape(probs.shape[:-1])


def sample_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    validate_temperature(temperature)
    if temperature == 0:
        return logits.argmax(dim=-1)  # Preserve the greedy path's RNG state.
    return sample_from_probs(logits_to_probs(logits, temperature))
