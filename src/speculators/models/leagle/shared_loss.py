"""Opt-in FP32 normalization shared by eager main losses and candidate KL."""

from typing import NamedTuple

import torch
from torch.nn import functional


class CandidateTeacher(NamedTuple):
    log_probs: torch.Tensor
    probabilities: torch.Tensor
    ids: torch.Tensor | None


def normalized_distribution(logits):
    log_probs = logits.log_softmax(-1, dtype=torch.float32)
    return log_probs, log_probs.exp()


def shared_terms(logits, probabilities, ids, loss_config, metrics, candidate):
    log_q, q = normalized_distribution(logits)
    overlap = None
    terms = []
    for name in loss_config:
        if name == "ce":
            value = -log_q.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        elif name == "tv":
            value = 1.0 - torch.minimum(q, probabilities).sum(-1)
            if metrics:
                overlap = (1.0 - value.detach()).clamp(0, 1)
        else:  # The caller validates eager forward KL / CE / TV only.
            value = functional.kl_div(
                log_q, probabilities, reduction="none", log_target=False
            ).sum(-1)
        terms.append(value)
    if metrics and overlap is None:
        overlap = torch.minimum(q.detach(), probabilities).sum(-1).clamp(0, 1)
    teacher = (
        CandidateTeacher(
            log_q.detach(), q.detach(), logits.argmax(-1) if metrics else None
        )
        if candidate
        else None
    )
    return tuple(terms), overlap, teacher
