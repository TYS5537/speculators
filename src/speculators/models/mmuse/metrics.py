"""MMUSE logging schema over the shared sequential distillation metrics."""

import torch

from speculators.models.dspark.metrics import compute_metrics
from speculators.models.metrics import select_metric_schema

__all__ = ["compute_metrics", "select_logged_metrics"]

_CORE_LOGGED_METRICS = frozenset(
    {
        "loss",
        "full_acc",
        "accept_len",
        "eal",
        "confidence_loss",
        "correction_hidden_aux_loss",
        "dflash2_selector_loss",
        "collaboration_accept_len_gain",
        "collaboration_markov_gate_mean",
        "collaboration_markov_change_accuracy",
        "collaboration_markov_harmed_count",
        "rollout_full_acc",
        "rollout_accept_len",
    }
)


def select_logged_metrics(
    metrics: dict[str, torch.Tensor],
    *,
    include_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Keep the compact MMUSE train/validation logging schema."""
    return select_metric_schema(
        metrics, _CORE_LOGGED_METRICS, include_diagnostics=include_diagnostics
    )
