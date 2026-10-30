"""Metrics and loss functions for Eagle3 draft model."""

from functools import partial

import torch

from speculators.losses import (
    LossConfig,
    compound_loss,
    exp_loss_decay,
    kl_div_loss,
)
from speculators.models.metrics import compute_accuracy_single_step

_OVERLAP_CHUNK_ELEMENTS = 4 * 1024 * 1024


@torch.no_grad()
def _distribution_overlap(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """FP32 overlap without materializing full-sequence probability tensors."""
    result = torch.empty(logits.shape[:-1], device=logits.device, dtype=torch.float32)
    chunk_size = max(1, _OVERLAP_CHUNK_ELEMENTS // logits.shape[-1])
    for start in range(0, logits.shape[1], chunk_size):
        end = start + chunk_size
        draft_p = torch.softmax(logits[:, start:end], dim=-1, dtype=torch.float32)
        target_p = torch.softmax(targets[:, start:end], dim=-1, dtype=torch.float32)
        result[:, start:end] = torch.minimum(draft_p, target_p).sum(-1).clamp(0, 1)
        del draft_p, target_p
    return result


class EagleTrainingMetrics:
    """DSpark-style metrics over teacher-forced TTT trajectories.

    A trajectory starts at a supervised token. Masks, packed-document boundaries
    and sequence tails truncate its accepted prefix. ``eal`` counts consecutive
    greedy matches; ``accept_len`` sums products of distribution overlaps. Both
    include one verifier token and are training proxies, not generation results.
    With a reduced teacher head these refer to the conditional draft vocabulary.
    Only detached per-trajectory vectors and scalar counts survive each update.
    """

    def __init__(self, document_ids, loss_mask, num_steps):
        self.document_ids = document_ids
        self.valid = document_ids >= 0
        if loss_mask is not None:
            self.valid = self.valid & loss_mask.bool()
        self.correct_prefix = self.valid.clone()
        self.overlap_prefix = self.valid.float()
        count = self.valid.sum(dtype=torch.float32)
        self.counts = {
            "full_acc_sum": count.new_zeros(()),
            "full_acc_total": count.new_zeros(()),
            "eal_sum": count.clone(),
            "eal_total": count.clone(),
            "accept_len_sum": count.clone(),
            "accept_len_total": count.clone(),
        }
        # All ranks must return the same scalar keys, even with short sequences.
        for step in range(num_steps):
            self.counts[f"position_{step}_acc_sum"] = count.new_zeros(())
            self.counts[f"position_{step}_acc_total"] = count.new_zeros(())

    @torch.no_grad()
    def update(
        self, logits, targets, step, *, draft_ids=None, target_ids=None, overlap=None
    ):
        """Reuse optional full-sequence IDs and an already aligned overlap vector."""
        length = self.valid.shape[1] - step
        if length <= 0:
            return
        logits = logits[:, :length]
        targets = targets[:, step:]
        draft_ids = logits.argmax(-1) if draft_ids is None else draft_ids[:, :length]
        target_ids = targets.argmax(-1) if target_ids is None else target_ids[:, step:]
        if overlap is None:
            overlap = _distribution_overlap(logits, targets)
        self.update_aligned(draft_ids, target_ids, overlap, step)

    @torch.no_grad()
    def update_aligned(self, draft_ids, target_ids, overlap, step):
        """Accept per-position summaries from a streamed output head."""
        length = self.valid.shape[1] - step
        valid = (
            self.valid[:, :length]
            & self.valid[:, step:]
            & (self.document_ids[:, :length] == self.document_ids[:, step:])
        )
        correct = (draft_ids == target_ids) & valid
        correct_count = correct.sum(dtype=torch.float32)
        total = valid.sum(dtype=torch.float32)
        self.counts[f"position_{step}_acc_sum"] = correct_count
        self.counts[f"position_{step}_acc_total"] = total
        self.counts["full_acc_sum"] += correct_count
        self.counts["full_acc_total"] += total

        # Never restart a prefix after a masked or incorrect intermediate slot.
        self.correct_prefix[:, :length] &= correct
        self.counts["eal_sum"] += self.correct_prefix[:, :length].sum()
        self.overlap_prefix[:, :length] *= torch.where(valid, overlap, 0.0)
        self.counts["accept_len_sum"] += self.overlap_prefix[:, :length].sum()

    def finish(self, loss):
        return {
            "loss_sum": loss.detach().clone(),
            "loss_total": loss.new_ones(()),
            **self.counts,
        }


def align_for_step(
    logits: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, total_seq_len, draft_vocab_size]
    loss_mask: torch.Tensor | None,  # shape: [1, total_seq_len]
    prev_correct: torch.Tensor | None,  # shape: [1, total_seq_len]
    ttt_step: int,
):
    """Align logits, targets, loss_mask, and prev_correct for a given ttt_step.

    There are no target values for the last ttt_step tokens, so we mask them out
    before computing the loss/accuracy. Likewise, there are no logits for the first
    ttt_step tokens, so we mask them out.
    This is equivalent to shifting the target values by ttt_step + 1 to the left
    which puts them in the correct position for the generated tokens.
    e.g.
        indices of targets = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        indices of logits for ttt_step_0 = [1, 2, 3, 4, 5, 6, 7, 8, 9] # no shift
        indices of logits for ttt_step_1 = [2, 3, 4, 5, 6, 7, 8, 9, 10] # shift by 1
        indices of logits for ttt_step_2 = [3, 4, 5, 6, 7, 8, 9, 10, 11] # shift by 2
    The indices for the loss_mask need to be kept in line with the targets indices
    """
    logits = logits[:, :-ttt_step] if ttt_step > 0 else logits
    # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    targets = targets[:, ttt_step:]
    # shape: [1, total_seq_len - ttt_step, draft_vocab_size]
    if loss_mask is not None:
        loss_mask = loss_mask[:, ttt_step:]
        # shape: [1, total_seq_len - ttt_step]
    if prev_correct is not None:
        # Align with draft starts
        prev_correct = prev_correct[:, :-ttt_step] if ttt_step > 0 else prev_correct
        # shape: [1, total_seq_len - ttt_step]
    return logits, targets, loss_mask, prev_correct


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    prev_correct: torch.Tensor | None,
    ttt_step: int,
    ttt_step_loss_decay: float,
    loss_config: LossConfig | None = None,
    *,
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Compute metrics for a given ttt_step.

    Args:
        logits: The logits for the current ttt_step.
        targets: The targets for the current ttt_step.
        loss_mask: The loss mask for the current ttt_step.
        prev_correct: The previous correct predictions for the current ttt_step.
        ttt_step: The current ttt_step.
        ttt_step_loss_decay: The loss decay for the current ttt_step.
        loss_config: Mapping of ``{name: (loss_fn, weight)}``.
        collect_metrics: False skips legacy accuracy/diagnostic construction.

    Effects:
        Modifies prev_correct in place.

    Returns:
        Loss value and metrics dictionary.
    """
    if loss_config is None:
        loss_config = {"kl_div": (kl_div_loss, 1.0)}
    s_logits, s_targets, s_loss_mask, s_prev_correct = align_for_step(
        logits, targets, loss_mask, prev_correct, ttt_step
    )

    seq_len = s_logits.shape[1]
    if s_loss_mask is None:
        s_loss_mask = torch.ones(1, seq_len, device=s_logits.device, dtype=torch.bool)

    pos_idx = torch.full(
        (1, seq_len), ttt_step, device=s_logits.device, dtype=torch.long
    )

    s_loss, term_losses = compound_loss(
        s_logits,
        s_targets,
        s_loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=partial(exp_loss_decay, gamma=ttt_step_loss_decay),
    )
    if not collect_metrics:
        return s_loss, {}

    pred_ids = torch.argmax(s_logits, dim=-1)
    target_ids = torch.argmax(s_targets, dim=-1)

    full_correct, full_total, cond_correct, cond_total = compute_accuracy_single_step(
        pred_ids, target_ids, s_loss_mask, s_prev_correct
    )

    ones = torch.tensor(1.0, device=s_loss.device)
    s_metrics = {}
    s_metrics[f"loss_{ttt_step}_sum"] = s_loss.detach().clone()
    s_metrics[f"loss_{ttt_step}_total"] = ones
    for term_name, term_val in term_losses.items():
        s_metrics[f"{term_name}_{ttt_step}_sum"] = term_val
        s_metrics[f"{term_name}_{ttt_step}_total"] = ones.clone()
    s_metrics[f"full_acc_{ttt_step}_sum"] = full_correct
    s_metrics[f"full_acc_{ttt_step}_total"] = full_total
    s_metrics[f"cond_acc_{ttt_step}_sum"] = cond_correct
    s_metrics[f"cond_acc_{ttt_step}_total"] = cond_total

    return s_loss, s_metrics
