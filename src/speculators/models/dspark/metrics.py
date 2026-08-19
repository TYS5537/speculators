"""Loss and metrics for the DSpark draft model.

loss = compound_loss(logits, targets)
     + conf_alpha * BCE(confidence, accept_rate)
     + length_alpha * SmoothL1(pred_len, target_len)
     + focal_alpha * CE(first greedy error)

Optional adaptive position weights (CAT / SSAL) replace fixed decay.
For the correction head, an optional short curriculum mixes the unchanged base
objective into the corrected objective and reports both acceptance lengths.
"""

from functools import partial
from typing import Any, Literal

import torch
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    cross_entropy,
    smooth_l1_loss,
    softmax,
)

from speculators.models.metrics import (
    LossConfig,
    compound_loss,
    compute_accuracy_multi_step,
    dpace_loss_decay,
    position_weights,
)

__all__ = [
    "compute_metrics",
    "select_logged_metrics",
]

_EPS = 1e-8
_CORE_LOGGED_METRICS = frozenset(
    {
        "loss",
        "full_acc",
        "accept_len",
        "confidence_loss",
        "correction_hidden_aux_loss",
        "correction_moe_balance_loss",
        "correction_moe_router_entropy",
        "dflash2_selector_loss",
        "collaboration_accept_len_gain",
        "collaboration_markov_gate_mean",
        "collaboration_markov_change_accuracy",
        "collaboration_markov_harmed_count",
        "rollout_full_acc",
        "rollout_accept_len",
    }
)

AdaptiveLoss = Literal["none", "cat", "ssal"]
ConfidenceLossWeighting = Literal["uniform", "match-draft"]


def select_logged_metrics(
    metrics: dict[str, torch.Tensor],
    *,
    include_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Keep the compact DSpark train/validation logging schema."""
    if include_diagnostics:
        return metrics

    def should_keep(key: str) -> bool:
        name = key
        if name.endswith("_sum"):
            name = name.removesuffix("_sum")
        elif name.endswith("_total"):
            name = name.removesuffix("_total")
        return name in _CORE_LOGGED_METRICS or (
            name.startswith("position_") and name.endswith("_acc")
        )

    return {key: value for key, value in metrics.items() if should_keep(key)}


def _masked_weighted_mean(
    elementwise: torch.Tensor,  # [1, T]
    loss_mask: torch.Tensor,  # [1, T]
    weights: torch.Tensor | None = None,  # [1, T]
) -> torch.Tensor:
    """Masked mean; optional weights scale the numerator only (same as draft decay)."""
    loss_mask = loss_mask.to(elementwise.dtype)
    weighted = elementwise * loss_mask
    if weights is not None:
        weighted = weighted * weights.to(elementwise.dtype)
    return (weighted.sum(dim=1) / (loss_mask.sum(dim=1) + _EPS)).mean()


def _first_error_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    weights: torch.Tensor,
    start_pos: int,
) -> torch.Tensor:
    """CE at each block's first greedy mismatch (chain breaker)."""
    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)
    vocab = logits.shape[-1]
    ce = cross_entropy(
        logits.reshape(-1, vocab),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_ids)

    num_blocks = logits.shape[1] // block_size
    wrong = ((pred_ids != target_ids) & loss_mask.bool()).view(num_blocks, block_size)
    if start_pos > 0:
        wrong = wrong.clone()
        wrong[:, :start_pos] = False

    first_idx = wrong.to(torch.int64).argmax(dim=-1)  # first True; 0 if none
    fe_mask = torch.zeros_like(wrong, dtype=ce.dtype)
    rows = torch.arange(num_blocks, device=logits.device)
    has_err = wrong.any(dim=-1)
    fe_mask[rows[has_err], first_idx[has_err]] = 1.0
    fe_mask = fe_mask.reshape_as(ce)
    return _masked_weighted_mean(ce, fe_mask, weights)


def _accept_length(
    probs: torch.Tensor,  # [num_blocks, draft_slots]
    draft_mask: torch.Tensor,  # [num_blocks, draft_slots]
) -> torch.Tensor:
    prefix = (probs * draft_mask).cumprod(dim=-1)
    return prefix.sum(dim=-1) + 1.0


def _change_outcome_metrics(
    prefix: str,
    candidate_ids: torch.Tensor,
    base_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Count whether changed argmax predictions became correct or wrong."""
    valid = loss_mask.bool()
    changed = (candidate_ids != base_ids) & valid
    candidate_correct = candidate_ids == target_ids
    base_correct = base_ids == target_ids

    changed_to_correct = changed & candidate_correct
    changed_to_wrong = changed & ~candidate_correct
    harmed = changed & base_correct & ~candidate_correct
    dtype = torch.float32
    correct_count = changed_to_correct.sum().to(dtype)
    wrong_count = changed_to_wrong.sum().to(dtype)
    harmed_count = harmed.sum().to(dtype)
    changed_count = changed.sum().to(dtype)
    one = torch.ones((), device=loss_mask.device, dtype=dtype)

    return {
        f"{prefix}_change_correct_count_sum": correct_count,
        f"{prefix}_change_correct_count_total": one,
        f"{prefix}_change_wrong_count_sum": wrong_count,
        f"{prefix}_change_wrong_count_total": one.clone(),
        f"{prefix}_harmed_count_sum": harmed_count,
        f"{prefix}_harmed_count_total": one.clone(),
        f"{prefix}_change_accuracy_sum": correct_count.clone(),
        f"{prefix}_change_accuracy_total": changed_count,
    }


def compute_metrics(  # noqa: C901
    logits: torch.Tensor,  # [1, T, draft_vocab_size]
    targets: torch.Tensor,  # [1, T, draft_vocab_size]
    confidence_logits: torch.Tensor | None,  # [1, T] or None
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    loss_config: LossConfig,
    gamma: float = 7.0,
    confidence_head_alpha: float = 1.0,
    confidence_length_alpha: float = 0.0,
    confidence_loss_weighting: ConfidenceLossWeighting = "match-draft",
    first_error_focal_alpha: float = 0.0,
    adaptive_loss: AdaptiveLoss = "none",
    ssal_decay_weight: torch.Tensor | float = 0.0,
    base_logits: torch.Tensor | None = None,
    rollout_logits: torch.Tensor | None = None,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = True,
    collaboration_base_logits: torch.Tensor | None = None,
    collaboration_gate: torch.Tensor | None = None,
    proposal_candidate_ids: torch.Tensor | None = None,
    proposal_candidate_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs)."""

    device = logits.device
    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)
    start_pos = 0 if sample_from_anchor else 1
    if (proposal_candidate_ids is None) != (proposal_candidate_logits is None):
        raise ValueError(
            "proposal_candidate_ids and proposal_candidate_logits must be set together"
        )
    if proposal_candidate_ids is not None:
        if proposal_candidate_logits is None:
            raise RuntimeError("Proposal candidate logits are missing")
        if proposal_candidate_ids.shape != proposal_candidate_logits.shape:
            raise ValueError("Proposal candidate IDs and logits must align")
        if proposal_candidate_ids.shape[:-1] != logits.shape[:-1]:
            raise ValueError("Proposal candidates must align with draft positions")

    # Analytical overlap (also SSAL score); needed for confidence / accept metrics.
    with torch.no_grad():
        target_p = softmax(targets.float(), dim=-1)
        target_ids = torch.argmax(targets, dim=-1)
        if proposal_candidate_ids is not None:
            if proposal_candidate_logits is None:
                raise RuntimeError("Proposal candidate logits are missing")
            draft_p = softmax(proposal_candidate_logits.float(), dim=-1)
            candidate_target_p = target_p.gather(-1, proposal_candidate_ids)
            accept_rate = torch.minimum(draft_p, candidate_target_p).sum(dim=-1)
            selected = proposal_candidate_logits.argmax(dim=-1, keepdim=True)
            pred_ids = proposal_candidate_ids.gather(-1, selected).squeeze(-1)
        else:
            draft_p = softmax(logits.float(), dim=-1)
            accept_rate = torch.minimum(draft_p, target_p).sum(dim=-1)
            pred_ids = torch.argmax(logits, dim=-1)
        rollout_accept_rate = None
        if rollout_logits is not None:
            if rollout_logits.shape != logits.shape:
                raise ValueError("rollout_logits and logits must have the same shape")
            rollout_p = softmax(rollout_logits.float(), dim=-1)
            rollout_accept_rate = torch.minimum(rollout_p, target_p).sum(dim=-1)
        collaboration_base_accept_rate = None
        if collaboration_base_logits is not None:
            if collaboration_base_logits.shape != logits.shape:
                raise ValueError(
                    "collaboration_base_logits and logits must have the same shape"
                )
            collaboration_base_p = softmax(collaboration_base_logits.float(), dim=-1)
            collaboration_base_accept_rate = torch.minimum(
                collaboration_base_p, target_p
            ).sum(dim=-1)

    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
        draft_weights = None
    else:
        adaptive_scores = None
        if adaptive_loss == "ssal":
            adaptive_scores = accept_rate
        elif adaptive_loss == "cat":
            with torch.no_grad():
                adaptive_scores = target_p.gather(-1, target_ids.unsqueeze(-1)).squeeze(
                    -1
                )
        draft_weights = position_weights(
            pos_idx.to(logits.dtype),
            block_size=block_size,
            gamma=gamma,
            sample_from_anchor=sample_from_anchor,
            adaptive_scores=(
                None if adaptive_scores is None else adaptive_scores.to(logits.dtype)
            ),
            decay_mix=ssal_decay_weight if adaptive_loss == "ssal" else 0.0,
        )
        decay_fn = lambda pos, **_kw: draft_weights  # noqa: E731

    final_draft_loss, term_losses = compound_loss(
        logits, targets, loss_mask, pos_idx, loss_config=loss_config, decay_fn=decay_fn
    )
    final_objective = final_draft_loss

    if first_error_focal_alpha > 0.0:
        fe_weights = (
            draft_weights
            if draft_weights is not None
            else position_weights(
                pos_idx.to(logits.dtype),
                block_size=block_size,
                gamma=gamma,
                sample_from_anchor=sample_from_anchor,
            )
        )
        fe_loss = _first_error_focal_loss(
            logits, targets, loss_mask, block_size, fe_weights, start_pos
        )
        final_objective = final_objective + first_error_focal_alpha * fe_loss

    loss = final_objective
    base_loss = None
    if base_logits is not None:
        with torch.no_grad():
            base_loss, _ = compound_loss(
                base_logits.detach(),
                targets,
                loss_mask,
                pos_idx,
                loss_config=loss_config,
                decay_fn=decay_fn,
            )
            if first_error_focal_alpha > 0.0:
                base_fe_loss = _first_error_focal_loss(
                    base_logits.detach(),
                    targets,
                    loss_mask,
                    block_size,
                    fe_weights,
                    start_pos,
                )
                base_loss = base_loss + first_error_focal_alpha * base_fe_loss

    num_blocks = seq_len // block_size
    with torch.no_grad():
        accept_blocks = accept_rate.view(num_blocks, block_size)
        draft_mask = loss_mask.to(accept_rate.dtype).view(num_blocks, block_size)[
            :, start_pos:
        ]
        accept_prefix = (accept_blocks[:, start_pos:] * draft_mask).cumprod(dim=-1)

    metrics: dict[str, Any] = {}
    if base_loss is not None:
        metrics["base_loss_sum"] = base_loss.detach().clone()
        metrics["base_loss_total"] = torch.ones((), device=device)
        metrics["corrected_loss_sum"] = final_objective.detach().clone()
        metrics["corrected_loss_total"] = torch.ones((), device=device)
    if confidence_logits is not None:
        c_star = accept_rate.detach().to(confidence_logits.dtype)
        bce = binary_cross_entropy_with_logits(
            confidence_logits, c_star, reduction="none"
        )
        conf_weights = (
            draft_weights if confidence_loss_weighting == "match-draft" else None
        )
        conf_loss = _masked_weighted_mean(bce, loss_mask, conf_weights)
        loss = loss + confidence_head_alpha * conf_loss

        conf_prob = confidence_logits.float().sigmoid()
        with torch.no_grad():
            mask_f = loss_mask.to(accept_rate.dtype)
            mask_total = mask_f.sum().clamp_min(1.0)
            metrics["confidence_loss_sum"] = conf_loss.detach().clone()
            metrics["confidence_loss_total"] = torch.ones((), device=device)
            metrics["confidence_abs_error_sum"] = (
                (conf_prob - accept_rate).abs() * mask_f
            ).sum()
            metrics["confidence_abs_error_total"] = mask_total
            metrics["confidence_pred_mean_sum"] = (conf_prob * mask_f).sum()
            metrics["confidence_pred_mean_total"] = mask_total.clone()
            conf_prefix = (
                conf_prob.view(num_blocks, block_size)[:, start_pos:] * draft_mask
            ).cumprod(dim=-1)
            metrics["confidence_cumprod_bias_sum"] = (
                (conf_prefix - accept_prefix) * draft_mask
            ).sum()
            metrics["confidence_cumprod_bias_total"] = draft_mask.sum().clamp_min(1.0)

        if confidence_length_alpha > 0.0:
            pred_len = _accept_length(
                conf_prob.view(num_blocks, block_size)[:, start_pos:], draft_mask
            )
            target_len = _accept_length(accept_blocks[:, start_pos:], draft_mask)
            block_valid = (draft_mask.sum(dim=-1) > 0).to(pred_len.dtype)
            length_loss = smooth_l1_loss(pred_len, target_len, reduction="none")
            length_loss = (length_loss * block_valid).sum() / (block_valid.sum() + _EPS)
            loss = loss + confidence_length_alpha * length_loss
            with torch.no_grad():
                metrics["confidence_length_loss_sum"] = length_loss.detach().clone()
                metrics["confidence_length_loss_total"] = torch.ones((), device=device)
                metrics["confidence_accept_len_pred_sum"] = (
                    pred_len * block_valid
                ).sum()
                metrics["confidence_accept_len_pred_total"] = (
                    block_valid.sum().clamp_min(1.0)
                )

    ones = torch.ones((), device=device)
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    with torch.no_grad():
        mask_f = loss_mask.to(accept_rate.dtype)
        metrics["accept_rate_sum"] = (accept_rate * mask_f).sum()
        metrics["accept_rate_total"] = mask_f.sum().clamp_min(1.0)
        per_block_len = accept_prefix.sum(dim=-1) + 1.0
        block_valid = (draft_mask.sum(dim=-1) > 0).to(accept_rate.dtype)
        metrics["accept_len_sum"] = (per_block_len * block_valid).sum()
        metrics["accept_len_total"] = block_valid.sum().clamp_min(1.0)

        if base_logits is not None:
            base_p = softmax(base_logits.float(), dim=-1)
            base_accept_rate = torch.minimum(base_p, target_p).sum(dim=-1)
            base_accept_blocks = base_accept_rate.view(num_blocks, block_size)
            base_prefix = (base_accept_blocks[:, start_pos:] * draft_mask).cumprod(
                dim=-1
            )
            base_per_block_len = base_prefix.sum(dim=-1) + 1.0
            metrics["base_accept_rate_sum"] = (base_accept_rate * mask_f).sum()
            metrics["base_accept_rate_total"] = mask_f.sum().clamp_min(1.0)
            metrics["correction_accept_rate_gain_sum"] = (
                (accept_rate - base_accept_rate) * mask_f
            ).sum()
            metrics["correction_accept_rate_gain_total"] = mask_f.sum().clamp_min(1.0)
            metrics["base_accept_len_sum"] = (base_per_block_len * block_valid).sum()
            metrics["base_accept_len_total"] = block_valid.sum().clamp_min(1.0)
            metrics["correction_accept_len_gain_sum"] = (
                (per_block_len - base_per_block_len) * block_valid
            ).sum()
            metrics["correction_accept_len_gain_total"] = block_valid.sum().clamp_min(
                1.0
            )

            delta_logits = logits.float() - base_logits.float()
            delta_rms = delta_logits.square().mean(dim=-1).sqrt()
            metrics["correction_logit_rms_sum"] = (delta_rms * mask_f).sum()
            metrics["correction_logit_rms_total"] = mask_f.sum().clamp_min(1.0)
            argmax_changed = (logits.argmax(dim=-1) != base_logits.argmax(dim=-1)).to(
                mask_f.dtype
            )
            metrics["correction_argmax_change_rate_sum"] = (
                argmax_changed * mask_f
            ).sum()
            metrics["correction_argmax_change_rate_total"] = mask_f.sum().clamp_min(1.0)
            base_ids = base_logits.argmax(dim=-1)
            metrics.update(
                _change_outcome_metrics(
                    "dspark_head",
                    logits.argmax(dim=-1),
                    base_ids,
                    target_ids,
                    loss_mask,
                )
            )

        if rollout_accept_rate is not None:
            rollout_accept_blocks = rollout_accept_rate.view(num_blocks, block_size)
            rollout_prefix = (
                rollout_accept_blocks[:, start_pos:] * draft_mask
            ).cumprod(dim=-1)
            rollout_per_block_len = rollout_prefix.sum(dim=-1) + 1.0
            metrics["rollout_accept_rate_sum"] = (rollout_accept_rate * mask_f).sum()
            metrics["rollout_accept_rate_total"] = mask_f.sum().clamp_min(1.0)
            metrics["rollout_accept_len_sum"] = (
                rollout_per_block_len * block_valid
            ).sum()
            metrics["rollout_accept_len_total"] = block_valid.sum().clamp_min(1.0)
            if base_logits is not None and rollout_logits is not None:
                metrics["rollout_accept_len_gain_sum"] = (
                    (rollout_per_block_len - base_per_block_len) * block_valid
                ).sum()
                metrics["rollout_accept_len_gain_total"] = block_valid.sum().clamp_min(
                    1.0
                )
                metrics.update(
                    _change_outcome_metrics(
                        "causal_rollout",
                        rollout_logits.argmax(dim=-1),
                        base_ids,
                        target_ids,
                        loss_mask,
                    )
                )

        if collaboration_base_accept_rate is not None:
            collaboration_base_blocks = collaboration_base_accept_rate.view(
                num_blocks, block_size
            )
            collaboration_base_prefix = (
                collaboration_base_blocks[:, start_pos:] * draft_mask
            ).cumprod(dim=-1)
            collaboration_base_len = collaboration_base_prefix.sum(dim=-1) + 1.0
            metrics["correction_only_accept_rate_sum"] = (
                collaboration_base_accept_rate * mask_f
            ).sum()
            metrics["correction_only_accept_rate_total"] = mask_f.sum().clamp_min(1.0)
            metrics["correction_only_accept_len_sum"] = (
                collaboration_base_len * block_valid
            ).sum()
            metrics["correction_only_accept_len_total"] = block_valid.sum().clamp_min(
                1.0
            )
            metrics["collaboration_accept_rate_gain_sum"] = (
                (accept_rate - collaboration_base_accept_rate) * mask_f
            ).sum()
            metrics["collaboration_accept_rate_gain_total"] = mask_f.sum().clamp_min(
                1.0
            )
            metrics["collaboration_accept_len_gain_sum"] = (
                (per_block_len - collaboration_base_len) * block_valid
            ).sum()
            metrics["collaboration_accept_len_gain_total"] = (
                block_valid.sum().clamp_min(1.0)
            )
            metrics.update(
                _change_outcome_metrics(
                    "collaboration_markov",
                    logits.argmax(dim=-1),
                    collaboration_base_logits.argmax(dim=-1),
                    target_ids,
                    loss_mask,
                )
            )

        if collaboration_gate is not None:
            gate_values = collaboration_gate.float().reshape(1, -1)
            if gate_values.shape != loss_mask.shape:
                raise ValueError(
                    "collaboration_gate must contain one value per draft position"
                )
            metrics["collaboration_markov_gate_mean_sum"] = (gate_values * mask_f).sum()
            metrics["collaboration_markov_gate_mean_total"] = mask_f.sum().clamp_min(
                1.0
            )

    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()
    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    if rollout_logits is not None:
        rollout_ids = torch.argmax(rollout_logits, dim=-1)
        rollout_correct, rollout_total = compute_accuracy_multi_step(
            rollout_ids, target_ids, loss_mask, pos_idx, block_size
        )
        metrics["rollout_full_acc_sum"] = rollout_correct[start_pos:].sum()
        metrics["rollout_full_acc_total"] = rollout_total[start_pos:].sum()

    return loss, metrics
