"""Opt-in checkpointed vocabulary projection and per-position loss chunks.

Only the small loss/ID/overlap vectors survive between chunks. Backward recomputes
the vocabulary intermediates. The final masked reduction uses the original full
sequence denominator, decay and term order.
"""

from functools import partial

import torch
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from speculators import losses
from speculators.losses import eager
from speculators.losses.utils import exp_loss_decay, reduce_masked_loss
from speculators.models.eagle3.metrics import _distribution_overlap
from speculators.models.leagle.head_gradients import (
    FrozenHeadGradients,
    supports_fp32_head,
)
from speculators.models.leagle.shared_loss import normalized_distribution, shared_terms
from speculators.models.leagle.teacher_cache import cached_terms
from speculators.models.leagle.training import TVOverlapProbe
from speculators.models.metrics import tv_loss_fused_or_eager


def validate_chunked_loss(loss_config):
    known = {
        "ce": (eager.ce_loss, losses.ce_loss),
        "tv": (eager.tv_loss, losses.tv_loss, tv_loss_fused_or_eager),
        "kl_div": (eager.kl_div_loss, losses.kl_div_loss),
    }
    if not loss_config or any(
        function not in known.get(name, ())
        for name, (function, _) in loss_config.items()
    ):
        raise ValueError(
            "LEAGLE chunked loss supports only built-in CE, TV and forward KL"
        )


def _project_chunk(
    hidden,
    teacher_hidden,
    weight,
    teacher_weight,
    *,
    loss_config,
    metrics,
    candidate,
    teacher_cache=None,
    teacher_plan=None,
    target_start=0,
    target_indices=None,
    predict=True,
    shared_normalization=False,
):
    logits = functional.linear(hidden, weight)
    if shared_normalization:
        if teacher_cache is not None:
            probabilities, target_ids = teacher_cache.get(
                teacher_hidden,
                teacher_weight,
                start=target_start,
                indices=target_indices,
                plan=teacher_plan,
            )
        else:
            with torch.no_grad():
                targets = functional.linear(teacher_hidden, teacher_weight)
                probabilities = normalized_distribution(targets)[1]
                target_ids = targets.argmax(-1)
        terms, overlap, teacher = shared_terms(
            logits, probabilities, target_ids, loss_config, metrics, candidate
        )
        return (
            terms,
            logits.argmax(-1) if predict else None,
            target_ids if metrics else None,
            overlap,
            teacher,
        )
    if teacher_cache is not None:
        probabilities, target_ids = teacher_cache.get(
            teacher_hidden,
            teacher_weight,
            start=target_start,
            indices=target_indices,
            plan=teacher_plan,
        )
        terms, overlap = cached_terms(
            logits, probabilities, target_ids, loss_config, metrics
        )
        return (
            terms,
            logits.argmax(-1) if predict else None,
            target_ids if metrics else None,
            overlap,
            logits.detach() if candidate else None,
        )
    with torch.no_grad():
        targets = functional.linear(teacher_hidden, teacher_weight)
    probe = TVOverlapProbe(loss_config) if metrics else None
    config = probe.loss_config if probe is not None else loss_config
    terms = tuple(function(logits, targets) for function, _ in config.values())
    target_ids = targets.argmax(-1) if metrics else None
    overlap = None
    if metrics:
        overlap = probe.overlap
        if overlap is None:
            overlap = _distribution_overlap(logits, targets)
    return (
        terms,
        logits.argmax(-1) if predict else None,
        target_ids,
        overlap,
        logits.detach() if candidate else None,
    )


def _reduce_terms(parts, loss_config, mask, step, decay):
    total = torch.zeros((), dtype=torch.float32, device=mask.device)
    positions = torch.full(mask.shape, step, dtype=torch.long, device=mask.device)
    for vectors, (_, weight) in zip(parts, loss_config.values(), strict=True):
        vector = torch.cat(vectors, dim=1)
        mean = reduce_masked_loss(
            vector, mask, positions, partial(exp_loss_decay, gamma=decay)
        )
        total = total + weight * mean
    return total


@torch.no_grad()
def predict_chunks(hidden, weight, chunk_size):
    return torch.cat(
        [
            functional.linear(hidden[:, start : start + chunk_size], weight).argmax(-1)
            for start in range(0, hidden.shape[1], chunk_size)
        ],
        dim=1,
    )


def chunked_training_step(objective, hidden, step, step_mask):
    model = objective.model
    width = hidden.shape[1] - step
    plans = (
        objective.teacher_cache.plan_chunks(width, objective.chunk_size, start=step)
        if objective.teacher_cache is not None
        else None
    )
    gradients = main_head_gradients(objective, hidden, step, step_mask[:, step:])
    parts = [[] for _ in objective.loss_config]
    predictions, target_ids, overlaps = [], [], []
    candidate_mask = objective.valid[:, :width] & step_mask[:, step:]
    candidate_scale = objective.decay**step / candidate_mask.sum(
        dtype=torch.float32
    ).clamp_min(1)
    function = partial(
        _project_chunk,
        loss_config=objective.loss_config,
        metrics=objective.collect_metrics,
        candidate=model.candidate_head is not None,
        predict=objective.need_predictions or objective.collect_metrics,
        shared_normalization=objective.shared_normalization,
    )
    for start in range(0, width, objective.chunk_size):
        end = min(start + objective.chunk_size, width)
        project = (
            partial(
                function,
                teacher_cache=objective.teacher_cache,
                target_start=start + step,
                teacher_plan=plans[start // objective.chunk_size]
                if plans is not None
                else None,
            )
            if objective.teacher_cache is not None
            else function
        )
        arguments = (
            hidden[:, start:end],
            objective.targets[:, start + step : end + step],
            objective.draft_weight,
            objective.teacher_weight,
        )
        result = (
            gradients.project(project, arguments, slice(start, end))
            if gradients is not None
            else checkpoint(
                project, *arguments, use_reentrant=False, preserve_rng_state=True
            )
            if torch.is_grad_enabled()
            else project(*arguments)
        )
        terms, ids, labels, overlap, logits = result
        for destination, values in zip(parts, terms, strict=True):
            destination.append(values)
        if ids is not None:
            predictions.append(ids)
        if objective.collect_metrics:
            target_ids.append(labels)
            overlaps.append(overlap)
        if model.candidate_head is not None:
            # Use the full step's denominator for every chunk, never a per-chunk mean.
            objective.add_candidate(
                hidden[:, start:end],
                logits,
                candidate_mask[:, start:end],
                loss_scale=candidate_scale,
            )
    if objective.collect_metrics:
        objective.metrics.update_aligned(
            torch.cat(predictions, dim=1),
            torch.cat(target_ids, dim=1),
            torch.cat(overlaps, dim=1),
            step,
        )
    if step and objective.need_predictions:
        predictions.append(
            predict_chunks(
                hidden[:, width:], objective.draft_weight, objective.chunk_size
            )
        )
    loss = _reduce_terms(
        parts, objective.loss_config, step_mask[:, step:], step, objective.decay
    )
    if gradients is not None:
        loss = gradients.attach(loss)
    return (
        torch.cat(predictions, dim=1) if objective.need_predictions else None,
        loss,
    )


def main_head_gradients(objective, hidden, step, mask):
    return (
        FrozenHeadGradients(hidden, objective.loss_config, mask, step, objective.decay)
        if objective.stream_main_loss
        and torch.is_grad_enabled()
        and supports_fp32_head(hidden)
        else None
    )
