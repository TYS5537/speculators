"""Opt-in output projection on supervised positions with full-step reductions."""

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from speculators.models.leagle.chunked_loss import (
    _project_chunk,
    _reduce_terms,
    main_head_gradients,
    predict_chunks,
)


def masked_training_step(objective, hidden, step, step_mask):
    model = objective.model
    width = hidden.shape[1] - step
    mask = step_mask[:, step:]
    gradients = main_head_gradients(objective, hidden, step, mask)
    indices = mask[0].nonzero(as_tuple=True)[0]
    all_target_indices = indices + step
    plans = (
        objective.teacher_cache.plan_chunks(
            indices.numel(), objective.chunk_size, indices=all_target_indices
        )
        if objective.teacher_cache is not None
        else None
    )
    parts = [[] for _ in objective.loss_config]
    ids, labels, overlaps = [], [], []
    candidate_mask = objective.valid[:, :width] & mask
    candidate_scale = objective.decay**step / candidate_mask.sum(
        dtype=torch.float32
    ).clamp_min(1)
    for start in range(0, indices.numel(), objective.chunk_size):
        selected = indices[start : start + objective.chunk_size]
        target_indices = all_target_indices[start : start + objective.chunk_size]
        function = partial(
            _project_chunk,
            loss_config=objective.loss_config,
            metrics=objective.collect_metrics,
            candidate=model.candidate_head is not None,
            teacher_cache=objective.teacher_cache,
            teacher_plan=plans[start // objective.chunk_size]
            if plans is not None
            else None,
            target_indices=target_indices,
            predict=objective.collect_metrics,
            shared_normalization=objective.shared_normalization,
        )
        arguments = (
            hidden[:, selected],
            objective.targets[:, target_indices],
            objective.draft_weight,
            objective.teacher_weight,
        )
        result = (
            gradients.project(function, arguments, selected)
            if gradients is not None
            else checkpoint(
                function, *arguments, use_reentrant=False, preserve_rng_state=True
            )
            if torch.is_grad_enabled()
            else function(*arguments)
        )
        terms, predicted, targets, overlap, logits = result
        for destination, values in zip(parts, terms, strict=True):
            destination.append(values)
        if objective.collect_metrics:
            ids.append(predicted)
            labels.append(targets)
            overlaps.append(overlap)
        objective.add_candidate(
            arguments[0],
            logits,
            candidate_mask[:, selected],
            loss_scale=candidate_scale,
        )

    def aligned(values, dtype):
        full = torch.zeros((1, width), dtype=dtype, device=hidden.device)
        return full.index_copy(1, indices, torch.cat(values, dim=1)) if values else full

    if indices.numel():
        vectors = [[aligned(values, values[0].dtype)] for values in parts]
        loss = _reduce_terms(
            vectors, objective.loss_config, mask, step, objective.decay
        )
    else:
        # Keep zero gradients connected to the whole backbone and LM head. Empty
        # selections must not skip distributed backward or change optimizer state.
        loss = (hidden[:, :1].sum() + model.lm_head.weight[:1, :1].sum()).float() * 0.0
    if gradients is not None:
        loss = gradients.attach(loss)
    if objective.collect_metrics:
        objective.metrics.update_aligned(
            aligned(ids, torch.long),
            aligned(labels, torch.long),
            aligned(overlaps, torch.float32),
            step,
        )
    predictions = (
        predict_chunks(hidden, objective.draft_weight, objective.chunk_size)
        if objective.need_predictions
        else None
    )
    return predictions, loss
