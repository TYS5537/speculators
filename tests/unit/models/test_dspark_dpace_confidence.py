"""Confidence match-draft reuses the main CE's D-PACE position weights."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[3]


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def metrics_modules(monkeypatch):
    common = _load_file(
        "dpace_confidence_common", ROOT / "src/speculators/models/metrics.py"
    )
    monkeypatch.setitem(sys.modules, "speculators.models.metrics", common)
    dspark = _load_file(
        "dpace_confidence_dspark", ROOT / "src/speculators/models/dspark/metrics.py"
    )
    return common, dspark


@pytest.mark.parametrize("weighting", ["uniform", "match-draft"])
@pytest.mark.parametrize("base_diagnostics", [False, True])
def test_dpace_confidence_loss_and_gradient_use_main_draft_weights(
    metrics_modules, weighting, base_diagnostics
):
    common, dspark = metrics_modules
    probabilities = torch.tensor([[[0.2, 0.8], [0.4, 0.6], [0.8, 0.2]]])
    logits = probabilities.log().requires_grad_()
    targets = torch.tensor([[[100.0, -100.0]]]).expand(1, 3, 2)
    confidence = torch.zeros(1, 3, requires_grad=True)
    mask = torch.ones(1, 3)
    loss, metrics = dspark.compute_metrics(
        logits,
        targets,
        confidence,
        mask,
        block_size=3,
        loss_config=common.resolve_loss_config("ce"),
        per_position_loss_weight="dpace",
        dpace_alpha=0.5,
        confidence_loss_weighting=weighting,
        base_logits=torch.tensor([[[-100.0, 100.0]]]).expand_as(logits)
        if base_diagnostics
        else None,
    )
    loss.backward()

    weights = (
        torch.tensor([[1.398, 0.798, 0.378]])
        if weighting == "match-draft"
        else torch.ones(1, 3)
    )
    expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        confidence.detach(), probabilities[:, :, 0], reduction="none"
    )
    torch.testing.assert_close(
        metrics["confidence_loss_sum"], (expected_bce * weights).sum() / 3
    )
    torch.testing.assert_close(
        confidence.grad, (0.5 - probabilities[:, :, 0]) * weights / 3
    )
    # Confidence targets and D-PACE weights are detached; sharing them does not
    # introduce a second gradient through draft probabilities or CE weights.
    reference_logits = probabilities.log().requires_grad_()
    reference_loss, _ = dspark.compute_metrics(
        reference_logits,
        targets,
        None,
        mask,
        block_size=3,
        loss_config=common.resolve_loss_config("ce"),
        per_position_loss_weight="dpace",
        dpace_alpha=0.5,
    )
    reference_loss.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        loss.detach(), reference_loss + metrics["confidence_loss_sum"]
    )


@pytest.mark.parametrize("sample_from_anchor", [False, True])
@pytest.mark.parametrize("mask_kind", ["partial", "empty"])
def test_dpace_shared_weights_respect_anchor_and_partial_supervision(
    metrics_modules, sample_from_anchor, mask_kind
):
    common, dspark = metrics_modules
    logits = torch.tensor([[[0.2, 0.8], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]]]).log()
    targets = torch.tensor([[[100.0, -100.0]]]).expand_as(logits)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    if not sample_from_anchor:
        mask[:, 0] = 0
    if mask_kind == "empty":
        mask.zero_()
    confidence = torch.zeros(1, 4, requires_grad=True)
    loss, metrics = dspark.compute_metrics(
        logits,
        targets,
        confidence,
        mask,
        block_size=4,
        loss_config=common.resolve_loss_config("ce"),
        per_position_loss_weight="dpace",
        confidence_loss_weighting="match-draft",
        sample_from_anchor=sample_from_anchor,
    )
    loss.backward()
    weights = common.dpace_loss_decay(
        torch.arange(4).unsqueeze(0),
        loss_mask=mask,
        block_size=4,
        dpace_alpha=0.5,
        elementwise_loss=common.ce_loss(logits, targets) * mask,
    )
    expected = (
        (0.5 - logits.softmax(-1)[:, :, 0]) * weights * mask / (mask.sum() + 1e-8)
    )
    torch.testing.assert_close(confidence.grad, expected)
    assert torch.isfinite(loss)
    assert torch.count_nonzero(confidence.grad[mask == 0]) == 0
    if mask_kind == "empty":
        assert loss.item() == 0
        assert metrics["confidence_loss_sum"].item() == 0


def test_dpace_confidence_keeps_pruned_full_target_acceptance(metrics_modules):
    common, dspark = metrics_modules
    targets = torch.tensor([[[0.10, 0.05], [0.10, 0.05]]]).log()
    confidence = torch.zeros(1, 2, requires_grad=True)
    loss, metrics = dspark.compute_metrics(
        targets,
        targets,
        confidence,
        torch.ones(1, 2),
        block_size=2,
        loss_config=common.resolve_loss_config("ce"),
        per_position_loss_weight="dpace",
        confidence_loss_weighting="match-draft",
        target_log_normalizer=torch.zeros(1, 2),
        target_argmax_ids=torch.full((1, 2), -1),
    )
    loss.backward()
    # Conditional CE sees q(target)=2/3, while real acceptance remains 0.15.
    weights = torch.tensor([[5 / 6 + 25 / 36, 25 / 36]])
    torch.testing.assert_close(confidence.grad, 0.35 * weights / 2)
    assert metrics["accept_rate_sum"].item() == pytest.approx(0.30)
    assert metrics["full_acc_sum"].item() == 0


@pytest.mark.parametrize("weighting", ["uniform", "match-draft"])
def test_default_fixed_decay_ce_tv_loss_and_gradients_are_unchanged(
    metrics_modules, weighting
):
    common, dspark = metrics_modules
    torch.manual_seed(46)
    logits = torch.randn(1, 4, 5, requires_grad=True)
    targets = torch.randn(1, 4, 5)
    confidence = torch.randn(1, 4, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    config = common.resolve_loss_config('{"ce":0.1,"tv":0.9}')
    actual, _ = dspark.compute_metrics(
        logits,
        targets,
        confidence,
        mask,
        block_size=4,
        loss_config=config,
        confidence_loss_weighting=weighting,
    )
    actual_gradients = torch.autograd.grad(actual, (logits, confidence))
    positions = torch.arange(4).unsqueeze(0)
    weights = common.position_weights(positions.float(), 4, 7.0)
    expected, _ = common.compound_loss(
        logits, targets, mask, positions, config, decay_fn=lambda *_args, **_kw: weights
    )
    overlap = torch.minimum(logits.softmax(-1), targets.softmax(-1)).sum(-1).detach()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        confidence, overlap, reduction="none"
    )
    confidence_weights = weights if weighting == "match-draft" else None
    expected = expected + dspark._masked_weighted_mean(bce, mask, confidence_weights)
    expected_gradients = torch.autograd.grad(expected, (logits, confidence))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
