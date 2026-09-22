"""Validation-only outputs and ordered auxiliary-loss accounting contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional

from speculators.models.muse.core import MuseDraftModel


class _RecordingProjection(nn.Linear):
    def __init__(self):
        super().__init__(4, 6, bias=False, dtype=torch.float64)
        self.calls = []

    def forward(self, hidden):
        self.calls.append((hidden, torch.is_grad_enabled()))
        return super().forward(hidden)


class _ValidationHarness(nn.Module):
    _validation_correction_outputs = MuseDraftModel._validation_correction_outputs

    def __init__(self, diagnostics, rollout, *, sample=False):
        super().__init__()
        self.config = SimpleNamespace(
            correction_base_diagnostics=diagnostics,
            correction_rollout_metrics=rollout,
            sample_from_anchor=sample,
        )
        self.candidate_selector = None
        self.lm_head = _RecordingProjection()
        self.rollout_gain = nn.Parameter(torch.arange(6, dtype=torch.float64))
        self.rollout_calls, self.step_calls = [], []
        self.rollout_blocks = None

    def rollout_correction(self, hidden, **kwargs):
        self.rollout_calls.append((hidden, kwargs, torch.is_grad_enabled()))
        # Exercise the real public decorator rather than faking a no-grad rollout.
        return MuseDraftModel.rollout_correction(self, hidden, **kwargs)

    def _rollout_correction_steps(self, hidden, anchor_token_ids, **kwargs):
        self.step_calls.append(
            (hidden, anchor_token_ids, kwargs, torch.is_grad_enabled())
        )
        self.rollout_blocks = hidden.sum(dim=-1, keepdim=True) + self.rollout_gain
        return self.rollout_blocks.argmax(dim=-1), self.rollout_blocks, hidden, hidden


def _validation_inputs(with_base):
    hidden = torch.randn(1, 6, 4, requires_grad=True)
    base = hidden[..., :1].expand(1, 6, 6) if with_base else None
    return {
        "hidden": hidden,
        "hidden_blocks": hidden.view(2, 3, 4),
        "base_logits": base,
        "base_logits_blocks": None if base is None else base.view(2, 3, 6),
        "targets": torch.randn(1, 6, 6, dtype=torch.float64, requires_grad=True),
        "anchor_token_ids": torch.tensor([8, 9]),
        "correction_output_mode": "logits",
    }


@pytest.mark.parametrize(
    ("diagnostics", "rollout"),
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("with_base", [False, True])
def test_training_validation_flags_do_no_work_or_detach_existing_base(
    diagnostics, rollout, with_base
):
    model = _ValidationHarness(diagnostics, rollout).train()
    inputs = _validation_inputs(with_base)
    base, rollout_logits = model._validation_correction_outputs(**inputs)
    assert base is inputs["base_logits"]
    assert rollout_logits is None
    assert model.lm_head.calls == []
    assert model.rollout_calls == []
    assert model.step_calls == []
    if with_base:
        assert base.requires_grad
        base.sum().backward()
        expected = torch.zeros_like(inputs["hidden"])
        expected[..., 0] = 6
        torch.testing.assert_close(inputs["hidden"].grad, expected)


@pytest.mark.parametrize(
    ("diagnostics", "rollout"),
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("with_base", [False, True])
def test_eval_diagnostics_and_rollout_are_independent(diagnostics, rollout, with_base):
    model = _ValidationHarness(diagnostics, rollout).eval()
    inputs = _validation_inputs(with_base)
    base, rollout_logits = model._validation_correction_outputs(**inputs)
    assert len(model.lm_head.calls) == int(diagnostics and not with_base)
    if with_base:
        assert base is inputs["base_logits"]
        assert base.requires_grad
        assert base.grad_fn is not None
    elif diagnostics:
        projected_hidden, grad_enabled = model.lm_head.calls[0]
        assert projected_hidden.shape == (1, 6, 4)
        assert projected_hidden.dtype == model.lm_head.weight.dtype
        assert not projected_hidden.requires_grad
        assert not grad_enabled
        torch.testing.assert_close(projected_hidden, inputs["hidden"].detach().double())
        torch.testing.assert_close(
            base, functional.linear(projected_hidden, model.lm_head.weight)
        )
        assert not base.requires_grad
    else:
        assert base is None
    assert len(model.rollout_calls) == int(rollout)
    assert len(model.step_calls) == int(rollout)
    if rollout:
        hidden, kwargs, grad_enabled = model.rollout_calls[0]
        assert grad_enabled  # The helper itself must not widen the no-grad region.
        assert not hidden.requires_grad
        assert hidden.data_ptr() == inputs["hidden_blocks"].data_ptr()
        assert kwargs["anchor_token_ids"] is inputs["anchor_token_ids"]
        # Newly created diagnostic logits must not become rollout's original base.
        assert kwargs["base_logits"] is inputs["base_logits_blocks"]
        assert (
            model.step_calls[0][2]["precomputed_base_logits"]
            is inputs["base_logits_blocks"]
        )
        assert not model.step_calls[0][3]
        assert rollout_logits.shape == (1, 6, 6)
        torch.testing.assert_close(
            rollout_logits, model.rollout_blocks.reshape(1, 6, 6)
        )
        assert not rollout_logits.requires_grad
    else:
        assert rollout_logits is None


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("mode", ["hidden", "logits"])
def test_rollout_initial_logits_only_for_reserved_anchor_logit_mode(sample, mode):
    model = _ValidationHarness(False, True, sample=sample).eval()
    inputs = _validation_inputs(False)
    inputs["correction_output_mode"] = mode
    model._validation_correction_outputs(**inputs)
    initial = model.rollout_calls[0][1]["initial_previous_logits"]
    if not sample and mode == "logits":
        torch.testing.assert_close(initial, inputs["targets"].view(2, 3, 6)[:, 0])
        assert initial.requires_grad
        assert initial.grad_fn is not None
        initial.sum().backward()
        expected = torch.zeros_like(inputs["targets"]).view(2, 3, 6)
        expected[:, 0] = 1
        torch.testing.assert_close(inputs["targets"].grad.view_as(expected), expected)
    else:
        assert initial is None


class _RecordingNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([0.7, 1.2, 0.4, -0.5], dtype=torch.float64)
        )
        self.calls = []

    def forward(self, hidden):
        self.calls.append((hidden, torch.is_grad_enabled()))
        return hidden * self.weight


class _LossHarness(nn.Module):
    _add_auxiliary_losses = MuseDraftModel._add_auxiliary_losses

    def __init__(
        self, *, auxiliary, sample=True, selector_weight=0.3, hidden_weight=0.7
    ):
        super().__init__()
        self.config = SimpleNamespace(
            correction_hidden_aux_loss=auxiliary,
            sample_from_anchor=sample,
            dflash2_selector_loss_weight=selector_weight,
            correction_hidden_aux_weight=hidden_weight,
        )
        self.verifier_norm = _RecordingNorm()
        self.alignment_calls = []

    def _hidden_alignment_loss(self, corrected, target, mask):
        loss = MuseDraftModel._hidden_alignment_loss(corrected, target, mask)
        self.alignment_calls.append((corrected, target, mask, loss))
        return loss


class _RecordingMetrics(dict):
    def __init__(self, **values):
        super().__init__(values)
        self.writes = []

    def __setitem__(self, key, value):
        self.writes.append((key, value))
        super().__setitem__(key, value)


def _loss_inputs(*, selector, zero_mask=False):
    return {
        "loss": torch.tensor(1.25, dtype=torch.float64, requires_grad=True),
        "metrics": _RecordingMetrics(untouched=torch.tensor(19.0)),
        "selector_loss": torch.tensor(0.875, dtype=torch.float64, requires_grad=True)
        if selector
        else None,
        "corrected_hidden": torch.linspace(-0.2, 1.3, 16)
        .reshape(2, 2, 4)
        .requires_grad_(),
        "verifier_last_hidden_states": (
            torch.arange(28, dtype=torch.float16).reshape(1, 7, 4) / 5
        ).requires_grad_(),
        "anchored_block_indices": torch.tensor([2, 5, 1, 4]),
        "aligned_loss_mask": torch.zeros(1, 4)
        if zero_mask
        else torch.tensor([[1.0, 0.0, 1.0, 1.0]]),
    }


def _expected_hidden_loss(model, inputs, corrected):
    teacher = (
        inputs["verifier_last_hidden_states"].detach().double()
        * model.verifier_norm.weight.detach()
    )
    if not model.config.sample_from_anchor:
        teacher = torch.roll(teacher, 1, dims=1)
    teacher = teacher[:, inputs["anchored_block_indices"]].view_as(corrected)
    mask = inputs["aligned_loss_mask"].view(2, 2)
    per_token = functional.smooth_l1_loss(
        corrected.float(), teacher.float(), reduction="none"
    ).mean(dim=-1)
    return (per_token * mask).sum() / mask.sum().clamp_min(1), teacher


def test_no_auxiliary_terms_preserve_loss_and_metrics_identity():
    model = _LossHarness(auxiliary=False)
    inputs = _loss_inputs(selector=False)
    inputs["corrected_hidden"] = None
    loss, metrics = model._add_auxiliary_losses(**inputs)
    assert loss is inputs["loss"]
    assert metrics is inputs["metrics"]
    assert metrics.writes == []
    assert model.verifier_norm.calls == []
    assert model.alignment_calls == []


@pytest.mark.parametrize(
    (
        "selector",
        "auxiliary",
        "sample",
        "selector_weight",
        "hidden_weight",
        "zero_mask",
    ),
    [
        (True, False, True, 0.3, 0.7, False),
        (False, True, True, 0.3, 0.7, False),
        (True, True, False, 0.3, 0.7, False),
        (True, False, True, 0.0, 0.7, False),
        (False, True, False, 0.3, 0.0, False),
        (True, True, True, 0.0, 0.0, False),
        (False, True, False, 0.3, 0.7, True),
        (True, True, True, 0.3, 0.7, True),
    ],
)
def test_auxiliary_terms_preserve_order_metrics_and_gradient_boundaries(
    *, selector, auxiliary, sample, selector_weight, hidden_weight, zero_mask
):
    model = _LossHarness(
        auxiliary=auxiliary,
        sample=sample,
        selector_weight=selector_weight,
        hidden_weight=hidden_weight,
    )
    inputs = _loss_inputs(selector=selector, zero_mask=zero_mask)
    loss, metrics = model._add_auxiliary_losses(**inputs)
    assert metrics is inputs["metrics"]
    assert metrics["untouched"] == 19
    expected = inputs["loss"].detach().clone().requires_grad_()
    expected_initial = expected
    expected_selector = None
    writes, subtotal = [], []
    if selector:
        expected_selector = inputs["selector_loss"].detach().clone().requires_grad_()
        expected = expected + selector_weight * expected_selector
        subtotal.append(expected.detach())
        writes.extend(
            ["loss_sum", "dflash2_selector_loss_sum", "dflash2_selector_loss_total"]
        )
        torch.testing.assert_close(
            metrics["dflash2_selector_loss_sum"], inputs["selector_loss"].detach()
        )
        assert (
            metrics["dflash2_selector_loss_sum"].data_ptr()
            != inputs["selector_loss"].data_ptr()
        )
    expected_corrected = inputs["corrected_hidden"].detach().clone().requires_grad_()
    assert len(model.verifier_norm.calls) == int(auxiliary)
    assert len(model.alignment_calls) == int(auxiliary)
    if auxiliary:
        expected_auxiliary, teacher = _expected_hidden_loss(
            model, inputs, expected_corrected
        )
        expected = expected + hidden_weight * expected_auxiliary
        subtotal.append(expected.detach())
        writes.extend(
            [
                "loss_sum",
                "correction_hidden_aux_loss_sum",
                "correction_hidden_aux_loss_total",
            ]
        )
        norm_input, grad_enabled = model.verifier_norm.calls[0]
        assert norm_input.dtype == model.verifier_norm.weight.dtype
        torch.testing.assert_close(
            norm_input, inputs["verifier_last_hidden_states"].double()
        )
        assert not grad_enabled
        corrected, target, mask, raw_auxiliary = model.alignment_calls[0]
        assert corrected is inputs["corrected_hidden"]
        assert not target.requires_grad
        torch.testing.assert_close(target, teacher)
        torch.testing.assert_close(mask, inputs["aligned_loss_mask"].view(2, 2))
        torch.testing.assert_close(
            metrics["correction_hidden_aux_loss_sum"], expected_auxiliary.detach()
        )
        assert (
            metrics["correction_hidden_aux_loss_sum"].data_ptr()
            != raw_auxiliary.data_ptr()
        )
    assert [name for name, _value in metrics.writes] == writes
    recorded_totals = [value for name, value in metrics.writes if name == "loss_sum"]
    for value, expected_subtotal in zip(recorded_totals, subtotal, strict=True):
        torch.testing.assert_close(value, expected_subtotal)
    torch.testing.assert_close(loss, expected)
    assert metrics["loss_sum"].data_ptr() != loss.data_ptr()
    for name, value in metrics.writes:
        assert not value.requires_grad
        assert value.grad_fn is None
        if name.endswith("_total"):
            assert value.dtype == torch.float32
            assert value.shape == ()
            assert value == 1
            assert value.device == loss.device
    loss.backward()
    expected.backward()
    torch.testing.assert_close(inputs["loss"].grad, expected_initial.grad)
    if selector:
        torch.testing.assert_close(inputs["selector_loss"].grad, expected_selector.grad)
    if auxiliary:
        torch.testing.assert_close(
            inputs["corrected_hidden"].grad, expected_corrected.grad
        )
        masked = ~inputs["aligned_loss_mask"].view(2, 2).bool()
        assert torch.count_nonzero(inputs["corrected_hidden"].grad[masked]) == 0
    else:
        assert inputs["corrected_hidden"].grad is None
    assert inputs["verifier_last_hidden_states"].grad is None
    assert model.verifier_norm.weight.grad is None


@pytest.mark.parametrize("selector", [False, True])
def test_missing_corrected_hidden_raises_after_selector_accounting(selector):
    model = _LossHarness(auxiliary=True)
    inputs = _loss_inputs(selector=selector)
    inputs["corrected_hidden"] = None
    with pytest.raises(RuntimeError, match="requires corrected DFlash hidden"):
        model._add_auxiliary_losses(**inputs)
    assert model.verifier_norm.calls == []
    assert model.alignment_calls == []
    metrics = inputs["metrics"]
    if selector:
        assert [name for name, _value in metrics.writes] == [
            "loss_sum",
            "dflash2_selector_loss_sum",
            "dflash2_selector_loss_total",
        ]
        expected = (
            inputs["loss"]
            + model.config.dflash2_selector_loss_weight * inputs["selector_loss"]
        )
        torch.testing.assert_close(metrics["loss_sum"], expected.detach())
        assert not metrics["loss_sum"].requires_grad
    else:
        assert metrics.writes == []
