"""Single-step Correction projection, casting and cache ownership contracts."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional

from speculators.models.muse.core import MuseDraftModel
from speculators.models.muse.correction import CausalCorrectionHead


class _CountingLinear(nn.Linear):
    def __init__(self):
        super().__init__(4, 6, bias=False, dtype=torch.float64)
        self.inputs = []

    def forward(self, value):
        self.inputs.append(value)
        return super().forward(value)


class _RecordingHead(nn.Module):
    def __init__(self, output_mode):
        super().__init__()
        self.output_mode = output_mode
        self.state_scale = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.residual_scale = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
        self.auxiliary_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        self.register_buffer(
            "projection", torch.arange(24, dtype=torch.float64).reshape(6, 4) / 29
        )
        self.calls, self.auxiliary_calls, self.fused_calls = [], [], []
        self.next_cache = object()

    def forward(self, previous, hidden, positions, *, cache, use_cache, **kwargs):
        self.calls.append((previous, hidden, positions, cache, use_cache, kwargs))
        states = (
            hidden * self.state_scale + previous * 0.1 + positions.unsqueeze(-1) * 0.01
        )
        for name, value in kwargs.items():
            if not name.endswith("mask"):
                states = states + value.mean(dim=-1, keepdim=True) * 0.07
        residual = (
            states
            if self.output_mode == "hidden"
            else functional.linear(states, self.projection)
        )
        return residual * self.residual_scale, states, self.next_cache

    def auxiliary_hidden_residual(self, states):
        self.auxiliary_calls.append(states)
        return states * self.auxiliary_scale

    def fused_lm_head_residual(self, states, weight):
        self.fused_calls.append((states, weight))
        scale = (
            self.residual_scale
            if self.output_mode == "hidden"
            else self.auxiliary_scale
        )
        return functional.linear(states * scale, weight)


class _StepHarness(nn.Module):
    _rollout_correction_step = MuseDraftModel._rollout_correction_step

    def __init__(
        self, mode, *, dual=False, auxiliary=False, feedback=False, training=True
    ):
        super().__init__()
        self.config = SimpleNamespace(
            correction_project_corrected_hidden=dual,
            correction_hidden_aux_loss=auxiliary,
            correction_hidden_feedback=feedback,
        )
        self.correction_head = _RecordingHead(mode)
        self.lm_head = _CountingLinear()
        self.train(training)


def _input_tensors(mode, options, bases):
    # Distinct current/correction tensors catch accidental rebasing of the residual.
    inputs = {
        "current": torch.randn(2, 4, dtype=torch.float32, requires_grad=True),
        "correction": torch.randn(2, 1, 4, dtype=torch.float64, requires_grad=True),
        "previous": torch.randn(2, 1, 4, dtype=torch.float64, requires_grad=True),
        "current_token_embeddings": torch.randn(
            2, 1, 4, dtype=torch.float64, requires_grad=True
        ),
    }
    if mode == "logits":
        name, width = (
            ("previous_rank_features", 2)
            if bases in {"fused", "both"}
            else ("previous_logits", 6)
        )
        inputs[name] = torch.randn(2, 1, width, dtype=torch.float64, requires_grad=True)
        inputs["previous_logits_mask"] = torch.ones(2, 1, dtype=torch.bool)
    if options.get("feedback"):
        inputs["previous_corrected_hidden"] = torch.randn(
            2, 1, 4, dtype=torch.float64, requires_grad=True
        )
        inputs["previous_corrected_hidden_mask"] = torch.ones(2, 1, dtype=torch.bool)
    if bases in {"precomputed", "both"}:
        inputs["precomputed"] = torch.randn(
            2, 3, 6, dtype=torch.float16, requires_grad=True
        )
    if bases in {"fused", "both"}:
        inputs["fused"] = torch.randn(2, 3, 6, dtype=torch.float32, requires_grad=True)
    return inputs


def _conditioning(inputs):
    return {
        name: value
        for name, value in inputs.items()
        if name not in {"current", "correction", "previous", "precomputed", "fused"}
    }


def _formula_reference(model, inputs, positions, source, use_auxiliary):
    residual, states, _ = model.correction_head(
        inputs["previous"],
        inputs["correction"],
        positions,
        cache=None,
        use_cache=True,
        **_conditioning(inputs),
    )
    hidden_mode = model.correction_head.output_mode == "hidden"
    delta_hidden = (
        residual if hidden_mode else states * model.correction_head.auxiliary_scale
    )
    corrected = inputs["current"]
    if hidden_mode or use_auxiliary:
        corrected = corrected + delta_hidden[:, 0].to(corrected.dtype)
    if source == "precomputed":
        projected = inputs["precomputed"][:, 1]
    elif source == "fused":
        projected = inputs["fused"][:, 1]
        hidden_logits = functional.linear(delta_hidden[:, 0], model.lm_head.weight)
        projected = projected + hidden_logits.to(projected.dtype)
    else:
        projection_input = corrected if source == "corrected" else inputs["current"]
        projected = functional.linear(
            projection_input.to(model.lm_head.weight.dtype), model.lm_head.weight
        )
    logits = (
        projected if hidden_mode else projected + residual[:, 0].to(projected.dtype)
    )
    return logits, states[:, 0], corrected


def _assert_gradient_equal(actual, expected, name):
    assert (actual.grad is None) == (expected.grad is None), name
    if actual.grad is not None:
        assert torch.isfinite(actual.grad).all(), name
        torch.testing.assert_close(
            actual.grad, expected.grad, rtol=1e-5, atol=1e-7, msg=name
        )


@pytest.mark.parametrize(
    ("mode", "options", "bases", "source"),
    [
        pytest.param("hidden", {}, "none", "corrected", id="hidden"),
        pytest.param(
            "hidden", {}, "precomputed", "corrected", id="hidden-ignores-precomputed"
        ),
        pytest.param("hidden", {}, "fused", "fused", id="hidden-fused"),
        pytest.param("hidden", {}, "both", "fused", id="hidden-fused-priority"),
        pytest.param("logits", {}, "none", "original", id="logits"),
        pytest.param(
            "logits", {}, "precomputed", "precomputed", id="logits-precomputed"
        ),
        pytest.param("logits", {}, "fused", "original", id="logits-ignores-fused"),
        pytest.param(
            "logits", {}, "both", "precomputed", id="logits-precomputed-priority"
        ),
        pytest.param("logits", {"auxiliary": True}, "none", "original", id="aux-train"),
        pytest.param(
            "logits",
            {"auxiliary": True, "training": False},
            "none",
            "original",
            id="aux-eval",
        ),
        pytest.param(
            "logits",
            {"feedback": True, "training": False},
            "none",
            "original",
            id="feedback-eval",
        ),
        pytest.param("logits", {"dual": True}, "none", "corrected", id="dual"),
        pytest.param(
            "logits",
            {"dual": True},
            "precomputed",
            "corrected",
            id="dual-ignores-precomputed",
        ),
        pytest.param("logits", {"dual": True}, "fused", "fused", id="dual-fused"),
        pytest.param(
            "logits", {"dual": True}, "both", "fused", id="dual-fused-priority"
        ),
        pytest.param(
            "logits",
            {"dual": True, "auxiliary": True, "training": False},
            "fused",
            "fused",
            id="dual-aux-eval",
        ),
    ],
)
def test_single_step_matches_formula_casts_cache_and_gradients(
    mode, options, bases, source
):
    torch.manual_seed(129)
    model = _StepHarness(mode, **options)
    reference = copy.deepcopy(model)
    inputs = _input_tensors(mode, options, bases)
    reference_inputs = {
        name: value.detach().clone().requires_grad_(value.requires_grad)
        for name, value in inputs.items()
    }
    positions = torch.ones(2, 1, dtype=torch.long)
    cache = [object()]
    kwargs = _conditioning(inputs)
    result = model._rollout_correction_step(
        inputs["previous"],
        inputs["current"],
        inputs["correction"],
        positions,
        position=1,
        cache=cache,
        precomputed_base_logits=inputs.get("precomputed"),
        fused_base_logits=inputs.get("fused"),
        **kwargs,
    )
    use_auxiliary = mode == "logits" and (
        options.get("dual", False)
        or options.get("feedback", False)
        or (options.get("auxiliary", False) and options.get("training", True))
    )
    expected = _formula_reference(
        reference, reference_inputs, positions, source, use_auxiliary
    )
    for actual_value, expected_value in zip(result[:3], expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value)
    assert result[0].shape == (2, 6)
    assert result[1].shape == (2, 4)
    assert result[2].dtype == inputs["current"].dtype
    assert result[3] is model.correction_head.next_cache
    if mode == "logits" and not use_auxiliary:
        assert result[2] is inputs["current"]
    assert len(model.correction_head.calls) == 1
    (
        previous,
        correction,
        recorded_positions,
        recorded_cache,
        use_cache,
        recorded_kwargs,
    ) = model.correction_head.calls[0]
    assert previous is inputs["previous"]
    assert correction is inputs["correction"]
    assert recorded_positions is positions
    assert recorded_cache is cache
    assert use_cache is True
    assert len(cache) == 1
    assert recorded_kwargs.keys() == kwargs.keys()
    for name, value in kwargs.items():
        assert recorded_kwargs[name] is value
    assert len(model.correction_head.auxiliary_calls) == int(use_auxiliary)
    assert len(model.correction_head.fused_calls) == int(source == "fused")
    assert len(model.lm_head.inputs) == int(source in {"original", "corrected"})
    if model.lm_head.inputs:
        assert model.lm_head.inputs[0].dtype == model.lm_head.weight.dtype
        assert model.lm_head.inputs[0].shape == (2, 4)
    if model.correction_head.fused_calls:
        states, weight = model.correction_head.fused_calls[0]
        assert states.shape == (2, 1, 4)
        assert weight is model.lm_head.weight
    sum(value.double().square().mean() for value in result[:3]).backward()
    sum(value.double().square().mean() for value in expected).backward()
    for name, parameter in model.named_parameters():
        _assert_gradient_equal(
            parameter, dict(reference.named_parameters())[name], name
        )
    for name, value in inputs.items():
        _assert_gradient_equal(value, reference_inputs[name], name)
    assert model.correction_head.state_scale.grad.abs() > 0
    for name, value in kwargs.items():
        if value.requires_grad:
            assert value.grad is not None, name
            assert torch.count_nonzero(value.grad) > 0, name


def test_training_auxiliary_runs_even_without_autograd():
    model = _StepHarness("logits", auxiliary=True)
    inputs = _input_tensors("logits", {}, "none")
    with torch.no_grad():
        _, _, corrected, _ = model._rollout_correction_step(
            inputs["previous"],
            inputs["current"],
            inputs["correction"],
            torch.ones(2, 1, dtype=torch.long),
            position=1,
            cache=None,
            **_conditioning(inputs),
        )
    assert len(model.correction_head.auxiliary_calls) == 1
    assert not torch.equal(corrected, inputs["current"])
    assert not corrected.requires_grad


@pytest.mark.parametrize("mode", ["hidden", "logits"])
def test_real_head_two_steps_use_only_caller_supplied_feedback_and_cache(mode):
    torch.manual_seed(300)
    model = _StepHarness(mode, feedback=True).float()
    model.correction_head = CausalCorrectionHead(
        input_hidden_size=4,
        token_embedding_size=4,
        block_size=3,
        correction_hidden_size=4,
        correction_rank=2,
        num_heads=2,
        output_mode=mode,
        draft_vocab_size=6,
        enable_hidden_feedback=True,
    )
    with torch.no_grad():
        model.correction_head.correction_up.weight.normal_(std=0.1)
        if mode == "logits":
            model.correction_head.auxiliary_hidden_up.weight.normal_(std=0.1)
    reference = copy.deepcopy(model)
    cache, reference_cache = None, None
    calls = []
    handle = model.correction_head.register_forward_pre_hook(
        lambda _module, args, kwargs: calls.append((args, kwargs)),
        with_kwargs=True,
    )
    try:
        for position in range(2):
            current, previous = torch.randn(2, 4), torch.randn(2, 1, 4)
            correction = current.unsqueeze(1)
            positions = torch.full((2, 1), position, dtype=torch.long)
            # Deliberately independent of the prior output: the caller owns feedback.
            feedback = torch.randn(2, 1, 4)
            kwargs = {
                "previous_corrected_hidden": feedback,
                "previous_corrected_hidden_mask": torch.ones(2, 1, dtype=torch.bool),
            }
            if mode == "logits":
                kwargs["previous_logits"] = torch.randn(2, 1, 6)
                kwargs["previous_logits_mask"] = torch.ones(2, 1, dtype=torch.bool)
            logits, states, corrected, next_cache = model._rollout_correction_step(
                previous,
                current,
                correction,
                positions,
                position=position,
                cache=cache,
                **kwargs,
            )
            residual, expected_states, reference_cache = reference.correction_head(
                previous,
                correction,
                positions,
                cache=reference_cache,
                use_cache=True,
                **kwargs,
            )
            delta = (
                residual
                if mode == "hidden"
                else reference.correction_head.auxiliary_hidden_residual(
                    expected_states
                )
            )
            expected_hidden = current + delta[:, 0]
            expected_logits = functional.linear(
                expected_hidden if mode == "hidden" else current,
                reference.lm_head.weight,
            )
            if mode == "logits":
                expected_logits = expected_logits + residual[:, 0]
            torch.testing.assert_close(logits, expected_logits)
            torch.testing.assert_close(states, expected_states[:, 0])
            torch.testing.assert_close(corrected, expected_hidden)
            assert calls[-1][1]["cache"] is cache
            assert calls[-1][1]["previous_corrected_hidden"] is feedback
            assert calls[-1][1]["use_cache"] is True
            assert next_cache[0][0].shape[-2] == position + 1
            if cache is not None:
                assert cache[0][0].shape[-2] == position
                torch.testing.assert_close(
                    next_cache[0][0][..., :position, :], cache[0][0]
                )
            for actual_layer, expected_layer in zip(
                next_cache, reference_cache, strict=True
            ):
                for actual_tensor, expected_tensor in zip(
                    actual_layer, expected_layer, strict=True
                ):
                    torch.testing.assert_close(actual_tensor, expected_tensor)
            cache = next_cache
    finally:
        handle.remove()
    assert len(calls) == 2
