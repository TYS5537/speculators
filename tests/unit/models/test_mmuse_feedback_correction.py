"""Teacher-forced recurrence, conditioning, cache and final projection contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional

from speculators.models.mmuse.core import MMuseDraftModel
from speculators.models.mmuse.feedback_correction import build_feedback_conditioning


@pytest.mark.parametrize("mode", ["hidden", "logits"])
@pytest.mark.parametrize("source", ["dense", "rank", "both"])
@pytest.mark.parametrize("position", [0, 2])
@pytest.mark.parametrize("with_current", [False, True])
def test_conditioning_preserves_slot_views_priority_and_gradients(
    mode, source, position, with_current
):
    inputs = {
        name: torch.randn(4, 2, width, dtype=torch.float64)
        .transpose(0, 1)
        .requires_grad_()
        for name, width in (("current", 4), ("dense", 6), ("rank", 2))
    }
    mask = torch.rand(2, 4, dtype=torch.float64)
    result = build_feedback_conditioning(
        position,
        output_mode=mode,
        current_token_embeddings=inputs["current"] if with_current else None,
        previous_target_logits=inputs["dense"] if source != "rank" else None,
        previous_target_logits_mask=mask,
        previous_rank_features=inputs["rank"] if source != "dense" else None,
    )
    selected = {}
    expected_keys = []
    if with_current:
        selected["current_token_embeddings"] = "current"
        expected_keys.append("current_token_embeddings")
    if mode == "logits":
        expected_keys.append("previous_logits_mask")
        name = "previous_logits" if source == "dense" else "previous_rank_features"
        selected[name] = "dense" if source == "dense" else "rank"
        expected_keys.append(name)
        torch.testing.assert_close(
            result["previous_logits_mask"], mask[:, position : position + 1]
        )
        assert result["previous_logits_mask"].dtype == mask.dtype
    assert list(result) == expected_keys
    for key, name in selected.items():
        expected = inputs[name][:, position : position + 1]
        assert result[key].stride() == expected.stride()
        torch.testing.assert_close(result[key], expected, rtol=0, atol=0)
    if selected:
        sum(result[key].sum() for key in selected).backward()
    for name, value in inputs.items():
        if name in selected.values():
            expected_grad = torch.zeros_like(value)
            expected_grad[:, position] = 1
            torch.testing.assert_close(value.grad, expected_grad, rtol=0, atol=0)
        else:
            assert value.grad is None


@pytest.mark.parametrize("has_mask", [False, True])
def test_conditioning_validates_mask_before_dense_source(has_mask):
    message = (
        "Logit-aware Correction requires previous logits"
        if has_mask
        else "Logit-aware Correction requires previous feature masks"
    )
    with pytest.raises(RuntimeError) as error:
        build_feedback_conditioning(
            0,
            output_mode="logits",
            current_token_embeddings=None,
            previous_target_logits=None,
            previous_target_logits_mask=torch.ones(2, 3) if has_mask else None,
            previous_rank_features=None,
        )
    assert str(error.value) == message


class _FeedbackHead(nn.Module):
    """A differentiable cache and hidden recurrence, with observable head calls."""

    def __init__(self, output_mode):
        super().__init__()
        self.output_mode = output_mode
        self.state_scale = nn.Parameter(torch.tensor(0.7))
        self.residual_scale = nn.Parameter(torch.tensor(0.3))
        self.auxiliary_scale = nn.Parameter(torch.tensor(0.2))
        self.register_buffer("projection", torch.arange(24).reshape(6, 4) / 29)
        self.calls, self.caches = [], []
        self.auxiliary_calls = 0

    def forward(self, previous, hidden, positions, **kwargs):
        self.calls.append((previous, hidden, positions, kwargs))
        dtype = self.state_scale.dtype
        states = hidden.to(dtype) * self.state_scale + previous.to(dtype) * 0.1
        states = states + positions.to(dtype).unsqueeze(-1) * 0.01
        feedback = kwargs["previous_corrected_hidden"].to(dtype)
        feedback = feedback * kwargs["previous_corrected_hidden_mask"].unsqueeze(-1)
        states = states + feedback * 0.2
        if kwargs.get("current_token_embeddings") is not None:
            states = states + kwargs["current_token_embeddings"].to(dtype) * 0.11
        if self.output_mode == "logits":
            source = kwargs.get("previous_logits")
            if source is None:
                source = kwargs["previous_rank_features"]
            feature = source.to(dtype).mean(-1) * kwargs["previous_logits_mask"]
            states = states + feature.unsqueeze(-1) * 0.13
        cache = kwargs["cache"]
        if cache is not None:
            states = states + cache[0][1].sum(dim=1, keepdim=True) * 0.07
        memory = states if cache is None else torch.cat([cache[0][1], states], dim=1)
        next_cache = [(memory, memory)]
        self.caches.append(next_cache)
        residual = (
            states
            if self.output_mode == "hidden"
            else functional.linear(states, self.projection)
        )
        return residual * self.residual_scale, states, next_cache

    def auxiliary_hidden_residual(self, states):
        self.auxiliary_calls += 1
        return states * self.auxiliary_scale


class _FeedbackHarness(nn.Module):
    _teacher_forced_hidden_feedback_correction = (
        MMuseDraftModel._teacher_forced_hidden_feedback_correction
    )
    _project_hidden_feedback_logits = MMuseDraftModel._project_hidden_feedback_logits

    def __init__(self, sample_from_anchor, mode):
        super().__init__()
        self.config = SimpleNamespace(
            sample_from_anchor=sample_from_anchor,
            correction_hidden_feedback=True,
            correction_hidden_size=4,
            correction_project_corrected_hidden=mode == "dual",
        )
        self.draft_vocab_size = 6
        self.correction_head = _FeedbackHead("hidden" if mode == "hidden" else "logits")
        self.lm_head = nn.Linear(4, 6, bias=False)


def _feedback_inputs(block=4, dtype=torch.float64):
    inputs = {
        name: torch.randn(2, block, width, dtype=dtype, requires_grad=True)
        for name, width in (
            ("dflash_hidden", 4),
            ("previous_token_embeddings", 4),
            ("base_logits", 6),
            ("previous_target_logits", 6),
            ("previous_rank_features", 2),
            ("current_token_embeddings", 4),
        )
    }
    inputs["block_positions"] = torch.arange(block).expand(2, -1)
    inputs["previous_target_logits_mask"] = inputs["block_positions"] > 0
    return inputs


@pytest.mark.parametrize("sample_from_anchor", [False, True])
@pytest.mark.parametrize("mode", ["hidden", "logits", "dual"])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("block", [1, 4])
def test_recurrence_preserves_anchor_cache_identity_and_cross_slot_gradients(
    sample_from_anchor, mode, compact, block
):
    model = _FeedbackHarness(sample_from_anchor, mode).double()
    inputs = _feedback_inputs(block)
    inputs["previous_target_logits" if compact else "previous_rank_features"] = None
    projections = []
    handle = model.lm_head.register_forward_hook(
        lambda _module, args, _output: projections.append(args[0])
    )
    try:
        result = model._teacher_forced_hidden_feedback_correction(**inputs)
    finally:
        handle.remove()
    assert type(result) is tuple
    logits, states, corrected = result
    start = 0 if sample_from_anchor else 1
    assert len(model.correction_head.calls) == block - start
    assert model.correction_head.auxiliary_calls == (
        block - start if mode != "hidden" else 0
    )
    assert len(projections) == (0 if mode == "logits" else 1)
    if projections:
        torch.testing.assert_close(projections[0], corrected.reshape(1, 2 * block, 4))
    if start:
        torch.testing.assert_close(corrected[:, 0], inputs["dflash_hidden"][:, 0])
        assert torch.count_nonzero(states[:, 0]) == 0
    _assert_feedback_calls(model, inputs, corrected, start=start)
    scale = (
        model.correction_head.residual_scale
        if mode == "hidden"
        else model.correction_head.auxiliary_scale
    )
    torch.testing.assert_close(
        corrected, inputs["dflash_hidden"] + states * scale, rtol=0, atol=0
    )
    expected_logits = (
        inputs["base_logits"]
        if mode == "logits"
        else functional.linear(corrected, model.lm_head.weight)
    )
    if mode != "hidden":
        # Preserve one GEMM per position: a single batched projection can have
        # different floating-point rounding from the recurrent head calls.
        delta_logits = torch.cat(
            [
                functional.linear(
                    states[:, position : position + 1].contiguous(),
                    model.correction_head.projection,
                )
                * model.correction_head.residual_scale
                for position in range(block)
            ],
            dim=1,
        )
        expected_logits = expected_logits + delta_logits
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    for cache in model.correction_head.caches[:-1]:
        cache[0][1].retain_grad()
    corrected[:, -1].sum().backward()
    # Only the final hidden slot is supervised here: earlier slots must receive
    # gradients through recurrence, not through a separate direct loss term.
    assert torch.count_nonzero(inputs["dflash_hidden"].grad[:, 0]) > 0
    for cache in model.correction_head.caches[:-1]:
        assert cache[0][1].grad is not None
        assert torch.count_nonzero(cache[0][1].grad) > 0


def _assert_feedback_calls(model, inputs, corrected, *, start):
    for index, (previous, hidden, positions, kwargs) in enumerate(
        model.correction_head.calls
    ):
        position = index + start
        torch.testing.assert_close(
            previous, inputs["previous_token_embeddings"][:, position : position + 1]
        )
        torch.testing.assert_close(
            hidden, inputs["dflash_hidden"][:, position : position + 1]
        )
        torch.testing.assert_close(
            positions, inputs["block_positions"][:, position : position + 1]
        )
        assert kwargs["use_cache"] is True
        assert kwargs["cache"] is (
            None if index == 0 else model.correction_head.caches[index - 1]
        )
        assert model.correction_head.caches[index][0][1].shape[1] == index + 1
        feedback = kwargs["previous_corrected_hidden"]
        assert torch.all(kwargs["previous_corrected_hidden_mask"] == (position > 0))
        if position:
            torch.testing.assert_close(feedback[:, 0], corrected[:, position - 1])
            assert feedback.requires_grad
        else:
            assert torch.count_nonzero(feedback) == 0
            assert not feedback.requires_grad


@pytest.mark.parametrize(
    ("failure", "message", "calls"),
    [
        ("head", "Hidden feedback requires Correction", 0),
        ("disabled", "Correction hidden feedback is not enabled", 0),
        ("mask", "Logit-aware Correction requires previous feature masks", 0),
        ("source", "Logit-aware Correction requires previous logits", 0),
        ("base", "Logit-residual Correction requires base logits", 3),
    ],
)
def test_feedback_failures_preserve_head_call_and_projection_order(
    failure, message, calls
):
    model = _FeedbackHarness(False, "logits").double()
    inputs = _feedback_inputs()
    head = model.correction_head
    if failure == "head":
        model.correction_head = None
        model.config.correction_hidden_feedback = False
    elif failure == "disabled":
        model.config.correction_hidden_feedback = False
    elif failure == "base":
        inputs["base_logits"] = None
    else:
        inputs["previous_target_logits"] = None
        inputs["previous_rank_features"] = None
        if failure == "mask":
            inputs["previous_target_logits_mask"] = None
    projections = []
    handle = model.lm_head.register_forward_hook(
        lambda *_args: projections.append(True)
    )
    try:
        with pytest.raises(RuntimeError) as error:
            model._teacher_forced_hidden_feedback_correction(**inputs)
    finally:
        handle.remove()
    assert str(error.value) == message
    assert len(head.calls) == calls
    assert head.auxiliary_calls == calls
    assert projections == []


@pytest.mark.parametrize("mode", ["hidden", "logits", "dual"])
def test_reserved_anchor_only_does_not_require_previous_features(mode):
    model = _FeedbackHarness(False, mode).double()
    inputs = _feedback_inputs(block=1)
    for name in (
        "previous_target_logits",
        "previous_target_logits_mask",
        "previous_rank_features",
    ):
        inputs[name] = None
    logits, states, corrected = model._teacher_forced_hidden_feedback_correction(
        **inputs
    )
    assert model.correction_head.calls == []
    assert model.correction_head.caches == []
    assert torch.count_nonzero(states) == 0
    torch.testing.assert_close(corrected, inputs["dflash_hidden"], rtol=0, atol=0)
    expected = inputs["base_logits"] if mode == "logits" else model.lm_head(corrected)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["hidden", "logits", "dual"])
@pytest.mark.parametrize(
    ("hidden_dtype", "projection_dtype", "base_dtype", "residual_dtype"),
    [
        (torch.float32, torch.float64, torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.float32, torch.float64, torch.float32),
        (torch.float64, torch.float32, torch.float32, torch.float64),
    ],
)
def test_final_projection_preserves_casts_and_gradients(
    mode, hidden_dtype, projection_dtype, base_dtype, residual_dtype
):
    model = _FeedbackHarness(True, mode).to(dtype=projection_dtype)
    hidden = torch.randn(2, 3, 4, dtype=hidden_dtype, requires_grad=True)
    base = torch.randn(2, 3, 6, dtype=base_dtype, requires_grad=True)
    residual = [
        torch.randn(2, 6, dtype=residual_dtype, requires_grad=True) for _ in range(3)
    ]
    result = model._project_hidden_feedback_logits(
        hidden, residual, base, num_blocks=2, block_size=3, hidden_size=4
    )
    expected = (
        base
        if mode == "logits"
        else functional.linear(
            hidden.reshape(1, 6, 4).to(projection_dtype), model.lm_head.weight
        ).view(2, 3, 6)
    )
    if mode != "hidden":
        expected = expected + torch.stack(residual, dim=1).to(expected.dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    leaves = [hidden, base, *residual, model.lm_head.weight]
    actual_gradients = torch.autograd.grad(
        result.square().sum(), leaves, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        expected.square().sum(), leaves, allow_unused=True
    )
    for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("compact", [False, True])
def test_conditioning_helper_compiles_without_breaking_gradients(compact):
    def condition(current, source, mask):
        return build_feedback_conditioning(
            1,
            output_mode="logits",
            current_token_embeddings=current,
            previous_target_logits=None if compact else source,
            previous_target_logits_mask=mask,
            previous_rank_features=source if compact else None,
        )

    current = torch.randn(2, 3, 4, requires_grad=True)
    source = torch.randn(2, 3, 2 if compact else 6, requires_grad=True)
    mask = torch.ones(2, 3, dtype=torch.bool)
    compiled = torch.compile(condition, backend="eager", fullgraph=True)
    actual, expected = compiled(current, source, mask), condition(current, source, mask)
    assert list(actual) == list(expected)
    for key, value in actual.items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    loss = sum(value.sum() for value in actual.values())
    gradients = torch.autograd.grad(loss, (current, source))
    for gradient in gradients:
        assert torch.count_nonzero(gradient[:, 0]) == 0
        assert torch.count_nonzero(gradient[:, 2]) == 0
        assert torch.all(gradient[:, 1] == 1)
