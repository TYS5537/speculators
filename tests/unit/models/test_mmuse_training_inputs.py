"""Teacher-forced block layout, dispatch, projection and gradient contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional

from speculators.models.mmuse.core import MMuseDraftModel
from speculators.models.mmuse.runtime_types import (
    SelectorConditioning,
    TeacherForcedCorrectionOutput,
    TrainingBlocks,
)
from speculators.models.mmuse.training_inputs import prepare_training_blocks
from tests.unit.models.test_mmuse_anchor_correction import _tiny_model


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("block", [1, 2, 7])
@pytest.mark.parametrize("with_base", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_training_blocks_keep_token_alignment_views_dtype_and_gradients(
    sample, block, with_base, strided
):
    ids = torch.arange(4 * block + 6, dtype=torch.int32).view(2, -1)
    indices = torch.arange(2 * block).roll(1)
    indices[1] = indices[0]  # Repeated/unsorted anchor indices must not be normalized.
    hidden = torch.randn(1, 2 * block, 8 if strided else 4)
    hidden = (hidden[..., ::2] if strided else hidden).requires_grad_()
    base = (
        torch.randn(1, 2 * block, 6, dtype=torch.float64, requires_grad=True)
        if with_base
        else None
    )
    result = prepare_training_blocks(
        ids,
        indices,
        hidden,
        base,
        num_blocks=2,
        block_size=block,
        sample_from_anchor=sample,
    )
    assert isinstance(result, TrainingBlocks)
    tokens = ids[0, indices].view(2, block)
    expected_previous = (
        tokens if sample else torch.cat([tokens[:, :1], tokens[:, :-1]], dim=1)
    )
    torch.testing.assert_close(result.token_ids, tokens, rtol=0, atol=0)
    torch.testing.assert_close(
        result.previous_token_ids, expected_previous, rtol=0, atol=0
    )
    assert result.previous_token_ids.dtype == torch.int32
    assert (result.previous_token_ids is result.token_ids) == sample
    assert result.hidden.data_ptr() == hidden.data_ptr()
    assert result.hidden.stride() == hidden.view(2, block, 4).stride()
    assert result.hidden.dtype == hidden.dtype
    torch.testing.assert_close(result.positions, torch.arange(block).expand(2, -1))
    assert result.positions.device == hidden.device
    objective = result.hidden.square().sum()
    if with_base:
        assert result.base_logits.data_ptr() == base.data_ptr()
        assert result.base_logits.dtype == base.dtype
        objective = objective + result.base_logits.square().sum()
    else:
        assert result.base_logits is None
    objective.backward()
    torch.testing.assert_close(hidden.grad, hidden.detach() * 2, rtol=0, atol=0)
    if with_base:
        torch.testing.assert_close(base.grad, base.detach() * 2, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("failure", "error_type", "message"),
    [
        ("index", IndexError, "out of bounds"),
        ("tokens", RuntimeError, "input of size 5"),
        ("hidden", RuntimeError, "input of size 20"),
        ("base", RuntimeError, "input of size 17"),
    ],
)
def test_block_preparation_keeps_index_then_hidden_then_base_failure_order(
    failure, error_type, message
):
    indices = torch.arange(6)
    if failure == "index":
        indices[-1] = 99
    elif failure == "tokens":
        indices = indices[:-1]
    hidden = torch.zeros(1, 6, 4) if failure == "base" else torch.zeros(20)
    with pytest.raises(error_type, match=message):
        prepare_training_blocks(
            torch.arange(12).view(1, 12),
            indices,
            hidden,
            torch.zeros(17),
            num_blocks=2,
            block_size=3,
            sample_from_anchor=False,
        )


@pytest.mark.parametrize("sample", [False, True])
def test_training_block_preparation_compiles_without_graph_breaks(sample):
    def prepare(hidden, base):
        return prepare_training_blocks(
            torch.arange(10).view(1, 10),
            torch.tensor([1, 2, 3, 5, 6, 7]),
            hidden,
            base,
            num_blocks=2,
            block_size=3,
            sample_from_anchor=sample,
        )

    hidden = torch.randn(1, 6, 4, dtype=torch.float64, requires_grad=True)
    base = torch.randn(1, 6, 6, dtype=torch.float64, requires_grad=True)
    compiled = torch.compile(prepare, backend="eager", fullgraph=True)
    actual, expected = compiled(hidden, base), prepare(hidden, base)
    assert type(actual) is TrainingBlocks
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(
        actual.hidden.square().sum() + actual.base_logits.square().sum(), (hidden, base)
    )
    expected_grad = torch.autograd.grad(
        expected.hidden.square().sum() + expected.base_logits.square().sum(),
        (hidden, base),
    )
    for left, right in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


class _DispatchHarness(nn.Module):
    _run_teacher_forced_correction = MMuseDraftModel._run_teacher_forced_correction

    def __init__(self, *, feedback, mode):
        super().__init__()
        self.config = SimpleNamespace(correction_hidden_feedback=feedback)
        self.mode = mode
        self.embed_tokens = nn.Embedding(8, 4, dtype=torch.float64)
        self.lm_head = nn.Linear(4, 6, bias=False, dtype=torch.float64)
        self.states = torch.randn(2, 3, 5, requires_grad=True)
        self.corrected = torch.randn(2, 3, 4, requires_grad=True)
        self.logits = torch.randn(2, 3, 6, dtype=torch.float64, requires_grad=True)
        self.calls = []
        self.events = []
        self.embed_tokens.register_forward_hook(self._record_embedding)
        self.lm_head.register_forward_hook(self._record_projection)

    def _record_embedding(self, module, args, output):
        self.events.append(("embedding", torch.is_grad_enabled(), args[0], output))

    def _record_projection(self, module, args, output):
        self.events.append(("projection", torch.is_grad_enabled(), args[0], output))

    def _teacher_forced_hidden_feedback_correction(self, *args, **kwargs):
        self.events.append(("feedback", torch.is_grad_enabled()))
        self.calls.append((args, kwargs))
        return self.logits, self.states, self.corrected

    def _teacher_forced_parallel_correction(self, *args, **kwargs):
        self.events.append(("parallel", torch.is_grad_enabled()))
        self.calls.append((args, kwargs))
        logits = None if self.mode == "hidden" else self.logits.view(1, 6, 6)
        return logits, self.states, self.corrected


def _dispatch_inputs(compact, mode):
    blocks = prepare_training_blocks(
        torch.arange(8).view(1, 8),
        torch.arange(6),
        torch.randn(1, 6, 4, requires_grad=True),
        torch.randn(1, 6, 6, requires_grad=True),
        num_blocks=2,
        block_size=3,
        sample_from_anchor=False,
    )
    conditioning = SelectorConditioning(
        selector_loss=torch.tensor(0.5, requires_grad=True),
        previous_token_ids=torch.tensor([[7, 6, 5], [4, 3, 2]]),
        current_token_embeddings=torch.randn(2, 3, 4, requires_grad=True),
        previous_rank_features=torch.randn(2, 3, 2, requires_grad=True),
        previous_logits_mask=torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.0, 1.0]])
        if compact
        else None,
    )
    # Hidden/compact paths must not view or otherwise consume the dense target.
    targets = (
        torch.empty(1)
        if mode == "hidden" or compact
        else torch.randn(1, 6, 6, requires_grad=True)
    )
    return blocks, targets, conditioning


@pytest.mark.parametrize("mode", ["hidden", "logits"])
@pytest.mark.parametrize("feedback", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_dispatch_keeps_branch_inputs_frozen_lookup_and_projection_count(
    mode, feedback, compact
):
    model = _DispatchHarness(feedback=feedback, mode=mode)
    blocks, targets, conditioning = _dispatch_inputs(compact, mode)
    result = model._run_teacher_forced_correction(
        blocks, targets, conditioning, correction_output_mode=mode
    )
    assert isinstance(result, TeacherForcedCorrectionOutput)
    assert result.causal_states is model.states
    assert result.corrected_hidden is model.corrected
    assert result.logits.shape == (1, 6, 6)
    if mode == "hidden" and not feedback:
        expected = functional.linear(
            model.corrected.reshape(1, 6, 4).double(), model.lm_head.weight
        )
    else:
        expected = model.logits.view(1, 6, 6)
    torch.testing.assert_close(result.logits, expected, rtol=0, atol=0)
    names = [event[0] for event in model.events]
    assert names == (
        ["embedding", "feedback"]
        if feedback
        else ["parallel"] + (["projection"] if mode == "hidden" else [])
    )
    assert len(model.calls) == 1
    args, kwargs = model.calls[0]
    assert args[0] is blocks.hidden
    assert args[2] is blocks.positions
    if feedback:
        assert model.events[0][1] is False
        assert model.events[0][2] is conditioning.previous_token_ids
        assert args[1] is model.events[0][3]
        assert not args[1].requires_grad
        assert args[3] is blocks.base_logits
        assert (
            kwargs["current_token_embeddings"] is conditioning.current_token_embeddings
        )
        _assert_recurrent_logit_inputs(
            args, kwargs, targets, conditioning, mode=mode, compact=compact
        )
    else:
        assert args[1] is conditioning.previous_token_ids
        assert kwargs["targets"] is targets
        assert kwargs["base_logits_blocks"] is blocks.base_logits
        assert (
            kwargs["selector_current_embeddings"]
            is conditioning.current_token_embeddings
        )
        assert (
            kwargs["selector_previous_rank_features"]
            is conditioning.previous_rank_features
        )
        assert (
            kwargs["selector_previous_logits_mask"] is conditioning.previous_logits_mask
        )


def _assert_recurrent_logit_inputs(
    args, kwargs, targets, conditioning, *, mode, compact
):
    dense, mask = args[4:6]
    rank = kwargs["previous_rank_features"]
    if mode == "hidden":
        assert dense is mask is rank is None
    elif compact:
        assert dense is None
        assert mask is conditioning.previous_logits_mask
        assert rank is conditioning.previous_rank_features
        rank.sum().backward()
        torch.testing.assert_close(rank.grad, torch.ones_like(rank), rtol=0, atol=0)
    else:
        assert rank is None
        target_blocks = targets.view(2, 3, 6)
        expected = torch.cat(
            [torch.zeros_like(target_blocks[:, :1]), target_blocks[:, :-1]], dim=1
        )
        torch.testing.assert_close(dense, expected, rtol=0, atol=0)
        torch.testing.assert_close(mask, torch.arange(3).expand(2, -1) > 0)
        dense.sum().backward()
        expected_gradient = torch.ones_like(target_blocks)
        expected_gradient[:, -1] = 0
        torch.testing.assert_close(
            targets.grad.view_as(expected_gradient), expected_gradient, rtol=0, atol=0
        )


def test_recurrent_dense_target_failure_still_follows_frozen_embedding_lookup():
    model = _DispatchHarness(feedback=True, mode="logits")
    blocks, _, conditioning = _dispatch_inputs(False, "logits")
    with pytest.raises(RuntimeError, match="input of size 1"):
        model._run_teacher_forced_correction(
            blocks, torch.empty(1), conditioning, correction_output_mode="logits"
        )
    assert len(model.events) == 1
    assert model.events[0][:2] == ("embedding", False)
    assert model.calls == []


def test_parallel_logit_dispatch_preserves_absent_corrected_hidden():
    model = _DispatchHarness(feedback=False, mode="logits")
    model.corrected = None
    blocks, targets, conditioning = _dispatch_inputs(False, "logits")
    result = model._run_teacher_forced_correction(
        blocks, targets, conditioning, correction_output_mode="logits"
    )
    assert result.corrected_hidden is None
    assert [event[0] for event in model.events] == ["parallel"]


@pytest.mark.parametrize(
    ("mode", "compact"),
    [("hidden", False), ("logits", False), ("logits", True), ("dual", True)],
)
def test_real_parallel_dispatch_compiles_with_exact_outputs_and_gradients(
    mode, compact
):
    sample = not compact
    model = _tiny_model(sample, mode, with_selector=False).double()
    output_mode = "hidden" if mode == "hidden" else "logits"

    def dispatch(hidden, base, targets, rank):
        blocks = prepare_training_blocks(
            torch.arange(8).view(1, 8),
            torch.arange(6),
            hidden,
            base,
            num_blocks=2,
            block_size=3,
            sample_from_anchor=sample,
        )
        conditioning = SelectorConditioning(
            None,
            blocks.previous_token_ids,
            None,
            rank if compact else None,
            blocks.positions > 0 if compact else None,
        )
        return model._run_teacher_forced_correction(
            blocks, targets, conditioning, correction_output_mode=output_mode
        )

    shapes = ((1, 6, 16), (1, 6, 32), (1, 6, 32), (2, 3, 4))
    inputs = tuple(
        torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in shapes
    )
    compiled = torch.compile(dispatch, backend="eager", fullgraph=True)
    actual, expected = compiled(*inputs), dispatch(*inputs)
    assert type(actual) is TeacherForcedCorrectionOutput
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    leaves = inputs + tuple(
        value for value in model.parameters() if value.requires_grad
    )
    actual_grad = torch.autograd.grad(
        sum(value.square().sum() for value in actual if value is not None),
        leaves,
        allow_unused=True,
    )
    expected_grad = torch.autograd.grad(
        sum(value.square().sum() for value in expected if value is not None),
        leaves,
        allow_unused=True,
    )
    for left, right in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
