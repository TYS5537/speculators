"""Rollout feedback precedence, detach boundaries, lifetimes and frozen lookups."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from speculators.models.mmuse.core import MMuseDraftModel
from speculators.models.mmuse.rollout_state import RolloutFeedbackState
from speculators.models.mmuse.runtime_types import InitialLogitFeedback
from tests.unit.models.test_mmuse_core import (
    _RecordingCorrectionHead,
    _RolloutHarness,
)
from tests.unit.models.test_mmuse_rollout_inputs import _RecordingEncoder


def _populated_state():
    return RolloutFeedbackState(
        dense_logits=torch.randn(2, 1, 6, dtype=torch.float64, requires_grad=True),
        dense_mask=torch.ones(2, 1, dtype=torch.bool),
        online_rank_features=torch.randn(
            2, 1, 2, dtype=torch.float64, requires_grad=True
        ),
        online_mask=torch.ones(2, 1, dtype=torch.bool),
        corrected_hidden=torch.randn(2, 1, 4, dtype=torch.float64, requires_grad=True),
        hidden_mask=torch.ones(2, 1, dtype=torch.bool),
    )


@pytest.mark.parametrize("source", ["dense", "online", "none"])
@pytest.mark.parametrize("hidden_enabled", [False, True])
def test_initial_state_preserves_references_and_only_allocates_hidden_seed(
    source, hidden_enabled
):
    populated = _populated_state()
    initial = InitialLogitFeedback(
        populated.dense_logits if source == "dense" else None,
        populated.dense_mask if source == "dense" else None,
        populated.online_rank_features if source == "online" else None,
        populated.online_mask if source == "online" else None,
    )
    hidden = torch.randn(2, 3, 4, dtype=torch.bfloat16, requires_grad=True)
    state = RolloutFeedbackState.from_initial(
        initial, hidden, hidden_feedback_enabled=hidden_enabled
    )
    for name, value in zip(initial._fields, initial, strict=True):
        assert getattr(state, name) is value
    assert not isinstance(state, nn.Module)
    assert not hasattr(state, "__dict__")
    if hidden_enabled:
        assert state.corrected_hidden.shape == (2, 1, 4)
        assert state.corrected_hidden.dtype == hidden.dtype
        assert state.corrected_hidden.device == hidden.device
        assert not state.corrected_hidden.requires_grad
        assert torch.count_nonzero(state.corrected_hidden) == 0
        assert state.hidden_mask.shape == (2, 1)
        assert state.hidden_mask.dtype == torch.bool
        assert not state.hidden_mask.any()
    else:
        assert state.corrected_hidden is None
        assert state.hidden_mask is None


@pytest.mark.parametrize("hidden_enabled", [False, True])
@pytest.mark.parametrize("logit_enabled", [False, True])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("has_rank", [False, True])
def test_step_kwargs_keep_precedence_keyword_order_and_feature_gradients(
    hidden_enabled, logit_enabled, static, online, has_rank
):
    state = _populated_state()
    if not has_rank:
        state.online_rank_features = None
    rank = torch.randn(3, 2, 2, dtype=torch.float64).transpose(0, 1).requires_grad_()
    mask = torch.rand(2, 3, dtype=torch.float64)
    result = state.correction_kwargs(
        1,
        hidden_feedback_enabled=hidden_enabled,
        logit_feedback_enabled=logit_enabled,
        has_conditioning_features=static,
        online_selector=online,
        conditioning_previous_rank_features=rank if has_rank else None,
        conditioning_previous_logits_mask=mask,
    )
    expected = {}
    if hidden_enabled:
        expected["previous_corrected_hidden"] = state.corrected_hidden
        expected["previous_corrected_hidden_mask"] = state.hidden_mask
    if logit_enabled:
        sources = {
            "static": {"previous_logits_mask": mask[:, 1:2]},
            "online": {"previous_logits_mask": state.online_mask},
            "dense": {
                "previous_logits": state.dense_logits,
                "previous_logits_mask": state.dense_mask,
            },
        }
        if has_rank:
            sources["static"]["previous_rank_features"] = rank[:, 1:2]
            sources["online"]["previous_rank_features"] = state.online_rank_features
        source = "static" if static else ("online" if online else "dense")
        expected.update(sources[source])
    assert list(result) == list(expected)
    for name, value in result.items():
        reference = expected[name]
        assert value.dtype == reference.dtype
        assert value.stride() == reference.stride()
        assert value.data_ptr() == reference.data_ptr()
        assert value.requires_grad == reference.requires_grad
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    differentiable = [value for value in result.values() if value.requires_grad]
    if differentiable:
        sum(value.sum() for value in differentiable).backward()
    if logit_enabled and static and has_rank:
        expected_gradient = torch.zeros_like(rank)
        expected_gradient[:, 1] = 1
        torch.testing.assert_close(rank.grad, expected_gradient, rtol=0, atol=0)
    else:
        assert rank.grad is None


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("logit_enabled", [False, True])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("hidden_enabled", [False, True])
@pytest.mark.parametrize("with_grad", [False, True])
def test_advance_preserves_update_rules_and_gradient_boundaries(
    active, logit_enabled, static, online, hidden_enabled, with_grad
):
    state = _populated_state()
    previous = {
        name: getattr(state, name) for name in RolloutFeedbackState.__dataclass_fields__
    }
    logits = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    hidden = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    encoder = _RecordingEncoder()
    with torch.set_grad_enabled(with_grad):
        state.advance(
            logits,
            hidden,
            position=1 if active else 0,
            start_position=1,
            logit_feedback_enabled=logit_enabled,
            has_conditioning_features=static,
            online_selector=online,
            hidden_feedback_enabled=hidden_enabled,
            correction_head=encoder,
        )
    update_dense = active and logit_enabled and not static and not online
    update_online = active and logit_enabled and online
    changed = set()
    if update_dense:
        changed.update(("dense_logits", "dense_mask"))
        assert not state.dense_logits.requires_grad
        assert state.dense_logits.data_ptr() == logits.data_ptr()
        torch.testing.assert_close(state.dense_logits[:, 0], logits, rtol=0, atol=0)
        assert state.dense_mask.dtype == torch.bool
        assert state.dense_mask.all()
    if update_online:
        changed.update(("online_rank_features", "online_mask"))
        assert len(encoder.calls) == 1
        call_mask, kwargs = encoder.calls[0]
        assert call_mask is state.online_mask
        assert call_mask.dtype == torch.bool
        assert call_mask.all()
        assert not kwargs["previous_logits"].requires_grad
        assert kwargs["previous_logits"].data_ptr() == logits.data_ptr()
        assert state.online_rank_features.requires_grad == with_grad
    else:
        assert encoder.calls == []
    if hidden_enabled:
        changed.update(("corrected_hidden", "hidden_mask"))
        assert state.corrected_hidden.data_ptr() == hidden.data_ptr()
        torch.testing.assert_close(state.corrected_hidden[:, 0], hidden, rtol=0, atol=0)
        assert state.hidden_mask.dtype == torch.bool
        assert state.hidden_mask.all()
    for name, value in previous.items():
        if name not in changed:
            assert getattr(state, name) is value
    losses = []
    if with_grad and hidden_enabled:
        losses.append(state.corrected_hidden.sum())
    if with_grad and update_online:
        losses.append(state.online_rank_features.sum())
    if losses:
        sum(losses).backward()
    assert logits.grad is None
    if with_grad and hidden_enabled:
        torch.testing.assert_close(hidden.grad, torch.ones_like(hidden), rtol=0, atol=0)
    else:
        assert hidden.grad is None
    assert (encoder.projection.weight.grad is not None) == (with_grad and update_online)


def test_replaced_initial_dense_logits_are_not_retained_in_state():
    initial = InitialLogitFeedback(torch.randn(2, 1, 6), torch.ones(2, 1), None, None)
    initial_ref = weakref.ref(initial.dense_logits)
    state = RolloutFeedbackState.from_initial(
        initial, torch.zeros(2, 3, 4), hidden_feedback_enabled=False
    )
    del initial
    state.advance(
        torch.randn(2, 6),
        torch.randn(2, 4),
        position=1,
        start_position=1,
        logit_feedback_enabled=True,
        has_conditioning_features=False,
        online_selector=False,
        hidden_feedback_enabled=False,
        correction_head=None,
    )
    gc.collect()
    assert initial_ref() is None


class _EmbeddingHarness(nn.Module):
    _rollout_token_embeddings = MMuseDraftModel._rollout_token_embeddings

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4, dtype=torch.float64)
        self.selector_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        self.events = []

    def dflash2_select_candidates(self, logits, hidden, previous_ids):
        self.events.append(("selector", torch.is_grad_enabled(), previous_ids))
        ids = torch.tensor([[[2, 1]], [[3, 0]]])
        scores = logits[..., :2] * self.selector_scale + hidden.mean(-1, keepdim=True)
        return ids, scores

    def _draft_ids_to_verifier(self, ids):
        self.events.append(("mapping", torch.is_grad_enabled(), ids))
        return ids + 1


@pytest.mark.parametrize("mode", ["none", "static", "online"])
@pytest.mark.parametrize("with_grad", [False, True])
@pytest.mark.parametrize("position", [0, 2])
def test_token_embeddings_keep_order_tie_breaking_mapping_and_no_grad(
    mode, with_grad, position
):
    model = _EmbeddingHarness()
    handle = model.embed_tokens.register_forward_hook(
        lambda _module, args, _output: model.events.append(
            ("embedding", torch.is_grad_enabled(), args[0])
        )
    )
    previous_ids = torch.tensor([6, 7])
    current_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    try:
        with torch.set_grad_enabled(with_grad):
            previous, current = model._rollout_token_embeddings(
                previous_ids,
                torch.zeros(2, 3, 4, dtype=torch.float64, requires_grad=True),
                position=position,
                online_selector=mode == "online",
                precomputed_base_logits=torch.zeros(
                    2, 3, 6, dtype=torch.float64, requires_grad=True
                ),
                conditioning_current_ids=current_ids if mode == "static" else None,
            )
    finally:
        handle.remove()
    assert not previous.requires_grad
    torch.testing.assert_close(previous[:, 0], model.embed_tokens(previous_ids))
    expected_events = ["embedding"]
    if mode == "online":
        expected_events.extend(("selector", "mapping", "embedding"))
    elif mode == "static":
        expected_events.append("embedding")
    assert [event[0] for event in model.events] == expected_events
    assert not any(event[1] for event in model.events)
    if mode == "none":
        assert current is None
    else:
        assert not current.requires_grad
        # Equal Selector scores choose the first candidate, then map vocabularies.
        selected = (
            torch.tensor([3, 4]) if mode == "online" else current_ids[:, position]
        )
        torch.testing.assert_close(current[:, 0], model.embed_tokens(selected))
    assert model.embed_tokens.weight.grad is None
    assert model.selector_scale.grad is None


def test_reserved_anchor_is_sampled_but_real_anchor_seeds_next_slot(monkeypatch):
    model = _RolloutHarness()
    model.block_size = 3
    model.draft_vocab_size = 8
    model.config = SimpleNamespace(sample_from_anchor=False, correction_hidden_size=4)
    model.correction_head = _RecordingCorrectionHead()
    model.embed_tokens = nn.Embedding(8, 4)
    model.lm_head = nn.Linear(4, 8, bias=False)
    model.d2t = torch.tensor([2, 3, 4, 5, 6, 7, 0, 1])
    draws = []

    def draw(probabilities, num_samples):
        draws.append((probabilities, num_samples, torch.is_grad_enabled()))
        return torch.zeros(probabilities.shape[0], 1, dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", draw)
    output = model._rollout_correction_steps(
        torch.zeros(2, 3, 4), torch.tensor([6, 7]), temperature=0.7
    )
    assert len(draws) == 3
    assert all(count == 1 and not grad for _, count, grad in draws)
    assert torch.count_nonzero(output.token_ids) == 0
    previous = model.correction_head.previous_embeddings
    torch.testing.assert_close(
        previous[0][:, 0], model.embed_tokens(torch.tensor([6, 7]))
    )
    torch.testing.assert_close(
        previous[1][:, 0], model.embed_tokens(torch.tensor([2, 2]))
    )
