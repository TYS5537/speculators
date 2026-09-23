"""Named MMuse boundaries preserve tensor identity and positional compatibility."""

import pytest
import torch

from speculators.models.mmuse import MMuseDraftModel, MMuseSpeculatorConfig
from speculators.models.mmuse.runtime_types import (
    CorrectionRolloutOutput,
    CorrectionStepOutput,
    InitialLogitFeedback,
    SelectorConditioning,
    SelectorCorrectionInputs,
    SelectorPreviousCandidates,
    TeacherForcedCorrectionOutput,
    TrainingBlocks,
)
from tests.unit.models.test_mmuse_architecture import _config
from tests.unit.models.test_mmuse_selector_conditioning import (
    _ConditioningHarness,
    _inputs,
)


@pytest.mark.parametrize(
    ("record_type", "fields"),
    [
        (
            SelectorConditioning,
            (
                "selector_loss",
                "previous_token_ids",
                "current_token_embeddings",
                "previous_rank_features",
                "previous_logits_mask",
            ),
        ),
        (
            SelectorCorrectionInputs,
            (
                "current_token_ids",
                "previous_token_ids",
                "previous_rank_features",
                "previous_logits_mask",
            ),
        ),
        (
            SelectorPreviousCandidates,
            ("candidate_ids", "candidate_logits", "mask"),
        ),
        (
            TrainingBlocks,
            ("token_ids", "previous_token_ids", "hidden", "positions", "base_logits"),
        ),
        (
            TeacherForcedCorrectionOutput,
            ("logits", "causal_states", "corrected_hidden"),
        ),
        (
            InitialLogitFeedback,
            (
                "dense_logits",
                "dense_mask",
                "online_rank_features",
                "online_mask",
            ),
        ),
        (
            CorrectionStepOutput,
            ("logits", "causal_states", "corrected_hidden", "cache"),
        ),
        (
            CorrectionRolloutOutput,
            (
                "token_ids",
                "logits",
                "causal_states",
                "corrected_hidden",
            ),
        ),
    ],
)
def test_named_containers_preserve_field_order_references_and_autograd(
    record_type, fields
):
    source = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    computed = source * 2
    values = tuple(computed for _ in fields)
    record = record_type(**dict(zip(fields, values, strict=True)))
    assert isinstance(record, tuple)
    assert record._fields == fields
    for name, unpacked in zip(fields, record, strict=True):
        assert unpacked is computed
        assert getattr(record, name) is computed
        assert unpacked.grad_fn is computed.grad_fn
    record[0].sum().backward()
    torch.testing.assert_close(source.grad, torch.full_like(source, 2), rtol=0, atol=0)
    with pytest.raises(AttributeError):
        setattr(record, fields[0], None)


def test_cache_and_disabled_fields_are_not_copied_or_materialized():
    ids = torch.tensor([[1, 2]])
    conditioning = SelectorConditioning(None, ids, None, None, None)
    assert conditioning == (None, ids, None, None, None)
    assert conditioning.previous_token_ids is ids
    assert InitialLogitFeedback(None, None, None, None) == (None, None, None, None)
    hidden = torch.randn(1, 2)
    cache = [(hidden, hidden)]
    step = CorrectionStepOutput(hidden, hidden, hidden, cache)
    assert step.cache is cache
    assert step.cache[0][0] is hidden


@pytest.mark.parametrize("mode", ["hidden", "logits"])
@pytest.mark.parametrize("feedback", ["static", "corrected"])
@pytest.mark.parametrize("sample", [False, True])
def test_selector_producers_return_named_contracts(mode, feedback, sample):
    model = _ConditioningHarness(mode=mode, feedback=feedback, sample=sample)
    result = model._prepare_selector_conditioning(
        **_inputs(), correction_output_mode=mode
    )
    assert isinstance(result, SelectorConditioning)
    assert result.selector_loss is model.selector_loss
    assert result.current_token_embeddings.shape == (2, 3, 4)
    assert not result.current_token_embeddings.requires_grad
    assert (result.previous_rank_features is not None) == (mode == "logits")
    if feedback == "static":
        selector_inputs = model.static_calls[0][2]
        assert isinstance(selector_inputs, SelectorCorrectionInputs)
        assert result.previous_token_ids is selector_inputs.previous_token_ids
        assert result.previous_rank_features is selector_inputs.previous_rank_features
        assert result.previous_logits_mask is selector_inputs.previous_logits_mask


@pytest.mark.parametrize("mode", ["hidden", "logits"])
@pytest.mark.parametrize("selector", [None, "static", "corrected"])
@pytest.mark.parametrize("sample", [False, True])
def test_public_rollout_keeps_plain_pair_and_registers_no_state(
    monkeypatch, mode, selector, sample
):
    torch.manual_seed(54)
    model = MMuseDraftModel(
        _config(
            MMuseSpeculatorConfig,
            sample_from_anchor=sample,
            enable_correction_head=True,
            correction_output_mode=mode,
            correction_hidden_size=16,
            correction_num_heads=4,
            correction_rank=8,
            dflash2_candidate_selector=selector is not None,
            selector_correction_feedback=selector or "static",
            dflash2_selector_rank=4,
            dflash2_selector_top_k=4,
        )
    ).eval()
    with torch.no_grad():
        for name in ("embed_tokens", "lm_head"):
            getattr(model, name).weight.normal_(std=0.1)
    state_keys = tuple(model.state_dict())
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    outputs = []
    original = model._rollout_correction_steps

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        outputs.append(result)
        return result

    monkeypatch.setattr(model, "_rollout_correction_steps", capture)
    result = model.rollout_correction(
        torch.randn(2, 3, 16),
        torch.tensor([1, 2]),
        initial_previous_logits=torch.randn(2, 32) if not sample else None,
    )
    assert type(result) is tuple
    assert len(result) == 2
    assert len(outputs) == 1
    rollout = outputs[0]
    assert isinstance(rollout, CorrectionRolloutOutput)
    assert result[0] is rollout.token_ids
    assert result[1] is rollout.logits
    assert rollout.token_ids.shape == (2, 3)
    assert rollout.logits.shape == (2, 3, 32)
    assert rollout.causal_states.shape == rollout.corrected_hidden.shape == (2, 3, 16)
    assert not rollout.logits.requires_grad
    assert tuple(model.state_dict()) == state_keys
    assert {
        name: id(value) for name, value in model.named_parameters()
    } == parameter_ids


def test_named_tensor_access_compiles_without_graph_breaks():
    def function(value):
        output = CorrectionStepOutput(value * 2, value + 1, value, None)
        return output.logits.square().sum() + output.causal_states.sum()

    compiled = torch.compile(function, backend="eager", fullgraph=True)
    source = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    reference = source.detach().clone().requires_grad_()
    actual, expected = compiled(source), function(reference)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(source.grad, reference.grad, rtol=0, atol=0)
