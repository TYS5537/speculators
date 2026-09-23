"""Static Selector alignment, sparse proposal semantics and encoding boundaries."""

import pytest
import torch
from torch.nn import functional

from speculators.models.mmuse.runtime_types import (
    SelectorCorrectionInputs,
    SelectorPreviousCandidates,
)
from speculators.models.mmuse.selector_inputs import (
    align_selector_token_ids,
    shift_selector_candidates,
)
from tests.unit.models.test_mmuse_selector_conditioning import _ConditioningHarness


@pytest.mark.parametrize("block", [1, 2, 5])
@pytest.mark.parametrize("start", [0, 1])
@pytest.mark.parametrize("mapped", [False, True])
def test_token_alignment_preserves_mapping_calls_and_anchor_slots(block, start, mapped):
    selected = (torch.arange(block * 2, dtype=torch.int32) % 6).view(block, 2).t()
    anchors = torch.tensor([13, 14], dtype=torch.int32)
    mapping = torch.tensor([3, 11, 8, 2, 7, 1])
    calls, results = [], []

    def to_verifier(ids):
        calls.append(ids)
        result = mapping[ids.long()] if mapped else ids.long()
        results.append(result)
        return result

    selected_before, anchors_before = selected.clone(), anchors.clone()
    output = align_selector_token_ids(
        selected,
        anchors,
        block_size=block,
        start_position=start,
        draft_to_verifier=to_verifier,
    )
    assert isinstance(output, SelectorCorrectionInputs)
    assert output.current_token_ids is results[0]
    assert calls[0] is selected
    assert len(calls) == 1 + int(start + 1 < block)
    if len(calls) > 1:
        torch.testing.assert_close(calls[1], selected[:, start:-1])
    for position in range(block):
        expected = anchors.long() if position <= start else results[0][:, position - 1]
        torch.testing.assert_close(output.previous_token_ids[:, position], expected)
    assert output.previous_token_ids.dtype == torch.long
    assert output.previous_rank_features is output.previous_logits_mask is None
    torch.testing.assert_close(selected, selected_before)
    torch.testing.assert_close(anchors, anchors_before)


@pytest.mark.parametrize("block", [1, 2, 5])
@pytest.mark.parametrize("sample", [False, True])
def test_sparse_rows_keep_draft_ids_dtype_and_reserved_positions(block, sample):
    ids = torch.arange(block * 4, dtype=torch.int32).view(block, 2, 2).transpose(0, 1)
    logits = torch.randn(block, 2, 2, dtype=torch.float64).transpose(0, 1)
    before_ids, before_logits = ids.clone(), logits.clone()
    output = shift_selector_candidates(
        ids,
        logits,
        sample_from_anchor=sample,
        initial_previous_logits=None,
    )
    assert isinstance(output, SelectorPreviousCandidates)
    assert output.candidate_ids.dtype == ids.dtype
    assert output.candidate_logits.dtype == logits.dtype
    assert output.mask.dtype == torch.bool
    assert output.mask.device == ids.device
    start = 0 if sample else 1
    for position in range(block):
        valid = position > start
        assert bool(output.mask[:, position].all()) == valid
        expected_ids = ids[:, position - 1] if valid else torch.zeros_like(ids[:, 0])
        expected_logits = (
            logits[:, position - 1] if valid else torch.zeros_like(logits[:, 0])
        )
        torch.testing.assert_close(output.candidate_ids[:, position], expected_ids)
        torch.testing.assert_close(
            output.candidate_logits[:, position], expected_logits
        )
    torch.testing.assert_close(ids, before_ids)
    torch.testing.assert_close(logits, before_logits)


def _arguments(model, sample, **overrides):
    return {
        "candidate_ids": model.candidate_ids,
        "selector_logits": model.realized_logits,
        "selected_draft_ids": model.selected_ids,
        "anchor_token_ids": torch.tensor([9, 10]),
        "initial_previous_logits": (
            None
            if sample
            else torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
        ),
        **overrides,
    }


@pytest.mark.parametrize(
    ("sample", "block"), [(True, 1), (True, 2), (True, 5), (False, 2), (False, 5)]
)
@pytest.mark.parametrize("search", ["greedy", "global"])
@pytest.mark.parametrize("mode", ["hidden", "logits"])
def test_compact_features_match_proposal_rows_and_keep_encoder_gradients(
    sample, block, search, mode
):
    model = _ConditioningHarness(mode=mode, sample=sample, search=search, block=block)
    arguments = _arguments(model, sample)
    output = model._selector_correction_inputs(**arguments)
    if mode == "hidden":
        assert output.previous_rank_features is output.previous_logits_mask is None
        assert model.correction_head.calls == []
        return
    assert len(model.correction_head.calls) == 1 + int(not sample)
    assert not model.correction_head.calls[0][1]["candidate_logits"].requires_grad
    expected = torch.zeros(2, block, 2, dtype=torch.float64)
    with torch.no_grad():
        for position in range(1, block):
            if not sample and position == 1:
                expected[:, position] = functional.linear(
                    arguments["initial_previous_logits"].softmax(dim=-1),
                    model.correction_head.projection.weight,
                )
            else:
                probabilities = model.realized_logits[:, position - 1].softmax(dim=-1)
                if search == "global":
                    probabilities = (
                        model.candidate_ids[:, position - 1]
                        == model.selected_ids[:, position - 1, None]
                    ).to(probabilities.dtype)
                codes = model.correction_head.projection.weight.t()[
                    model.candidate_ids[:, position - 1]
                ]
                expected[:, position] = (codes * probabilities.unsqueeze(-1)).sum(
                    dim=-2
                )
    torch.testing.assert_close(output.previous_rank_features, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output.previous_logits_mask, (torch.arange(block) > 0).expand(2, -1)
    )
    output.previous_rank_features.sum().backward()
    assert model.realized_logits.grad is None
    assert model.correction_head.projection.weight.grad is not None
    assert torch.isfinite(model.correction_head.projection.weight.grad).all()
    if not sample:
        assert arguments["initial_previous_logits"].grad is not None


@pytest.mark.parametrize(
    ("sample", "search", "has_head", "overrides", "error_type", "message", "calls"),
    [
        (
            True,
            "greedy",
            False,
            {"selector_logits": torch.zeros(1)},
            RuntimeError,
            "Selector conditioning requires Correction",
            0,
        ),
        (
            True,
            "greedy",
            True,
            {"selector_logits": torch.zeros(1), "selected_draft_ids": torch.zeros(1)},
            ValueError,
            "Selector candidate IDs and logits must align",
            0,
        ),
        (
            True,
            "greedy",
            True,
            {"selected_draft_ids": torch.zeros(1), "anchor_token_ids": torch.zeros(1)},
            ValueError,
            "Selector path must align with candidate block positions",
            0,
        ),
        (
            True,
            "unsupported",
            True,
            {"anchor_token_ids": torch.zeros(1)},
            ValueError,
            "Selector conditioning requires one anchor per block",
            0,
        ),
        (
            True,
            "unsupported",
            True,
            {"initial_previous_logits": torch.zeros(1)},
            ValueError,
            "Unsupported DFlash2 selector search mode: 'unsupported'",
            0,
        ),
        (
            True,
            "global",
            True,
            {
                "selected_draft_ids": torch.full((2, 3), 100),
                "initial_previous_logits": torch.zeros(1),
            },
            RuntimeError,
            "DFlash2 global path selected an ID outside its Top-K",
            0,
        ),
        (
            True,
            "greedy",
            True,
            {"initial_previous_logits": torch.zeros(1)},
            ValueError,
            "Selector initial logits are only valid when sample_from_anchor=False",
            0,
        ),
        (
            False,
            "greedy",
            True,
            {"initial_previous_logits": None},
            ValueError,
            "Logit-aware Selector Correction with sample_from_anchor=False "
            "requires initial verifier logits",
            1,
        ),
        (
            False,
            "greedy",
            True,
            {"initial_previous_logits": torch.zeros(2, 5)},
            ValueError,
            "Expected selector initial logits shape (2, 6), got (2, 5)",
            1,
        ),
    ],
)
def test_input_failures_keep_priority_and_sparse_encoding_order(
    sample, search, has_head, overrides, error_type, message, calls
):
    model = _ConditioningHarness(sample=sample, search=search)
    encoder = model.correction_head
    if not has_head:
        model.correction_head = None
    with pytest.raises(error_type) as error:
        model._selector_correction_inputs(**_arguments(model, sample, **overrides))
    assert str(error.value) == message
    assert len(encoder.calls) == calls


@pytest.mark.parametrize("sample", [False, True])
def test_alignment_helpers_compile_with_mapping_callback(sample):
    def to_verifier(ids):
        return ids.long() + 7

    def function(ids, anchors, candidates, logits):
        tokens = align_selector_token_ids(
            ids,
            anchors,
            block_size=ids.shape[1],
            start_position=0 if sample else 1,
            draft_to_verifier=to_verifier,
        )
        previous = shift_selector_candidates(
            candidates,
            logits.detach(),
            sample_from_anchor=sample,
            initial_previous_logits=None,
        )
        return (
            tokens.current_token_ids,
            tokens.previous_token_ids,
            previous.candidate_ids,
            previous.candidate_logits,
            previous.mask,
        )

    inputs = (
        torch.tensor([[0, 2, 4]]),
        torch.tensor([9]),
        torch.arange(6).view(1, 3, 2),
        torch.randn(1, 3, 2, requires_grad=True),
    )
    compiled = torch.compile(function, backend="eager", fullgraph=True)
    for actual, expected in zip(compiled(*inputs), function(*inputs), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.requires_grad == expected.requires_grad
