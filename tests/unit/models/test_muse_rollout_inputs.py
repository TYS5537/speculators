"""Contracts for rollout validation and its first dense/compact feedback state."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from speculators.models.muse.core import MuseDraftModel
from speculators.models.muse.correction import CausalCorrectionHead


class _RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(6, 2, bias=False, dtype=torch.float64)
        self.calls = []

    def encode_previous_distribution(self, mask, **kwargs):
        self.calls.append((mask, kwargs))
        dense = kwargs.get("previous_logits")
        if dense is not None:
            probabilities = dense.softmax(dim=-1).to(self.projection.weight.dtype)
            result = self.projection(probabilities)
        else:
            ids = kwargs["candidate_ids"]
            probabilities = kwargs["candidate_logits"].softmax(dim=-1)
            codes = self.projection.weight.transpose(0, 1)[ids]
            result = (codes * probabilities.unsqueeze(-1)).sum(dim=-2)
        return result * mask.unsqueeze(-1)


class _InputHarness:
    _validate_rollout_inputs = MuseDraftModel._validate_rollout_inputs
    _initial_rollout_logit_feedback = MuseDraftModel._initial_rollout_logit_feedback
    _rollout_correction_steps = MuseDraftModel._rollout_correction_steps
    _draft_ids_to_verifier = MuseDraftModel._draft_ids_to_verifier
    rollout_correction = MuseDraftModel.rollout_correction

    def __init__(self, *, sample_from_anchor=False, online=False):
        self.block_size = 3
        self.draft_vocab_size = 6
        self.config = SimpleNamespace(
            sample_from_anchor=sample_from_anchor,
            selector_correction_feedback="corrected" if online else "static",
            correction_hidden_size=4,
        )
        self.candidate_selector = SimpleNamespace(top_k=2) if online else None
        self.correction_head = _RecordingEncoder()
        self.lm_head = nn.Linear(4, 6, bias=False, dtype=torch.float64)
        self.embed_tokens = nn.Embedding(6, 4, dtype=torch.float64)
        self.d2t = None
        self.training = False


def _validation_inputs(**overrides):
    return {
        "dflash_hidden": torch.zeros(2, 3, 4),
        "anchor_token_ids": torch.tensor([0, 1]),
        "precomputed_base_logits": None,
        "conditioning_current_ids": None,
        "conditioning_previous_ids": None,
        "conditioning_previous_rank_features": None,
        "conditioning_previous_logits_mask": None,
        **overrides,
    }


@pytest.mark.parametrize(
    ("overrides", "online", "message"),
    [
        ({"dflash_hidden": torch.zeros(2, 4)}, False, "rank-3"),
        ({"dflash_hidden": torch.zeros(2, 2, 4)}, False, "block_size"),
        ({"anchor_token_ids": torch.zeros(2, 1)}, False, "anchor_token_ids shape"),
        ({"conditioning_current_ids": torch.zeros(2, 2)}, False, "current_ids shape"),
        ({"conditioning_current_ids": torch.zeros(2, 3)}, False, "provided together"),
        ({"conditioning_previous_ids": torch.zeros(2, 3)}, False, "provided together"),
        (
            {
                "conditioning_current_ids": torch.zeros(2, 3),
                "conditioning_previous_ids": torch.zeros(2, 2),
            },
            False,
            "previous_ids shape",
        ),
        (
            {"conditioning_previous_rank_features": torch.zeros(2, 3, 2)},
            False,
            "provided together",
        ),
        (
            {"conditioning_previous_logits_mask": torch.ones(2, 3, dtype=torch.bool)},
            False,
            "provided together",
        ),
        (
            {
                "conditioning_previous_rank_features": torch.zeros(2, 3, 2),
                "conditioning_previous_logits_mask": torch.ones(2, 2, dtype=torch.bool),
            },
            False,
            "mask must align",
        ),
        (
            {
                "conditioning_previous_rank_features": torch.zeros(2, 2, 2),
                "conditioning_previous_logits_mask": torch.ones(2, 3, dtype=torch.bool),
            },
            False,
            "rank features must align",
        ),
        ({"precomputed_base_logits": torch.zeros(2, 3, 5)}, False, "base_logits shape"),
        ({}, True, "requires DFlash base logits"),
        (
            {
                "conditioning_current_ids": torch.zeros(2, 3),
                "conditioning_previous_ids": torch.zeros(2, 3),
                "precomputed_base_logits": torch.zeros(2, 3, 6),
            },
            True,
            "cannot use a static path",
        ),
    ],
)
def test_validation_rejects_misaligned_or_conflicting_inputs(
    overrides, online, message
):
    model = _InputHarness(online=online)
    with pytest.raises(ValueError, match=message):
        model._validate_rollout_inputs(**_validation_inputs(**overrides))
    assert model.correction_head.calls == []


def test_validation_requires_a_correction_head():
    model = _InputHarness()
    model.correction_head = None
    with pytest.raises(RuntimeError, match="enable_correction_head"):
        model._validate_rollout_inputs(**_validation_inputs())


@pytest.mark.parametrize(
    ("online", "compact", "static_ids"),
    [
        (False, False, False),
        (False, False, True),
        (False, True, True),
        (True, False, False),
        (True, True, False),
    ],
)
def test_validation_classifies_without_encoding(online, compact, static_ids):
    model = _InputHarness(online=online)
    arguments = _validation_inputs()
    if online:
        arguments["precomputed_base_logits"] = torch.zeros(2, 3, 6)
    if static_ids:
        arguments["conditioning_current_ids"] = torch.zeros(2, 3, dtype=torch.long)
        arguments["conditioning_previous_ids"] = torch.zeros(2, 3, dtype=torch.long)
    if compact:
        arguments["conditioning_previous_rank_features"] = torch.zeros(2, 3, 2)
        arguments["conditioning_previous_logits_mask"] = torch.ones(
            2, 3, dtype=torch.bool
        )
    assert model._validate_rollout_inputs(**arguments) == (online, compact)
    assert model.correction_head.calls == []


def test_validation_defers_compact_width_and_allows_features_without_token_path():
    model = _InputHarness()
    arguments = _validation_inputs(
        conditioning_previous_rank_features=torch.zeros(2, 3, 7),
        conditioning_previous_logits_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    # Only block alignment belongs here; the head validates feature width later.
    assert model._validate_rollout_inputs(**arguments) == (False, True)
    assert model.correction_head.calls == []


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_dense_initial_feedback_is_detached_and_uses_projection_dtype(
    sample_from_anchor,
):
    model = _InputHarness(sample_from_anchor=sample_from_anchor)
    hidden = torch.zeros(2, 3, 4, dtype=torch.float32)
    initial = None if sample_from_anchor else torch.randn(2, 6, requires_grad=True)
    dense, mask, rank, rank_mask = model._initial_rollout_logit_feedback(
        hidden,
        initial,
        logit_feedback_enabled=True,
        has_conditioning_features=False,
        online_selector=False,
    )
    assert rank is None
    assert rank_mask is None
    assert dense.shape == (2, 1, 6)
    assert dense.dtype == model.lm_head.weight.dtype
    assert dense.device == hidden.device
    assert not dense.requires_grad
    assert mask.dtype == torch.bool
    assert mask.device == hidden.device
    assert mask.shape == (2, 1)
    assert bool(mask.all()) is not sample_from_anchor
    if initial is None:
        assert torch.count_nonzero(dense) == 0
    else:
        torch.testing.assert_close(dense[:, 0], initial.detach().double())
    assert model.correction_head.calls == []


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_online_initial_feedback_keeps_encoder_gradient_not_initial_logits(
    sample_from_anchor,
):
    model = _InputHarness(sample_from_anchor=sample_from_anchor, online=True)
    hidden = torch.zeros(2, 3, 4, dtype=torch.float32)
    initial = None if sample_from_anchor else torch.randn(2, 6, requires_grad=True)
    dense, mask, rank, rank_mask = model._initial_rollout_logit_feedback(
        hidden,
        initial,
        logit_feedback_enabled=True,
        has_conditioning_features=False,
        online_selector=True,
    )
    assert dense is None
    assert mask is None
    assert rank.shape == (2, 1, 2)
    assert rank.requires_grad
    assert rank.device == hidden.device
    assert rank_mask.dtype == torch.bool
    assert rank_mask.device == hidden.device
    assert bool(rank_mask.all()) is not sample_from_anchor
    assert len(model.correction_head.calls) == 1
    encoded_mask, kwargs = model.correction_head.calls[0]
    assert encoded_mask is rank_mask
    if sample_from_anchor:
        assert "previous_logits" not in kwargs
        assert kwargs["candidate_ids"].shape == (2, 1, 2)
        assert kwargs["candidate_ids"].dtype == torch.long
        assert kwargs["candidate_ids"].device == hidden.device
        assert kwargs["candidate_logits"].dtype == model.lm_head.weight.dtype
        assert torch.count_nonzero(kwargs["candidate_ids"]) == 0
        assert torch.count_nonzero(kwargs["candidate_logits"]) == 0
        assert torch.count_nonzero(rank) == 0
    else:
        source = kwargs["previous_logits"]
        assert source.dtype == initial.dtype  # Online encoding owns conversion.
        assert source.device == initial.device
        assert not source.requires_grad
        torch.testing.assert_close(source[:, 0], initial.detach())
        assert "candidate_ids" not in kwargs
    rank.sum().backward()
    gradient = model.correction_head.projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert bool(torch.count_nonzero(gradient)) is not sample_from_anchor
    if initial is not None:
        assert initial.grad is None


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_initial_logits_required_only_for_reserved_anchor(online, sample_from_anchor):
    model = _InputHarness(sample_from_anchor=sample_from_anchor, online=online)
    initial = torch.zeros(2, 6) if sample_from_anchor else None
    with pytest.raises(ValueError, match="sample_from_anchor=False"):
        model._initial_rollout_logit_feedback(
            torch.zeros(2, 3, 4),
            initial,
            logit_feedback_enabled=True,
            has_conditioning_features=False,
            online_selector=online,
        )


def test_dense_initial_logits_shape_is_checked_before_use():
    model = _InputHarness()
    with pytest.raises(ValueError, match="initial_previous_logits shape"):
        model._initial_rollout_logit_feedback(
            torch.zeros(2, 3, 4),
            torch.zeros(2, 1, 6),
            logit_feedback_enabled=True,
            has_conditioning_features=False,
            online_selector=False,
        )


@pytest.mark.parametrize("hidden_output", [False, True])
@pytest.mark.parametrize("initial", [None, torch.zeros(1)])
def test_hidden_or_static_compact_feedback_does_not_read_initial_logits(
    hidden_output, initial
):
    model = _InputHarness()
    result = model._initial_rollout_logit_feedback(
        torch.zeros(2, 3, 4),
        initial,
        logit_feedback_enabled=not hidden_output,
        has_conditioning_features=not hidden_output,
        online_selector=False,
    )
    assert result == (None, None, None, None)
    assert model.correction_head.calls == []


def test_online_initialization_is_not_skipped_when_compact_features_also_exist():
    model = _InputHarness(online=True)
    kwargs = {
        "logit_feedback_enabled": True,
        "has_conditioning_features": True,
        "online_selector": True,
    }
    with pytest.raises(ValueError, match="requires verifier logits"):
        model._initial_rollout_logit_feedback(torch.zeros(2, 3, 4), None, **kwargs)
    result = model._initial_rollout_logit_feedback(
        torch.zeros(2, 3, 4),
        torch.zeros(2, 6),
        **kwargs,
    )
    assert result[0] is None
    assert result[1] is None
    assert result[2] is not None
    assert result[3].all()
    assert len(model.correction_head.calls) == 1


@pytest.mark.parametrize(
    ("bad_shape", "fusion", "track_grad", "precomputed", "projection_count"),
    [
        (True, True, False, False, 0),
        (False, True, False, False, 1),
        (False, False, False, False, 0),
        (False, True, True, False, 0),
        (False, True, False, True, 0),
    ],
)
def test_rollout_preserves_validation_projection_initialization_order(
    bad_shape, fusion, track_grad, precomputed, projection_count
):
    model = _InputHarness()
    model.correction_head.output_mode = "logits"
    model.config.correction_lm_head_fusion = fusion
    hidden = torch.zeros(2, 4) if bad_shape else torch.zeros(2, 3, 4)
    inputs = []
    handle = model.lm_head.register_forward_pre_hook(
        lambda _module, args: inputs.append(args[0])
    )
    message = "rank-3" if bad_shape else "requires verifier logits"
    try:
        with (
            torch.set_grad_enabled(track_grad),
            pytest.raises(ValueError, match=message),
        ):
            model._rollout_correction_steps(
                hidden,
                torch.tensor([0, 1]),
                precomputed_base_logits=torch.zeros(2, 3, 6) if precomputed else None,
            )
    finally:
        handle.remove()
    assert len(inputs) == projection_count
    if inputs:
        assert inputs[0].shape == (2, 3, 4)
        assert inputs[0].dtype == model.lm_head.weight.dtype
    assert model.correction_head.calls == []


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_public_rollout_passes_initial_feedback_to_real_correction(sample_from_anchor):
    torch.manual_seed(622)
    model = _InputHarness(sample_from_anchor=sample_from_anchor)
    model.lm_head = model.lm_head.float()
    model.embed_tokens = model.embed_tokens.float()
    model.correction_head = CausalCorrectionHead(
        input_hidden_size=4,
        token_embedding_size=4,
        block_size=3,
        correction_hidden_size=4,
        correction_rank=2,
        num_heads=2,
        output_mode="logits",
        draft_vocab_size=6,
    ).eval()
    with torch.no_grad():
        model.correction_head.correction_up.weight.normal_(std=0.1)
    hidden, base = torch.randn(2, 3, 4), torch.randn(2, 3, 6)
    initial = None if sample_from_anchor else torch.randn(2, 6, requires_grad=True)
    calls = []
    handle = model.correction_head.register_forward_pre_hook(
        lambda _module, args, kwargs: calls.append((args, kwargs)),
        with_kwargs=True,
    )
    try:
        tokens, logits = model.rollout_correction(
            hidden,
            torch.tensor([0, 1]),
            initial_previous_logits=initial,
            base_logits=base,
        )
    finally:
        handle.remove()
    start = 0 if sample_from_anchor else 1
    assert len(calls) == 3 - start
    assert tokens.shape == (2, 3)
    assert logits.shape == (2, 3, 6)
    assert torch.isfinite(logits).all()
    assert not logits.requires_grad
    first_args, first_kwargs = calls[0]
    torch.testing.assert_close(first_args[2], torch.full((2, 1), start))
    first_feedback = first_kwargs["previous_logits"]
    assert not first_feedback.requires_grad
    assert bool(first_kwargs["previous_logits_mask"].all()) is not sample_from_anchor
    if initial is None:
        assert torch.count_nonzero(first_feedback) == 0
    else:
        torch.testing.assert_close(first_feedback[:, 0], initial.detach())
        torch.testing.assert_close(logits[:, 0], base[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(calls[1][1]["previous_logits"][:, 0], logits[:, start])
    assert calls[1][1]["previous_logits_mask"].all()
