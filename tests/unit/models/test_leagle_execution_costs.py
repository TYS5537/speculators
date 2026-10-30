"""Execution savings preserve supervision, gradients and packed visibility."""

import copy
from collections import Counter

import pytest
import torch

from speculators.models.leagle import sparse_attention
from speculators.models.leagle.candidate_loss import CandidateLossAccumulator
from speculators.models.leagle.sparse_attention import (
    TrajectoryAttention,
    TrajectoryMaskCache,
)
from speculators.models.metrics import resolve_training_loss

from .test_leagle import make_model
from .test_leagle_training_diagnostics import packed_batch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(371)
    torch.set_num_threads(1)


def run(model, inputs, **options):
    result = model(
        **inputs,
        ttt_steps=11,
        loss_config=resolve_training_loss("kl_div", loss_implementation="legacy"),
        **options,
    )
    result[1].backward()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return result, gradients, torch.get_rng_state()


@pytest.mark.parametrize("path", ["dense", "chunked", "masked"])
@pytest.mark.parametrize("collect", [False, True])
@pytest.mark.parametrize("teacher_forcing", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unused_predictions_preserve_loss_gradients_and_rng(
    path, collect, teacher_forcing, dtype
):
    base = make_model(layers=1, loops=2, attention="sdpa", candidate_head=True).to(
        dtype
    )
    base.set_activation_checkpointing(True)
    inputs = packed_batch()
    for name in ("hidden_states", "verifier_last_hidden_states"):
        inputs[name] = inputs[name].to(dtype)
    options = {
        "collect_metrics": collect,
        "teacher_forcing": teacher_forcing,
        "leagle_chunked_loss": path == "chunked",
        "leagle_masked_head": path == "masked",
        "leagle_loss_chunk_size": 3,
    }
    reference = run(copy.deepcopy(base), inputs, return_predictions=True, **options)
    actual = run(base, inputs, return_predictions=False, **options)
    assert actual[0][0] == []
    torch.testing.assert_close(actual[0][1:], reference[0][1:], rtol=0, atol=0)
    torch.testing.assert_close(actual[1:], reference[1:], rtol=0, atol=0)


@pytest.mark.parametrize("chunked", [False, True])
def test_no_argmax_or_tail_predictions_when_unused(chunked, monkeypatch):
    model = make_model(layers=1, loops=1, attention="sdpa")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Unsupervised diagnostics must not compute argmax")

    monkeypatch.setattr(torch.Tensor, "argmax", unexpected)
    run(
        model,
        packed_batch(),
        return_predictions=False,
        collect_metrics=False,
        leagle_chunked_loss=chunked,
        leagle_loss_chunk_size=3,
    )


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_candidate_alignment_uses_requested_rows_and_preserves_objective(
    masked, shared, monkeypatch
):
    base = make_model(layers=1, loops=1, attention="sdpa", candidate_head=True)
    base.config.candidate_head_chunk_size = 5
    inputs = packed_batch()
    common = {
        "leagle_masked_head": masked,
        "leagle_shared_loss_normalization": shared,
        "return_predictions": False,
    }
    reference = run(
        copy.deepcopy(base),
        inputs,
        leagle_chunked_loss=True,
        leagle_loss_chunk_size=5,
        **common,
    )
    widths = []
    original = CandidateLossAccumulator._chunk

    def record(self, hidden, *args, **kwargs):
        widths.append(hidden.shape[0])
        return original(self, hidden, *args, **kwargs)

    monkeypatch.setattr(CandidateLossAccumulator, "_chunk", record)
    actual = run(
        base,
        inputs,
        leagle_align_candidate_chunks=True,
        leagle_loss_chunk_size=2,
        **common,
    )
    assert max(widths) == 5
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("selective", "cache"), [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("window", [None, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_execution_matches_original_with_packing_and_shared_loops(
    selective, cache, reuse, window, dtype
):
    base = make_model(layers=2, loops=2, attention="sdpa", window=window).to(dtype)
    base.set_activation_checkpointing(True)
    inputs = packed_batch()
    for name in ("hidden_states", "verifier_last_hidden_states"):
        inputs[name] = inputs[name].to(dtype)
    common = {
        "leagle_reuse_training_kv": reuse,
        "leagle_sparse_attention": True,
        "leagle_attn_chunk_size": 3,
    }
    reference = run(copy.deepcopy(base), inputs, **common)
    actual = run(
        base,
        inputs,
        leagle_selective_checkpoint=selective,
        leagle_cache_attention_masks=cache,
        **common,
    )
    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
    tolerance = (
        {"rtol": 0.04, "atol": 0.004}
        if dtype == torch.bfloat16
        else {"rtol": 5e-4, "atol": 3e-5}
    )
    torch.testing.assert_close(actual[1], reference[1], **tolerance)
    torch.testing.assert_close(actual[2], reference[2], rtol=0, atol=0)


def test_selective_checkpoint_reduces_recomputation_and_mask_construction(monkeypatch):
    base = make_model(layers=2, loops=2, attention="sdpa")
    base.set_activation_checkpointing(True)
    inputs = packed_batch()
    original_attend = sparse_attention._attend_chunk
    original_mask = sparse_attention._chunk_mask
    counts = Counter()

    def attend(*args, **kwargs):
        counts["attention"] += 1
        return original_attend(*args, **kwargs)

    def mask(*args, **kwargs):
        counts["mask"] += 1
        return original_mask(*args, **kwargs)

    monkeypatch.setattr(sparse_attention, "_attend_chunk", attend)
    monkeypatch.setattr(sparse_attention, "_chunk_mask", mask)
    common = {"leagle_sparse_attention": True, "leagle_attn_chunk_size": 3}
    run(copy.deepcopy(base), inputs, **common)
    before = counts.copy()
    counts.clear()
    run(
        base,
        inputs,
        leagle_selective_checkpoint=True,
        leagle_cache_attention_masks=True,
        **common,
    )
    assert counts["attention"] < before["attention"]
    assert counts["mask"] == 11 * 3  # One mask per TTT step/query block.
    assert counts["mask"] < before["mask"]


def test_mask_cache_is_bounded_and_respects_windows_and_steps():
    inputs = packed_batch()
    cache = TrajectoryMaskCache(budget_bytes=100)
    for step in range(3):
        for window in (None, 3):
            layout = TrajectoryAttention(
                inputs["document_ids"][0],
                inputs["position_ids"][0],
                step,
                window,
                3,
                cache,
            )
            for start in range(0, 9, 3):
                expected = sparse_attention._chunk_mask(layout, start, start + 3)
                for _ in range(2):
                    actual = cache.get(layout, start, start + 3)
                    torch.testing.assert_close(actual, expected)
                    assert cache.used_bytes <= cache.budget_bytes
    assert 0 < cache.used_bytes <= 100
    assert cache.used_bytes == sum(
        x.numel() * x.element_size() for x in cache.masks.values()
    )


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"leagle_align_candidate_chunks": True}, "candidate head"),
        ({"leagle_selective_checkpoint": True}, "sparse attention"),
        ({"leagle_cache_attention_masks": True}, "sparse attention"),
        (
            {"leagle_sparse_attention": True, "leagle_selective_checkpoint": True},
            "activation checkpointing",
        ),
        (
            {
                "leagle_sparse_attention": True,
                "leagle_cache_attention_masks": True,
                "leagle_attention_mask_cache_mb": 0,
            },
            "budget",
        ),
    ],
)
def test_execution_option_guards(options, match):
    model = make_model(layers=1, loops=1, attention="sdpa")
    with pytest.raises(ValueError, match=match):
        model(**packed_batch(), **options)


def test_candidate_chunk_cap_is_reported_at_trainer_setup(caplog):
    model = make_model()
    kwargs = {
        "candidate_head": True,
        "candidate_head_chunk_size": 512,
        "leagle_chunked_loss": True,
        "loss_fn": "kl_div",
    }
    train, val = model.get_trainer_kwargs(**kwargs)
    assert train["return_predictions"] is val["return_predictions"] is False
    assert "capped" in caplog.text
    caplog.clear()
    model.get_trainer_kwargs(**kwargs, leagle_align_candidate_chunks=True)
    assert "capped" not in caplog.text
