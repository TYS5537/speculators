"""Correction input ordering, feature arithmetic, gradient and cache contracts."""

import pytest
import torch
from torch import nn
from torch.nn import functional

from speculators.models.mmuse.correction import CausalCorrectionHead


def _head(representation, *, feedback=True, dtype=torch.float64):
    head = CausalCorrectionHead(
        input_hidden_size=8,
        token_embedding_size=6,
        block_size=4,
        correction_hidden_size=12,
        correction_rank=4,
        num_layers=2,
        num_heads=3,
        output_mode="hidden" if representation == "hidden" else "logits",
        draft_vocab_size=11,
        enable_hidden_feedback=feedback,
    ).to(dtype)
    nn.init.normal_(head.correction_up.weight, std=0.1)
    return head


def _inputs(representation, *, feedback=True, current=True, dtype=torch.float64):
    def leaf(width):
        # Strided inputs also exercise the projection/cast path without a copy.
        return torch.randn(2, 3, width * 2, dtype=dtype)[..., ::2].requires_grad_()

    inputs = {
        "previous_token_embeddings": leaf(6),
        "dflash_hidden": leaf(8),
        "block_positions": torch.arange(3, dtype=torch.int32).expand(2, -1),
    }
    if current:
        inputs["current_token_embeddings"] = leaf(8)
    if representation != "hidden":
        name = (
            "previous_logits" if representation == "dense" else "previous_rank_features"
        )
        inputs[name] = leaf(11 if representation == "dense" else 4)
        inputs["previous_logits_mask"] = torch.tensor(
            [[0.0, 1.0, 0.5], [1.0, 0.0, 1.0]], dtype=dtype, requires_grad=True
        )
    if feedback:
        inputs["previous_corrected_hidden"] = leaf(8)
        inputs["previous_corrected_hidden_mask"] = torch.tensor(
            [[0.0, 0.5, 1.0], [1.0, 0.0, 1.0]], dtype=dtype, requires_grad=True
        )
    return inputs


@pytest.mark.parametrize("first_failure", range(8))
def test_multiple_invalid_inputs_keep_embedding_cache_feedback_logit_check_order(
    first_failure,
):
    head = _head("dense")
    inputs = _inputs("dense")
    faults = [
        (
            "previous_token_embeddings",
            torch.empty(1, 1, 6),
            "previous-token embeddings",
        ),
        ("current_token_embeddings", torch.empty(1), "current selector-token"),
        ("block_positions", torch.empty(1), "block positions"),
        ("cache", [], "Expected 2 cache entries, got 0"),
        ("previous_corrected_hidden", torch.empty(1), "previous corrected hidden and"),
        (
            "previous_corrected_hidden_mask",
            torch.empty(1),
            "previous corrected hidden mask",
        ),
        ("previous_logits_mask", torch.empty(1), "previous logits mask"),
        ("previous_logits", torch.empty(1), "Expected previous logits shape"),
    ]
    for name, value, _ in faults[first_failure:]:
        inputs[name] = value
    with pytest.raises(ValueError, match=faults[first_failure][2]):
        head(**inputs)


@pytest.mark.parametrize("missing", ["hidden", "mask", "both"])
def test_feedback_pairing_precedes_shapes_and_logit_checks(missing):
    head = _head("dense")
    inputs = _inputs("dense")
    inputs["previous_corrected_hidden"] = None if missing != "mask" else torch.empty(1)
    inputs["previous_corrected_hidden_mask"] = (
        None if missing != "hidden" else torch.empty(1)
    )
    inputs["previous_logits_mask"] = None
    with pytest.raises(ValueError, match="requires previous corrected hidden and mask"):
        head(**inputs)


@pytest.mark.parametrize("provided", ["hidden", "mask", "both"])
def test_disabled_feedback_rejects_even_empty_inputs_before_logit_checks(provided):
    head = _head("dense", feedback=False)
    inputs = _inputs("dense", feedback=False)
    inputs["previous_corrected_hidden"] = None if provided == "mask" else torch.empty(0)
    inputs["previous_corrected_hidden_mask"] = (
        None if provided == "hidden" else torch.empty(0)
    )
    inputs["previous_logits_mask"] = None
    with pytest.raises(ValueError, match="only valid when hidden feedback is enabled"):
        head(**inputs)


@pytest.mark.parametrize(
    "name", ["previous_logits", "previous_logits_mask", "previous_rank_features"]
)
def test_hidden_output_rejects_even_empty_logit_inputs(name):
    inputs = _inputs("hidden")
    inputs[name] = torch.empty(0)
    with pytest.raises(ValueError, match="require logit-residual Correction mode"):
        _head("hidden")(**inputs)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("with_mask", [False, True])
def test_logit_mask_presence_precedes_representation_exclusivity_and_shape(
    dense, compact, with_mask
):
    inputs = _inputs("dense")
    inputs["previous_logits"] = torch.empty(0) if dense else None
    inputs["previous_rank_features"] = torch.empty(0) if compact else None
    inputs["previous_logits_mask"] = torch.empty(0) if with_mask else None
    message = "requires previous features and mask"
    if with_mask:
        message = (
            "exactly one dense or compact"
            if dense == compact
            else "previous logits mask and DFlash hidden must align"
        )
    with pytest.raises(ValueError, match=message):
        _head("dense")(**inputs)


@pytest.mark.parametrize("representation", ["dense", "compact"])
def test_missing_rank_projection_keeps_dense_encoder_vs_compact_error(representation):
    head = _head(representation)
    head.previous_logits_down = None
    message = (
        "Logit Correction is missing its rank projection"
        if representation == "dense"
        else "Logit Correction rank projection is missing"
    )
    with pytest.raises(RuntimeError, match=message):
        head(**_inputs(representation))


@pytest.mark.parametrize("outcome", ["wrong_shape", "none", "missing_module"])
def test_dense_encoding_runs_before_rank_validation(monkeypatch, outcome):
    head = _head("dense")
    inputs = _inputs("dense")
    calls = []

    def encode(mask, *, previous_logits):
        calls.append((mask, previous_logits))
        if outcome == "missing_module":
            head.previous_logits_down = None
        return None if outcome == "none" else torch.empty(1)

    monkeypatch.setattr(head, "encode_previous_distribution", encode)
    error = RuntimeError if outcome == "missing_module" else ValueError
    message = (
        "rank projection is missing" if outcome == "missing_module" else "rank features"
    )
    with pytest.raises(error, match=message):
        head(**inputs)
    assert len(calls) == 1
    assert calls[0][0] is inputs["previous_logits_mask"]
    assert calls[0][1] is inputs["previous_logits"]


def _expected_features(head, inputs):
    dtype = head.hidden_proj.weight.dtype
    hidden = inputs["dflash_hidden"].to(dtype)
    if "current_token_embeddings" in inputs:
        hidden = hidden + inputs["current_token_embeddings"].to(dtype)
    states = (
        functional.linear(hidden, head.hidden_proj.weight)
        + functional.linear(
            inputs["previous_token_embeddings"].to(dtype), head.token_proj.weight
        )
        + functional.embedding(
            inputs["block_positions"].long(), head.position_embedding.weight
        ).to(dtype)
    )
    if "previous_logits" in inputs:
        probabilities = inputs["previous_logits"].float().softmax(dim=-1)
        rank = functional.linear(
            probabilities.to(dtype), head.previous_logits_down.weight
        )
        rank = rank * inputs["previous_logits_mask"].to(dtype).unsqueeze(-1)
        states = states + functional.linear(rank, head.previous_logits_proj.weight)
    elif "previous_rank_features" in inputs:
        # Compact features have already been masked by their producer.
        states = states + functional.linear(
            inputs["previous_rank_features"].to(dtype), head.previous_logits_proj.weight
        )
    if "previous_corrected_hidden" in inputs:
        feedback = inputs["previous_corrected_hidden"].to(dtype)
        feedback = feedback * inputs["previous_corrected_hidden_mask"].to(
            dtype
        ).unsqueeze(-1)
        states = states + functional.linear(feedback, head.hidden_feedback_proj.weight)
    return states


def _leaves(head, inputs):
    return tuple(value for value in inputs.values() if value.requires_grad) + tuple(
        head.parameters()
    )


def _assert_same_gradients(actual, expected, leaves):
    actual_grads = torch.autograd.grad(actual, leaves, allow_unused=True)
    expected_grads = torch.autograd.grad(expected, leaves, allow_unused=True)
    for left, right in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("representation", ["hidden", "dense", "compact"])
@pytest.mark.parametrize("feedback", [False, True])
@pytest.mark.parametrize("current", [False, True])
@pytest.mark.parametrize(
    ("model_dtype", "input_dtype"),
    [
        (torch.float64, torch.float32),
        (torch.float32, torch.float64),
        (torch.bfloat16, torch.float32),
    ],
)
def test_projected_features_keep_addition_cast_mask_order_and_gradients(
    representation, feedback, current, model_dtype, input_dtype
):
    head = _head(representation, feedback=feedback, dtype=model_dtype)
    inputs = _inputs(
        representation, feedback=feedback, current=current, dtype=input_dtype
    )
    snapshots = {name: value.detach().clone() for name, value in inputs.items()}
    projected = []
    handle = head.layers[0].register_forward_pre_hook(
        lambda module, args: projected.append(args[0])
    )
    delta, states, cache = head(**inputs)
    handle.remove()
    expected = _expected_features(head, inputs)
    torch.testing.assert_close(projected[0], expected, rtol=0, atol=0)
    assert states.dtype == delta.dtype == model_dtype
    assert cache is None
    _assert_same_gradients(
        projected[0].square().sum(), expected.square().sum(), _leaves(head, inputs)
    )
    for name, value in inputs.items():
        torch.testing.assert_close(value, snapshots[name], rtol=0, atol=0)


def _past_cache():
    return [
        tuple(
            torch.randn(2, 3, 2, 4, dtype=torch.float64, requires_grad=True)
            for _ in range(2)
        )
        for _ in range(2)
    ]


@pytest.mark.parametrize("representation", ["hidden", "dense", "compact"])
@pytest.mark.parametrize("with_past", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
def test_head_passes_cache_entries_through_without_mutating_or_detaching(
    representation, with_past, use_cache
):
    head = _head(representation)
    inputs = _inputs(representation)
    past = _past_cache() if with_past else None
    snapshots = [[value.detach().clone() for value in entry] for entry in (past or [])]
    observed = []
    for layer in head.layers:
        layer.register_forward_hook(
            lambda module, args, result: observed.append((args[1], result[1]))
        )
    delta, states, cache = head(**inputs, cache=past, use_cache=use_cache)
    if use_cache:
        assert type(cache) is list
        assert cache is not past
        assert len(cache) == len(head.layers)
    else:
        assert cache is None
    for index, (incoming, outgoing) in enumerate(observed):
        assert incoming is (None if past is None else past[index])
        assert outgoing is (cache[index] if use_cache else None)
    (delta.square().sum() + states.square().sum()).backward()
    for entry, snapshot in zip(past or [], snapshots, strict=True):
        for value, unchanged in zip(entry, snapshot, strict=True):
            torch.testing.assert_close(value, unchanged, rtol=0, atol=0)
            assert value.grad is not None
            assert torch.isfinite(value.grad).all()
            assert value.grad.count_nonzero() > 0


@pytest.mark.parametrize("representation", ["hidden", "dense", "compact"])
@pytest.mark.parametrize("cached", [False, True])
def test_head_features_and_cache_compile_without_graph_breaks(representation, cached):
    head = _head(representation)
    inputs = _inputs(representation, dtype=torch.float32)
    past = _past_cache() if cached else None
    compiled = torch.compile(head, backend="eager", fullgraph=True)
    actual = compiled(**inputs, cache=past, use_cache=cached)
    expected = head(**inputs, cache=past, use_cache=cached)
    assert type(actual) is tuple
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    leaves = _leaves(head, inputs) + tuple(
        value for entry in (past or []) for value in entry
    )
    _assert_same_gradients(
        actual[0].square().sum() + actual[1].square().sum(),
        expected[0].square().sum() + expected[1].square().sum(),
        leaves,
    )
