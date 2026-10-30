"""Opt-in paths preserve the baseline objective and trajectory visibility."""

import copy
import itertools

import pytest
import torch

from speculators.models.metrics import resolve_training_loss

from .test_leagle import make_model
from .test_leagle_training_diagnostics import packed_batch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(37)
    torch.set_num_threads(1)


def switches(reuse, chunked, sparse):
    return {
        "leagle_reuse_training_kv": reuse,
        "leagle_chunked_loss": chunked,
        "leagle_sparse_attention": sparse,
        "leagle_loss_chunk_size": 3,
        "leagle_attn_chunk_size": 2,
    }


def run(model, inputs, flags, loss_fn, *, bf16=False, checkpointed=False):
    model.set_activation_checkpointing(checkpointed)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        predictions, loss, metrics = model(
            **inputs,
            ttt_steps=4,
            ttt_step_loss_decay=0.7,
            loss_config=resolve_training_loss(loss_fn, loss_implementation="legacy"),
            **flags,
        )
    loss.backward()
    gradients = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return predictions, loss.detach(), metrics, gradients


@pytest.mark.parametrize(
    ("reuse", "chunked", "sparse"), list(itertools.product([False, True], repeat=3))
)
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("loss_fn", ["kl_div", '{"ce": 0.1, "tv": 0.9}'])
def test_training_switches_match_baseline(
    reuse, chunked, sparse, checkpointed, loss_fn
):
    model = make_model(layers=2, loops=2, attention="sdpa", candidate_head=True)
    inputs = packed_batch()
    reference = run(
        copy.deepcopy(model), inputs, {}, loss_fn, checkpointed=checkpointed
    )
    actual = run(
        model,
        inputs,
        switches(reuse, chunked, sparse),
        loss_fn,
        checkpointed=checkpointed,
    )
    tolerance = (
        {"rtol": 5e-4, "atol": 3e-5}
        if any((reuse, chunked, sparse))
        else {"rtol": 0, "atol": 0}
    )
    torch.testing.assert_close(actual, reference, **tolerance)


@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_combined_bf16_and_reduced_vocab(reduced, empty):
    model = make_model(
        layers=1, loops=2, attention="sdpa", reduced=reduced, candidate_head=True
    )
    inputs = packed_batch()
    if empty:
        inputs["loss_mask"].zero_()
    loss_fn = '{"ce": 0.1, "tv": 0.9}'
    reference = run(
        copy.deepcopy(model), inputs, {}, loss_fn, bf16=True, checkpointed=True
    )
    actual = run(
        model, inputs, switches(True, True, True), loss_fn, bf16=True, checkpointed=True
    )
    torch.testing.assert_close(actual, reference, rtol=0.04, atol=0.004)


def test_context_projection_only_once_per_physical_layer():
    model = make_model(layers=1, loops=3, attention="sdpa")
    calls = []
    hook = model.layers[0].self_attn.k_proj.register_forward_hook(
        lambda *_: calls.append(1)
    )
    try:
        model(
            **packed_batch(),
            ttt_steps=4,
            leagle_reuse_training_kv=True,
            loss_config=resolve_training_loss("kl_div", loss_implementation="legacy"),
        )
    finally:
        hook.remove()
    assert len(calls) == 1 + 3 * 4


def test_chunked_head_never_projects_the_full_sequence(monkeypatch):
    model = make_model(layers=1, loops=1, attention="sdpa")
    original = torch.nn.functional.linear
    widths = []

    def linear(inputs, weight, *args, **kwargs):
        if weight is model.lm_head.weight or weight is model.verifier_lm_head.weight:
            widths.append(inputs.shape[1])
        return original(inputs, weight, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "linear", linear)
    run(model, packed_batch(), switches(False, True, False), "kl_div")
    assert widths
    assert max(widths) <= 3


@pytest.mark.parametrize("window", [None, 3])
def test_sparse_attention_visibility_with_padding_short_sequences_and_rollout(window):
    model = make_model(layers=1, loops=2, attention="sdpa", window=window)
    inputs = packed_batch()
    loss = resolve_training_loss("kl_div", loss_implementation="legacy")
    options = {"ttt_steps": 11, "teacher_forcing": False, "loss_config": loss}
    expected = copy.deepcopy(model)(**inputs, **options)
    actual = model(**inputs, **options, **switches(True, True, True))
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-4)


@pytest.mark.parametrize("collect", [False, True])
def test_optimizations_preserve_trainable_output_head_gradients(collect):
    model = make_model(layers=1, loops=2, attention="sdpa", candidate_head=True)
    model.lm_head.requires_grad_(True)
    inputs = packed_batch()
    loss_fn = '{"ce": 0.1, "tv": 0.9}'
    reference = run(copy.deepcopy(model), inputs, {"collect_metrics": collect}, loss_fn)
    flags = {**switches(True, True, True), "collect_metrics": collect}
    actual = run(model, inputs, flags, loss_fn)
    assert "lm_head.weight" in actual[-1]
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=5e-4)


def test_unsupported_experimental_paths_fail_explicitly():
    model = make_model(layers=1, loops=1, attention="sdpa")
    inputs = packed_batch()
    with pytest.raises(ValueError, match="built-in CE"):
        model(
            **inputs,
            leagle_chunked_loss=True,
            loss_config={"tv": (lambda x, _: x.sum(-1), 1.0)},
        )
    model.layers[0].self_attn.attention_dropout = 0.1
    with pytest.raises(ValueError, match="zero dropout"):
        model(**inputs, leagle_sparse_attention=True)
    model.rotary_emb.rope_type = "dynamic"
    with pytest.raises(ValueError, match="position-independent"):
        model(**inputs, leagle_reuse_training_kv=True)
