"""Sublayer recurrence matches untied expansion, gradients and AR trajectories."""

import copy
from contextlib import ExitStack

import pytest
import torch
from torch import nn

from .test_leagle import batch, forward_logits, make_model
from .test_leagle_loop_hidden_loss import (
    gradients,
    inputs,
    reference_loop_loss,
    settings,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(83)
    torch.set_num_threads(1)


class ZeroMLP(nn.Module):
    def forward(self, hidden):
        return torch.zeros_like(hidden)


class ZeroAttention(nn.Module):
    sliding_window = None

    def forward(self, hidden, *_args):
        return torch.zeros_like(hidden), None


def expanded_reference(shared):
    """Untied standard layers; zero residual branches implement omitted sublayers."""
    loops = shared.config.num_loops
    expanded = make_model(
        layers=len(shared.layers) * loops,
        loops=1,
        attention=shared.config.transformer_layer_config._attn_implementation,
    )
    source = shared.state_dict()
    weights = {}
    for name in expanded.state_dict():
        if name.startswith("layers."):
            _, index, tail = name.split(".", 2)
            weights[name] = source[f"layers.{int(index) // loops}.{tail}"]
        else:
            weights[name] = source[name]
    expanded.load_state_dict(weights)
    for index, layer in enumerate(expanded.layers):
        repeat = index % loops
        if shared.config.loop_mode == "attn" and repeat + 1 < loops:
            layer.mlp = ZeroMLP()
        if shared.config.loop_mode == "ffn" and repeat > 0:
            layer.self_attn = ZeroAttention()
    return expanded


def summed_reference_gradients(shared, reference):
    independent = dict(reference.named_parameters())
    result = {}
    for name, param in shared.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("layers."):
            _, index, tail = name.split(".", 2)
            copies = (
                independent.get(
                    f"layers.{int(index) * shared.config.num_loops + r}.{tail}"
                )
                for r in range(shared.config.num_loops)
            )
            result[name] = sum(
                (p.grad for p in copies if p is not None and p.grad is not None),
                torch.zeros_like(param),
            )
        else:
            result[name] = independent[name].grad
    return result


@pytest.mark.parametrize("mode", ["attn", "ffn"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("supervise", [False, True])
def test_matches_independent_depth_and_summed_gradients(mode, attention, supervise):
    shared = make_model(layers=2, loops=3, loop_mode=mode, attention=attention)
    reference = expanded_reference(shared)
    data = inputs()
    visits = []
    with ExitStack() as hooks:
        for layer in reference.layers:
            hooks.callback(
                layer.register_forward_hook(
                    lambda _m, _a, out: visits.append(out[0])
                ).remove
            )
        expected_logits, (_, main, _) = forward_logits(reference, data, **settings())
    hs = torch.zeros(())
    if supervise:
        for step in range(4):
            for index in (0, 1, 3, 4):
                hs = (
                    hs
                    + reference_loop_loss(
                        reference, data, visits[step * 6 + index], step
                    )
                    / 4
                )
    actual_logits, (_, loss, metrics) = forward_logits(
        shared, data, **settings(leagle_loop_hs_loss=supervise)
    )
    expected_loss = main + 0.1 * hs
    torch.testing.assert_close(actual_logits, expected_logits, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(loss, expected_loss, atol=2e-6, rtol=2e-5)
    if supervise:
        torch.testing.assert_close(metrics["loop_hs_loss_sum"], hs)
    loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(
        gradients(shared),
        summed_reference_gradients(shared, reference),
        atol=3e-5,
        rtol=3e-4,
    )
    assert data["verifier_last_hidden_states"].grad is None


@pytest.mark.parametrize("mode", ["attn", "ffn"])
def test_visit_order_and_one_cache_per_attention_visit(mode):
    model = make_model(layers=2, loops=3, loop_mode=mode)
    calls, cache_lengths = [], []
    with ExitStack() as hooks:
        for index, layer in enumerate(model.layers):

            def record_attention(_m, _a, out, index=index):
                calls.append((index, "A"))
                cache_lengths.append(out[1][0].shape[-2])

            hooks.callback(
                layer.self_attn.register_forward_hook(record_attention).remove
            )
            hooks.callback(
                layer.mlp.register_forward_hook(
                    lambda _m, _a, _o, index=index: calls.append((index, "F"))
                ).remove
            )
        model(**batch(length=4), ttt_steps=2)
    one_step = [
        (index, op)
        for index in range(2)
        for op in (["A"] * 3 + ["F"] if mode == "attn" else ["A"] + ["F"] * 3)
    ]
    assert calls == one_step * 2
    cache_count = 6 if mode == "attn" else 2
    assert model.config.attention_depth == cache_count
    assert cache_lengths == [4] * cache_count + [8] * cache_count
    assert model.config.ffn_depth == (2 if mode == "attn" else 6)
    assert model.config.attention_layer_indices == (
        (0, 0, 0, 1, 1, 1) if mode == "attn" else (0, 1)
    )


@pytest.mark.parametrize("mode", ["attn", "ffn"])
@pytest.mark.parametrize("bf16", [False, True])
def test_one_repeat_is_bitwise_equal_to_original_full_layer(mode, bf16):
    original = make_model(layers=2, loops=1, attention="sdpa")
    sublayer = make_model(layers=2, loops=1, loop_mode=mode, attention="sdpa")
    sublayer.load_state_dict(original.state_dict())
    data = inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        expected = original(**data, **settings(leagle_loop_hs_loss=True))
        actual = sublayer(**data, **settings(leagle_loop_hs_loss=True))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual[1].backward()
    expected[1].backward()
    torch.testing.assert_close(gradients(sublayer), gradients(original), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["attn", "ffn"])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("supervise", [False, True])
def test_recomputation_preserves_all_sublayer_gradients(
    mode, bf16, selective, supervise
):
    reference = make_model(layers=2, loops=3, loop_mode=mode, attention="sdpa")
    actual = copy.deepcopy(reference)
    actual.set_activation_checkpointing(True)
    data = inputs()
    kwargs = settings(
        leagle_loop_hs_loss=supervise,
        leagle_sparse_attention=selective,
        leagle_attn_chunk_size=2,
        leagle_reuse_training_kv=True,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        expected = reference(**data, **kwargs)
        observed = actual(**data, **kwargs, leagle_selective_checkpoint=selective)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    observed[1].backward()
    expected[1].backward()
    torch.testing.assert_close(
        gradients(actual), gradients(reference), rtol=3e-4, atol=2e-5
    )


@pytest.mark.parametrize("mode", ["attn", "ffn"])
@pytest.mark.parametrize("low_rank", [False, True])
def test_optimized_losses_keep_sublayer_supervision(mode, low_rank):
    reference = make_model(
        layers=2,
        loops=3,
        loop_mode=mode,
        attention="sdpa",
        reduced=True,
        candidate_head=True,
        candidate_rerank=not low_rank,
    )
    optimized = copy.deepcopy(reference)
    optimized.set_activation_checkpointing(True)
    data = inputs()
    expected = reference(**data, **settings(leagle_loop_hs_loss=True))
    actual = optimized(
        **data,
        **settings(
            leagle_loop_hs_loss=True,
            leagle_reuse_training_kv=True,
            leagle_chunked_loss=True,
            leagle_masked_head=True,
            leagle_teacher_cache=True,
            leagle_teacher_cache_mb=1,
            leagle_loss_chunk_size=2,
            leagle_sparse_attention=True,
            leagle_attn_chunk_size=2,
            leagle_selective_checkpoint=True,
        ),
    )
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=3e-5)
    actual[1].backward()
    expected[1].backward()
    torch.testing.assert_close(
        gradients(optimized), gradients(reference), rtol=5e-4, atol=3e-5
    )


@pytest.mark.parametrize("mode", ["attn", "ffn"])
@pytest.mark.parametrize("window", [None, 3])
def test_packed_training_trajectories_match_inference_and_do_not_leak(mode, window):
    model = make_model(
        layers=2, loops=3, loop_mode=mode, attention="sdpa", window=window
    ).eval()
    data = inputs()
    kwargs = settings(teacher_forcing=False, leagle_loop_hs_loss=True)
    logits, (_, loss, _) = forward_logits(model, data, **kwargs)
    assert torch.isfinite(loss)
    for start, end in ((0, 2), (3, 5)):
        _, scores = model.propose(
            data["hidden_states"][:, start : end + 1],
            data["input_ids"][:, end : end + 1],
            num_tokens=4,
        )
        torch.testing.assert_close(logits[:, :, end], scores, atol=2e-6, rtol=2e-5)
    changed = copy.deepcopy(data)
    changed["hidden_states"][:, 3:] += 100
    other, _ = forward_logits(model, changed, **kwargs)
    torch.testing.assert_close(logits[:, :, :3], other[:, :, :3], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["attn", "ffn"])
def test_empty_short_sequence_supervision_keeps_connected_zero_gradients(mode):
    model = make_model(layers=2, loops=3, loop_mode=mode, attention="sdpa")
    model.set_activation_checkpointing(True)
    data = inputs(length=2)
    data["loss_mask"].zero_()
    _, loss, metrics = model(**data, **settings(leagle_loop_hs_loss=True))
    assert loss.item() == metrics["loop_hs_loss_sum"].item() == 0
    loss.backward()
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad) == 0
