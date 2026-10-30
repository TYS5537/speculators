"""Backward candidate replay matches independent full-step native autograd."""

import copy
from contextlib import nullcontext

import pytest
import torch
from torch.nn import functional

from speculators.models.leagle.candidate_head import LowRankCandidateHead
from speculators.models.metrics import resolve_training_loss

from .test_leagle import make_model
from .test_leagle_training_diagnostics import packed_batch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(42)


def autocast(precision):
    return (
        torch.autocast("cpu", dtype=torch.bfloat16)
        if precision == "amp"
        else nullcontext()
    )


def reference_step(head, hidden, weight, mask, scale):
    teacher = functional.linear(hidden.detach(), weight.detach())
    log_p = teacher.float().log_softmax(-1)
    coarse = head.coarse_logits(hidden.detach())
    log_q = coarse.float().log_softmax(-1)
    raw = ((log_p.exp() * (log_p - log_q)).sum(-1) * mask).sum()
    coefficient = scale / mask.sum(dtype=torch.float32).clamp_min(1)
    return raw * coefficient


class NativeCandidateLoss:
    """Small-test oracle: full vocabulary, native backward, no replay helper."""

    def __init__(self, head):
        self.head = head
        self.value = head.down_proj.weight.new_zeros((), dtype=torch.float32)

    def add(self, hidden, full_weight, mask, *, loss_scale, **_kwargs):
        teacher = functional.linear(hidden.detach(), full_weight.detach())
        log_p = teacher.float().log_softmax(-1)
        coarse = self.head.coarse_logits(hidden.detach())
        log_q = coarse.float().log_softmax(-1)
        raw = ((log_p.exp() * (log_p - log_q)).sum(-1) * mask).sum()
        count = mask.sum(dtype=torch.float32)
        self.value = self.value + raw * (loss_scale / count.clamp_min(1))
        return raw.detach(), count.new_zeros(()), count

    def loss(self):
        return self.value


@pytest.mark.parametrize("seed", [42, 803])
@pytest.mark.parametrize("shape", [(16, 24, 4, 9), (64, 257, 16, 73)])
@pytest.mark.parametrize("precision", ["float32", "bfloat16", "amp"])
@pytest.mark.parametrize("chunk_size", [2, 32])
@pytest.mark.parametrize("scale", [0.0, -0.37, 0.37, 4 / 3])
def test_full_step_reference_with_arbitrary_outer_scaling(
    seed, shape, precision, chunk_size, scale
):
    torch.manual_seed(seed)
    hidden_size, vocab, rank, length = shape
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    model = LowRankCandidateHead(hidden_size, vocab, rank, initializer_range=0.1).to(
        dtype
    )
    reference = copy.deepcopy(model)
    weight = torch.randn(vocab, hidden_size, dtype=dtype, requires_grad=True)
    accumulator = model.loss_accumulator(backward_recompute=True)
    expected = torch.zeros(())
    inputs = []
    with autocast(precision):
        for step in range(7):
            hidden = torch.randn(
                1, length, hidden_size, dtype=dtype, requires_grad=True
            )
            inputs.append(hidden)
            mask = torch.rand(1, length) > 0.3
            if step == 3:
                mask.zero_()
            accumulator.add(
                hidden,
                weight,
                mask,
                top_k=3,
                chunk_size=chunk_size,
                normalize=True,
                loss_scale=0.7**step,
            )
            expected = expected + reference_step(
                reference, hidden, weight, mask, 0.7**step
            )
        actual = accumulator.loss()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
    rng = torch.get_rng_state()
    (actual * scale).backward()
    (expected * scale).backward()
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert weight.grad is None
    assert all(h.grad is None for h in inputs)
    actual_grads, expected_grads = (
        [p.grad for p in head.parameters()] for head in (model, reference)
    )
    if precision != "float32":
        torch.testing.assert_close(actual_grads, expected_grads, atol=0, rtol=0)
    else:
        torch.testing.assert_close(actual_grads, expected_grads, atol=2e-7, rtol=1e-4)
        left, right = (
            torch.cat([g.flatten() for g in items])
            for items in (actual_grads, expected_grads)
        )
        assert (left - right).norm() <= 1e-5 * right.norm()


@pytest.mark.parametrize("frozen", ["down_proj", "up_proj", "all"])
@pytest.mark.parametrize("precision", ["float32", "bfloat16", "amp"])
def test_frozen_parameters_and_empty_supervision(frozen, precision):
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    head = LowRankCandidateHead(16, 24, 4).to(dtype)
    (head if frozen == "all" else getattr(head, frozen)).requires_grad_(False)
    with autocast(precision):
        accumulator = head.loss_accumulator(backward_recompute=True)
        accumulator.add(
            torch.randn(1, 9, 16, dtype=dtype),
            torch.randn(24, 16, dtype=dtype),
            torch.zeros(1, 9, dtype=torch.bool),
            top_k=3,
            chunk_size=2,
            normalize=True,
        )
    if frozen == "all":
        assert not accumulator.loss().requires_grad
    else:
        (accumulator.loss() * 0.37).backward()
    for parameter in head.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert not parameter.grad.count_nonzero()
        else:
            assert parameter.grad is None


def test_replay_retains_no_vocabulary_activations_and_recomputes_in_backward(
    monkeypatch,
):
    head = LowRankCandidateHead(8, 128, 4)
    hidden, weight = torch.randn(1, 17, 8), torch.randn(128, 8)
    saved, widths = [], []
    original = functional.linear

    def record(input_tensor, matrix, *args, **kwargs):
        if matrix.data_ptr() == weight.data_ptr():
            widths.append(input_tensor.shape[0])
        return original(input_tensor, matrix, *args, **kwargs)

    def pack(value):
        saved.append(value)
        return value

    monkeypatch.setattr(functional, "linear", record)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        accumulator = head.loss_accumulator(backward_recompute=True)
        for _ in range(7):
            accumulator.add(
                hidden,
                weight,
                torch.ones(1, 17, dtype=torch.bool),
                top_k=3,
                chunk_size=2,
            )
    count = len(widths)
    accumulator.loss().backward()
    assert len(widths) == count * 2
    assert max(widths) == 2
    storage = {value.data_ptr() for value in (hidden, weight, *head.parameters())}
    assert all(value.data_ptr() in storage or value.numel() <= 17 for value in saved)


@pytest.mark.parametrize("precision", ["float32", "bfloat16", "amp"])
@pytest.mark.parametrize("profile", ["dense", "chunked", "masked", "all"])
def test_model_uses_one_candidate_backward_per_ttt_step(
    precision, profile, monkeypatch
):
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    model = make_model(layers=2, loops=2, attention="sdpa", candidate_head=True).to(
        dtype
    )
    model.config.candidate_head_chunk_size = 2
    model.config.candidate_head_loss_weight = 0.73
    model.set_activation_checkpointing(True)
    reference = copy.deepcopy(model.candidate_head)
    inputs = packed_batch()
    for name in ("hidden_states", "verifier_last_hidden_states"):
        inputs[name] = inputs[name].to(dtype)
    states = []
    original = model._step

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        states.append(result[3].detach())
        return result

    monkeypatch.setattr(model, "_step", record)
    flags = {
        "leagle_chunked_loss": profile != "dense",
        "leagle_masked_head": profile in ("masked", "all"),
        "leagle_loss_chunk_size": 3,
    }
    if profile == "all":
        flags.update(
            leagle_reuse_training_kv=True,
            leagle_sparse_attention=True,
            leagle_attn_chunk_size=3,
            leagle_teacher_cache=True,
            leagle_teacher_cache_supervised=True,
            leagle_candidate_grad_addmm=True,
            leagle_shared_loss_normalization=True,
            leagle_stream_main_loss=True,
            leagle_selective_checkpoint=True,
            leagle_cache_attention_masks=True,
            leagle_align_candidate_chunks=True,
        )
    with autocast(precision):
        result = model(
            **inputs,
            **flags,
            ttt_steps=11,
            ttt_step_loss_decay=0.7,
            leagle_candidate_backward_recompute=True,
            loss_config=resolve_training_loss(
                '{"ce":0.1,"tv":0.9}', loss_implementation="eager"
            ),
        )
        docs = inputs["document_ids"]
        valid = (docs >= 0) & inputs["loss_mask"].bool()
        expected = torch.zeros(())
        for step, hidden in enumerate(states):
            width = hidden.shape[1] - step
            if width <= 0:
                continue
            mask = (
                valid[:, :width] & valid[:, step:] & (docs[:, :width] == docs[:, step:])
            )
            expected = expected + reference_step(
                reference, hidden[:, :width], model.lm_head.weight, mask, 0.7**step
            )
    (result[1] * (4 / 3)).backward()
    (expected * model.config.candidate_head_loss_weight * (4 / 3)).backward()
    observed, wanted = (
        [p.grad for p in head.parameters()]
        for head in (model.candidate_head, reference)
    )
    if precision != "float32":
        torch.testing.assert_close(observed, wanted, atol=0, rtol=0)
    else:
        torch.testing.assert_close(observed, wanted, atol=2e-7, rtol=1e-4)


def test_recompute_requires_candidate_head_and_teacher_labels():
    with pytest.raises(ValueError, match="candidate head"):
        make_model()(**packed_batch(), leagle_candidate_backward_recompute=True)
    inputs = packed_batch()
    inputs.pop("verifier_last_hidden_states")
    with pytest.raises(ValueError, match="teacher hidden states"):
        make_model(candidate_head=True)(
            **inputs, leagle_candidate_backward_recompute=True
        )


def test_no_grad_validation_skips_replay_graph():
    head = LowRankCandidateHead(8, 12, 3)
    with torch.no_grad():
        accumulator = head.loss_accumulator(backward_recompute=True)
        raw, recall, count = accumulator.add(
            torch.randn(1, 4, 8),
            torch.randn(12, 8),
            torch.ones(1, 4, dtype=torch.bool),
            top_k=3,
            chunk_size=2,
            collect_recall=False,
        )
        assert not accumulator.loss().requires_grad
        torch.testing.assert_close(accumulator.loss(), raw)
        assert recall == 0
        assert count == 4


@pytest.mark.parametrize("precision", ["float32", "bfloat16", "amp"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("trainable_head", [False, True])
def test_complete_model_against_native_candidate_reference(
    precision, empty, trainable_head
):
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    base = make_model(layers=2, loops=2, attention="sdpa", candidate_head=True).to(
        dtype
    )
    base.set_activation_checkpointing(True)
    base.lm_head.requires_grad_(trainable_head)
    base.config.candidate_head_chunk_size = 2
    inputs = packed_batch()
    if empty:
        inputs["loss_mask"].zero_()
    for name in ("hidden_states", "verifier_last_hidden_states"):
        inputs[name] = inputs[name].to(dtype)
    results = []
    for native in (False, True):
        model = copy.deepcopy(base)
        if native:
            model.candidate_head.loss_accumulator = (
                lambda head=model.candidate_head, **_: NativeCandidateLoss(head)
            )
        with autocast(precision):
            result = model(
                **inputs,
                ttt_steps=7,
                ttt_step_loss_decay=0.7,
                collect_metrics=False,
                return_predictions=False,
                leagle_candidate_backward_recompute=True,
                leagle_reuse_training_kv=True,
                leagle_sparse_attention=True,
                leagle_attn_chunk_size=3,
                leagle_masked_head=True,
                leagle_loss_chunk_size=3,
                leagle_teacher_cache=True,
                leagle_selective_checkpoint=True,
                loss_config=resolve_training_loss(
                    '{"ce":0.1,"tv":0.9}', loss_implementation="eager"
                ),
            )
        (result[1] * 0.37).backward()
        results.append({n: p.grad for n, p in model.named_parameters()})
    tolerance = (
        {"atol": 2e-7, "rtol": 1e-4}
        if precision == "float32"
        else {"atol": 0, "rtol": 0}
    )
    torch.testing.assert_close(results[0], results[1], **tolerance)
