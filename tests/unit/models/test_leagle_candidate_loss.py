"""Streaming KL agrees with autograd without repeated teacher work or TTT graphs."""

import copy
from contextlib import nullcontext

import pytest
import torch
from torch.nn import functional

from speculators.losses import resolve_loss_config
from speculators.models.leagle.candidate_head import LowRankCandidateHead

from .test_leagle import batch, make_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


@pytest.mark.parametrize("precision", ["fp32", "bf16", "autocast"])
@pytest.mark.parametrize("chunk_size", [2, 32])
@pytest.mark.parametrize("frozen", [None, "down_proj", "up_proj"])
@pytest.mark.parametrize(
    ("grad_addmm", "valid_only"), [(False, False), (True, False), (True, True)]
)
def test_streamed_ttt_gradients_match_autograd_with_masks_decay_and_loss_scaling(
    precision, chunk_size, frozen, grad_addmm, valid_only
):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    head = LowRankCandidateHead(16, 24, 4, initializer_range=0.1).to(dtype)
    if frozen:
        getattr(head, frozen).requires_grad_(False)
    reference = copy.deepcopy(head)
    weight = torch.randn(24, 16, dtype=dtype, requires_grad=True)
    inputs, teachers = [], []
    accumulator = head.loss_accumulator(grad_addmm=grad_addmm, valid_only=valid_only)
    expected = torch.zeros(())
    context = (
        torch.autocast("cpu", dtype=torch.bfloat16)
        if precision == "autocast"
        else nullcontext()
    )
    with context:
        for step in range(7):
            hidden = torch.randn(1, 9, 16, dtype=dtype, requires_grad=True)
            mask = torch.rand(1, 9) > 0.3
            if step == 3:
                mask.zero_()
            teacher = functional.linear(hidden, weight)
            teacher.retain_grad()
            inputs.append(hidden)
            teachers.append(teacher)
            scale = 0.7**step
            accumulator.add(
                hidden,
                weight,
                mask,
                top_k=3,
                chunk_size=chunk_size,
                teacher_logits=teacher,
                normalize=True,
                loss_scale=scale,
            )
            log_p = teacher.detach().float().log_softmax(-1)
            log_q = reference.coarse_logits(hidden.detach()).float().log_softmax(-1)
            expected = expected + scale * (
                (log_p.exp() * (log_p - log_q)).sum(-1) * mask
            ).sum() / mask.sum().clamp_min(1)
        actual = accumulator.loss()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    (actual * 0.37).backward()
    (expected * 0.37).backward()
    assert weight.grad is None
    assert all(h.grad is None for h in inputs)
    assert all(t.grad is None for t in teachers)
    for observed, wanted in zip(head.parameters(), reference.parameters(), strict=True):
        if wanted.grad is None:
            assert observed.grad is None
        elif precision == "fp32":
            torch.testing.assert_close(observed.grad, wanted.grad, atol=1e-6, rtol=1e-4)
        else:
            # BF16 chunk products, FP32 accumulation, and delayed outer loss
            # scaling round differently near cancellations. Bound the whole
            # gradient error, rather than relative error near zero elements.
            error = torch.linalg.vector_norm(
                observed.grad.float() - wanted.grad.float()
            )
            norm = torch.linalg.vector_norm(wanted.grad.float())
            assert error < 0.01 * norm


def test_saved_memory_is_two_gradient_buffers_independent_of_ttt_and_token_count():
    head = LowRankCandidateHead(8, 128, 4)
    hidden, weight = torch.randn(1, 17, 8), torch.randn(128, 8)
    teacher = functional.linear(hidden, weight)
    saved = []
    calls = []
    hook = head.up_proj.register_forward_hook(lambda *_: calls.append(True))
    accumulator = head.loss_accumulator()

    def pack(value):
        saved.append(tuple(value.shape))
        return value

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            for _ in range(7):
                accumulator.add(
                    hidden,
                    weight,
                    torch.ones(1, 17, dtype=torch.bool),
                    top_k=3,
                    chunk_size=2,
                    teacher_logits=teacher,
                )
            loss = accumulator.loss()
        before = len(calls)
        loss.backward()
        assert len(calls) == before  # No backward recomputation of any chunk.
    finally:
        hook.remove()
    assert saved == [(4, 8), (128, 4)]


def test_model_reuses_dense_logits_and_delivers_each_head_gradient_once(monkeypatch):
    model = make_model(candidate_head=True)
    inputs = batch()
    inputs["verifier_last_hidden_states"] = torch.randn(1, 6, 16)
    calls, hooks = [], []
    original = functional.linear

    def record(input_tensor, weight, bias=None):
        if weight.data_ptr() == model.lm_head.weight.data_ptr():
            calls.append(tuple(input_tensor.shape))
        return original(input_tensor, weight, bias)

    monkeypatch.setattr(functional, "linear", record)
    handles = [
        p.register_hook(lambda grad: hooks.append(True))
        for p in model.candidate_head.parameters()
    ]
    try:
        _, loss, _ = model(
            **inputs,
            ttt_steps=7,
            loss_config=resolve_loss_config("kl_div", implementation="eager"),
        )
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [(1, 6, 16)] * 7
    assert len(hooks) == 2


def test_no_grad_validation_uses_no_gradient_buffers():
    head = LowRankCandidateHead(8, 12, 3)
    hidden, weight = torch.randn(1, 4, 8), torch.randn(12, 8)
    with torch.no_grad():
        accumulator = head.loss_accumulator()
        loss, _, _ = accumulator.add(
            hidden, weight, torch.ones(1, 4, dtype=torch.bool), top_k=3, chunk_size=2
        )
        torch.testing.assert_close(accumulator.loss(), loss)
        assert not accumulator.loss().requires_grad
        assert accumulator.up_grad is None
        assert accumulator.down_grad is None
