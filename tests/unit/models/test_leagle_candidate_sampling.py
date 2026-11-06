"""Candidate-only position budgets preserve backbone training and full validation."""

import copy

import pytest
import torch

from speculators.losses import resolve_loss_config
from speculators.models.leagle import LEagleDraftModel
from speculators.models.leagle.objective import select_candidate_positions

from .test_leagle import batch, make_model
from .test_leagle_objective import explicit_objective
from .test_leagle_training_diagnostics import packed_batch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(739)
    torch.set_num_threads(1)


def options(mode):
    result = {
        "ttt_steps": 11,
        "ttt_step_loss_decay": 0.7,
        "loss_config": resolve_loss_config('{"ce": 0.1, "tv": 0.9}', "eager"),
        "leagle_loss_chunk_size": 2,
    }
    if mode == "chunked":
        result["leagle_chunked_loss"] = True
    elif mode == "masked":
        result.update(leagle_masked_head=True, leagle_teacher_cache=True)
    elif mode == "streamed":
        result.update(
            leagle_masked_head=True,
            leagle_teacher_cache=True,
            leagle_shared_loss_normalization=True,
            leagle_stream_main_loss=True,
            leagle_candidate_grad_addmm=True,
        )
    elif mode == "replay":
        result.update(
            leagle_masked_head=True,
            leagle_candidate_backward_recompute=True,
        )
    return result


def test_sampling_counts_only_eligible_positions_rotates_and_keeps_short_steps():
    mask = torch.tensor([[False, True, False, True, True, False, True, True]])
    assert select_candidate_positions(mask, 1, 0) is mask
    for step, expected in enumerate(([1, 4, 7], [3, 6], [1, 4, 7])):
        selected = select_candidate_positions(mask, 2, step)
        assert selected[0].nonzero().flatten().tolist() == expected
    for step in range(12):
        selected = select_candidate_positions(mask, 8, step)
        assert selected.sum().item() == 1
        assert selected[0].nonzero().item() == [1, 3, 4, 6, 7][step % 5]
        assert not select_candidate_positions(torch.zeros_like(mask), 8, step).any()


@pytest.mark.parametrize("stride", [2, 8])
@pytest.mark.parametrize("mode", ["dense", "chunked", "masked", "streamed", "replay"])
def test_sampled_loss_and_gradients_match_independent_formula(
    stride, mode, monkeypatch
):
    model = make_model(layers=1, loops=2, attention="sdpa", candidate_head=True).train()
    model.set_activation_checkpointing(True)
    inputs = packed_batch()
    states = []
    original = model._step

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        states.append(result[3])
        return result

    monkeypatch.setattr(model, "_step", record)
    _, loss, _ = model(**inputs, candidate_head_train_stride=stride, **options(mode))
    expected = explicit_objective(model, states, inputs, 0.7, candidate_stride=stride)
    parameters = [p for p in model.parameters() if p.requires_grad]
    actual_grads = torch.autograd.grad(loss * 0.37, parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(expected * 0.37, parameters)
    torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_grads, expected_grads, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("precision", ["fp32", "bf16", "amp"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("mode", ["dense", "chunked", "masked", "replay"])
def test_backbone_gradients_clipping_updates_and_rng_stay_unchanged(
    precision, empty, mode
):
    base = make_model(layers=1, loops=2, attention="sdpa", candidate_head=True).train()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    base.to(dtype)
    inputs = packed_batch()
    if empty:
        inputs["loss_mask"].zero_()
    inputs = {
        name: value.to(dtype) if value.is_floating_point() else value
        for name, value in inputs.items()
    }
    results = []
    rng = torch.get_rng_state()
    for stride in (1, 8):
        model = copy.deepcopy(base)
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, foreach=False)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "amp"):
            _, loss, metrics = model(
                **inputs, candidate_head_train_stride=stride, **options(mode)
            )
        (loss * 0.37).backward()
        gradients = {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.requires_grad and not name.startswith("candidate_head.")
        }
        for parameter in model.candidate_head.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            if empty:
                assert not parameter.grad.count_nonzero()
        for group in model.gradient_clip_groups():
            torch.nn.utils.clip_grad_norm_(group, 1.0)
        optimizer.step()
        weights = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if not name.startswith("candidate_head.")
        }
        results.append((gradients, weights, metrics["draft_loss_sum"]))
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


def test_sampling_reduces_projected_rows_and_validation_restores_full_coverage():
    model = make_model(layers=1, loops=1, attention="sdpa", candidate_head=True)
    inputs = batch(length=64)
    inputs["verifier_last_hidden_states"] = torch.randn(1, 64, 16)
    rows = []
    hook = model.candidate_head.down_proj.register_forward_pre_hook(
        lambda _module, args: rows.append(args[0].shape[0])
    )
    kwargs = options("dense")
    kwargs["ttt_steps"] = 7
    model.train()
    model(**inputs, candidate_head_train_stride=8, **kwargs)
    sampled = sum(rows)
    assert sampled == 53
    rows.clear()
    model.eval()
    with torch.no_grad():
        _, sampled_val, sampled_metrics = model(
            **inputs, candidate_head_train_stride=8, **kwargs
        )
        assert sum(rows) == sum(range(58, 65))
        _, full_val, full_metrics = model(
            **inputs, candidate_head_train_stride=1, **kwargs
        )
    hook.remove()
    torch.testing.assert_close(sampled_val, full_val, atol=0, rtol=0)
    torch.testing.assert_close(sampled_metrics, full_metrics, atol=0, rtol=0)
    train, val = LEagleDraftModel.get_trainer_kwargs(
        candidate_head=True,
        candidate_head_train_stride=8,
        loss_fn='{"ce": 0.1, "tv": 0.9}',
        loss_implementation="eager",
    )
    assert train["candidate_head_train_stride"] == 8
    assert val["candidate_head_train_stride"] == 1
    assert "candidate_head_train_stride" not in model.config.to_dict()


@pytest.mark.parametrize("stride", [0, -1, 1.5])
def test_invalid_stride_fails(stride):
    model = make_model(candidate_head=True)
    with pytest.raises(ValueError, match="stride must be a positive integer"):
        model(**packed_batch(), candidate_head_train_stride=stride)


@pytest.mark.parametrize("candidate", [False, True])
def test_sampling_rejects_models_without_auxiliary_kl(candidate):
    model = make_model(candidate_head=candidate, candidate_rerank=False)
    with pytest.raises(ValueError, match="requires auxiliary distillation"):
        model(**packed_batch(), candidate_head_train_stride=8)
