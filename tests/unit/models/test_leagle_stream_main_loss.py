"""Streaming the frozen main head preserves supervision without head replay."""

import copy
from contextlib import nullcontext

import pytest
import torch

from speculators.models.leagle import chunked_loss
from speculators.models.leagle.head_gradients import FrozenHeadGradients
from speculators.models.metrics import resolve_training_loss

from .test_leagle import make_model
from .test_leagle_training_diagnostics import packed_batch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(47)
    torch.set_num_threads(1)


@pytest.mark.parametrize("scale", [2, 0.37, 4 / 3])
@pytest.mark.parametrize("precision", ["float32", "bfloat16", "autocast"])
@pytest.mark.parametrize("loss_name", ["ce", "tv", "kl_div", '{"ce":0.1,"tv":0.9}'])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("extras", [False, True])
def test_loss_gradients_metrics_rng_and_scaled_backward(
    precision, loss_name, masked, extras, scale
):
    model = make_model(layers=1, loops=2, attention="sdpa", candidate_head=extras)
    model.set_activation_checkpointing(True)
    inputs = packed_batch()
    if precision == "bfloat16":
        model = model.to(torch.bfloat16)
        for name in ("hidden_states", "verifier_last_hidden_states"):
            inputs[name] = inputs[name].to(torch.bfloat16)
    results = []
    for streaming in (False, True):
        draft = copy.deepcopy(model)
        context = (
            torch.autocast("cpu", dtype=torch.bfloat16)
            if precision == "autocast"
            else nullcontext()
        )
        with context:
            result = draft(
                **inputs,
                ttt_steps=11,
                ttt_step_loss_decay=0.7,
                loss_config=resolve_training_loss(
                    loss_name, loss_implementation="eager"
                ),
                return_predictions=not masked,
                collect_metrics=not extras,
                teacher_forcing=extras,
                leagle_chunked_loss=True,
                leagle_loss_chunk_size=3,
                leagle_masked_head=masked,
                leagle_stream_main_loss=streaming,
                leagle_reuse_training_kv=extras,
                leagle_sparse_attention=extras,
                leagle_attn_chunk_size=3,
                leagle_selective_checkpoint=extras,
                leagle_cache_attention_masks=extras,
                leagle_teacher_cache=extras,
                leagle_teacher_cache_supervised=extras,
                leagle_shared_loss_normalization=extras,
            )
        (result[1] * scale).backward()
        gradients = {
            name: p.grad.clone()
            for name, p in draft.named_parameters()
            if p.grad is not None
        }
        results.append((result, gradients, torch.get_rng_state()))
    torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
    torch.testing.assert_close(results[0][2], results[1][2], atol=0, rtol=0)
    tolerance = (
        {"atol": 2e-5, "rtol": 1e-4}
        if precision == "float32"
        else {"atol": 0, "rtol": 0}
    )
    torch.testing.assert_close(results[0][1], results[1][1], **tolerance)
    if precision == "float32":
        expected, actual = (
            torch.cat([grad.flatten() for grad in result[1].values()])
            for result in results
        )
        relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
        assert relative_error < 1e-5


@pytest.mark.parametrize("masked", [False, True])
def test_empty_supervision_retains_zero_gradients(masked):
    inputs = packed_batch()
    inputs["loss_mask"].zero_()
    model = make_model(layers=1, loops=1, attention="sdpa", candidate_head=True)
    gradients = []
    for enabled in (False, True):
        draft = copy.deepcopy(model)
        result = draft(
            **inputs,
            ttt_steps=11,
            return_predictions=False,
            leagle_chunked_loss=True,
            leagle_masked_head=masked,
            leagle_stream_main_loss=enabled,
            loss_config=resolve_training_loss("kl_div", loss_implementation="eager"),
        )
        assert result[1].item() == 0
        result[1].backward()
        gradients.append(
            {n: p.grad for n, p in draft.named_parameters() if p.grad is not None}
        )
    torch.testing.assert_close(*gradients, rtol=0, atol=0)


def test_removes_head_replay_and_keeps_no_grad_results(monkeypatch):
    model = make_model(layers=1, loops=1, attention="sdpa")
    inputs = packed_batch()
    calls = []
    original = chunked_loss._project_chunk

    def project(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(chunked_loss, "_project_chunk", project)
    values = []
    for enabled in (False, True):
        calls.clear()
        draft = copy.deepcopy(model)
        options = {
            "ttt_steps": 3,
            "return_predictions": False,
            "leagle_chunked_loss": True,
            "leagle_loss_chunk_size": 3,
            "leagle_stream_main_loss": enabled,
            "loss_config": resolve_training_loss("kl_div", loss_implementation="eager"),
        }
        result = draft(**inputs, **options)
        forward_calls = len(calls)
        result[1].backward()
        assert len(calls) == forward_calls * (1 if enabled else 2)
        with torch.no_grad():
            values.append(draft(**inputs, **options))
    torch.testing.assert_close(*values, rtol=0, atol=0)


@pytest.mark.parametrize("head", ["lm_head", "verifier_lm_head"])
def test_trainable_output_weights_fail_explicitly(head):
    model = make_model(layers=1, loops=1, attention="sdpa")
    getattr(model, head).weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen output"):
        model(
            **packed_batch(),
            leagle_stream_main_loss=True,
            loss_config=resolve_training_loss("kl_div", loss_implementation="eager"),
        )


def test_tv_equal_distributions_and_non_power_of_two_outer_scaling():
    dtype = torch.float32
    weight = torch.randn(24, 16, dtype=dtype)
    hidden = torch.randn(1, 5, 16, dtype=dtype, requires_grad=True)
    teacher = hidden.detach().clone()
    mask = torch.tensor([[1, 0, 1, 1, 0]])
    config = resolve_training_loss("tv", loss_implementation="eager")

    def project(value, *args):
        return chunked_loss._project_chunk(
            value, *args, loss_config=config, metrics=False, candidate=False
        )

    terms = project(hidden, teacher, weight, weight)[0]
    loss = chunked_loss._reduce_terms([[terms[0]]], config, mask, 2, 0.7)
    (expected,) = torch.autograd.grad(loss * 0.37, hidden)
    stream = FrozenHeadGradients(hidden, config, mask, 2, 0.7)
    result = stream.project(project, (hidden, teacher, weight, weight), slice(None))
    value = chunked_loss._reduce_terms([[result[0][0]]], config, mask, 2, 0.7)
    (actual,) = torch.autograd.grad(stream.attach(value) * 0.37, hidden)
    torch.testing.assert_close(loss, value, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("autocast", [False, True])
def test_low_precision_cannot_enter_hidden_gradient_precomputation(
    dtype, autocast, monkeypatch
):
    inputs = packed_batch()
    model = make_model(layers=1, loops=1, attention="sdpa")
    if not autocast:
        model = model.to(dtype)
        for name in ("hidden_states", "verifier_last_hidden_states"):
            inputs[name] = inputs[name].to(dtype)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Low precision must use the original checkpointed head")

    monkeypatch.setattr(chunked_loss, "FrozenHeadGradients", unexpected)
    context = torch.autocast("cpu", dtype=dtype) if autocast else nullcontext()
    with context:
        result = model(
            **inputs,
            leagle_stream_main_loss=True,
            loss_config=resolve_training_loss("kl_div", loss_implementation="eager"),
        )
    (result[1] * 0.37).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_trainer_reports_low_precision_fallback(caplog):
    make_model().get_trainer_kwargs(
        leagle_stream_main_loss=True,
        hidden_states_dtype="bfloat16",
        loss_fn="kl_div",
        loss_implementation="eager",
    )
    assert "FP32-only" in caplog.text
