"""Budgeted supervised caching, shared normalization and frozen-head identity."""

import copy
import itertools
from contextlib import nullcontext

import pytest
import torch
from safetensors.torch import load_file
from torch.nn import functional

from speculators.models.leagle import chunked_loss
from speculators.models.leagle.shared_loss import CandidateTeacher, shared_terms
from speculators.models.leagle.teacher_cache import TeacherCache
from speculators.models.metrics import resolve_training_loss

from .test_leagle import make_model
from .test_leagle_objective import explicit_objective
from .test_leagle_training_diagnostics import packed_batch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(803)
    torch.set_num_threads(1)


@pytest.mark.parametrize("shared", [False, True])
def test_priority_cache_interleaved_hits_misses_and_budget(shared):
    hidden, weight = torch.randn(1, 9, 16), torch.randn(24, 16)
    valid = torch.tensor([[0, 0, 1, 0, 1, 1, 0, 1, 0]])
    budget = 9 * 8 + 3 * (24 * 4 + 8)
    cache = TeacherCache(
        hidden,
        weight,
        budget_bytes=budget,
        chunk_size=2,
        priority_mask=valid,
        shared_normalization=shared,
    )
    assert cache.rows == 3
    assert cache.slots.tolist() == [-1, -1, 0, -1, 1, 2, -1, -1, -1]
    tensors = (cache.probs, cache.ids, cache.slots)
    assert sum(value.numel() * value.element_size() for value in tensors) <= budget
    logits = functional.linear(hidden, weight)
    probs = logits.log_softmax(-1).exp() if shared else logits.softmax(-1)
    expected = (probs, logits.argmax(-1))
    for indices in ([0, 2, 3, 4, 5, 7], [4, 0, 2, 5, 8], [2, 4, 5], [7, 8], []):
        indices = torch.tensor(indices, dtype=torch.long)
        actual = cache.get(hidden[:, indices], weight, indices=indices)
        torch.testing.assert_close(actual, tuple(x[:, indices] for x in expected))
    actual = cache.get(hidden[:, 1:8], weight, start=1)
    torch.testing.assert_close(actual, tuple(x[:, 1:8] for x in expected))
    empty = TeacherCache(
        hidden,
        weight,
        budget_bytes=9 * 8 + 1,
        chunk_size=2,
        priority_mask=torch.zeros_like(valid),
        shared_normalization=shared,
    )
    assert empty.rows == 0
    torch.testing.assert_close(empty.get(hidden, weight), expected)
    for bad_budget in (9 * 8, 9 * 8 + 1):
        with pytest.raises(ValueError, match="budget"):
            TeacherCache(
                hidden,
                weight,
                budget_bytes=bad_budget,
                chunk_size=2,
                priority_mask=valid,
            )


@pytest.mark.parametrize(
    ("shared", "masked", "cache"), list(itertools.product([False, True], repeat=3))
)
@pytest.mark.parametrize(
    ("dtype", "autocast"),
    [
        (torch.float32, False),
        (torch.bfloat16, False),
        (torch.float32, True),
    ],
)
@pytest.mark.parametrize("loss_fn", ["kl_div", '{"ce":0.1,"tv":0.9}'])
def test_training_paths_match_loss_metrics_gradients_and_rng(
    shared,
    masked,
    cache,
    dtype,
    autocast,
    loss_fn,
    monkeypatch,
):
    base = make_model(layers=1, loops=2, attention="sdpa", candidate_head=True).to(
        dtype
    )
    base.set_activation_checkpointing(True)
    inputs = packed_batch()
    for key in ("hidden_states", "verifier_last_hidden_states"):
        inputs[key] = inputs[key].to(dtype)
    original = TeacherCache.__init__

    def bounded(obj, hidden, weight, **kwargs):
        kwargs["budget_bytes"] = hidden.shape[1] * 8 + 2 * (weight.shape[0] * 4 + 8)
        original(obj, hidden, weight, **kwargs)

    monkeypatch.setattr(TeacherCache, "__init__", bounded)
    results = []
    reference_config = resolve_training_loss(loss_fn, loss_implementation="eager")
    for flags in (
        {},
        {
            "leagle_shared_loss_normalization": shared,
            "leagle_teacher_cache": cache,
            "leagle_teacher_cache_supervised": cache,
            "leagle_masked_head": masked,
        },
    ):
        model = copy.deepcopy(base)
        torch.manual_seed(473)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            result = model(
                **inputs,
                ttt_steps=7,
                ttt_step_loss_decay=0.7,
                loss_config=reference_config
                if not flags
                else resolve_training_loss(loss_fn, loss_implementation="eager"),
                leagle_loss_chunk_size=3,
                return_predictions=False,
                **flags,
            )
        result[1].backward()
        gradients = {
            name: p.grad for name, p in model.named_parameters() if p.grad is not None
        }
        results.append((result, gradients, torch.get_rng_state()))
    # Different candidate chunk shapes can still change low-precision rounding.
    # Shared normalization itself uses the reference loss kernels in BF16/AMP.
    low_precision = dtype == torch.bfloat16 or autocast
    atol, rtol = (0.015, 0.06) if low_precision else (3e-5, 5e-4)
    torch.testing.assert_close(results[0], results[1], atol=atol, rtol=rtol)
    if low_precision:
        for name, expected in results[0][1].items():
            error = (results[1][1][name].float() - expected.float()).norm()
            assert error <= 0.03 * expected.float().norm() + 1e-6, name


@pytest.mark.parametrize("masked", [False, True])
def test_shared_objective_matches_independent_formula(masked, monkeypatch):
    model = make_model(layers=2, loops=2, attention="sdpa", candidate_head=True)
    model.set_activation_checkpointing(True)
    inputs, states = packed_batch(), []
    original = model._step

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        states.append(result[3])
        return result

    monkeypatch.setattr(model, "_step", record)
    _, loss, _ = model(
        **inputs,
        ttt_steps=11,
        ttt_step_loss_decay=0.7,
        collect_metrics=False,
        return_predictions=False,
        loss_config=resolve_training_loss(
            '{"ce":0.1,"tv":0.9}', loss_implementation="eager"
        ),
        leagle_teacher_cache=True,
        leagle_teacher_cache_supervised=True,
        leagle_shared_loss_normalization=True,
        leagle_masked_head=masked,
        leagle_loss_chunk_size=3,
    )
    expected = explicit_objective(model, states, inputs, 0.7)
    parameters = [p for p in model.parameters() if p.requires_grad]
    actual_grads = torch.autograd.grad(loss, parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, parameters)
    torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_grads, expected_grads, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("empty", [False, True])
def test_supervised_cache_only_projects_valid_teacher_rows(empty, monkeypatch):
    model, inputs = make_model(layers=1, loops=1, attention="sdpa"), packed_batch()
    if empty:
        inputs["loss_mask"].zero_()
    rows, original = [], functional.linear

    def linear(hidden, weight, *args, **kwargs):
        if weight is model.verifier_lm_head.weight:
            rows.append(hidden.shape[1])
        return original(hidden, weight, *args, **kwargs)

    monkeypatch.setattr(functional, "linear", linear)
    _, loss, _ = model(
        **inputs,
        ttt_steps=7,
        leagle_teacher_cache=True,
        leagle_teacher_cache_supervised=True,
        leagle_masked_head=True,
        leagle_loss_chunk_size=3,
        return_predictions=False,
        loss_config=resolve_training_loss("kl_div", loss_implementation="eager"),
    )
    valid = (inputs["document_ids"] >= 0) & inputs["loss_mask"].bool()
    assert sum(rows) == valid.sum().item()
    loss.backward()
    assert sum(rows) == valid.sum().item()


@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_shared_tv_identical_distributions_have_zero_gradient(scale):
    logits = (torch.randn(1, 7, 24) * scale).requires_grad_()
    probabilities = logits.detach().log_softmax(-1).exp()
    terms, _, teacher = shared_terms(
        logits,
        probabilities,
        logits.argmax(-1),
        resolve_training_loss("tv", loss_implementation="eager"),
        True,
        True,
    )
    terms[0].sum().backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=2e-7, rtol=0)
    assert isinstance(teacher, CandidateTeacher)
    assert not teacher.log_probs.requires_grad
    assert not teacher.probabilities.requires_grad


def test_frozen_head_sharing_optimizer_checkpoint_and_resume(tmp_path):
    base = make_model(layers=1, loops=1, attention="sdpa", candidate_head=True)
    shared, inputs = copy.deepcopy(base), packed_batch()
    reference = shared.share_output_head()
    assert shared.lm_head.weight is shared.verifier_lm_head.weight
    shared.to(dtype=torch.bfloat16).to(dtype=torch.float32)
    # Use the same BF16-rounded weights on the reference for this dtype-move test.
    base.to(dtype=torch.bfloat16).to(dtype=torch.float32)
    reference = reference.to(torch.bfloat16).float()
    shared.validate_shared_output_head(reference)
    optimizers = [
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        for model in (base, shared)
    ]
    for model, optimizer in zip((base, shared), optimizers, strict=True):
        model(
            **inputs,
            ttt_steps=4,
            loss_config=resolve_training_loss("kl_div", loss_implementation="eager"),
        )[1].backward()
        optimizer.step()
    torch.testing.assert_close(base.state_dict(), shared.state_dict(), atol=0, rtol=0)
    shared.save_pretrained(tmp_path)
    saved = load_file(str(tmp_path / "model.safetensors"))
    assert "lm_head.weight" in saved
    assert "verifier_lm_head.weight" not in saved
    resumed = copy.deepcopy(base)
    resumed_reference = resumed.share_output_head()
    resumed.load_state_dict(saved, strict=False)
    resumed.validate_shared_output_head(resumed_reference)
    new_optimizer = torch.optim.AdamW(
        [p for p in resumed.parameters() if p.requires_grad]
    )
    new_optimizer.load_state_dict(optimizers[1].state_dict())
    torch.testing.assert_close(new_optimizer.state_dict(), optimizers[1].state_dict())
    with torch.no_grad():
        saved["lm_head.weight"].add_(1)
    resumed.load_state_dict(saved, strict=False)
    with pytest.raises(ValueError, match="differs"):
        resumed.validate_shared_output_head(resumed_reference)


@pytest.mark.parametrize("invalid", ["trainable", "different", "nonfinite"])
def test_head_sharing_rejects_unsafe_weights(invalid):
    model = make_model()
    if invalid == "trainable":
        model.lm_head.requires_grad_(True)
    else:
        with torch.no_grad():
            model.lm_head.weight[0, 0] = float("inf") if invalid == "nonfinite" else 10
            if invalid == "nonfinite":
                model.verifier_lm_head.weight.copy_(model.lm_head.weight)
    with pytest.raises(ValueError, match="sharing requires"):
        model.share_output_head()


def test_shared_normalization_rejects_fused_or_custom_losses():
    model = make_model(attention="sdpa")
    for config in (
        resolve_training_loss("kl_div", loss_implementation="fused"),
        {"tv": (lambda x, y: x.sum() + y.sum(), 1.0)},
    ):
        with pytest.raises(ValueError, match="eager|built-in"):
            model(
                **packed_batch(),
                leagle_shared_loss_normalization=True,
                loss_config=config,
            )


def test_output_sharing_rejects_equal_values_with_different_precision():
    model = make_model()
    model.verifier_lm_head.double()
    assert torch.equal(model.lm_head.weight, model.verifier_lm_head.weight)
    with pytest.raises(ValueError, match="matching matrices"):
        model.share_output_head()


@pytest.mark.parametrize("precision", ["bfloat16", "autocast"])
@pytest.mark.parametrize("scale", [0.37, 4 / 3])
@pytest.mark.parametrize("loss_name", ["ce", "tv", "kl_div", '{"ce":0.1,"tv":0.9}'])
@pytest.mark.parametrize("cached_masked", [False, True])
def test_shared_low_precision_fallback_is_exact(
    precision, scale, loss_name, cached_masked, monkeypatch
):
    model = make_model(layers=1, loops=2, attention="sdpa", candidate_head=True)
    model.set_activation_checkpointing(True)
    inputs = packed_batch()
    if precision != "autocast":
        dtype = getattr(torch, precision)
        model = model.to(dtype)
        for name in ("hidden_states", "verifier_last_hidden_states"):
            inputs[name] = inputs[name].to(dtype)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Low precision must retain the native loss kernels")

    monkeypatch.setattr(chunked_loss, "shared_terms", unexpected)
    results = []
    for shared in (False, True):
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
                return_predictions=False,
                leagle_chunked_loss=True,
                leagle_loss_chunk_size=3,
                leagle_shared_loss_normalization=shared,
                leagle_teacher_cache=cached_masked,
                leagle_teacher_cache_supervised=cached_masked,
                leagle_masked_head=cached_masked,
            )
        (result[1] * scale).backward()
        gradients = {
            n: p.grad for n, p in draft.named_parameters() if p.grad is not None
        }
        results.append((result, gradients, torch.get_rng_state()))
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)


def test_trainer_reports_shared_normalization_fallback(caplog):
    make_model().get_trainer_kwargs(
        leagle_shared_loss_normalization=True,
        hidden_states_dtype="bfloat16",
        loss_fn="kl_div",
        loss_implementation="eager",
    )
    assert "shared loss normalization is FP32-only" in caplog.text
