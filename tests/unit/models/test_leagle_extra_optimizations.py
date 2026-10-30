"""Optional output-head execution preserves supervision and optimizer gradients."""

import copy
import itertools

import pytest
import torch
from torch.nn import functional

from speculators.models.leagle.teacher_cache import TeacherCache
from speculators.models.metrics import resolve_training_loss

from .test_leagle import make_model
from .test_leagle_objective import explicit_objective
from .test_leagle_training_diagnostics import packed_batch

COMBINATIONS = list(itertools.product((False, True), repeat=3))


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(372)
    torch.set_num_threads(1)


def switches(addmm, cache, masked):
    return {
        "leagle_candidate_grad_addmm": addmm,
        "leagle_teacher_cache": cache,
        "leagle_masked_head": masked,
        "leagle_loss_chunk_size": 3,
        "leagle_attn_chunk_size": 3,
    }


@pytest.mark.parametrize(("addmm", "cache", "masked"), COMBINATIONS)
@pytest.mark.parametrize(
    ("dtype", "autocast"),
    [
        (torch.float32, False),
        (torch.bfloat16, False),
        (torch.float32, True),
    ],
)
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("loss_fn", ["kl_div", '{"ce":0.1,"tv":0.9}'])
def test_head_options_match_loss_metrics_gradients_and_rng(
    addmm, cache, masked, dtype, autocast, checkpointed, loss_fn
):
    base = make_model(
        layers=1, loops=2, attention="sdpa", candidate_head=True, reduced=masked
    ).to(dtype)
    base.lm_head.requires_grad_(True)
    base.set_activation_checkpointing(checkpointed)
    inputs = packed_batch()
    for name in ("hidden_states", "verifier_last_hidden_states"):
        inputs[name] = inputs[name].to(dtype)
    outcomes = []
    for flags in ({}, switches(addmm, cache, masked)):
        model = copy.deepcopy(base)
        torch.manual_seed(993)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            result = model(
                **inputs,
                ttt_steps=4,
                ttt_step_loss_decay=0.7,
                loss_config=resolve_training_loss(
                    loss_fn, loss_implementation="legacy"
                ),
                return_predictions=False,
                **flags,
            )
        result[1].backward()
        gradients = {
            name: p.grad for name, p in model.named_parameters() if p.grad is not None
        }
        outcomes.append((result, gradients, torch.get_rng_state()))
    atol, rtol = (0.005, 0.05) if dtype == torch.bfloat16 or autocast else (3e-5, 5e-4)
    torch.testing.assert_close(outcomes[0], outcomes[1], atol=atol, rtol=rtol)


@pytest.mark.parametrize(("addmm", "cache", "masked"), COMBINATIONS)
def test_new_paths_against_independent_objective(addmm, cache, masked, monkeypatch):
    model = make_model(layers=2, loops=2, attention="sdpa", candidate_head=True)
    model.set_activation_checkpointing(True)
    inputs, states = packed_batch(), []
    original = model._step

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        states.append(result[3])
        return result

    monkeypatch.setattr(model, "_step", record)
    _, actual, _ = model(
        **inputs,
        ttt_steps=11,
        ttt_step_loss_decay=0.7,
        loss_config=resolve_training_loss(
            '{"ce":0.1,"tv":0.9}', loss_implementation="legacy"
        ),
        collect_metrics=False,
        return_predictions=False,
        leagle_reuse_training_kv=True,
        leagle_sparse_attention=True,
        **switches(addmm, cache, masked),
    )
    expected = explicit_objective(model, states, inputs, decay=0.7)
    parameters = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(actual, parameters, retain_graph=True)
    reference = torch.autograd.grad(expected, parameters)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(grads, reference, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("empty", [False, True])
def test_masked_projection_visits_only_supervised_positions(empty, monkeypatch):
    model, inputs = make_model(layers=1, loops=1, attention="sdpa"), packed_batch()
    if empty:
        inputs["loss_mask"].zero_()
    rows = []
    original = functional.linear

    def linear(hidden, weight, *args, **kwargs):
        if weight is model.lm_head.weight:
            rows.append(hidden.shape[1])
        return original(hidden, weight, *args, **kwargs)

    monkeypatch.setattr(functional, "linear", linear)
    model(
        **inputs,
        ttt_steps=7,
        leagle_masked_head=True,
        leagle_loss_chunk_size=3,
        collect_metrics=False,
        return_predictions=False,
        loss_config=resolve_training_loss("kl_div", loss_implementation="eager"),
    )
    docs = inputs["document_ids"]
    valid = (docs >= 0) & inputs["loss_mask"].bool()
    expected = sum(
        int((valid[:, step:] & (docs[:, : 9 - step] == docs[:, step:])).sum())
        for step in range(7)
    )
    assert sum(rows) == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_empty_selection_keeps_zero_gradients_and_optimizer_state(dtype):
    base = make_model(layers=2, loops=2, attention="sdpa", candidate_head=True).to(
        dtype
    )
    base.lm_head.requires_grad_(True)
    inputs = packed_batch()
    inputs["loss_mask"].zero_()
    for name in ("hidden_states", "verifier_last_hidden_states"):
        inputs[name] = inputs[name].to(dtype)
    results = []
    for flags in ({}, switches(True, True, True)):
        model = copy.deepcopy(base)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        _, loss, _ = model(
            **inputs,
            ttt_steps=4,
            return_predictions=False,
            loss_config=resolve_training_loss("kl_div", loss_implementation="eager"),
            **flags,
        )
        loss.backward()
        grads = {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        optimizer.step()
        results.append((grads, model.state_dict()))
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)


@pytest.mark.parametrize("teacher_forcing", [False, True])
def test_masked_path_keeps_public_predictions_and_autoregression(teacher_forcing):
    base = make_model(layers=1, loops=2, attention="sdpa", candidate_head=True)
    base.config.norm_output = False
    inputs = packed_batch()
    kwargs = {
        "ttt_steps": 11,
        "teacher_forcing": teacher_forcing,
        "loss_config": resolve_training_loss("kl_div", loss_implementation="eager"),
    }
    expected = copy.deepcopy(base)(**inputs, **kwargs)
    actual = base(**inputs, **kwargs, **switches(True, True, True))
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-4)


def test_teacher_cache_budget_partial_hits_and_misses():
    hidden, weight = torch.randn(1, 9, 16), torch.randn(24, 16)
    budget = 4 * (24 * 4 + 8)
    cache = TeacherCache(hidden, weight, budget_bytes=budget, chunk_size=3)
    assert cache.probs.numel() * 4 + cache.ids.numel() * 8 <= budget
    logits = functional.linear(hidden, weight)
    expected_p, expected_ids = logits.softmax(-1), logits.argmax(-1)
    for start, end in ((0, 3), (2, 7), (5, 9)):
        actual = cache.get(hidden[:, start:end], weight, start=start)
        torch.testing.assert_close(
            actual, (expected_p[:, start:end], expected_ids[:, start:end])
        )
    for indices in (torch.tensor([0, 2, 5, 8]), torch.tensor([5, 8])):
        actual = cache.get(hidden[:, indices], weight, indices=indices)
        torch.testing.assert_close(
            actual, (expected_p[:, indices], expected_ids[:, indices])
        )
    with pytest.raises(ValueError, match="budget"):
        TeacherCache(hidden, weight, budget_bytes=1, chunk_size=3)


def test_full_teacher_cache_projects_once_and_never_again_in_backward(monkeypatch):
    model, inputs = make_model(layers=1, loops=1, attention="sdpa"), packed_batch()
    rows = []
    original = functional.linear

    def linear(hidden, weight, *args, **kwargs):
        if weight is model.verifier_lm_head.weight:
            rows.append(hidden.shape[1])
        return original(hidden, weight, *args, **kwargs)

    monkeypatch.setattr(functional, "linear", linear)
    _, loss, _ = model(
        **inputs,
        ttt_steps=7,
        leagle_teacher_cache=True,
        leagle_loss_chunk_size=3,
        return_predictions=False,
        loss_config=resolve_training_loss("kl_div", loss_implementation="eager"),
    )
    assert sum(rows) == 9
    loss.backward()
    assert sum(rows) == 9
    assert model.verifier_lm_head.weight.grad is None


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("rows", [1, 4])
def test_partial_cache_with_checkpointed_backward_matches_uncached(
    rows, masked, monkeypatch
):
    model = make_model(layers=1, loops=2, attention="sdpa", candidate_head=True)
    model.set_activation_checkpointing(True)
    reference, inputs = copy.deepcopy(model), packed_batch()
    original = TeacherCache.__init__

    def bounded(cache, hidden, weight, **kwargs):
        kwargs["budget_bytes"] = rows * (weight.shape[0] * 4 + 8)
        original(cache, hidden, weight, **kwargs)

    monkeypatch.setattr(TeacherCache, "__init__", bounded)
    args = {
        "ttt_steps": 7,
        "return_predictions": False,
        "loss_config": resolve_training_loss("kl_div", loss_implementation="eager"),
    }
    expected = reference(**inputs, **args)
    actual = model(**inputs, **args, **switches(True, True, masked))
    expected[1].backward()
    actual[1].backward()
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-4)
    for (name, observed), (_, wanted) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        torch.testing.assert_close(
            observed.grad, wanted.grad, atol=3e-5, rtol=5e-4, msg=name
        )


def test_invalid_head_options_fail_explicitly():
    model, inputs = make_model(attention="sdpa"), packed_batch()
    with pytest.raises(ValueError, match="candidate head"):
        model(**inputs, leagle_candidate_grad_addmm=True)
    with pytest.raises(ValueError, match="budget"):
        model(**inputs, leagle_teacher_cache=True, leagle_teacher_cache_mb=0)
    with pytest.raises(ValueError, match="eager"):
        model(
            **inputs,
            leagle_teacher_cache=True,
            loss_config=resolve_training_loss("kl_div", loss_implementation="fused"),
        )
