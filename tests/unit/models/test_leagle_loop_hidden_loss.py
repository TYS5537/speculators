"""Loop HS supervision preserves rollout, masks, head work and native gradients."""

import copy

import pytest
import torch

from speculators.losses import resolve_loss_config
from speculators.models.leagle.loop_hidden_loss import LoopHiddenAlignment

from .test_leagle import batch, make_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(83)
    torch.set_num_threads(1)


def inputs(length=7):
    data = batch(length)
    data["verifier_last_hidden_states"] = torch.randn(1, length, 16, requires_grad=True)
    data["loss_mask"] = torch.ones(1, length, dtype=torch.bool)
    if length == 7:  # Packed fixture with masked prompt and padding.
        data["document_ids"] = torch.tensor([[0, 0, 0, 1, 1, 1, -1]])
        data["position_ids"] = torch.tensor([[1, 2, 3, 1, 2, 3, 0]])
        data["loss_mask"] = torch.tensor([[0, 1, 0, 0, 1, 1, 1]]).bool()
    return data


def settings(**extra):
    return {
        "ttt_steps": 4,
        "ttt_step_loss_decay": 0.7,
        "loss_config": resolve_loss_config(
            '{"ce":0.1,"tv":0.9}', implementation="eager"
        ),
        "teacher_forcing": True,
        **extra,
    }


def reference_loop_loss(model, data, hidden, step, decay=0.7):
    """Independent normalized MSE, shifted/masked per the token objective."""
    with torch.no_grad():
        target = model.verifier_norm(data["verifier_last_hidden_states"])
    docs = data["document_ids"]
    valid = (docs >= 0) & data["loss_mask"]
    width = docs.shape[1] - step
    mask = valid[:, step:] & (docs[:, step:] == docs[:, :width])
    pred = model.norm(hidden[:, :width])
    error = (pred.float() - target[:, step:].float()).square().mean(-1)
    return (
        (error * mask.float() * decay ** torch.full_like(error, step)).sum(-1)
        / (mask.float().sum(-1) + 1e-5)
    ).mean()


def reference_hidden_loss(model, data, visits, steps=4):
    """Full-unroll reference for the scalar objective and FP32 gradients."""
    target = data["verifier_last_hidden_states"]
    total = target.new_zeros((), dtype=torch.float32)
    loops = model.config.num_loops
    for step in range(min(steps, target.shape[1])):
        for loop in range(loops - 1):
            per_step = reference_loop_loss(
                model, data, visits[step * loops + loop], step
            )
            total = total + per_step / (loops - 1)
    return total


def gradients(model):
    return {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_matches_explicit_intermediate_supervision_and_gradients(
    attention, checkpointed, bf16
):
    model = make_model(layers=2, loops=3, attention=attention)
    reference = copy.deepcopy(model)
    model.set_activation_checkpointing(checkpointed)
    data = inputs()
    visits, head_calls = [], []
    hs_loss = torch.zeros((), dtype=torch.float32)

    def supervise_visit(_module, _args, output):
        nonlocal hs_loss
        step, loop = divmod(len(visits), reference.config.num_loops)
        visits.append(output[0])
        if loop + 1 < reference.config.num_loops:
            # Attach each reference branch when that loop finishes. Deferring
            # all branches until after the unroll changes BF16 backward addition
            # order, even with identical hidden states and identical scalar loss.
            hs_loss = hs_loss + reference_loop_loss(
                reference, data, output[0], step
            ) / (reference.config.num_loops - 1)

    visit_hook = reference.layers[-1].register_forward_hook(supervise_visit)
    hooks = [
        m.lm_head.register_forward_hook(lambda _m, _a, _o: head_calls.append(1))
        for m in (model, reference)
    ]
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            actual, total, metrics = model(
                **data,
                **settings(leagle_loop_hs_loss=True, leagle_loop_hs_loss_weight=0.13),
            )
            expected, main_loss, baseline_metrics = reference(**data, **settings())
            expected_loss = main_loss + 0.13 * hs_loss
    finally:
        visit_hook.remove()
        for hook in hooks:
            hook.remove()
    # Adding supervision must not change the autoregressive states or logits.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(head_calls) == 8  # Four TTT steps on each model, not one per loop.
    torch.testing.assert_close(metrics["draft_loss_sum"], main_loss)
    torch.testing.assert_close(metrics["loop_hs_loss_sum"], hs_loss)
    torch.testing.assert_close(total, expected_loss)
    torch.testing.assert_close(metrics["loss_sum"], total)
    for name, value in baseline_metrics.items():
        if name != "loss_sum":
            torch.testing.assert_close(metrics[name], value, rtol=0, atol=0)
    total.backward()
    expected_loss.backward()
    assert data["verifier_last_hidden_states"].grad is None
    torch.testing.assert_close(
        gradients(model), gradients(reference), rtol=3e-4, atol=2e-5
    )


@pytest.mark.parametrize(
    ("enabled", "weight", "loops"), [(False, 0.1, 3), (True, 0.0, 3), (True, 0.1, 1)]
)
def test_disabled_zero_weight_and_single_loop_are_exact_noops(enabled, weight, loops):
    baseline = make_model(loops=loops, attention="sdpa")
    model = copy.deepcopy(baseline)
    data = inputs()
    expected = baseline(**data, **settings())
    actual = model(
        **data,
        **settings(leagle_loop_hs_loss=enabled, leagle_loop_hs_loss_weight=weight),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual[1].backward()
    expected[1].backward()
    torch.testing.assert_close(gradients(model), gradients(baseline), rtol=0, atol=0)


@pytest.mark.parametrize("empty", [False, True])
def test_sequence_tail_and_empty_supervision(empty):
    model = make_model(layers=1, loops=3, attention="sdpa")
    reference = copy.deepcopy(model)
    data = inputs(length=2)
    if empty:
        data["loss_mask"].zero_()
    visits = []
    hook = reference.layers[-1].register_forward_hook(
        lambda _m, _a, out: visits.append(out[0])
    )
    try:
        _, main, _ = reference(**data, **settings())
    finally:
        hook.remove()
    expected_hs = reference_hidden_loss(reference, data, visits)
    _, total, metrics = model(**data, **settings(leagle_loop_hs_loss=True))
    torch.testing.assert_close(total, main + 0.1 * expected_hs)
    torch.testing.assert_close(metrics["loop_hs_loss_sum"], expected_hs)
    total.backward()
    assert all(torch.isfinite(g).all() for g in gradients(model).values())
    if empty:
        assert total == 0
        assert all(torch.count_nonzero(g) == 0 for g in gradients(model).values())


@pytest.mark.parametrize("low_rank", [False, True])
@pytest.mark.parametrize("reduced", [False, True])
def test_existing_optimized_losses_and_candidate_modes(low_rank, reduced):
    reference = make_model(
        layers=1,
        loops=3,
        attention="sdpa",
        reduced=reduced,
        candidate_head=True,
        candidate_rerank=not low_rank,
    )
    model = copy.deepcopy(reference)
    model.set_activation_checkpointing(True)
    data = inputs()
    expected = reference(**data, **settings(leagle_loop_hs_loss=True))
    actual = model(
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
        gradients(model), gradients(reference), rtol=5e-4, atol=3e-5
    )


def test_intermediate_alignment_is_averaged_and_does_not_detach_student():
    model = make_model(layers=1, loops=4)
    data = inputs(length=3)
    hidden = torch.randn(1, 3, 16, requires_grad=True)
    target = model.verifier_norm(data["verifier_last_hidden_states"])
    objective = LoopHiddenAlignment(
        model, target, data["document_ids"], data["loss_mask"], 1.0
    )
    for _ in range(3):
        objective.add(hidden, 0)
    expected = (model.norm(hidden).float() - target.detach().float()).square().mean()
    # Existing main-loss reduction has a 1e-5 stabilizer in its valid-token count.
    expected = expected * 3 / (3 + 1e-5)
    torch.testing.assert_close(objective.loss, expected)
    objective.loss.backward()
    assert hidden.grad is not None
    assert hidden.grad.norm() > 0
    assert model.norm.weight.grad is not None
    assert data["verifier_last_hidden_states"].grad is None


def test_metrics_can_be_disabled_without_changing_loss_or_gradients():
    model = make_model(loops=3, attention="sdpa")
    reference = copy.deepcopy(model)
    data = inputs()
    _, expected, _ = reference(**data, **settings(leagle_loop_hs_loss=True))
    _, actual, metrics = model(
        **data, **settings(leagle_loop_hs_loss=True, collect_metrics=False)
    )
    assert set(metrics) == {"loss_sum", "loss_total"}
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(gradients(model), gradients(reference), rtol=0, atol=0)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        make_model()(**inputs(), leagle_loop_hs_loss_weight=weight)


def test_requires_teacher_and_backbone_training():
    model = make_model()
    with pytest.raises(ValueError, match="teacher hidden"):
        model(**batch(), leagle_loop_hs_loss=True)
    model.candidate_head_only = True
    with pytest.raises(ValueError, match="candidate-head-only"):
        model(**inputs(), leagle_loop_hs_loss=True)
