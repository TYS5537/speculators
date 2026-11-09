"""Stage-two gradients, frozen states and output-head-only work contracts."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from speculators.models.leagle.candidate_training import prepare_candidate_training
from speculators.models.leagle.objective import select_candidate_positions
from speculators.models.leagle.svd_head import svd_factors
from speculators.train.optimizers import build_optimizers
from tests.unit.models.test_leagle import make_model
from tests.unit.models.test_leagle_training_diagnostics import packed_batch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(723)
    torch.set_num_threads(1)


def stage_two(*, reduced=False, initialization="svd"):
    model = make_model(layers=1, loops=2, attention="sdpa", reduced=reduced)
    model.config.candidate_head = True
    prepare_candidate_training(model, initialization=initialization)
    return model.train()


@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("initialization", ["svd", "random"])
def test_initialization_preserves_all_source_weights_and_rng(reduced, initialization):
    model = make_model(attention="sdpa", reduced=reduced)
    original = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    model.config.candidate_head = True
    prepare_candidate_training(model, initialization=initialization)
    assert torch.equal(torch.get_rng_state(), rng)
    for name, before in original.items():
        torch.testing.assert_close(model.state_dict()[name], before, rtol=0, atol=0)
    if initialization == "svd":
        up, down = svd_factors(model.lm_head.weight, model.config.candidate_head_rank)
        torch.testing.assert_close(
            model.candidate_head.up_proj.weight, up, rtol=0, atol=0
        )
        torch.testing.assert_close(
            model.candidate_head.down_proj.weight, down, rtol=0, atol=0
        )
    assert {name for name, p in model.named_parameters() if p.requires_grad} == {
        "candidate_head.down_proj.weight",
        "candidate_head.up_proj.weight",
    }
    for mode in (True, False, True):
        model.train(mode)
        assert model.candidate_head.training is mode
        assert not model.layers.training
        assert all(not layer.training for layer in model.layers)
    assert "candidate_head_only" not in model.config.to_dict()
    assert "candidate_head_init" not in model.config.to_dict()


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("stride", [1, 3])
def test_candidate_loss_and_gradients_match_independent_native_reference(
    recompute, stride
):
    model = stage_two(initialization="random")
    inputs = packed_batch()
    # Stage two does not depend on target final hidden states, even when supplied.
    inputs["verifier_last_hidden_states"].fill_(float("nan"))
    hidden = []
    hook = model.norm.register_forward_hook(
        lambda _m, _a, out: hidden.append(out.detach())
    )
    _, actual, metrics = model(
        **inputs,
        ttt_steps=4,
        ttt_step_loss_decay=0.7,
        return_predictions=False,
        candidate_head_train_stride=stride,
        leagle_candidate_backward_recompute=recompute,
    )
    hook.remove()
    head = copy.deepcopy(model.candidate_head)
    valid = (inputs["document_ids"] >= 0) & inputs["loss_mask"].bool()
    expected = torch.zeros(())
    total = 0
    for step, value in enumerate(hidden):
        width = value.shape[1] - step
        mask = valid[:, :width] & valid[:, step:]
        mask &= inputs["document_ids"][:, :width] == inputs["document_ids"][:, step:]
        mask = select_candidate_positions(mask, stride, step)
        coarse = head.coarse_logits(value[:, :width])
        teacher = value[:, :width] @ model.lm_head.weight.T
        log_p, log_q = teacher.log_softmax(-1), coarse.log_softmax(-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)
        expected = expected + (kl * mask).sum() / mask.sum().clamp_min(1) * 0.7**step
        total += mask.sum()
    assert metrics["candidate_kl_total"] == total
    assert "draft_loss_sum" not in metrics
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    (actual * 0.37).backward()
    (expected * 0.37).backward()
    for ours, reference in zip(
        model.candidate_head.parameters(), head.parameters(), strict=True
    ):
        torch.testing.assert_close(ours.grad, reference.grad, rtol=5e-5, atol=2e-7)
    assert all(
        p.grad is None
        for name, p in model.named_parameters()
        if not name.startswith("candidate_head.")
    )


@pytest.mark.parametrize("precision", ["fp32", "bf16", "amp"])
@pytest.mark.parametrize("recompute", [False, True])
def test_updates_only_change_head_with_empty_supervision_and_validation(
    precision, recompute
):
    model = stage_two()
    if precision == "bf16":
        model = model.to(torch.bfloat16)
    before = copy.deepcopy(model.state_dict())
    optimizer = build_optimizers(
        model, SimpleNamespace(optimizer="adamw", lr=0.003, weight_decay=0.1)
    )[0]
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.candidate_head.parameters()
    }
    inputs = packed_batch()
    inputs.pop("verifier_last_hidden_states")
    inputs["hidden_states"] = (
        inputs["hidden_states"].to(model.lm_head.weight.dtype).requires_grad_()
    )
    for step in range(3):
        current = {name: value.clone() for name, value in inputs.items()}
        if step == 1:
            current["loss_mask"].zero_()
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast("cpu", dtype=torch.bfloat16)
            if precision == "amp"
            else nullcontext()
        )
        with context:
            _, loss, _ = model(
                **current,
                ttt_steps=3,
                return_predictions=False,
                candidate_head_train_stride=3,
                leagle_candidate_backward_recompute=recompute,
            )
        loss.backward()
        assert inputs["hidden_states"].grad is None
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.candidate_head.parameters()
        )
        if step == 1:
            assert loss == 0
            assert all(
                p.grad.count_nonzero() == 0 for p in model.candidate_head.parameters()
            )
        optimizer.step()
    assert any(
        not torch.equal(model.state_dict()[name], value)
        for name, value in before.items()
        if name.startswith("candidate_head.")
    )
    for name, value in before.items():
        if not name.startswith("candidate_head."):
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        baseline = model(
            **inputs, return_predictions=False, candidate_head_train_stride=1
        )
        sampled = model(
            **inputs, return_predictions=False, candidate_head_train_stride=3
        )
    torch.testing.assert_close(baseline, sampled, rtol=0, atol=0)


def test_head_only_skips_target_head_main_losses_and_decoder_backward(monkeypatch):
    model = stage_two().train()
    model.set_activation_checkpointing(True)

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "Stage two must not compute the target/main loss or decoder checkpoint"
        )

    monkeypatch.setattr(model.verifier_lm_head, "forward", forbidden)
    monkeypatch.setattr("speculators.models.leagle.core.TrainingObjective", forbidden)
    monkeypatch.setattr("speculators.models.leagle.core.checkpoint", forbidden)
    states = []
    hook = model.layers[0].register_forward_hook(
        lambda _m, _a, out: states.append(out[0].requires_grad)
    )
    _, loss, metrics = model(**packed_batch(), return_predictions=False, ttt_steps=3)
    hook.remove()
    loss.backward()
    assert states
    assert not any(states)
    assert "candidate_recall_sum" in metrics


def test_existing_head_is_never_reinitialized():
    model = make_model(candidate_head=True)
    before = copy.deepcopy(model.candidate_head.state_dict())
    prepare_candidate_training(model, initialization="svd")
    torch.testing.assert_close(
        model.candidate_head.state_dict(), before, rtol=0, atol=0
    )
