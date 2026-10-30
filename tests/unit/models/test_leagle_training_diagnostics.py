"""Diagnostic scheduling preserves training values, gradients, updates and RNG."""

import copy
from contextlib import nullcontext

import pytest
import torch

from speculators.models.eagle3 import metrics as eagle_metrics
from speculators.models.eagle3.metrics import EagleTrainingMetrics, compute_metrics
from speculators.models.leagle import LEagleDraftModel
from speculators.models.leagle.training import TVOverlapProbe
from speculators.models.metrics import resolve_training_loss

from .test_leagle import batch, make_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(37)
    torch.set_num_threads(1)


def packed_batch():
    inputs = batch(length=9)
    inputs["document_ids"] = torch.tensor([[0, 0, 0, 0, 1, 1, 1, -1, -1]])
    inputs["loss_mask"] = torch.tensor([[1, 0, 1, 1, 1, 1, 0, 0, 0]])
    inputs["verifier_last_hidden_states"] = torch.randn(1, 9, 16)
    return inputs


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("loss_fn", ["kl_div", '{"ce": 0.1, "tv": 0.9}'])
def test_scheduling_preserves_predictions_loss_gradients_updates_and_rng(
    autocast, checkpointed, candidate, reduced, loss_fn
):
    model = make_model(
        layers=1, loops=2, attention="sdpa", candidate_head=candidate, reduced=reduced
    )
    model.set_activation_checkpointing(checkpointed)
    inputs = packed_batch()
    results = []
    for collect_metrics in (True, False):
        draft = copy.deepcopy(model).train()
        optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-4)
        torch.manual_seed(71)
        context = (
            torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext()
        )
        with context:
            predictions, loss, metrics = draft(
                **inputs,
                ttt_steps=7,
                ttt_step_loss_decay=0.7,
                loss_config=resolve_training_loss(
                    loss_fn, loss_implementation="legacy"
                ),
                collect_metrics=collect_metrics,
            )
        loss.backward()
        gradients = {
            name: p.grad.clone()
            for name, p in draft.named_parameters()
            if p.grad is not None
        }
        optimizer.step()
        results.append(
            (
                predictions,
                loss.detach(),
                gradients,
                draft.state_dict(),
                torch.get_rng_state(),
            )
        )
        assert all(not value.requires_grad for value in metrics.values())
        if collect_metrics:
            assert "accept_len_sum" in metrics
            assert ("candidate_recall_sum" in metrics) is candidate
        else:
            assert set(metrics) == {"loss_sum", "loss_total"}
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)


@pytest.mark.parametrize("implementation", ["legacy", "eager"])
@pytest.mark.parametrize("reduced", [False, True])
def test_tv_reuse_matches_original_metrics_without_extra_overlap_pass(
    implementation, reduced, monkeypatch
):
    model = make_model(layers=1, loops=2, reduced=reduced)
    inputs = packed_batch()
    scores = []
    handle = model.lm_head.register_forward_hook(
        lambda _module, _args, output: scores.append(output.detach())
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            eagle_metrics,
            "_distribution_overlap",
            lambda *_: pytest.fail(
                "TV metrics recomputed full-vocabulary probabilities"
            ),
        )
        patch.setattr(
            eagle_metrics,
            "compute_accuracy_single_step",
            lambda *_: pytest.fail("Discarded legacy metrics were computed"),
        )
        try:
            _, loss, metrics = model(
                **inputs,
                ttt_steps=7,
                loss_config=resolve_training_loss(
                    '{"ce": 0.1, "tv": 0.9}', loss_implementation=implementation
                ),
            )
        finally:
            handle.remove()
    with torch.no_grad():
        targets = model.verifier_lm_head(
            model.verifier_norm(inputs["verifier_last_hidden_states"])
        )
    reference = EagleTrainingMetrics(inputs["document_ids"], inputs["loss_mask"], 7)
    for step, logits in enumerate(scores):
        reference.update(logits, targets, step)
    torch.testing.assert_close(metrics, reference.finish(loss), atol=1e-6, rtol=1e-6)


def test_unlogged_training_skips_topk_overlap_and_legacy_diagnostics(monkeypatch):
    model = make_model(layers=1, loops=2, candidate_head=True)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unlogged training executed a diagnostic-only operation")

    monkeypatch.setattr(torch.Tensor, "topk", forbidden)
    monkeypatch.setattr(eagle_metrics, "_distribution_overlap", forbidden)
    monkeypatch.setattr(eagle_metrics, "compute_accuracy_single_step", forbidden)
    _, loss, metrics = model(
        **packed_batch(),
        ttt_steps=7,
        collect_metrics=False,
        loss_config=resolve_training_loss("kl_div", loss_implementation="legacy"),
    )
    loss.backward()
    assert set(metrics) == {"loss_sum", "loss_total"}
    assert all(p.grad is not None for p in model.candidate_head.parameters())


@pytest.mark.parametrize("implementation", ["legacy", "eager"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("spec", ["kl_div", "tv", '{"ce": 0.1, "tv": 0.9}'])
def test_loss_only_path_preserves_legacy_loss_and_gradient(implementation, empty, spec):
    scores = torch.randn(1, 9, 24)
    targets = torch.randn_like(scores)
    mask = torch.tensor([[1, 0, 1, 1, 0, 1, 1, 0, 0]])
    if empty:
        mask.zero_()
    results = []
    for collect in (True, False):
        logits = scores.clone().requires_grad_()
        loss, metrics = compute_metrics(
            logits,
            targets,
            mask,
            mask.bool().clone(),
            2,
            0.7,
            loss_config=resolve_training_loss(spec, loss_implementation=implementation),
            collect_metrics=collect,
        )
        loss.backward()
        results.append((loss.detach(), logits.grad))
        assert bool(metrics) is collect
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)


def test_unknown_tv_callable_keeps_original_metric_fallback():
    custom = lambda logits, targets: logits.sum(-1)  # noqa: E731
    config = {"tv": (custom, 0.9)}
    probe = TVOverlapProbe(config)
    assert probe.loss_config is config
    assert probe.overlap is None


def test_training_and_validation_default_to_available_metrics():
    training, validation = LEagleDraftModel.get_trainer_kwargs(
        loss_fn="tv", loss_implementation="legacy"
    )
    assert training["collect_metrics"] is True
    assert validation["collect_metrics"] is True
    assert training is not validation
