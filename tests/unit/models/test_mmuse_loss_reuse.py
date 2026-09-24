"""MMuse's legacy losses reuse upstream primitives, not upstream dispatch policy."""

from functools import partial
from inspect import signature

import pytest
import torch

from speculators.losses import eager, utils
from speculators.models import metrics


@pytest.mark.parametrize(
    ("name", "owner"),
    [
        ("kl_div_loss", eager),
        ("reverse_kl_div_loss", eager),
        ("ce_loss", eager),
        ("tv_loss", eager),
        ("lk_hybrid_loss", eager),
        ("dflash_loss_decay", utils),
        ("exp_loss_decay", utils),
        ("dpace_loss_decay", utils),
    ],
)
def test_legacy_imports_reexport_shared_primitives(name, owner):
    assert getattr(metrics, name) is getattr(owner, name)


def test_numerically_distinct_legacy_policies_are_not_replaced():
    assert metrics.js_div_loss is not eager.js_div_loss
    assert metrics.neg_log_acceptance_loss is not eager.neg_log_acceptance_loss
    assert metrics.compound_loss is not utils.compound_loss
    assert signature(metrics.loss_function).parameters["loss_fn"].default is (
        eager.kl_div_loss
    )
    for spec, expected in (
        ("jsd", metrics.js_div_loss),
        ("tv", metrics.tv_loss_fused_or_eager),
        ("nla", metrics.nla_loss_fused_or_eager),
    ):
        assert metrics.resolve_training_loss(spec, training_recipe="legacy")[spec] == (
            expected,
            1.0,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "spec", ["kl_div", "rkl", "ce", "tv", "lk_hybrid", "jsd", "nla"]
)
def test_legacy_cpu_forward_backward_does_not_select_fused_kernels(
    dtype, spec, monkeypatch
):
    def forbidden_kernel(_name):
        pytest.fail("A legacy CPU loss selected a fused kernel")

    monkeypatch.setattr(utils, "_fused_kernel", forbidden_kernel)
    monkeypatch.setattr(metrics, "_fused_kernel", forbidden_kernel)
    logits = torch.tensor(
        [[[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]]], dtype=dtype, requires_grad=True
    )
    targets = logits.detach().flip(-1)
    loss, _ = metrics.compound_loss(
        logits,
        targets,
        torch.ones(1, 2),
        torch.arange(2).unsqueeze(0),
        metrics.resolve_training_loss(spec, training_recipe="legacy"),
        decay_fn=partial(metrics.dflash_loss_decay, gamma=7.0, sample_from_anchor=True),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
