"""Isolate optimizer grouping without silently changing the training recipe."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from speculators.models.dspark.model_definitions import MarkovHead
from speculators.train.config import TrainConfig
from speculators.train.optimizers import build_optimizers
from speculators.train.trainer import TrainerConfig


def build_fixture(recipe, policy, optimizer="muon"):
    model = nn.ModuleDict(
        {
            "backbone": nn.Linear(4, 4),
            "correction_markov_head": MarkovHead(
                verifier_vocab_size=16,
                draft_vocab_size=16,
                markov_rank=4,
                hidden_size=4,
                init_std=None,
            ),
        }
    )
    cfg = TrainerConfig(
        lr=6e-5,
        num_epochs=10,
        save_path="unused",
        optimizer=optimizer,
        muon_lr=6e-4,
        training_recipe=recipe,
        muon_parameter_policy=policy,
    )
    return model, cfg


@pytest.mark.parametrize(
    ("recipe", "policy", "markov_class"),
    [
        ("legacy", None, torch.optim.Muon),
        ("upstream", None, torch.optim.AdamW),
        ("legacy", "upstream", torch.optim.AdamW),
        ("upstream", "legacy", torch.optim.Muon),
    ],
)
def test_explicit_policy_overrides_only_parameter_grouping(
    recipe, policy, markov_class
):
    model, cfg = build_fixture(recipe, policy)
    weights = {name: value.detach().clone() for name, value in model.named_parameters()}
    rng = torch.get_rng_state()
    optimizers = build_optimizers(model, cfg)
    owners = {
        id(parameter): (type(optimizer), group["lr"], group["weight_decay"])
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert len(owners) == len(weights)
    for name, parameter in model.named_parameters():
        expected = (
            markov_class
            if "markov_w" in name
            else (torch.optim.AdamW if name.endswith("bias") else torch.optim.Muon)
        )
        lr, decay = (6e-4, 0.1) if expected is torch.optim.Muon else (6e-5, 0.01)
        assert owners[id(parameter)] == (expected, lr, decay)
        torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    # Exercise both real CPU optimizers, not just the classification helper.
    sum(parameter.square().mean() for parameter in model.parameters()).backward()
    for optimizer in optimizers:
        optimizer.step()
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all()
        assert not torch.equal(parameter, weights[name])


def test_adamw_runs_remain_all_adamw_with_an_explicit_policy():
    model, cfg = build_fixture("legacy", "upstream", optimizer="adamw")
    optimizers = build_optimizers(model, cfg)
    assert len(optimizers) == 1
    assert isinstance(optimizers[0], torch.optim.AdamW)
    assert len(optimizers[0].param_groups[0]["params"]) == len(list(model.parameters()))


def test_old_config_objects_without_policy_keep_their_grouping():
    model, cfg = build_fixture("legacy", None)
    old = SimpleNamespace(
        **{
            key: value
            for key, value in cfg._asdict().items()
            if key != "muon_parameter_policy"
        }
    )
    optimizers = build_optimizers(model, old)
    markov = model["correction_markov_head"].markov_w1.weight
    assert any(
        parameter is markov for parameter in optimizers[0].param_groups[0]["params"]
    )
    assert isinstance(optimizers[0], torch.optim.Muon)


def test_policy_override_does_not_change_loss_or_lr_defaults():
    old = TrainConfig.from_flat({"speculator_type": "mmuse", "lr": 6e-5}).flatten()
    new = TrainConfig.from_flat(
        {"speculator_type": "mmuse", "lr": 6e-5, "muon_parameter_policy": "upstream"}
    ).flatten()
    assert {key for key in old if old[key] != new[key]} == {"muon_parameter_policy"}
    assert new["training_recipe"] == new["loss_implementation"] == "legacy"
    assert new["muon_lr"] == pytest.approx(6e-4)
