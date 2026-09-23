"""Contract tests for the merge boundary between historical and upstream recipes."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from scripts import train as legacy_entry
from speculators.cli.generate_offline_data import _validate_and_publish
from speculators.models.dspark.model_definitions import MarkovHead
from speculators.models.metrics import resolve_loss_config, resolve_training_loss
from speculators.train import cli, vocab_setup
from speculators.train.config import TrainConfig
from speculators.train.optimizers import split_named_params_for_muon


@pytest.mark.parametrize("algorithm", ["eagle3", "dflash", "dspark", "mmuse"])
def test_legacy_entry_preserves_resolved_values_and_yaml(
    tmp_path, monkeypatch, algorithm
):
    argv = [
        "train.py",
        "--verifier-name-or-path",
        "target",
        "--speculator-type",
        algorithm,
        "--lr",
        "0.0006",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = legacy_entry.parse_args()
    captured = Mock()
    monkeypatch.setattr(cli, "main", captured)
    legacy_entry.main(args)
    cfg = captured.call_args.args[0]
    assert cfg.training_recipe == "legacy"
    assert list(cfg.argv) == argv
    flat = cfg.flatten()
    for key, value in vars(args).items():
        if not key.startswith("_") and value is not None:
            assert key in flat, key
            assert flat[key] == value, key
    path = tmp_path / "run.yaml"
    path.write_text(cfg.dump_yaml(), encoding="utf-8")
    restored = TrainConfig.from_sources(cli={}, config_path=str(path), argv=argv)
    # The old parser exposes all connector defaults, even for inactive backends.
    # YAML intentionally records only the selected connector's active settings.
    inactive = {key for key in flat if key.startswith("mooncake_")}
    assert {
        key: value for key, value in restored.flatten().items() if key not in inactive
    } == {key: value for key, value in flat.items() if key not in inactive}


@pytest.mark.parametrize(
    ("recipe", "block", "layers", "loss"),
    [("legacy", 8, 1, "legacy"), ("upstream", 16, 5, "fused")],
)
def test_recipe_defaults_are_separate(recipe, block, layers, loss):
    cfg = TrainConfig(speculator_type="dflash", training_recipe=recipe)
    assert cfg.dflash.block_size == block
    assert cfg.draft.num_layers == layers
    assert cfg.loss.loss_implementation == loss
    ratio = 10 if recipe == "legacy" else 1
    assert cfg.optimizer.muon_lr == ratio * cfg.optimizer.lr


def test_explicit_eager_loss_wins_over_legacy_recipe():
    cfg = TrainConfig.from_flat(
        {"speculator_type": "mmuse", "loss_implementation": "eager"}
    )
    assert cfg.training_recipe == "legacy"
    assert cfg.loss.loss_implementation == "eager"
    old = resolve_loss_config("tv")
    selected = resolve_training_loss("tv", training_recipe="legacy")
    assert selected == old
    assert resolve_training_loss("tv", loss_implementation="eager") != old


def test_legacy_markov_preserves_initialization_and_rng():
    torch.manual_seed(21)
    embedding = nn.Embedding(32, 4)
    projection = nn.Linear(4, 16, bias=False)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(21)
    head = MarkovHead(
        verifier_vocab_size=32,
        draft_vocab_size=16,
        markov_rank=4,
        hidden_size=8,
        init_std=None,
    )
    torch.testing.assert_close(head.markov_w1.weight, embedding.weight, rtol=0, atol=0)
    torch.testing.assert_close(head.markov_w2.weight, projection.weight, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_markov_optimizer_groups_follow_recipe():
    head = MarkovHead(
        verifier_vocab_size=32, draft_vocab_size=16, markov_rank=4, hidden_size=8
    )
    old_muon, old_adam = split_named_params_for_muon(head, training_recipe="legacy")
    new_muon, new_adam = split_named_params_for_muon(head, training_recipe="upstream")
    assert len(old_muon) == len(new_adam) == 2
    assert old_adam == new_muon == []


@pytest.mark.parametrize("explicit", [False, True])
def test_checkpoint_recipe_inference_preserves_explicit_override(monkeypatch, explicit):
    options = {
        "verifier_name_or_path": "target",
        "from_pretrained": "checkpoint",
        "lr": 0.0006,
    }
    if explicit:
        options["training_recipe"] = "upstream"
    cfg = TrainConfig.from_sources(cli=options, argv=["train"])
    monkeypatch.setattr(
        cli.SpeculatorModelConfig,
        "from_pretrained",
        lambda _path: SimpleNamespace(speculators_model_type="mmuse"),
    )
    resolved = cli._resolve_checkpoint_recipe(cfg)
    assert resolved.training_recipe == ("upstream" if explicit else "legacy")
    assert resolved.optimizer.lr == 0.0006
    assert resolved.optimizer.muon_lr == pytest.approx(0.0006 if explicit else 0.006)
    if not explicit:
        assert resolved.speculator_type == "mmuse"
        assert resolved.dflash.block_size == 7


def test_zero_requested_vocab_is_not_treated_as_unspecified(tmp_path):
    np.save(tmp_path / "d2t.npy", np.zeros(4, dtype=np.int64))
    np.save(tmp_path / "t2d.npy", np.ones(4, dtype=np.bool_))
    with pytest.raises(ValueError, match="requires 0"):
        vocab_setup._load_mappings(tmp_path / "d2t.npy", tmp_path / "t2d.npy", 0)


def test_new_offline_entry_cannot_publish_invalid_cache(tmp_path):
    source, target = tmp_path / "server.safetensors", tmp_path / "hs_0.safetensors"
    save_file(
        {
            "token_ids": torch.tensor([1, 2]),
            "hidden_states": torch.full((2, 3, 4), torch.nan),
        },
        source,
    )
    with pytest.raises(ValueError, match="NaN/Inf"):
        _validate_and_publish(str(source), target, {"input_ids": [1, 2]}, True)
    assert source.exists()
    assert not target.exists()


def test_new_offline_entry_publishes_only_valid_multimodal_prefix(tmp_path):
    source, target = tmp_path / "server.safetensors", tmp_path / "hs_0.safetensors"
    save_file(
        {"token_ids": torch.tensor([1, 2, 3]), "hidden_states": torch.ones(3, 3, 4)},
        source,
    )
    _validate_and_publish(
        str(source), target, {"input_ids": [1, 2], "messages": []}, False
    )
    assert not source.exists()
    assert load_file(target)["hidden_states"].shape == (2, 3, 4)
