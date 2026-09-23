"""Local experiments are explicit configs, not replacements for upstream examples."""

import sys
import warnings
from pathlib import Path

import pytest

from speculators.train import legacy_cli
from speculators.train.config import TrainConfig

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "examples/train/configs/local/dspark_qwen3_0_6b_sharegpt.yaml"

# Independent argv from the historical Qwen3-0.6B launcher, before its replacement
# by the upstream example. Keep the reference values out of the YAML loader.
HISTORICAL_ARGS = [
    "--verifier-name-or-path",
    "Qwen/Qwen3-0.6B",
    "--data-path",
    "./output/dspark_qwen3_0_6b_sharegpt",
    "--vllm-endpoint",
    "http://localhost:8000/v1",
    "--save-path",
    "./output/dspark_qwen3_0_6b_sharegpt/checkpoints",
    "--draft-vocab-size",
    "32000",
    "--epochs",
    "10",
    "--lr",
    "3e-4",
    "--total-seq-len",
    "4096",
    "--speculator-type",
    "dspark",
    "--block-size",
    "7",
    "--max-anchors",
    "3072",
    "--num-layers",
    "5",
    "--target-layer-ids",
    "2",
    "14",
    "25",
    "--markov-rank",
    "256",
    "--markov-head-type",
    "vanilla",
    "--enable-confidence-head",
    "--confidence-head-with-markov",
    "--loss-fn",
    '{"ce": 0.1, "tv": 0.9}',
    "--confidence-head-alpha",
    "1.0",
    "--confidence-loss-weighting",
    "match-draft",
    "--on-missing",
    "generate",
    "--on-generate",
    "delete",
]


def resolve_local(*overrides):
    # Unknown/misnested YAML fields must not be silently ignored by the recipe.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return TrainConfig.resolve(["--config", str(CONFIG), *overrides])


def test_local_config_preserves_historical_resolved_training_values(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", *HISTORICAL_ARGS])
    historical = legacy_cli.parse_train_args()
    expected = TrainConfig.from_flat(
        {**vars(historical), "training_recipe": "legacy"}
    ).flatten()
    actual = resolve_local().flatten()
    # New experiment directories deliberately avoid the upstream baseline's data
    # and auto-resume checkpoints. Inactive connector options are not YAML fields.
    excluded = {"data_path", "save_path"} | {
        key for key in expected if key.startswith("mooncake_")
    }
    assert {key: value for key, value in actual.items() if key not in excluded} == {
        key: value for key, value in expected.items() if key not in excluded
    }


def test_local_config_is_explicit_and_has_isolated_output_paths():
    cfg = resolve_local()
    assert cfg.training_recipe == "legacy"
    assert cfg.provenance["training_recipe"] == "yaml"
    assert cfg.loss.loss_implementation == "legacy"
    assert cfg.optimizer.muon_lr == pytest.approx(0.003)
    assert cfg.dflash.dflash_decay_gamma == 7.0
    assert cfg.data.data_path == "./output/dspark_qwen3_0_6b_sharegpt_local"
    assert cfg.trainer.save_path == f"{cfg.data.data_path}/checkpoints"


def test_local_config_cli_overrides_rederive_legacy_defaults():
    cfg = resolve_local("--lr", "0.0006", "--block-size", "11", "--epochs", "2")
    assert cfg.optimizer.lr == pytest.approx(0.0006)
    assert cfg.optimizer.muon_lr == pytest.approx(0.006)
    assert cfg.dflash.block_size == 11
    assert cfg.dflash.dflash_decay_gamma == 11.0
    assert cfg.trainer.epochs == 2
    assert cfg.draft.num_layers == 5
    assert cfg.provenance["lr"] == "flag"
    assert cfg.provenance["num_layers"] == "yaml"


def test_local_config_keeps_explicit_optimizer_and_decay_overrides():
    cfg = resolve_local("--muon-lr", "0.002", "--dflash-decay-gamma", "3")
    assert cfg.optimizer.muon_lr == pytest.approx(0.002)
    assert cfg.dflash.dflash_decay_gamma == 3.0


def test_local_config_can_explicitly_select_upstream_policy():
    cfg = resolve_local("--training-recipe", "upstream")
    assert cfg.training_recipe == "upstream"
    assert cfg.loss.loss_implementation == "fused"
    assert cfg.optimizer.muon_lr == cfg.optimizer.lr
    # Explicit experimental settings survive a change in the default policy.
    assert cfg.trainer.epochs == 10
    assert cfg.dflash.block_size == 7
    assert cfg.draft.num_layers == 5


def test_local_config_roundtrips_through_run_yaml(tmp_path):
    cfg = resolve_local()
    saved = tmp_path / "run.yaml"
    saved.write_text(cfg.dump_yaml(), encoding="utf-8")
    restored = TrainConfig.resolve(["--config", str(saved)])
    assert restored.flatten() == cfg.flatten()
