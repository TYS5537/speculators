"""Standalone DFlash2 CLI, checkpoint and shared-data migration contracts."""

import argparse
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from speculators.train import legacy_cli, model_config, vocab_setup
from speculators.train.config import TrainConfig

_STRUCTURAL_VALUES = {
    "conv_kernel_size": (2, 3),
    "conv_group_size": (16, 8),
    "selector_rank": (256, 128),
    "selector_top_k": (16, 12),
}


def _argv(*extra):
    return [
        "--verifier-name-or-path",
        "unused-verifier",
        "--speculator-type",
        "dflash2",
        *extra,
    ]


@pytest.mark.parametrize("typed", [False, True])
def test_legacy_recipe_preserves_standalone_dflash2_defaults(typed):
    if typed:
        values = TrainConfig.resolve(_argv("--training-recipe", "legacy")).flatten()
    else:
        values = vars(legacy_cli.parse_train_args(_argv()))
    assert values["num_layers"] == 5
    assert values["block_size"] == 8
    assert values["loss_fn"] == "kl_div"
    assert values["sliding_window_non_causal"] is True
    assert values["sample_from_anchor"] in (None, False)


def test_legacy_parser_accepts_old_dflash2_recipe_and_tracks_checkpoint_fields():
    extra = []
    for field, (_, value) in _STRUCTURAL_VALUES.items():
        extra.extend(["--" + field.replace("_", "-"), str(value)])
    args = legacy_cli.parse_train_args(
        _argv(
            *extra,
            "--selector-loss-alpha",
            "0.1",
            "--num-layers",
            "3",
            "--no-sliding-window-non-causal",
            "--sample-from-anchor",
        )
    )
    for field, (_, value) in _STRUCTURAL_VALUES.items():
        assert getattr(args, field) == value
        assert field in args._provided_model_config_dests
    assert args.selector_loss_alpha == pytest.approx(0.1)
    assert args.num_layers == 3
    assert args.sliding_window_non_causal is False
    assert args.sample_from_anchor is True


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize(
    ("flag", "value"),
    [("--" + field.replace("_", "-"), "0") for field in _STRUCTURAL_VALUES]
    + [("--selector-loss-alpha", "-0.1")],
)
def test_invalid_dflash2_dimensions_fail_before_model_initialization(
    typed, flag, value
):
    resolve = TrainConfig.resolve if typed else legacy_cli.parse_train_args
    with pytest.raises(SystemExit) as exc:
        resolve(_argv(flag, value))
    assert exc.value.code == 2


@pytest.mark.parametrize("field", _STRUCTURAL_VALUES)
def test_checkpoint_rejects_changed_standalone_structure_without_mutating(field):
    saved_value, requested = _STRUCTURAL_VALUES[field]
    args = argparse.Namespace(
        **{field: requested}, _provided_model_config_dests={field}
    )
    config = SimpleNamespace(speculators_model_type="dflash2", **{field: saved_value})
    with pytest.raises(ValueError, match="checkpoint architecture"):
        model_config.reconcile_pretrained_config_args(
            args, config, logger=logging.getLogger(__name__)
        )
    assert getattr(args, field) == requested
    assert getattr(config, field) == saved_value


@pytest.mark.parametrize("field", _STRUCTURAL_VALUES)
def test_checkpoint_inherits_nondefault_standalone_structure(field):
    default_value, saved_value = _STRUCTURAL_VALUES[field]
    args = argparse.Namespace(
        **{field: default_value}, _provided_model_config_dests=set()
    )
    config = SimpleNamespace(speculators_model_type="dflash2", **{field: saved_value})
    model_config.reconcile_pretrained_config_args(
        args, config, logger=logging.getLogger(__name__)
    )
    assert getattr(args, field) == saved_value


def _vocab_args(tmp_path, **overrides):
    values = {
        "speculator_type": "dflash2",
        "from_pretrained": "",
        "training_recipe": "legacy",
        "data_path": str(tmp_path),
        "verifier_name_or_path": "unused-verifier",
        "d2t_path": None,
        "t2d_path": None,
        "draft_vocab_size": None,
        "token_freq_path": None,
        "trust_remote_code": False,
    }
    return argparse.Namespace(**(values | overrides))


@pytest.mark.parametrize("recipe", ["legacy", "upstream"])
@pytest.mark.parametrize("from_checkpoint", [False, True])
def test_full_vocab_bypasses_shared_cache_for_resolved_dflash2(
    tmp_path, monkeypatch, recipe, from_checkpoint
):
    # Corrupt cache files must never even be read, including on resume when the
    # user leaves --speculator-type at its unrelated parser default.
    for name in ("d2t.npy", "t2d.npy", "d2t-None.npy", "t2d-None.npy"):
        (tmp_path / name).write_bytes(b"shared DSpark cache")
    forbidden = Mock(side_effect=AssertionError("DFlash2 must not load vocab mappings"))
    monkeypatch.setattr(vocab_setup, "_load_mappings", forbidden)
    monkeypatch.setattr(vocab_setup, "is_distributed", lambda: False)
    checkpoint = Mock(return_value=SimpleNamespace(speculators_model_type="dflash2"))
    monkeypatch.setattr(
        vocab_setup.SpeculatorModelConfig, "from_pretrained", checkpoint
    )
    verifier = Mock(return_value=SimpleNamespace(vocab_size=151936))
    monkeypatch.setattr(vocab_setup, "get_verifier_config", verifier)
    args = _vocab_args(
        tmp_path,
        training_recipe=recipe,
        trust_remote_code=True,
        from_pretrained="checkpoint" if from_checkpoint else "",
        speculator_type="eagle3" if from_checkpoint else "dflash2",
    )

    assert vocab_setup.parse_vocab_mappings(args) == (None, None, 151936)
    forbidden.assert_not_called()
    verifier.assert_called_once_with("unused-verifier", trust_remote_code=True)
    if from_checkpoint:
        checkpoint.assert_called_once_with("checkpoint")
    else:
        checkpoint.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [{"draft_vocab_size": 32000}, {"d2t_path": "d2t.npy", "t2d_path": "t2d.npy"}],
)
def test_dflash2_rejects_explicit_pruned_vocab_before_reading_files(
    tmp_path, monkeypatch, overrides
):
    forbidden = Mock(side_effect=AssertionError("Invalid vocabulary requires no I/O"))
    monkeypatch.setattr(vocab_setup, "get_verifier_config", forbidden)
    monkeypatch.setattr(vocab_setup, "is_distributed", lambda: False)
    with pytest.raises(ValueError, match="full verifier vocabulary"):
        vocab_setup.parse_vocab_mappings(_vocab_args(tmp_path, **overrides))
    forbidden.assert_not_called()
