"""Historical CLI policy stays distinct when argument metadata shares a schema."""

import argparse

import pytest

from speculators.train import legacy_cli
from speculators.train.config import resolution
from speculators.train.config.resolution import build_parser


def _argv(*flags):
    return ["--verifier-name-or-path", "unused-verifier", *flags]


@pytest.mark.parametrize(
    "parser_factory", [legacy_cli.build_train_parser, build_parser]
)
def test_runtime_optimization_flags_share_names_and_types(parser_factory):
    args = parser_factory().parse_args(
        _argv(
            "--profile-freq",
            "100",
            "--activation-checkpointing",
            "--activation-checkpoint-layers",
            "0",
            "2",
        )
    )
    assert args.profile_freq == 100
    assert args.activation_checkpoint_layers == [0, 2]
    assert args.activation_checkpointing is True


def test_runtime_optimization_defaults_keep_all_layer_policy():
    args = legacy_cli.build_train_parser().parse_args(_argv())
    assert args.profile_freq == 0
    assert args.activation_checkpoint_layers is None


def test_legacy_optimizer_defaults_preserve_existing_runs():
    args = legacy_cli.build_train_parser().parse_args(_argv())
    assert (args.adam_beta1, args.adam_beta2) == (0.9, 0.999)
    assert args.max_grad_norm == 1.0


def test_legacy_raw_defaults_do_not_inherit_schema_policy():
    parser = legacy_cli.build_train_parser()
    historical = {
        "verifier_name_or_path": None,
        "lr": 1e-4,
        "num_layers": 1,
        "loss_fn": "kl_div",
        "block_size": 8,
        "max_anchors": 3072,
        "per_position_loss_weight": "fixed-exp-decay",
        "sliding_window_non_causal": False,
    }
    assert {name: parser.get_default(name) for name in historical} == historical
    raw = vars(parser.parse_args(_argv()))
    assert {name: raw[name] for name in historical} == {
        **historical,
        "verifier_name_or_path": "unused-verifier",
    }
    # These are deliberately pre-finalization values even for the MMuse recipe.
    mmuse = parser.parse_args(_argv("--speculator-type", "mmuse"))
    assert (mmuse.num_layers, mmuse.block_size, mmuse.loss_fn) == (1, 8, "kl_div")


def test_legacy_still_requires_verifier_during_argument_parsing(capsys):
    with pytest.raises(SystemExit) as caught:
        legacy_cli.build_train_parser().parse_args([])
    assert caught.value.code == 2
    assert "--verifier-name-or-path" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0.25", "1", "2.0"])
def test_checkpoint_frequency_keeps_custom_converter_and_float_result(value):
    parser = legacy_cli.build_train_parser()
    action = next(a for a in parser._actions if a.dest == "checkpoint_freq")
    assert action.type is legacy_cli._checkpoint_freq
    parsed = parser.parse_args(_argv("--checkpoint-freq", value))
    assert type(parsed.checkpoint_freq) is float
    assert parsed.checkpoint_freq == float(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("0", "--checkpoint-freq must be > 0"),
        ("-1", "--checkpoint-freq must be > 0"),
        ("1.5", "Values > 1 are treated as epoch counts and must be whole numbers"),
        ("invalid", "invalid _checkpoint_freq value"),
    ],
)
def test_checkpoint_frequency_invalid_values_fail_in_argparse(value, message, capsys):
    with pytest.raises(SystemExit) as caught:
        legacy_cli.build_train_parser().parse_args(_argv("--checkpoint-freq", value))
    assert caught.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("name", ["activation-checkpointing"])
def test_legacy_store_true_flags_do_not_gain_negative_spellings(name, capsys):
    parser = legacy_cli.build_train_parser()
    dest = name.replace("-", "_")
    action = next(a for a in parser._actions if a.dest == dest)
    assert type(action) is argparse._StoreTrueAction
    assert action.option_strings == [f"--{name}"]
    assert getattr(parser.parse_args(_argv()), dest) is False
    assert getattr(parser.parse_args(_argv(f"--{name}")), dest) is True
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(_argv(f"--no-{name}"))
    assert caught.value.code == 2
    assert f"unrecognized arguments: --no-{name}" in capsys.readouterr().err


def test_legacy_sliding_window_keeps_default_and_allows_explicit_causal_override():
    parser = legacy_cli.build_train_parser()
    assert parser.parse_args(_argv()).sliding_window_non_causal is False
    assert parser.parse_args(
        _argv("--sliding-window-non-causal")
    ).sliding_window_non_causal
    assert not parser.parse_args(
        _argv("--no-sliding-window-non-causal")
    ).sliding_window_non_causal


@pytest.mark.parametrize(
    "flag",
    [
        "--config",
        "--dump-config",
        "--training-recipe",
        "--loss-implementation",
        "--muon-parameter-policy",
        "--max-steps",
        "--gradient-checkpointing",
        "--generation-validation-retries",
        "--max-consecutive-generation-failures",
    ],
)
def test_schema_only_flags_do_not_expand_legacy_cli(flag, capsys):
    parser = legacy_cli.build_train_parser()
    assert flag not in parser._option_string_actions
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(_argv(flag))
    assert caught.value.code == 2
    assert f"unrecognized arguments: {flag}" in capsys.readouterr().err


def test_each_legacy_parser_gets_its_own_full_attention_default_list():
    first = legacy_cli.build_train_parser().parse_args(_argv())
    assert first.full_attention_indices == []
    first.full_attention_indices.append(7)
    second = legacy_cli.build_train_parser().parse_args(_argv())
    assert second.full_attention_indices == []
    assert second.full_attention_indices is not first.full_attention_indices
    explicit = legacy_cli.build_train_parser().parse_args(
        _argv("--full-attention-indices", "0", "2")
    )
    assert explicit.full_attention_indices == [0, 2]


def test_draft_architecture_choices_follow_live_registry(monkeypatch):
    first = legacy_cli.build_train_parser()
    registry = {"fixture_decoder": object, "second_decoder": object}
    monkeypatch.setattr(legacy_cli, "DRAFT_ARCH_CONFIGS", registry)
    second = legacy_cli.build_train_parser()
    first_action = next(a for a in first._actions if a.dest == "draft_arch")
    second_action = next(a for a in second._actions if a.dest == "draft_arch")
    assert "fixture_decoder" not in first_action.choices
    assert second_action.choices == list(registry)
    assert (
        second.parse_args(_argv("--draft-arch", "fixture_decoder")).draft_arch
        == "fixture_decoder"
    )


def test_typed_parser_still_suppresses_unsupplied_fields_for_precedence():
    parser = build_parser()
    assert vars(parser.parse_args([])) == {"config": None, "dump_config": False}
    parsed = vars(
        parser.parse_args(
            [
                "--lr",
                "0",
                "--no-activation-checkpointing",
                "--no-sliding-window-non-causal",
                "--full-attention-indices",
                "0",
                "2",
            ]
        )
    )
    assert parsed == {
        "config": None,
        "dump_config": False,
        "lr": 0.0,
        "activation_checkpointing": False,
        "sliding_window_non_causal": False,
        "full_attention_indices": [0, 2],
    }


def test_typed_parser_empty_dynamic_registry_keeps_choices_unrestricted(monkeypatch):
    monkeypatch.setitem(resolution._CLI_CHOICE_REGISTRIES, "hidden_states_backends", {})
    parser = build_parser()
    action = next(a for a in parser._actions if a.dest == "hidden_states_backend")
    assert action.choices is None
    assert (
        parser.parse_args(["--hidden-states-backend", "fixture"]).hidden_states_backend
        == "fixture"
    )
