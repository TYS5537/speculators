"""Training CLI construction, explicit argv isolation, and finalization order."""

import argparse
import inspect
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import train
from speculators.train import cli


def _argv(*extra):
    return ["--verifier-name-or-path", "unused-verifier", *extra]


@pytest.mark.parametrize(
    "features", [[], ["--enable-correction-head", "--dflash-gated-layer-fusion"]]
)
def test_legacy_muse_alias_has_identical_defaults_and_explicit_features(features):
    canonical = cli.parse_train_args(_argv("--speculator-type", "mmuse", *features))
    legacy = cli.parse_train_args(_argv("--speculator-type", "muse", *features))
    assert legacy.speculator_type == "mmuse"
    assert vars(legacy) == vars(canonical)


def test_script_preserves_constants_and_checkpoint_converter_identity():
    for name in (
        "DSPARK_PAPER_LOSS_FN",
        "DSPARK_PAPER_BLOCK_SIZE",
        "DSPARK_PAPER_NUM_LAYERS",
        "DSPARK_PAPER_EPOCHS",
        "_checkpoint_freq",
    ):
        assert getattr(train, name) is getattr(cli, name)


def test_script_parser_remains_a_zero_argument_dynamic_wrapper(monkeypatch):
    implementation = Mock(return_value=argparse.Namespace(marker=object()))
    monkeypatch.setattr(cli, "parse_train_args", implementation)
    assert not inspect.signature(train.parse_args).parameters
    assert train.parse_args() is implementation.return_value
    implementation.assert_called_once_with()


@pytest.mark.parametrize("argv", [None, _argv("--speculator-type", "mmuse")])
def test_parse_entrypoint_preserves_initial_parse_call_and_argv_identity(
    monkeypatch, argv
):
    parser = Mock()
    raw_args = argparse.Namespace(marker=object())
    parser.parse_args.return_value = raw_args
    builder = Mock(return_value=parser)
    finalize_signature = inspect.signature(cli.finalize_train_args)
    finalizer = Mock(return_value=object())
    monkeypatch.setattr(cli, "build_train_parser", builder)
    monkeypatch.setattr(cli, "finalize_train_args", finalizer)

    result = cli.parse_train_args(argv)

    assert result is finalizer.return_value
    builder.assert_called_once_with()
    if argv is None:
        parser.parse_args.assert_called_once_with()
    else:
        parser.parse_args.assert_called_once_with(argv)
    finalizer.assert_called_once()
    bound = finalize_signature.bind(
        *finalizer.call_args.args, **finalizer.call_args.kwargs
    )
    bound.apply_defaults()
    assert bound.arguments["parser"] is parser
    assert bound.arguments["args"] is raw_args
    assert bound.arguments["argv"] is argv


def test_parser_builder_only_declares_options_without_reading_argv_or_validating(
    monkeypatch,
):
    monkeypatch.setattr("sys.argv", ["train.py", "--unknown-ambient-option"])
    parse = Mock(side_effect=AssertionError("Builder must not parse arguments"))
    defaults = Mock(side_effect=AssertionError("Builder must not finalize defaults"))
    validate = Mock(side_effect=AssertionError("Builder must not validate settings"))
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    monkeypatch.setattr(cli, "_apply_training_defaults", defaults)
    monkeypatch.setattr(cli, "_validate_training_args", validate)

    parser = cli.build_train_parser()

    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.get_default("draft_arch") is None
    assert parser.get_default("muon_lr") is None
    assert "--verifier-name-or-path" in parser.format_help()
    parse.assert_not_called()
    defaults.assert_not_called()
    validate.assert_not_called()


def test_backend_options_follow_live_registry_and_registration_order(monkeypatch):
    calls = []

    def add_file(parser):
        calls.append("file")
        parser.add_argument("--fixture-file", default="file-default")

    def add_extra(parser):
        calls.append("extra")
        parser.add_argument("--fixture-extra", type=int, default=7)

    registry = {"file": SimpleNamespace(add_train_args=add_file)}
    monkeypatch.setattr(cli.HiddenStatesBackend, "registry", registry)
    first = cli.build_train_parser()
    monkeypatch.setitem(registry, "fixture", SimpleNamespace(add_train_args=add_extra))
    second = cli.build_train_parser()

    assert calls == ["file", "file", "extra"]
    first_action = next(
        action for action in first._actions if action.dest == "hidden_states_backend"
    )
    second_action = next(
        action for action in second._actions if action.dest == "hidden_states_backend"
    )
    assert first_action.choices == ["file"]
    assert second_action.choices == ["file", "fixture"]
    assert first.parse_args(_argv()).fixture_file == "file-default"
    assert (
        second.parse_args(_argv("--hidden-states-backend", "fixture")).fixture_extra
        == 7
    )


@pytest.mark.parametrize(
    "extra",
    [
        [
            "--speculator-type",
            "mmuse",
            "--num-layers",
            "1",
            "--block-size",
            "4",
            "--no-correction-base-diagnostics",
        ],
        [
            "--from-pretrained",
            "checkpoint",
            "--no-correction-base-diagnostics",
            "--correction-output-mode",
            "logits",
        ],
        ["--speculator-type", "dspark"],
    ],
)
def test_explicit_argv_matches_script_despite_conflicting_ambient_arguments(
    monkeypatch, extra
):
    argv = _argv(*extra)
    original_argv = argv.copy()
    monkeypatch.setattr("sys.argv", ["train.py", *argv])
    script_args = train.parse_args()
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--from-pretrained",
            "ambient",
            "--draft-config",
            "conflict",
            "--unknown-ambient",
        ],
    )

    actual = cli.parse_train_args(argv)

    assert vars(actual) == vars(script_args)
    assert argv == original_argv
    if "--no-correction-base-diagnostics" in extra:
        assert actual.correction_base_diagnostics is False
        assert "correction_base_diagnostics" in actual._provided_model_config_dests
    if "--num-layers" in extra:
        assert (
            actual.num_layers == 1
        )  # Explicit raw default must not become paper default.
        assert actual.block_size == 4
        assert "block_size" in actual._provided_model_config_dests


def test_finalizer_uses_same_argv_for_all_tracking_and_preserves_stage_order(
    monkeypatch,
):
    argv = _argv("--speculator-type", "mmuse", "--no-correction-base-diagnostics")
    parser = cli.build_train_parser()
    args = parser.parse_args(argv)
    events, tracking_argv = [], []
    original_tracking = cli.explicitly_provided_dests

    def track(parser, dests, argv=None):
        tracking_argv.append(argv)
        events.append(f"tracking-{len(tracking_argv)}")
        return original_tracking(parser, dests, argv)

    def observe(name, original):
        def invoke(*positional, **kwargs):
            events.append(name)
            return original(*positional, **kwargs)

        return invoke

    monkeypatch.setattr(cli, "explicitly_provided_dests", track)
    for name, event in (
        ("_apply_training_defaults", "defaults"),
        ("_validate_training_args", "validation"),
        ("validate_draft_init_args", "init"),
        ("resolve_loss_config", "loss"),
        ("validate_mmuse_options", "mmuse"),
    ):
        monkeypatch.setattr(cli, name, observe(event, getattr(cli, name)))

    assert cli.finalize_train_args(parser, args, argv) is args

    assert events == [
        "tracking-1",
        "tracking-2",
        "defaults",
        "tracking-3",
        "validation",
        "init",
        "loss",
        "mmuse",
    ]
    assert len(tracking_argv) == 3
    assert all(observed is argv for observed in tracking_argv)
    assert args.draft_arch == "qwen3"
    assert args.muon_lr == 10 * args.lr


@pytest.mark.parametrize(
    ("extra", "error_type", "fragment"),
    [
        (
            [
                "--dsv4-external-arrow",
                "--no-correction-base-diagnostics",
                "--loss-fn",
                "invalid",
            ],
            SystemExit,
            "requires DSV4",
        ),
        (
            [
                "--no-correction-base-diagnostics",
                "--draft-config",
                "decoder",
                "--num-layers",
                "1",
                "--loss-fn",
                "invalid",
            ],
            SystemExit,
            "now belong to MMuse",
        ),
        (
            [
                "--speculator-type",
                "mmuse",
                "--from-pretrained",
                "checkpoint",
                "--draft-config",
                "decoder",
                "--loss-fn",
                "invalid",
            ],
            SystemExit,
            "takes precedence",
        ),
        (
            [
                "--speculator-type",
                "mmuse",
                "--loss-fn",
                "invalid",
                "--correction-num-heads",
                "0",
            ],
            ValueError,
            "Unknown loss function",
        ),
        (
            [
                "--speculator-type",
                "mmuse",
                "--correction-num-heads",
                "0",
                "--per-position-loss-weight",
                "dpace",
                "--dpace-alpha",
                "0",
            ],
            SystemExit,
            "correction_num_heads",
        ),
        (
            [
                "--per-position-loss-weight",
                "dpace",
                "--loss-fn",
                "kl_div",
                "--dpace-alpha",
                "0",
            ],
            SystemExit,
            "requires --loss-fn=ce",
        ),
        (
            [
                "--per-position-loss-weight",
                "dpace",
                "--loss-fn",
                "ce",
                "--dpace-alpha",
                "0",
            ],
            ValueError,
            "alpha must be in",
        ),
        (
            ["--checkpoint-freq", "0", "--loss-fn", "invalid"],
            SystemExit,
            "checkpoint-freq must be > 0",
        ),
    ],
)
def test_first_error_and_exception_type_are_preserved(
    extra, error_type, fragment, capsys
):
    with pytest.raises(error_type) as caught:
        cli.parse_train_args(_argv(*extra))

    assert type(caught.value) is error_type
    if error_type is SystemExit:
        assert caught.value.code == 2
        assert fragment in capsys.readouterr().err
    else:
        assert fragment in str(caught.value)
        assert capsys.readouterr().err == ""


@pytest.mark.parametrize("stage", ["dsv4", "ownership", "initialization"])
def test_failed_finalization_keeps_original_stage_specific_mutations(stage):
    extra = {
        "dsv4": ["--dsv4-external-arrow"],
        "ownership": ["--no-correction-base-diagnostics"],
        "initialization": [
            "--speculator-type",
            "mmuse",
            "--draft-config",
            "decoder",
            "--num-layers",
            "1",
        ],
    }[stage]
    argv = _argv(*extra)
    parser = cli.build_train_parser()
    args = parser.parse_args(argv)
    before = deepcopy(vars(args))

    with pytest.raises(SystemExit):
        cli.finalize_train_args(parser, args, argv)

    if stage == "dsv4":
        assert vars(args) == before
    elif stage == "ownership":
        assert vars(args) == {
            **before,
            "_provided_model_config_dests": {"correction_base_diagnostics"},
        }
    else:
        assert args.draft_arch == "qwen3"
        assert args.muon_lr == 10 * args.lr
        assert args.num_layers == 1
        # Decoder-shaping provenance is checked separately from saved-model fields.
        assert args._provided_model_config_dests == set()
