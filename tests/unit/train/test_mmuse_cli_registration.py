"""Lightweight MMuse option registration and training-parser boundary contracts."""

import argparse
import json
import subprocess
import sys
from types import MappingProxyType
from unittest.mock import Mock

import pytest

from scripts import train
from speculators.models.mmuse.config import MMUSE_OPTION_FIELDS, mmuse_option_defaults
from speculators.train import cli, mmuse_args
from speculators.utils.argparse_utils import explicitly_provided_dests


def _parser(defaults=None):
    parser = argparse.ArgumentParser(add_help=False)
    values = mmuse_option_defaults() if defaults is None else defaults
    mmuse_args.add_mmuse_backbone_args(parser, values)
    mmuse_args.add_mmuse_correction_args(parser, values)
    return parser


def _parse_training(monkeypatch, flags):
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--verifier-name-or-path", "unused-local-verifier", *flags],
    )
    return train.parse_args()


def test_registration_groups_cover_only_schema_owned_destinations():
    groups = []
    for register in (
        mmuse_args.add_mmuse_backbone_args,
        mmuse_args.add_mmuse_correction_args,
    ):
        parser = argparse.ArgumentParser(add_help=False)
        assert register(parser, mmuse_option_defaults()) is None
        groups.append({action.dest for action in parser._actions})
    assert groups[0].isdisjoint(groups[1])
    assert groups[0] | groups[1] == MMUSE_OPTION_FIELDS
    excluded = {
        "markov_rank",
        "markov_head_type",
        "enable_confidence_head",
        "confidence_head_with_markov",
        "confidence_detach_features",
        "sample_from_anchor",
        "block_size",
    }
    assert MMUSE_OPTION_FIELDS.isdisjoint(excluded)
    assert vars(_parser().parse_args([])) == mmuse_option_defaults()


def test_registration_consumes_custom_read_only_defaults_without_mutation():
    original = mmuse_option_defaults()
    alternate_strings = {
        "dflash2_selector_search_mode": "global",
        "correction_output_mode": "logits",
        "selector_correction_feedback": "corrected",
    }
    custom = {
        key: (
            not value
            if isinstance(value, bool)
            else value + 7
            if isinstance(value, int)
            else value + 0.25
            if isinstance(value, float)
            else alternate_strings[key]
        )
        for key, value in original.items()
    }
    snapshot = custom.copy()
    parser = _parser(MappingProxyType(custom))
    # Registration consumes values; cross-field architectural validation is elsewhere.
    assert vars(parser.parse_args([])) == custom
    assert custom == snapshot
    assert mmuse_option_defaults() == original


def test_each_boolean_option_accepts_both_spellings_and_tracks_explicit_defaults():
    parser = _parser()
    assert explicitly_provided_dests(parser, MMUSE_OPTION_FIELDS, []) == set()
    actions = [
        action
        for action in parser._actions
        if isinstance(action, argparse.BooleanOptionalAction)
    ]
    assert actions
    for action in actions:
        positive = next(
            flag for flag in action.option_strings if not flag.startswith("--no-")
        )
        negative = next(
            flag for flag in action.option_strings if flag.startswith("--no-")
        )
        assert getattr(parser.parse_args([positive]), action.dest) is True
        assert getattr(parser.parse_args([negative]), action.dest) is False
        for flag in (positive, negative):
            assert explicitly_provided_dests(parser, MMUSE_OPTION_FIELDS, [flag]) == {
                action.dest
            }


@pytest.mark.parametrize("default_mode", ["greedy", "global"])
def test_search_switches_are_mutually_exclusive_with_caller_owned_default(default_mode):
    defaults = {**mmuse_option_defaults(), "dflash2_selector_search_mode": default_mode}
    parser = _parser(defaults)
    assert parser.parse_args([]).dflash2_selector_search_mode == default_mode
    flags = ["--dflash2-selector-greedy", "--dflash2-selector-global"]
    for mode, flag in zip(("greedy", "global"), flags, strict=True):
        assert parser.parse_args([flag]).dflash2_selector_search_mode == mode
        assert explicitly_provided_dests(parser, MMUSE_OPTION_FIELDS, [flag]) == {
            "dflash2_selector_search_mode"
        }
    for conflicting in (flags, list(reversed(flags))):
        with pytest.raises(SystemExit):
            parser.parse_args(conflicting)


def test_scalar_types_and_choices_are_registered_by_both_groups():
    args = _parser().parse_args(
        [
            "--dflash2-selector-top-k",
            "7",
            "--correction-num-heads",
            "4",
            "--dflash2-selector-loss-weight",
            "0.25",
            "--correction-gate-bias",
            "-1.25",
            "--correction-output-mode",
            "logits",
            "--selector-correction-feedback",
            "corrected",
        ]
    )
    assert args.dflash2_selector_top_k == 7
    assert args.correction_num_heads == 4
    assert isinstance(args.dflash2_selector_top_k, int)
    assert isinstance(args.correction_num_heads, int)
    assert args.dflash2_selector_loss_weight == 0.25
    assert args.correction_gate_bias == -1.25
    assert args.correction_output_mode == "logits"
    assert args.selector_correction_feedback == "corrected"


@pytest.mark.parametrize(
    "flags",
    [
        ["--dflash2-selector-top-k", "not-an-integer"],
        ["--correction-num-heads", "2.5"],
        ["--correction-output-mode", "dense"],
        ["--selector-correction-feedback", "autoregressive"],
    ],
)
def test_invalid_scalar_types_or_choices_are_rejected(flags):
    with pytest.raises(SystemExit):
        _parser().parse_args(flags)


def test_registration_module_runs_with_only_the_standard_library():
    # Load this module directly: package __init__ imports are outside its contract.
    program = """
import argparse
import json
import runpy
import sys
module = runpy.run_path(sys.argv[1])
defaults = json.loads(sys.argv[2])
parser = argparse.ArgumentParser(add_help=False)
module['add_mmuse_backbone_args'](parser, defaults)
module['add_mmuse_correction_args'](parser, defaults)
assert vars(parser.parse_args([])) == defaults
assert 'torch' not in sys.modules
assert not any(name.startswith('speculators.models') for name in sys.modules)
"""
    result = subprocess.run(  # noqa: S603 -- Trusted Python and repository module; no shell.
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            program,
            mmuse_args.__file__,
            json.dumps(mmuse_option_defaults()),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_full_parser_shares_one_defaults_mapping_between_registration_groups(
    monkeypatch,
):
    defaults = mmuse_option_defaults()
    supplier = Mock(return_value=defaults)
    backbone = Mock(wraps=mmuse_args.add_mmuse_backbone_args)
    correction = Mock(wraps=mmuse_args.add_mmuse_correction_args)
    monkeypatch.setattr(cli, "mmuse_option_defaults", supplier)
    monkeypatch.setattr(cli, "add_mmuse_backbone_args", backbone)
    monkeypatch.setattr(cli, "add_mmuse_correction_args", correction)
    args = _parse_training(
        monkeypatch,
        [
            "--speculator-type",
            "mmuse",
            "--dflash2-conv-group-size",
            "12",
            "--correction-hidden-size",
            "96",
        ],
    )
    supplier.assert_called_once_with()
    backbone.assert_called_once()
    correction.assert_called_once()
    assert backbone.call_args.args[0] is correction.call_args.args[0]
    assert backbone.call_args.args[1] is defaults
    assert correction.call_args.args[1] is defaults
    assert args.dflash2_conv_group_size == 12
    assert args.correction_hidden_size == 96


@pytest.mark.parametrize(
    ("algorithm", "flag"),
    [
        ("dflash", "--no-dflash-gated-layer-fusion"),
        ("dspark", "--no-correction-base-diagnostics"),
    ],
)
def test_explicit_disabled_flags_still_belong_to_mmuse_in_full_parser(
    monkeypatch, algorithm, flag
):
    with pytest.raises(SystemExit):
        _parse_training(monkeypatch, ["--speculator-type", algorithm, flag])


def test_checkpoint_parsing_keeps_provided_tracking_and_defers_cross_group_rules(
    monkeypatch,
):
    flags = [
        "--no-dflash-gated-layer-fusion",
        "--no-correction-base-diagnostics",
        "--dflash2-selector-greedy",
        "--correction-hidden-feedback",
        "--selector-correction-feedback",
        "corrected",
    ]
    # A pure registrar accepts the incomplete combination; fresh training must not.
    standalone = _parser().parse_args(flags)
    assert standalone.correction_hidden_feedback
    with pytest.raises(SystemExit):
        _parse_training(monkeypatch, ["--speculator-type", "mmuse", *flags])
    restored = _parse_training(
        monkeypatch,
        [
            "--speculator-type",
            "mmuse",
            "--from-pretrained",
            "not-loaded-by-parser",
            *flags,
        ],
    )
    assert restored._provided_model_config_dests & MMUSE_OPTION_FIELDS == {
        "dflash_gated_layer_fusion",
        "correction_base_diagnostics",
        "dflash2_selector_search_mode",
        "correction_hidden_feedback",
        "selector_correction_feedback",
    }
    assert restored.dflash_gated_layer_fusion is False
    assert restored.correction_base_diagnostics is False
    assert restored.dflash2_selector_search_mode == "greedy"
    assert restored.correction_hidden_feedback is True
    assert restored.selector_correction_feedback == "corrected"
