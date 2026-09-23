"""Training-config module boundaries, transactional overlays and compatibility."""

import argparse
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import train
from speculators.train import model_config


class _TrackedNamespace(SimpleNamespace):
    def __init__(self, events, label, **values):
        self.__dict__.update(_events=events, _label=label, **values)

    def __setattr__(self, name, value):
        self._events.append(("set", self._label, name, value))
        super().__setattr__(name, value)


def _snapshot(namespace):
    return copy.deepcopy(
        {
            key: value
            for key, value in vars(namespace).items()
            if key not in {"_events", "_label"}
        }
    )


def _transaction_objects():
    events = []
    args = _TrackedNamespace(
        events,
        "args",
        block_size=99,
        correction_hidden_aux_weight=0.25,
        correction_rollout_metrics=False,
        correction_base_diagnostics=False,
        correction_rank=9,
        _provided_model_config_dests={
            "correction_hidden_aux_weight",
            "correction_rollout_metrics",
            "correction_rank",
        },
    )
    config = _TrackedNamespace(
        events,
        "config",
        speculators_model_type="mmuse",
        block_size=3,
        correction_hidden_aux_weight=0.1,
        correction_rollout_metrics=True,
        correction_base_diagnostics=True,
        correction_hidden_size=24,
    )
    return events, args, config


def test_scripts_preserve_constant_identity_and_callable_aliases():
    for name in (
        "DECODER_SHAPING_FLAGS",
        "PRETRAINED_MODEL_CONFIG_FLAGS",
        "PRETRAINED_RUNTIME_CONFIG_FIELDS",
        "MMUSE_MODEL_CONFIG_FIELDS",
    ):
        assert getattr(train, name) is getattr(model_config, name)
    assert (
        train._plan_pretrained_config_overrides
        is model_config.plan_pretrained_config_overrides
    )
    assert train.validate_draft_init_args is model_config.validate_draft_init_args


def test_planner_is_pure_and_skips_fields_missing_from_either_side():
    args = argparse.Namespace(
        block_size=99,
        correction_rollout_metrics=False,
        dflash2_selector_top_k=4,
        correction_rank=9,
    )
    config = SimpleNamespace(
        block_size=3,
        correction_rollout_metrics=True,
        dflash2_selector_top_k=4,
        correction_hidden_size=24,
    )
    provided = {
        "correction_rollout_metrics",
        "dflash2_selector_top_k",
        "correction_rank",
    }
    before = (_snapshot(args), _snapshot(config), provided.copy())
    inherited, overrides = model_config.plan_pretrained_config_overrides(
        args, config, provided
    )
    assert inherited == {"block_size": 3}
    assert overrides == {"correction_rollout_metrics": False}
    assert (_snapshot(args), _snapshot(config), provided) == before


def test_planner_reports_all_structure_conflicts_in_registration_order():
    args = argparse.Namespace(
        correction_rank=9,
        block_size=7,
        markov_rank=16,
        correction_rollout_metrics=False,
    )
    config = SimpleNamespace(
        correction_rank=4, block_size=3, markov_rank=8, correction_rollout_metrics=True
    )
    provided = set(vars(args))
    before = (_snapshot(args), _snapshot(config))
    with pytest.raises(ValueError) as error:
        model_config.plan_pretrained_config_overrides(args, config, provided)
    message = str(error.value)
    conflicts = [
        "--block-size=7 (checkpoint: 3)",
        "--markov-rank=16 (checkpoint: 8)",
        "--correction-rank=9 (checkpoint: 4)",
    ]
    assert all(fragment in message for fragment in conflicts)
    assert [message.index(fragment) for fragment in conflicts] == sorted(
        message.index(fragment) for fragment in conflicts
    )
    assert "--correction-rollout-metrics" not in message
    assert (_snapshot(args), _snapshot(config)) == before


def test_reconcile_validates_candidate_before_inheritance_overrides_and_caller_log(
    monkeypatch,
):
    events, args, config = _transaction_objects()
    before = (_snapshot(args), _snapshot(config))

    def validate(candidate):
        assert events == []
        assert (_snapshot(args), _snapshot(config)) == before
        assert candidate == {
            "block_size": 3,
            "correction_hidden_size": 24,
            "correction_hidden_aux_weight": 0.25,
            "correction_rollout_metrics": False,
            "correction_base_diagnostics": True,
        }
        events.append(("validate",))

    validator = Mock(side_effect=validate)
    monkeypatch.setattr(model_config, "validate_mmuse_options", validator)
    logger = Mock()
    logger.info.side_effect = lambda *args: events.append(("log",))
    assert (
        model_config.reconcile_pretrained_config_args(args, config, logger=logger)
        is None
    )
    validator.assert_called_once()
    assert events == [
        ("validate",),
        ("set", "args", "block_size", 3),
        ("set", "args", "correction_base_diagnostics", True),
        ("set", "config", "correction_hidden_aux_weight", 0.25),
        ("set", "config", "correction_rollout_metrics", False),
        ("log",),
    ]
    assert args.correction_rollout_metrics is False
    assert config.correction_rollout_metrics is False
    assert args.correction_rank == 9
    assert not hasattr(config, "correction_rank")
    assert not hasattr(args, "correction_hidden_size")
    logger.info.assert_called_once()
    assert (
        logger.info.call_args.args[1]
        == "--correction-hidden-aux-weight=0.25, --correction-rollout-metrics=False"
    )


@pytest.mark.parametrize("failure", ["ownership", "structure", "combination"])
def test_reconcile_failure_never_mutates_or_logs(monkeypatch, failure):
    events, args, config = _transaction_objects()
    if failure == "ownership":
        config.__dict__["speculators_model_type"] = "dspark"
    elif failure == "structure":
        args._provided_model_config_dests.add("block_size")
    before = (_snapshot(args), _snapshot(config))

    def reject(_candidate):
        assert events == []
        assert (_snapshot(args), _snapshot(config)) == before
        events.append(("validate",))
        raise ValueError("invalid complete candidate")

    validator = Mock(side_effect=reject)
    monkeypatch.setattr(model_config, "validate_mmuse_options", validator)
    logger = Mock()
    expected = {
        "ownership": "MMuse checkpoint",
        "structure": "cannot change checkpoint",
        "combination": "invalid complete candidate",
    }
    with pytest.raises(ValueError, match=expected[failure]):
        model_config.reconcile_pretrained_config_args(args, config, logger=logger)
    assert (_snapshot(args), _snapshot(config)) == before
    assert events == ([("validate",)] if failure == "combination" else [])
    assert validator.call_count == int(failure == "combination")
    assert logger.mock_calls == []


@pytest.mark.parametrize("explicit", [False, True])
def test_omitted_tracking_or_matching_runtime_value_does_not_log(explicit):
    args = argparse.Namespace(confidence_detach_features=explicit)
    if explicit:
        args._provided_model_config_dests = {"confidence_detach_features"}
    config = SimpleNamespace(
        speculators_model_type="dspark", confidence_detach_features=True
    )
    logger = Mock()
    model_config.reconcile_pretrained_config_args(args, config, logger=logger)
    assert args.confidence_detach_features is True
    assert config.confidence_detach_features is True
    assert logger.mock_calls == []


def test_legacy_two_argument_wrapper_uses_current_scripts_logger(monkeypatch):
    script_logger = Mock()
    monkeypatch.setattr(train, "logger", script_logger)
    args = argparse.Namespace(
        confidence_detach_features=False,
        _provided_model_config_dests={"confidence_detach_features"},
    )
    config = SimpleNamespace(
        speculators_model_type="dspark", confidence_detach_features=True
    )
    assert train._reconcile_pretrained_config_args(args, config) is None
    assert config.confidence_detach_features is False
    script_logger.info.assert_called_once()
    assert script_logger.info.call_args.args[1] == "--confidence-detach-features=False"


class _ParserError(ValueError):
    pass


def _raise_parser_error(message):
    raise _ParserError(message)


@pytest.mark.parametrize(
    ("checkpoint", "draft_config", "algorithm", "provided", "error_fragment"),
    [
        ("checkpoint", None, "mmuse", set(), None),
        (None, "decoder", "mmuse", set(), None),
        (None, None, "mmuse", {"draft_arch", "num_layers"}, None),
        (None, None, "mtp", set(), None),
        (
            "checkpoint",
            "decoder",
            "mmuse",
            {"draft_arch", "num_layers"},
            "--from-pretrained",
        ),
        (None, "decoder", "mtp", {"draft_arch", "num_layers"}, "--speculator-type mtp"),
        (None, "decoder", "mmuse", {"draft_arch", "num_layers"}, "--draft-config"),
        ("checkpoint", None, "mtp", set(), None),
    ],
)
def test_initialization_source_rules_match_direct_module_and_legacy_entrypoint(
    checkpoint, draft_config, algorithm, provided, error_fragment
):
    args = argparse.Namespace(
        from_pretrained=checkpoint, draft_config=draft_config, speculator_type=algorithm
    )
    before = (_snapshot(args), provided.copy())
    messages = []
    for validator in (
        model_config.validate_draft_init_args,
        train.validate_draft_init_args,
    ):
        parser = Mock(spec=argparse.ArgumentParser)
        parser.error.side_effect = _raise_parser_error
        if error_fragment is None:
            assert validator(parser, args, provided) is None
            parser.error.assert_not_called()
        else:
            with pytest.raises(_ParserError, match=error_fragment):
                validator(parser, args, provided)
            parser.error.assert_called_once()
            message = parser.error.call_args.args[0]
            assert error_fragment in message
            assert message.index("--num-layers") < message.index("--draft-arch")
            messages.append(message)
        assert (_snapshot(args), provided) == before
    if messages:
        assert messages[0] == messages[1]
