"""Model-initialization boundaries and side-effect ordering, without downloads."""

import inspect
import logging
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import train
from speculators.train import model_init


def _args(**overrides):
    values = {
        "speculator_type": "mmuse",
        "from_pretrained": "",
        "draft_config": "",
        "verifier_name_or_path": "cli-verifier",
        "draft_attn_impl": "sdpa",
        "num_layers": 3,
        "draft_arch": "qwen3",
        "draft_hidden_act": "gelu",
        "sliding_window": 64,
        "full_attention_indices": [],
        "draft_mrope_full_head_hack": False,
        "mask_token_id": 99,
        "trust_remote_code": True,
        "draft_vocab_size": 77,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("script_name", "module_name", "arguments"),
    [
        (
            "_build_from_config_only",
            "build_from_config_only",
            (object(), "checkpoint", object(), object(), "verifier", "eager", _args()),
        ),
        ("build_draft_model", "build_draft_model", (_args(), object(), None, None, 32)),
    ],
)
def test_legacy_wrappers_forward_all_arguments_and_dynamic_logger(
    monkeypatch, script_name, module_name, arguments
):
    implementation_signature = inspect.signature(getattr(model_init, module_name))
    wrapper = getattr(train, script_name)
    wrapper_signature = inspect.signature(wrapper)
    assert list(wrapper_signature.parameters) == [
        name for name in implementation_signature.parameters if name != "logger"
    ]
    implementation = Mock(return_value=object())
    monkeypatch.setattr(model_init, module_name, implementation)

    for caller_logger in (Mock(), Mock()):
        monkeypatch.setattr(train, "logger", caller_logger)
        assert wrapper(*arguments) is implementation.return_value
        bound = implementation_signature.bind(
            *implementation.call_args.args, **implementation.call_args.kwargs
        )
        expected = wrapper_signature.bind(*arguments)
        assert bound.arguments == {**expected.arguments, "logger": caller_logger}
    assert implementation.call_count == 2


@pytest.mark.parametrize("source", ["synthesized", "decoder_config", "mtp"])
def test_fresh_sources_route_decoder_then_mask_then_training_factory(
    monkeypatch, source
):
    args = _args(
        speculator_type="mtp" if source == "mtp" else "mmuse",
        draft_config="decoder-source" if source == "decoder_config" else "",
    )
    decoder = SimpleNamespace(vocab_size=128)
    events = []

    def record(name, result):
        def invoke(*_args, **_kwargs):
            events.append(name)
            return result

        return Mock(side_effect=invoke)

    create = record("create", decoder)
    load = record("load", decoder)
    verifier = record("verifier", decoder)
    mask = record("mask", 5)
    expected_model = object()
    factory = record("factory", expected_model)
    monkeypatch.setattr(model_init, "create_transformer_layer_config", create)
    monkeypatch.setattr(model_init, "load_draft_transformer_layer_config", load)
    monkeypatch.setattr(model_init, "get_verifier_config", verifier)
    monkeypatch.setattr(model_init, "resolve_mask_token_id", mask)
    caller_logger = None if source == "synthesized" else Mock()
    effective_logger = caller_logger or logging.getLogger(model_init.__name__)
    t2d, d2t = object(), object()

    result = model_init.build_draft_model(
        args,
        SimpleNamespace(from_training_args=factory),
        t2d,
        d2t,
        16,
        logger=caller_logger,
    )

    assert result is expected_model
    assert (
        events
        == {
            "synthesized": ["create", "mask", "factory"],
            "decoder_config": ["load", "mask", "factory"],
            "mtp": ["verifier", "factory"],
        }[source]
    )
    if source == "synthesized":
        create.assert_called_once_with(
            verifier_name_or_path="cli-verifier",
            num_layers=3,
            draft_arch="qwen3",
            hidden_act="gelu",
            sliding_window=64,
            full_attention_indices=[],
            mrope_full_head_hack=False,
            logger=effective_logger,
        )
    elif source == "decoder_config":
        load.assert_called_once_with(
            "decoder-source", "cli-verifier", logger=effective_logger
        )
    else:
        verifier.assert_called_once_with("cli-verifier")
    assert args.mask_token_id == (99 if source == "mtp" else 5)
    assert args.draft_vocab_size == 16
    factory.assert_called_once_with(
        verifier_config=decoder, t2d=t2d, d2t=d2t, **vars(args)
    )
    if source != "mtp":
        mask.assert_called_once_with("cli-verifier", 128, 99, trust_remote_code=True)


@pytest.mark.parametrize("failure", ["decoder", "mask"])
def test_fresh_loading_failure_does_not_construct_or_commit_args(monkeypatch, failure):
    args = _args(draft_config="decoder-source")
    before = deepcopy(vars(args))
    decoder = Mock(return_value=SimpleNamespace(vocab_size=128))
    mask = Mock(return_value=5)
    (decoder if failure == "decoder" else mask).side_effect = ValueError("load failed")
    factory = Mock()
    monkeypatch.setattr(model_init, "load_draft_transformer_layer_config", decoder)
    monkeypatch.setattr(model_init, "resolve_mask_token_id", mask)

    with pytest.raises(ValueError, match="load failed"):
        model_init.build_draft_model(
            args, SimpleNamespace(from_training_args=factory), None, None, 16
        )

    factory.assert_not_called()
    if failure == "decoder":
        mask.assert_not_called()
    assert vars(args) == before


@pytest.fixture
def restore_harness(monkeypatch):
    events, failures = [], set()
    validation_snapshots, construction_snapshots = [], []
    config = SimpleNamespace(
        transformer_layer_config=SimpleNamespace(
            _attn_implementation="saved-attention"
        ),
        speculators_config=SimpleNamespace(verifier=SimpleNamespace(name_or_path="")),
    )

    def record(name, result=None):
        def invoke(*_args, **_kwargs):
            events.append(name)
            if name == "validate":
                validation_snapshots.append(deepcopy(config))
            elif name == "construct":
                construction_snapshots.append(deepcopy(config))
            if name in failures:
                raise ValueError(f"{name} failed")
            return result

        return Mock(side_effect=invoke)

    instance = SimpleNamespace(
        load_vocab_mappings=record("mappings"),
        load_verifier_weights=record("verifier_weights"),
    )
    selected_class = record("construct", instance)
    read_config = record("read_config", config)
    registry = record("registry", selected_class)
    validate = record("validate")
    caller_logger = Mock()
    caller_logger.info.side_effect = lambda *_args, **_kwargs: events.append("info")
    monkeypatch.setattr(
        model_init.SpeculatorModelConfig, "from_pretrained", read_config
    )
    monkeypatch.setattr(
        model_init.SpeculatorModel, "registered_model_class_from_config", registry
    )
    monkeypatch.setattr(model_init, "reconcile_pretrained_config_args", validate)
    monkeypatch.setattr(model_init, "is_config_only_dir", Mock(return_value=True))
    return SimpleNamespace(
        events=events,
        failures=failures,
        config=config,
        instance=instance,
        selected_class=selected_class,
        read_config=read_config,
        registry=registry,
        validate=validate,
        logger=caller_logger,
        validation_snapshots=validation_snapshots,
        construction_snapshots=construction_snapshots,
    )


@pytest.mark.parametrize("mtp", [False, True])
def test_config_only_uses_saved_class_then_mappings_and_verifier_weights(
    restore_harness, mtp
):
    state = restore_harness
    args = _args(
        from_pretrained="config-only", speculator_type="mtp" if mtp else "mmuse"
    )
    state.config.speculators_config.verifier.name_or_path = (
        "saved-verifier" if mtp else ""
    )
    config_before = deepcopy(state.config)
    requested_class = Mock(side_effect=AssertionError("CLI class must not construct"))
    t2d, d2t = object(), object()

    result = model_init.build_draft_model(
        args, requested_class, t2d, d2t, 16, logger=state.logger
    )

    assert result is state.instance
    assert state.events == [
        "info",
        "read_config",
        "registry",
        "validate",
        "construct",
        "mappings",
        "verifier_weights",
    ]
    state.read_config.assert_called_once_with("config-only")
    state.registry.assert_called_once_with(state.config)
    state.validate.assert_called_once_with(args, state.config, logger=state.logger)
    state.selected_class.assert_called_once_with(config=state.config)
    assert state.validation_snapshots == [config_before]
    assert state.construction_snapshots == [state.config]
    state.instance.load_vocab_mappings.assert_called_once_with(t2d, d2t)
    requested_class.assert_not_called()
    assert state.config.transformer_layer_config._attn_implementation == (
        "saved-attention" if mtp else "sdpa"
    )
    assert state.config.speculators_config.verifier.name_or_path == (
        "saved-verifier" if mtp else "cli-verifier"
    )
    assert args.mask_token_id == 99
    assert args.draft_vocab_size == 77


@pytest.mark.parametrize(
    ("failure", "expected_events"),
    [
        ("read_config", ["info", "read_config"]),
        ("validate", ["info", "read_config", "registry", "validate"]),
    ],
)
def test_config_only_failure_keeps_discovery_log_but_never_constructs(
    restore_harness, failure, expected_events
):
    state = restore_harness
    state.failures.add(failure)
    before = deepcopy(state.config)
    args = _args(from_pretrained="config-only")

    with pytest.raises(ValueError, match=f"{failure} failed"):
        model_init.build_draft_model(args, Mock(), None, None, 16, logger=state.logger)

    assert state.events == expected_events
    state.selected_class.assert_not_called()
    state.instance.load_vocab_mappings.assert_not_called()
    state.instance.load_verifier_weights.assert_not_called()
    assert state.config == before


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("reject_overlay", [False, True])
def test_weight_restore_validates_before_selecting_loader(
    monkeypatch, restore_harness, mtp, reject_overlay
):
    state = restore_harness
    monkeypatch.setattr(model_init, "is_config_only_dir", Mock(return_value=False))
    args = _args(from_pretrained="weights", speculator_type="mtp" if mtp else "mmuse")
    args_before = deepcopy(vars(args))
    restored_model = object()

    def load_weights(*_args, **_kwargs):
        state.events.append("weights")
        assert state.config.transformer_layer_config._attn_implementation == (
            "saved-attention" if mtp else "sdpa"
        )
        return restored_model

    generic_loader = Mock(side_effect=load_weights)
    mtp_loader = Mock(side_effect=load_weights)
    monkeypatch.setattr(model_init.SpeculatorModel, "from_pretrained", generic_loader)
    requested_class = SimpleNamespace(from_pretrained=mtp_loader)
    t2d, d2t = object(), object()
    if reject_overlay:
        state.failures.add("validate")
        with pytest.raises(ValueError, match="validate failed"):
            model_init.build_draft_model(
                args, requested_class, t2d, d2t, 16, logger=state.logger
            )
        generic_loader.assert_not_called()
        mtp_loader.assert_not_called()
    else:
        result = model_init.build_draft_model(
            args, requested_class, t2d, d2t, 16, logger=state.logger
        )
        assert result is restored_model
        selected_loader = mtp_loader if mtp else generic_loader
        unused_loader = generic_loader if mtp else mtp_loader
        selected_loader.assert_called_once_with(
            "weights", config=state.config, t2d=t2d, d2t=d2t, verifier="cli-verifier"
        )
        unused_loader.assert_not_called()
    assert state.events == ["read_config", "validate"] + (
        [] if reject_overlay else ["weights"]
    )
    state.validate.assert_called_once_with(args, state.config, logger=state.logger)
    state.registry.assert_not_called()
    assert state.config.transformer_layer_config._attn_implementation == (
        "saved-attention" if mtp or reject_overlay else "sdpa"
    )
    assert state.config.speculators_config.verifier.name_or_path == ""
    assert vars(args) == args_before
