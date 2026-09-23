"""Decoder-config module boundaries and script compatibility, without downloads."""

import inspect
import logging
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from scripts import train
from speculators.train import draft_config


def _verifier(**overrides):
    values = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 96,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 256,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "head_dim": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _create(builder, **overrides):
    kwargs = {
        "verifier_name_or_path": "verifier-source",
        "num_layers": 2,
        "draft_arch": "llama",
        "hidden_act": None,
        "sliding_window": 64,
        "full_attention_indices": [1],
    }
    kwargs.update(overrides)
    return builder(**kwargs)


def test_script_constants_keep_identity_and_registered_config_classes():
    assert train.DRAFT_ARCH_CONFIGS is draft_config.DRAFT_ARCH_CONFIGS
    assert train.MROPE_INVERSE_TOLERANCE is draft_config.MROPE_INVERSE_TOLERANCE
    registered_configs = draft_config.DRAFT_ARCH_CONFIGS
    assert registered_configs == {
        "llama": LlamaConfig,
        "qwen3": Qwen3Config,
    }


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        (
            "create_transformer_layer_config",
            ("verifier", 3, "qwen3", "gelu", 128, [0, 2], False),
        ),
        ("load_draft_transformer_layer_config", ("decoder/config.json", "verifier")),
        ("_maybe_apply_mrope_full_head_hack", ({"mrope_section": [1, 1]}, 8, False)),
    ],
)
def test_script_wrappers_forward_original_arguments_and_dynamic_logger(
    monkeypatch, name, arguments
):
    implementation_signature = inspect.signature(getattr(draft_config, name))
    wrapper = getattr(train, name)
    wrapper_signature = inspect.signature(wrapper)
    assert "logger" not in wrapper_signature.parameters
    assert list(wrapper_signature.parameters) == [
        key for key in implementation_signature.parameters if key != "logger"
    ]
    implementation = Mock(return_value=None if name.startswith("_") else object())
    monkeypatch.setattr(draft_config, name, implementation)

    for caller_logger in (Mock(), Mock()):
        monkeypatch.setattr(train, "logger", caller_logger)
        assert wrapper(*arguments) is implementation.return_value
        bound = implementation_signature.bind(
            *implementation.call_args.args, **implementation.call_args.kwargs
        )
        expected = wrapper_signature.bind(*arguments)
        assert bound.arguments == {**expected.arguments, "logger": caller_logger}
    assert implementation.call_count == 2


@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
def test_direct_and_script_builders_match_for_nested_verifier_geometry(
    monkeypatch, architecture
):
    # Incompatible verifier head counts are adapted to its explicit head width.
    text_config = _verifier(
        hidden_size=24,
        num_attention_heads=5,
        hidden_act=None,
        hidden_activation="gelu",
        rope_parameters={
            "full_attention": {"rope_theta": 90000.0},
            "sliding_attention": {
                "rope_theta": 20000.0,
                "partial_rotary_factor": 0.5,
            },
        },
    )
    original = deepcopy(vars(text_config))
    loader = Mock(return_value=SimpleNamespace(text_config=text_config))
    monkeypatch.setattr(draft_config.AutoConfig, "from_pretrained", loader)
    monkeypatch.setattr(draft_config.transformers, "__version__", "5.0.0")

    direct = _create(
        draft_config.create_transformer_layer_config, draft_arch=architecture
    )
    legacy = _create(train.create_transformer_layer_config, draft_arch=architecture)

    assert type(direct) is draft_config.DRAFT_ARCH_CONFIGS[architecture]
    assert direct.to_dict() == legacy.to_dict()
    assert direct.hidden_size == 24
    assert direct.num_attention_heads == direct.num_key_value_heads == 3
    assert direct.head_dim == 8
    assert direct.hidden_act == "gelu"
    assert direct.layer_types == ["sliding_attention", "full_attention"]
    assert direct.rope_parameters == {"rope_type": "default", "rope_theta": 20000.0}
    assert vars(text_config) == original
    direct.rope_parameters["rope_theta"] = 1.0
    assert vars(text_config) == original
    assert legacy.rope_parameters["rope_theta"] == 20000.0
    assert loader.call_args_list == [(("verifier-source",), {})] * 2


@pytest.mark.parametrize(
    ("entrypoint", "transformers_version", "rope_field"),
    [
        ("module", "5.0.0", "rope_parameters"),
        ("script", "4.57.6", "rope_scaling"),
        ("caller_logger", "5.0.0", "rope_parameters"),
    ],
)
def test_mrope_copy_and_warning_logger_are_preserved(
    monkeypatch, caplog, entrypoint, transformers_version, rope_field
):
    rope = {
        "rope_type": "default",
        "rope_theta": 1000000.0,
        "mrope_section": [1, 1],
        "partial_rotary_factor": 0.5,
        "type": "mrope",
        "mrope_interleaved": True,
    }
    verifier = _verifier(**{rope_field: rope}, rope_theta=1000000.0)
    original = deepcopy(vars(verifier))
    monkeypatch.setattr(
        draft_config.AutoConfig, "from_pretrained", Mock(return_value=verifier)
    )
    monkeypatch.setattr(draft_config.transformers, "__version__", transformers_version)
    supplied_logger = Mock()
    builder = (
        train.create_transformer_layer_config
        if entrypoint == "script"
        else draft_config.create_transformer_layer_config
    )
    kwargs = {"logger": supplied_logger} if entrypoint == "caller_logger" else {}

    with caplog.at_level(logging.WARNING):
        result = _create(builder, **kwargs)

    output_rope = getattr(result, rope_field)
    assert output_rope["mrope_section"] == [2, 2]
    assert output_rope["partial_rotary_factor"] == 1.0
    assert "type" not in output_rope
    assert "mrope_interleaved" not in output_rope
    assert vars(verifier) == original
    output_rope["mrope_section"][0] = 99
    assert vars(verifier) == original
    if entrypoint == "caller_logger":
        supplied_logger.warning.assert_called_once()
        assert (
            "MRoPE full-head hack applied" in supplied_logger.warning.call_args.args[0]
        )
        assert not caplog.records
    else:
        assert len(caplog.records) == 1
        expected_name = (
            train.__name__ if entrypoint == "script" else draft_config.__name__
        )
        assert caplog.records[0].name == expected_name
        assert "MRoPE full-head hack applied" in caplog.records[0].message


@pytest.mark.parametrize("hidden_size_matches", [True, False])
def test_direct_loader_reconciles_model_without_rewriting_source_dimensions(
    monkeypatch, hidden_size_matches
):
    source = "opaque-decoder-source"
    decoder_dict = Qwen3Config(
        hidden_size=32, vocab_size=64, num_hidden_layers=3
    ).to_dict()
    full_config = {"transformer_layer_config": decoder_dict, "unrelated": [1, 2]}
    original = deepcopy(full_config)
    # HF from_dict may normalize its input (e.g. attention settings); only the
    # loader's later verifier reconciliation must not rewrite source dimensions.
    Qwen3Config.from_dict(original["transformer_layer_config"])
    read_config = Mock(return_value=(full_config, {}))
    verifier_loader = Mock(
        return_value=_verifier(hidden_size=32 if hidden_size_matches else 48)
    )
    supplied_logger = Mock()
    monkeypatch.setattr(draft_config.PretrainedConfig, "get_config_dict", read_config)
    monkeypatch.setattr(draft_config, "get_verifier_config", verifier_loader)

    if hidden_size_matches:
        result = draft_config.load_draft_transformer_layer_config(
            source, "verifier", logger=supplied_logger
        )
        assert isinstance(result, Qwen3Config)
        assert result.hidden_size == 32
        assert result.num_hidden_layers == 3
        assert result.vocab_size == 128
        supplied_logger.warning.assert_called_once()
        assert supplied_logger.warning.call_args.args[1:] == (64, 128)
    else:
        with pytest.raises(ValueError, match="hidden_size"):
            draft_config.load_draft_transformer_layer_config(
                source, "verifier", logger=supplied_logger
            )
        supplied_logger.warning.assert_not_called()

    read_config.assert_called_once_with(source)
    verifier_loader.assert_called_once_with("verifier")
    assert full_config == original
    assert decoder_dict["vocab_size"] == 64
    assert decoder_dict["hidden_size"] == 32


def test_unknown_architecture_fails_before_loading_verifier(monkeypatch):
    loader = Mock(side_effect=AssertionError("Verifier loading must not be reached"))
    monkeypatch.setattr(draft_config.AutoConfig, "from_pretrained", loader)

    with pytest.raises(ValueError, match="Unknown draft architecture"):
        _create(
            draft_config.create_transformer_layer_config, draft_arch="not-registered"
        )

    loader.assert_not_called()
