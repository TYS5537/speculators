"""DFlash2 conversion keeps architecture, selector weights and vocabulary intact."""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import PretrainedConfig

from speculators.convert.dflash2.converter import DFlash2Converter
from speculators.convert.entrypoints import (
    convert_model,
    maybe_convert_external_checkpoint,
)
from speculators.models.dflash2 import DFlash2DraftModel


def _source_config():
    return {
        "model_type": "qwen3",
        "architectures": ["DFlash2DraftModel"],
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "dflash_config": {
            "block_size": 4,
            "target_layer_ids": [0],
            "mask_token_id": 0,
            "conv_kernel_size": 2,
            "conv_group_size": 4,
            "selector_rank": 8,
            "selector_top_k": 3,
        },
    }


@pytest.fixture
def verifier_config(monkeypatch):
    original = PretrainedConfig.get_config_dict

    def get_config(path, *args, **kwargs):
        if str(path) != "verifier":
            return original(path, *args, **kwargs)
        return (
            {
                "hidden_size": 16,
                "num_hidden_layers": 8,
                "architectures": ["Qwen3ForCausalLM"],
            },
            {},
        )

    monkeypatch.setattr(
        "speculators.convert.dflash.converter.PretrainedConfig.get_config_dict",
        get_config,
    )


@pytest.mark.parametrize("nested", [False, True])
def test_config_preserves_dflash2_dimensions(verifier_config, nested):
    source = _source_config()
    if not nested:
        source.update(source.pop("dflash_config"))
    config = DFlash2Converter()._build_config(source, "verifier", None)
    assert (
        config.speculators_model_type
        == config.speculators_config.algorithm
        == "dflash2"
    )
    assert config.architectures == ["DFlash2DraftModel"]
    assert config.aux_hidden_state_layer_ids == [1]
    assert (config.conv_kernel_size, config.conv_group_size) == (2, 4)
    assert (config.selector_rank, config.selector_top_k) == (8, 3)
    assert config.sliding_window_non_causal
    assert not config.sample_from_anchor
    assert config.speculators_config.proposal_methods[0].speculative_tokens == 3
    assert not hasattr(config.transformer_layer_config, "selector_rank")


def test_nested_options_override_top_level(verifier_config):
    source = _source_config()
    source.update(block_size=16, selector_rank=256, mask_token_id=31)
    config = DFlash2Converter()._build_config(source, "verifier", None)
    assert (config.block_size, config.selector_rank, config.mask_token_id) == (4, 8, 0)


@pytest.mark.parametrize(("draft_layers", "expected"), [(1, [5]), (3, [2, 4, 6])])
def test_source_default_target_layers_are_preserved(
    verifier_config, draft_layers, expected
):
    source = _source_config()
    del source["dflash_config"]["target_layer_ids"]
    source.update(num_target_layers=8, num_hidden_layers=draft_layers)
    config = DFlash2Converter()._build_config(source, "verifier", None)
    assert config.aux_hidden_state_layer_ids == expected


def test_last_target_layer_is_not_silently_dropped(verifier_config):
    source = _source_config()
    source["dflash_config"]["target_layer_ids"] = [0, 7]
    config = DFlash2Converter()._build_config(source, "verifier", None)
    assert config.aux_hidden_state_layer_ids == [1, 8]
    assert DFlash2DraftModel(config).fc.in_features == 32


@pytest.mark.parametrize("causal", [None, False, True])
def test_source_sliding_windows_are_rejected_before_predictions_change(
    verifier_config, causal
):
    source = _source_config()
    source.update(
        layer_types=["sliding_attention"], sliding_window=16, is_causal=causal
    )
    with pytest.raises(ValueError, match="full.attention"):
        DFlash2Converter()._build_config(source, "verifier", None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_embedding_scale", 2.0),
        ("output_multiplier", 2.0),
        ("final_logit_softcapping", 30.0),
        ("sample_from_anchor", True),
    ],
)
def test_unsupported_semantics_are_rejected(field, value):
    source = _source_config()
    source["dflash_config"][field] = value
    with pytest.raises(ValueError, match="scaling|sample_from_anchor"):
        DFlash2Converter()._build_config(source, "unused", None)


def test_missing_dimensions_are_rejected():
    source = _source_config()
    del source["dflash_config"]["selector_rank"]
    with pytest.raises(ValueError, match="missing: selector_rank"):
        DFlash2Converter()._build_config(source, "unused", None)


@pytest.mark.parametrize("embedding_keys", [False, True])
def test_external_checkpoint_round_trip(
    verifier_config, monkeypatch, tmp_path, embedding_keys
):
    def fill_verifier_weights(model):
        with torch.no_grad():
            for name in (
                "embed_tokens",
                "lm_head",
                "verifier_lm_head",
                "verifier_norm",
            ):
                getattr(model, name).weight.fill_(0.25)

    monkeypatch.setattr(
        DFlash2DraftModel, "load_verifier_weights", fill_verifier_weights
    )
    converter = DFlash2Converter()
    config = converter._build_config(_source_config(), "verifier", None)
    original = DFlash2DraftModel(config)
    weights = {
        key: value.clone()
        for key, value in original.state_dict().items()
        if not key.startswith(("embed_tokens.", "lm_head.", "verifier_"))
    }
    if embedding_keys:
        for name in ("predecessor_codebook", "successor_codebook"):
            key = f"candidate_selector.{name}"
            weights[f"{key}.weight"] = weights.pop(key)
    source = tmp_path / "external"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(_source_config()))
    save_file(weights, source / "model.safetensors")
    output = tmp_path / "converted"
    converter.convert(source, output, "verifier", validate=True)
    loaded = DFlash2DraftModel.from_pretrained(output)
    assert isinstance(loaded, DFlash2DraftModel)
    for key, value in original.state_dict().items():
        if key.startswith(("layers.", "candidate_selector.", "fc.")):
            torch.testing.assert_close(loaded.state_dict()[key], value)


def test_duplicate_codebook_keys_are_rejected():
    key = "candidate_selector.predecessor_codebook"
    with pytest.raises(ValueError, match="Duplicate selector"):
        DFlash2Converter()._remap_weights(
            {key: torch.ones(2, 2), f"{key}.weight": torch.ones(2, 2)}, None, None
        )


def test_missing_trained_module_is_rejected(verifier_config, tmp_path):
    converter = DFlash2Converter()
    config = converter._build_config(_source_config(), "verifier", None)
    weights = DFlash2DraftModel(config).state_dict()
    del weights["candidate_selector.successor_codebook"]
    with pytest.raises(ValueError, match="Draft weights missing.*successor_codebook"):
        converter._save(config, weights, tmp_path)


@pytest.mark.parametrize("architecture", ["DFlash2DraftModel", "DFlashDraftModel"])
def test_auto_conversion_selects_dflash2(monkeypatch, architecture):
    calls = []
    monkeypatch.setattr(
        "speculators.convert.entrypoints.convert_model", lambda **kw: calls.append(kw)
    )
    source = _source_config()
    source["architectures"] = [architecture]
    maybe_convert_external_checkpoint(
        "external", "verifier", output_path="converted", config_dict=source
    )
    assert calls[0]["algorithm"] == "dflash2"


def test_explicit_conversion_selects_dflash2(monkeypatch):
    calls = []
    monkeypatch.setattr(
        DFlash2Converter, "convert", lambda *args, **kwargs: calls.append(args)
    )
    convert_model("external", "verifier", "dflash2", output_path="converted")
    assert isinstance(calls[0][0], DFlash2Converter)
