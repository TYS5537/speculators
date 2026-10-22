"""Existing checkpoint metadata can be normalized without changing the model."""

import copy
import json

import pytest
import torch

from scripts.normalize_mmuse_config import main
from speculators import SpeculatorModelConfig
from speculators.models.dflash.model_definitions import Qwen3DFlashAttention
from speculators.models.mmuse.compat import normalize_mmuse_checkpoint_config
from speculators.models.mmuse.config import MMuseSpeculatorConfig


def _checkpoint(*, dsv4=False):
    return {
        "architectures": ["DSparkDraftModel" if dsv4 else "MMuseDraftModel"],
        "speculators_model_type": "dspark" if dsv4 else "mmuse",
        "speculators_version": "0.6.0.dev0" if dsv4 else "0.8.0.dev0",
        "transformers_version": "5.14.1",
        "auto_map": {"": "config.DSparkSpeculatorConfig"},
        "enable_correction_head": True,
        "correction_output_mode": "logits",
        "correction_hidden_feedback": dsv4,
        "correction_project_corrected_hidden": not dsv4,
        "dflash2_dynamic_conv": True,
        "sample_from_anchor": True,
        "block_size": 7,
        "target_training_contract": {"format": "deepseek_v4_mean_hc_head"}
        if dsv4
        else None,
        "future_extension": {"enabled": True, "values": [1, None, "custom"]},
        "speculators_config": {
            "algorithm": "dspark" if dsv4 else "mmuse",
            "default_proposal_method": "greedy",
            "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": 7}],
            "verifier": {
                "name_or_path": "target-dsv4" if dsv4 else "target-qwen",
                "architectures": ["DeepseekV4ForCausalLM"]
                if dsv4
                else ["Qwen3ForCausalLM"],
            },
        },
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 4096 if dsv4 else 2560,
            "head_dim": 256 if dsv4 else 128,
            "num_hidden_layers": 5,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "layer_types": ["sliding_attention" if dsv4 else "full_attention"] * 5,
            "use_sliding_window": dsv4,
            "sliding_window": 2048 if dsv4 else None,
        },
    }


@pytest.mark.parametrize("dsv4", [False, True])
def test_normalization_preserves_model_settings_and_original_versions(dsv4):
    original = _checkpoint(dsv4=dsv4)
    untouched = copy.deepcopy(original)
    normalized = normalize_mmuse_checkpoint_config(original)
    expected = copy.deepcopy(original)
    expected.update(
        architectures=["MMuseDraftModel"],
        speculators_model_type="mmuse",
        markov_init_std=None,
    )
    expected["speculators_config"]["algorithm"] = "mmuse"
    expected.pop("auto_map")
    assert normalized == expected
    assert original == untouched
    assert normalize_mmuse_checkpoint_config(normalized) == normalized

    config = SpeculatorModelConfig.from_dict(normalized)
    assert type(config) is MMuseSpeculatorConfig
    with torch.device("meta"):
        attention = Qwen3DFlashAttention(config.transformer_layer_config, layer_idx=0)
    assert attention.q_proj.weight.shape == ((8192, 4096) if dsv4 else (4096, 2560))
    assert config.correction_hidden_feedback is dsv4
    assert config.correction_project_corrected_hidden is not dsv4
    assert config.sample_from_anchor


def test_explicit_initialization_and_valid_auto_mapping_are_retained():
    original = _checkpoint()
    original["auto_map"]["AutoConfig"] = "custom.Config"
    original["markov_init_std"] = 0.01
    normalized = normalize_mmuse_checkpoint_config(original)
    assert normalized["auto_map"] == {"AutoConfig": "custom.Config"}
    assert normalized["markov_init_std"] == 0.01


@pytest.mark.parametrize("model_type", ["dspark", "dflash", "dflash2"])
def test_other_algorithms_are_not_relabeled(model_type):
    original = _checkpoint()
    original["speculators_model_type"] = model_type
    original["speculators_config"]["algorithm"] = model_type
    if model_type == "dspark":
        for key in (
            "enable_correction_head",
            "dflash2_dynamic_conv",
            "correction_project_corrected_hidden",
        ):
            original[key] = False
        original["correction_output_mode"] = "hidden"
    with pytest.raises(ValueError, match="algorithm|cannot become MMuse"):
        normalize_mmuse_checkpoint_config(original)


@pytest.mark.parametrize("payload", [[], {}, {"speculators_model_type": "mmuse"}])
def test_non_checkpoint_json_is_rejected(payload):
    with pytest.raises(ValueError, match="checkpoint"):
        normalize_mmuse_checkpoint_config(payload)


def test_cli_in_place_backs_up_config_and_leaves_other_files_untouched(tmp_path):
    source = tmp_path / "config.json"
    original = json.dumps(_checkpoint(dsv4=True)).encode()
    source.write_bytes(original)
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weight bytes must stay unchanged")
    provenance = tmp_path / "train_command.txt"
    provenance.write_bytes(b"original training command")
    main([str(tmp_path), "--in-place"])
    assert (tmp_path / "config.json.before-normalize.bak").read_bytes() == original
    assert json.loads(source.read_text())["speculators_model_type"] == "mmuse"
    assert weights.read_bytes() == b"weight bytes must stay unchanged"
    assert provenance.read_bytes() == b"original training command"
    first_result = source.read_bytes()
    main([str(tmp_path), "--in-place"])
    assert source.read_bytes() == first_result
    assert (tmp_path / "config.json.before-normalize.bak").read_bytes() == original


def test_cli_preview_and_output_preserve_source(tmp_path, capsys):
    source = tmp_path / "config.json"
    original = json.dumps(_checkpoint()).encode()
    source.write_bytes(original)
    main([str(source)])
    preview = json.loads(capsys.readouterr().out)
    assert "auto_map" not in preview
    output = tmp_path / "normalized.json"
    main([str(source), "--output", str(output)])
    assert json.loads(output.read_text()) == preview
    assert source.read_bytes() == original
    with pytest.raises(SystemExit) as exc:
        main([str(source), "--output", str(output)])
    assert exc.value.code == 2
    assert json.loads(output.read_text()) == preview


def test_cli_existing_backup_is_not_overwritten(tmp_path):
    source = tmp_path / "config.json"
    original = json.dumps(_checkpoint()).encode()
    source.write_bytes(original)
    backup = tmp_path / "config.json.before-normalize.bak"
    backup.write_bytes(b"older backup")
    with pytest.raises(SystemExit) as exc:
        main([str(source), "--in-place"])
    assert exc.value.code == 2
    assert source.read_bytes() == original
    assert backup.read_bytes() == b"older backup"
