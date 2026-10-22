"""Checkpoint names must identify the same model before and after serialization."""

import copy
import json
from unittest.mock import Mock

import pytest
import torch
from transformers import PreTrainedModel, Qwen3Config

from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models import (
    DFlash2SpeculatorConfig,
    DFlashSpeculatorConfig,
    DSparkSpeculatorConfig,
    Eagle3SpeculatorConfig,
    MMuseSpeculatorConfig,
    MTPSpeculatorConfig,
    PEagleSpeculatorConfig,
)
from speculators.models.dflash import DFlashDraftModel
from speculators.models.dflash2 import DFlash2DraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig

CONFIG_CASES = [
    (DFlashSpeculatorConfig, "dflash", "DFlashDraftModel", "DFlashSpeculator"),
    (DFlash2SpeculatorConfig, "dflash2", "DFlash2DraftModel", None),
    (DSparkSpeculatorConfig, "dspark", "DSparkDraftModel", "DSparkSpeculator"),
    (MMuseSpeculatorConfig, "mmuse", "MMuseDraftModel", "MMuseSpeculator"),
    (Eagle3SpeculatorConfig, "eagle3", "Eagle3DraftModel", "Eagle3Speculator"),
    (PEagleSpeculatorConfig, "peagle", "PEagleDraftModel", "PEagleSpeculator"),
    (MTPSpeculatorConfig, "mtp", "MTPDraftModel", None),
]


def _config(config_class, algorithm, **kwargs):
    return config_class(
        speculators_config=SpeculatorsConfig(
            algorithm=algorithm,
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None, architectures=["Qwen3ForCausalLM"]
            ),
        ),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("config_class", "algorithm", "architecture", "legacy"), CONFIG_CASES
)
def test_config_only_save_identifies_registered_model(
    tmp_path, config_class, algorithm, architecture, legacy
):
    config = _config(config_class, algorithm)
    config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["architectures"] == [architecture]
    assert (
        saved["speculators_model_type"]
        == saved["speculators_config"]["algorithm"]
        == algorithm
    )
    assert "auto_map" not in saved
    assert not (tmp_path / "config.py").exists()
    restored = SpeculatorModelConfig.from_pretrained(tmp_path)
    assert type(restored) is config_class
    assert (
        SpeculatorModel.registered_model_class_from_config(restored).__name__
        == architecture
    )


@pytest.mark.parametrize(
    ("config_class", "algorithm", "architecture", "legacy"),
    [case for case in CONFIG_CASES if case[3]],
)
def test_old_architecture_names_and_empty_auto_map_resave_cleanly(
    tmp_path, config_class, algorithm, architecture, legacy
):
    old = _config(config_class, algorithm).to_dict()
    old["architectures"] = [legacy]
    old["auto_map"] = {"": f"config.{config_class.__name__}"}
    original = copy.deepcopy(old)
    restored = SpeculatorModelConfig.from_dict(old)
    restored.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["architectures"] == [architecture]
    assert "auto_map" not in saved
    assert saved["speculators_config"] == original["speculators_config"]
    assert old == original


def test_valid_auto_map_entries_are_preserved(tmp_path):
    mappings = {"": "config.MMuseSpeculatorConfig", "AutoConfig": "custom.Config"}
    config = _config(MMuseSpeculatorConfig, "mmuse", auto_map=mappings)
    config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["auto_map"] == {"AutoConfig": "custom.Config"}
    assert mappings == {
        "": "config.MMuseSpeculatorConfig",
        "AutoConfig": "custom.Config",
    }


def test_explicit_hf_auto_class_registration_still_exports_config(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(MMuseSpeculatorConfig, "_auto_class", "AutoConfig")
    config = _config(MMuseSpeculatorConfig, "mmuse")
    config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["auto_map"] == {"AutoConfig": "config.MMuseSpeculatorConfig"}
    assert (tmp_path / "config.py").is_file()


@pytest.mark.parametrize(
    "config_class",
    [DSparkSpeculatorConfig, MMuseSpeculatorConfig, DFlash2SpeculatorConfig],
)
def test_real_checkpoint_load_keeps_identity_and_trainable_weights(
    tmp_path, config_class
):
    algorithm = config_class.model_fields["speculators_model_type"].default
    transformer = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        layer_types=["full_attention"],
    )
    transformer._attn_implementation = "eager"
    config = _config(
        config_class,
        algorithm,
        transformer_layer_config=transformer,
        draft_vocab_size=32,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
    )
    model_class = SpeculatorModel.registered_model_class_from_config(config)
    original = model_class(config)
    original.save_pretrained(tmp_path)
    restored = SpeculatorModel.from_pretrained(tmp_path, local_files_only=True)
    assert type(restored) is model_class
    assert restored.config.architectures == [model_class.__name__]
    assert restored.config.speculators_config.algorithm == algorithm
    assert restored.state_dict().keys() == original.state_dict().keys()
    for name, value in original.named_parameters():
        if value.requires_grad:
            torch.testing.assert_close(
                restored.get_parameter(name), value, rtol=0, atol=0
            )


@pytest.mark.parametrize(
    ("loader", "config_class", "algorithm"),
    [
        (DFlashDraftModel, DSparkSpeculatorConfig, "dspark"),
        (DFlashDraftModel, DFlash2SpeculatorConfig, "dflash2"),
        (DFlash2DraftModel, MMuseSpeculatorConfig, "mmuse"),
    ],
)
def test_explicit_wrong_model_class_fails_before_loading_weights(
    monkeypatch, loader, config_class, algorithm
):
    load_weights = Mock(
        side_effect=AssertionError("must reject before loading weights")
    )
    monkeypatch.setattr(PreTrainedModel, "from_pretrained", load_weights)
    with pytest.raises(ValueError, match="checkpoint.*type|config.*type"):
        loader.from_pretrained("unused", config=_config(config_class, algorithm))
    load_weights.assert_not_called()
