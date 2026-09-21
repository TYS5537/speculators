"""Architecture boundaries, disabled-feature parity and local Muse round trips."""

import copy
import json
from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.models.muse import MuseDraftModel, MuseSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _config(config_class, **features):
    algorithm = config_class.model_fields["speculators_model_type"].default
    transformer = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        layer_types=["full_attention"] * 2,
    )
    transformer._attn_implementation = "eager"
    return config_class(
        transformer_layer_config=transformer,
        draft_vocab_size=32,
        block_size=3,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm=algorithm,
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=2)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None, architectures=["Qwen3ForCausalLM"]
            ),
        ),
        **features,
    )


def test_muse_is_registered_and_config_roundtrip_preserves_features():
    config = _config(
        MuseSpeculatorConfig,
        enable_correction_head=True,
        correction_hidden_size=16,
        correction_num_heads=4,
        correction_rank=8,
        dflash_gated_layer_fusion=True,
        dflash2_dynamic_conv=True,
        dflash2_conv_group_size=4,
    )
    restored = SpeculatorModelConfig.from_dict(config.to_dict())
    assert type(restored) is MuseSpeculatorConfig
    assert restored.to_dict() == config.to_dict()
    assert (
        SpeculatorModel.registered_model_class_from_config(restored) is MuseDraftModel
    )
    assert "enable_correction_head" not in DSparkSpeculatorConfig.model_fields
    assert "dflash_gated_layer_fusion" not in DFlashSpeculatorConfig.model_fields
    assert not hasattr(DSparkDraftModel, "rollout_correction")
    assert not hasattr(DFlashDraftModel, "dflash2_select_candidates")


@pytest.mark.parametrize("model_class", [DFlashDraftModel, DSparkDraftModel])
@pytest.mark.parametrize("feature", ["enable_correction_head", "dflash2_dynamic_conv"])
def test_baseline_training_factory_rejects_extensions_before_loading_verifier(
    monkeypatch, model_class, feature
):
    verifier_loader = Mock(side_effect=AssertionError("verifier must not load"))
    monkeypatch.setattr(VerifierConfig, "from_pretrained", verifier_loader)
    with pytest.raises(ValueError, match="muse"):
        model_class.from_training_args(
            _config(DFlashSpeculatorConfig).transformer_layer_config,
            verifier_name_or_path="unused-local-verifier",
            draft_vocab_size=32,
            target_layer_ids=[0, 1],
            **{feature: True},
        )
    verifier_loader.assert_not_called()


@pytest.mark.parametrize("model_class", [DFlashDraftModel, DSparkDraftModel])
@pytest.mark.parametrize("disabled", [None, False])
def test_baseline_training_factory_accepts_disabled_legacy_flags(
    monkeypatch, model_class, disabled
):
    template = _config(DFlashSpeculatorConfig)
    monkeypatch.setattr(
        VerifierConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: template.speculators_config.verifier,
    )
    monkeypatch.setattr(model_class, "load_verifier_weights", lambda _self: None)
    model = model_class.from_training_args(
        template.transformer_layer_config,
        verifier_name_or_path="unused-local-verifier",
        draft_vocab_size=32,
        target_layer_ids=[0, 1],
        enable_correction_head=disabled,
        dflash2_dynamic_conv=disabled,
        draft_attn_impl="eager",
    )
    assert type(model) is model_class


@pytest.mark.parametrize("feature", ["enable_correction_head", "dflash2_dynamic_conv"])
def test_muse_training_factory_keeps_enabled_features(monkeypatch, feature):
    template = _config(DFlashSpeculatorConfig)
    monkeypatch.setattr(
        VerifierConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: template.speculators_config.verifier,
    )
    monkeypatch.setattr(MuseDraftModel, "load_verifier_weights", lambda _self: None)
    model = MuseDraftModel.from_training_args(
        template.transformer_layer_config,
        verifier_name_or_path="unused-local-verifier",
        draft_vocab_size=32,
        target_layer_ids=[0, 1],
        markov_rank=4,
        correction_hidden_size=16,
        correction_num_heads=4,
        correction_rank=8,
        dflash2_conv_group_size=4,
        draft_attn_impl="eager",
        **{feature: True},
    )
    assert getattr(model.config, feature) is True
    assert model.config.speculators_config.algorithm == "muse"
    assert model.config.sample_from_anchor


@pytest.mark.parametrize("baseline", ["dflash", "vanilla", "gated", "rnn"])
@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_disabled_muse_preserves_baseline_initialization_and_backbone(
    baseline, sample_from_anchor
):
    if baseline == "dflash":
        config_class, model_class = DFlashSpeculatorConfig, DFlashDraftModel
        baseline_features = {}
        muse_features = {"markov_rank": 0, "enable_confidence_head": False}
    else:
        config_class, model_class = DSparkSpeculatorConfig, DSparkDraftModel
        baseline_features = {"markov_rank": 4, "markov_head_type": baseline}
        muse_features = baseline_features
    torch.manual_seed(91)
    original = model_class(
        _config(
            config_class, sample_from_anchor=sample_from_anchor, **baseline_features
        )
    ).eval()
    torch.manual_seed(91)
    muse = MuseDraftModel(
        _config(
            MuseSpeculatorConfig,
            sample_from_anchor=sample_from_anchor,
            **muse_features,
        )
    ).eval()
    assert original.state_dict().keys() == muse.state_dict().keys()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(
            value, muse.state_dict()[name], rtol=0, atol=0, equal_nan=True, msg=name
        )
    with torch.no_grad():
        for name in ("embed_tokens", "lm_head", "verifier_lm_head"):
            weight = getattr(original, name).weight
            weight.normal_(std=0.1)
            getattr(muse, name).weight.copy_(weight)
    inputs = {
        "hidden_states": torch.randn(1, 8, 32),
        "input_ids": torch.arange(1, 9).unsqueeze(0),
        "loss_mask": torch.ones(1, 8),
        "verifier_last_hidden_states": torch.randn(1, 8, 16),
        "document_ids": torch.zeros(1, 8, dtype=torch.long),
        "max_anchors": 2,
    }
    torch.manual_seed(123)
    expected = original._backbone_forward(**inputs)
    torch.manual_seed(123)
    actual = muse._backbone_forward(**inputs)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    if baseline != "dflash":
        torch.manual_seed(123)
        _, expected_loss, expected_metrics = original(**inputs)
        torch.manual_seed(123)
        _, actual_loss, actual_metrics = muse(**inputs)
        torch.testing.assert_close(expected_loss, actual_loss, rtol=0, atol=0)
        for name, value in expected_metrics.items():
            torch.testing.assert_close(value, actual_metrics[name], rtol=0, atol=0)


@pytest.mark.parametrize("legacy_loader", [None, "generic", "dspark", "dspark_config"])
def test_muse_local_checkpoint_roundtrip_preserves_trainable_weights(
    tmp_path, legacy_loader
):
    model = MuseDraftModel(
        _config(
            MuseSpeculatorConfig,
            enable_correction_head=True,
            correction_hidden_size=16,
            correction_num_heads=4,
            correction_rank=8,
            correction_output_mode="logits",
            correction_hidden_feedback=True,
            dflash_gated_layer_fusion=True,
            dflash2_dynamic_conv=True,
            dflash2_conv_group_size=4,
            dflash2_candidate_selector=True,
            dflash2_selector_rank=4,
            dflash2_selector_top_k=4,
        )
    )
    expected = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    model.save_pretrained(tmp_path)
    loader = SpeculatorModel
    loader_kwargs = {}
    if legacy_loader is not None:
        legacy = model.config.to_dict()
        legacy["speculators_model_type"] = "dspark"
        legacy["architectures"] = ["DSparkSpeculator"]
        legacy["speculators_config"]["algorithm"] = "dspark"
        (tmp_path / "config.json").write_text(json.dumps(legacy), encoding="utf-8")
        if legacy_loader in {"dspark", "dspark_config"}:
            loader = DSparkDraftModel
        if legacy_loader == "dspark_config":
            # Direct legacy config objects must not bypass the migration path.
            loader_kwargs["config"] = DSparkSpeculatorConfig(**legacy)
    warning_context = (
        pytest.warns(UserWarning, match="legacy DSpark")
        if legacy_loader is not None
        else nullcontext()
    )
    with warning_context:
        loaded = loader.from_pretrained(
            tmp_path, local_files_only=True, **loader_kwargs
        )
    assert type(loaded) is MuseDraftModel
    assert loaded.config.speculators_model_type == "muse"
    assert loaded.config.speculators_config.algorithm == "muse"
    for name, value in loaded.named_parameters():
        if name in expected:
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)


def test_legacy_enhanced_dspark_config_migrates_without_changing_options():
    config = _config(MuseSpeculatorConfig, enable_correction_head=True)
    legacy = config.to_dict()
    legacy["speculators_model_type"] = "dspark"
    legacy["architectures"] = ["DSparkSpeculator"]
    legacy["speculators_config"]["algorithm"] = "dspark"
    untouched = copy.deepcopy(legacy)
    with pytest.warns(UserWarning, match="legacy DSpark"):
        restored = SpeculatorModelConfig.from_dict(legacy)
    assert type(restored) is MuseSpeculatorConfig
    assert restored.enable_correction_head
    assert restored.correction_rank == config.correction_rank
    assert restored.sample_from_anchor == config.sample_from_anchor
    assert legacy == untouched
