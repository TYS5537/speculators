"""Offline evaluation must reload borrowed weights from the selected verifier."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.models.mmuse import MMuseDraftModel, MMuseSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators_eval.cli import parse_args


@pytest.mark.parametrize(
    ("model_class", "config_class"),
    [
        (DSparkDraftModel, DSparkSpeculatorConfig),
        (MMuseDraftModel, MMuseSpeculatorConfig),
        (DFlash2DraftModel, DFlash2SpeculatorConfig),
    ],
)
@pytest.mark.parametrize("old_target_exists", [False, True])
def test_relocated_verifier_supplies_draft_weights(
    tmp_path, monkeypatch, model_class, config_class, old_target_exists
):
    path = Path(__file__).parents[3] / "scripts/evaluate/dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("offline_verifier_test", path)
    evaluator = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, evaluator)
    spec.loader.exec_module(evaluator)
    evaluator.torch = torch

    target_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        layer_types=["full_attention", "full_attention"],
    )
    target = Qwen3ForCausalLM(target_config).eval()
    target_path = tmp_path / "evaluation-target"
    target.save_pretrained(target_path)
    old_path = tmp_path / "unavailable-training-target"
    if old_target_exists:
        # An existing old path must not silently supply another model's weights.
        old_target = Qwen3ForCausalLM(target_config)
        old_target.save_pretrained(old_path)
    algorithm = config_class.model_fields["speculators_model_type"].default
    config = config_class(
        transformer_layer_config=target_config,
        draft_vocab_size=32,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm=algorithm,
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=str(old_path), architectures=["Qwen3ForCausalLM"]
            ),
        ),
    )
    original = model_class(config).eval()
    checkpoint_path = tmp_path / "draft"
    original.save_pretrained(checkpoint_path)
    saved_config = (checkpoint_path / "config.json").read_bytes()
    args = parse_args(
        [
            "--verifier-model",
            str(target_path),
            "--draft-model",
            str(checkpoint_path),
            "--datasets-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--draft-attn-impl",
            "sdpa",
        ]
    )
    loaded, loaded_config = evaluator._load_evaluation_draft(
        args,
        evaluator._TargetSetup("hf", target_config.to_dict(), None, None),
        target,
        torch.device("cpu"),
        model_class=SpeculatorModel,
    )

    assert isinstance(loaded, model_class)
    assert loaded_config.speculators_config.verifier.name_or_path == str(target_path)
    torch.testing.assert_close(
        loaded.embed_tokens.weight, target.model.embed_tokens.weight
    )
    torch.testing.assert_close(loaded.lm_head.weight, target.lm_head.weight)
    torch.testing.assert_close(loaded.verifier_lm_head.weight, target.lm_head.weight)
    torch.testing.assert_close(loaded.verifier_norm.weight, target.model.norm.weight)
    torch.testing.assert_close(loaded.fc.weight, original.fc.weight)
    assert (checkpoint_path / "config.json").read_bytes() == saved_config
