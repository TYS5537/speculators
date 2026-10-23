"""Standalone DFlash2 backbone and sequential-selector integration in offline eval."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators_eval.draft_cache import DraftContextCache


@pytest.fixture
def evaluator():
    path = Path(__file__).parents[3] / "scripts/evaluate/dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("dflash2_eval_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.torch = torch
    yield module
    sys.modules.pop(spec.name, None)


def _draft(attention):
    torch.manual_seed(71)
    decoder = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        layer_types=["sliding_attention"],
        sliding_window=5,
    )
    decoder._attn_implementation = attention
    model = DFlash2DraftModel(
        DFlash2SpeculatorConfig(
            transformer_layer_config=decoder,
            draft_vocab_size=32,
            block_size=4,
            aux_hidden_state_layer_ids=[0, 1],
            mask_token_id=0,
            conv_group_size=4,
            selector_rank=4,
            selector_top_k=4,
            speculators_config=SpeculatorsConfig(
                algorithm="dflash2",
                proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=None, architectures=["Qwen3ForCausalLM"]
                ),
            ),
        )
    ).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.fill_(1)
            elif "base_kernel" in name:
                parameter.add_(torch.randn_like(parameter) * 0.05)
            else:
                parameter.normal_(std=0.15)
    return model


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_standalone_backbone_selector_and_context_reuse(
    evaluator, attention, temperature
):
    draft = _draft(attention)
    # These optional DSpark/MMuse modules must not be required by the runner.
    assert not hasattr(draft, "correction_head")
    assert not hasattr(draft, "markov_head")
    runner = evaluator.DSparkOfflineRunner(
        torch.nn.Linear(1, 1),
        draft,
        None,
        SimpleNamespace(device="cpu", draft_kv_reuse=True, temperature=temperature),
    )
    assert runner.first_draft_slot == 1
    assert runner.max_proposal_tokens == 3
    cache = DraftContextCache()
    states = torch.randn(1, 14, 32)
    tokens = torch.randint(0, 32, (1, 14))
    with torch.inference_mode():
        for start in (3, 7, 8, 11):
            hidden, logits = runner._single_anchor_backbone(
                states[:, :start], tokens, start
            )
            cached_hidden, cached_logits = runner._single_anchor_backbone(
                states[:, :start], tokens, start, cache=cache
            )
            torch.testing.assert_close(cached_hidden, hidden, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(cached_logits, logits, atol=2e-5, rtol=2e-5)
            proposed, probabilities = runner._sample_dspark_tokens(
                logits, hidden, tokens[:, start]
            )
            assert len(proposed) == 3
            assert probabilities.shape == (1, 3, 32)
            previous = tokens[:, start : start + 1]
            for offset, token in enumerate(proposed):
                slot = offset + 1
                candidates, scores = draft.candidate_selector.select(
                    logits[:, slot : slot + 1],
                    hidden[:, slot : slot + 1],
                    previous,
                )
                expected = torch.zeros_like(logits[:, slot : slot + 1]).float()
                expected.scatter_(
                    -1, candidates, evaluator.logits_to_probs(scores, temperature)
                )
                torch.testing.assert_close(
                    probabilities[:, offset : offset + 1], expected
                )
                assert probabilities[0, offset, token] > 0
                previous = torch.tensor([[token]])


def test_saved_standalone_config_dispatches_through_shared_loader(evaluator, tmp_path):
    draft = _draft("eager")
    draft.config.save_pretrained(tmp_path)

    loaded = evaluator._load_draft_config(tmp_path)

    assert isinstance(loaded, DFlash2SpeculatorConfig)
    assert loaded.speculators_model_type == "dflash2"
    assert loaded.sample_from_anchor is False
    assert loaded.block_size == 4
