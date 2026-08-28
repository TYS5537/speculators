"""Focused coverage for the standalone upstream DFlash2 implementation."""

from typing import Any

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from speculators.models.dflash2.metrics import selector_training_candidates
from speculators.models.dflash2.model_definitions import (
    CandidateSelector,
    GroupedDynamicCausalConv,
    grouped_dynamic_conv,
)


def test_grouped_dynamic_conv_stays_inside_proposal_blocks():
    hidden = torch.zeros(1, 8, 4)
    hidden[:, 3] = 7
    delta = torch.zeros(1, 8, 2, 2)
    base = torch.zeros(2, 4)
    base[1] = 1

    output = grouped_dynamic_conv(
        hidden,
        delta,
        base,
        block_size=4,
        group_size=2,
    )

    assert torch.count_nonzero(output) == 0


def test_grouped_dynamic_conv_identity_start_has_gradients():
    torch.manual_seed(0)
    module = GroupedDynamicCausalConv(
        8,
        block_size=4,
        kernel_size=2,
        group_size=2,
    )
    hidden = torch.randn(2, 4, 8, requires_grad=True)

    prepared, output_kernel = module.prepare(hidden)
    output = module.finish(prepared.square(), output_kernel)
    torch.testing.assert_close(prepared, hidden)
    torch.testing.assert_close(output, hidden.square())

    output.sum().backward()
    assert module.base_kernel.grad is not None
    assert module.kernel_projection.weight.grad is not None
    assert torch.isfinite(module.base_kernel.grad).all()
    assert torch.isfinite(module.kernel_projection.weight.grad).all()


def test_candidate_selector_only_reranks_unary_top_k():
    selector = CandidateSelector(
        vocab_size=7,
        hidden_size=4,
        rank=3,
        top_k=2,
    )
    unary = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    hidden = torch.randn(1, 1, 4)
    predecessor_ids = torch.tensor([[1]])

    candidate_ids, scores = selector.select(unary, hidden, predecessor_ids)

    assert candidate_ids.tolist() == [[[6, 5]]]
    assert scores.shape == (1, 1, 2)


def _tiny_config(**overrides) -> DFlash2SpeculatorConfig:
    transformer_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    values: dict[str, Any] = {
        "transformer_layer_config": transformer_config,
        "draft_vocab_size": 64,
        "block_size": 4,
        "aux_hidden_state_layer_ids": [0, 1],
        "mask_token_id": 0,
        "conv_kernel_size": 2,
        "conv_group_size": 4,
        "selector_rank": 8,
        "selector_top_k": 4,
    }
    values.update(overrides)
    return DFlash2SpeculatorConfig(**values)


def test_model_checkpoint_keys_match_upstream_contract():
    model = DFlash2DraftModel(_tiny_config())
    state_keys = set(model.state_dict())

    expected = {
        "layers.0.attention_conv.base_kernel",
        "layers.0.attention_conv.kernel_projection.weight",
        "layers.0.mlp_conv.base_kernel",
        "layers.0.mlp_conv.kernel_projection.weight",
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
        "candidate_selector.hidden_projection.weight",
    }
    assert expected <= state_keys


def test_model_requires_full_vocab_and_anchor_as_reserved_slot():
    with pytest.raises(ValueError, match="full verifier vocabulary"):
        DFlash2DraftModel(_tiny_config(draft_vocab_size=32))
    with pytest.raises(ValueError, match="sample_from_anchor=False"):
        DFlash2DraftModel(_tiny_config(sample_from_anchor=True))


def test_predecessor_ids_match_teacher_forced_runtime_alignment():
    model = DFlash2DraftModel(_tiny_config())
    input_ids = torch.arange(16).unsqueeze(0)
    block_indices = torch.tensor([2, 3, 4, 5, 8, 9, 10, 11])

    predecessor_ids = model._predecessor_ids(input_ids, block_indices)

    expected = torch.tensor([[2, 2, 3, 4], [8, 8, 9, 10]])
    torch.testing.assert_close(predecessor_ids, expected)


def test_selector_loss_injects_missing_target_without_mutating_runtime_top_k():
    candidate_ids = torch.tensor([[[5, 4, 3], [5, 4, 3]]])
    original = candidate_ids.clone()
    target_ids = torch.tensor([[4, 0]])

    training_ids, target_positions, contains_target = (
        selector_training_candidates(candidate_ids, target_ids)
    )

    assert training_ids.tolist() == [[[5, 4, 3], [5, 4, 0]]]
    assert target_positions.tolist() == [[1, 2]]
    assert contains_target.tolist() == [[True, False]]
    torch.testing.assert_close(candidate_ids, original)
