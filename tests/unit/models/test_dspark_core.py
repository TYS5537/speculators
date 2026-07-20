"""Focused alignment and rollout tests for the DSpark correction path."""

from types import MethodType

import torch
from torch import nn
from transformers import Qwen3Config

from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.metrics import resolve_loss_config


class _RecordingCorrectionHead(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.seen_token_ids: list[torch.Tensor] = []

    def forward(
        self,
        previous_token_embeddings,
        dflash_hidden,
        cache=None,
        *,
        use_cache=False,
    ):
        self.seen_token_ids.append(previous_token_embeddings.argmax(dim=-1).detach())
        shape = (*dflash_hidden.shape[:-1], self.vocab_size)
        correction = dflash_hidden.new_zeros(shape)
        states = dflash_hidden
        next_cache = [] if use_cache else None
        return correction, states, next_cache


def _tiny_model() -> DSparkDraftModel:
    transformer_config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=32,
        layer_types=["full_attention"],
    )
    config = DSparkSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=32,
        block_size=4,
        max_anchors=1,
        aux_hidden_state_layer_ids=[0],
        mask_token_id=0,
        markov_rank=0,
        enable_correction_head=True,
        correction_hidden_size=32,
        correction_rank=8,
        correction_num_layers=1,
        correction_num_heads=4,
        enable_confidence_head=False,
        confidence_head_with_markov=False,
        # Exercise the classic slots-1 teacher-forcing path.
        sample_from_anchor=False,
    )
    model = DSparkDraftModel(config)
    with torch.no_grad():
        model.embed_tokens.weight.copy_(torch.eye(32))
        model.lm_head.weight.copy_(torch.eye(32))
        model.verifier_lm_head.weight.copy_(torch.eye(32))
    return model


def test_teacher_forcing_shifts_only_gt_tokens():
    model = _tiny_model()
    recording_head = _RecordingCorrectionHead(vocab_size=32)
    model.correction_head = recording_head

    hidden = torch.randn(1, 4, 32)
    base_logits = model.lm_head(hidden)
    targets = torch.randn(1, 4, 32)
    aligned_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    anchored_indices = torch.arange(4)

    def fake_backbone(self, *args, **kwargs):
        return hidden, base_logits, targets, aligned_mask, anchored_indices

    model._backbone_forward = MethodType(fake_backbone, model)
    input_ids = torch.tensor([[7, 11, 13, 17]])
    model(
        hidden_states=torch.empty(1, 4, 32),
        input_ids=input_ids,
        loss_mask=torch.ones(1, 4),
        verifier_last_hidden_states=torch.empty(1, 4, 32),
        document_ids=torch.zeros(1, 4, dtype=torch.long),
        loss_config=resolve_loss_config("tv"),
        max_anchors=1,
    )

    seen = recording_head.seen_token_ids[0]
    assert torch.equal(seen, torch.tensor([[7, 11, 13]]))
    assert torch.equal(base_logits, model.lm_head(hidden))


def test_rollout_feeds_generated_token_to_next_position():
    model = _tiny_model()
    recording_head = _RecordingCorrectionHead(vocab_size=32)
    model.correction_head = recording_head

    base_logits = torch.full((1, 4, 32), -100.0)
    base_logits[0, 1, 3] = 100.0
    base_logits[0, 2, 5] = 100.0
    base_logits[0, 3, 9] = 100.0
    hidden = torch.randn(1, 4, 32)

    tokens, corrected_logits = model.rollout_correction(
        base_logits, hidden, anchor_token_ids=torch.tensor([2])
    )
    assert torch.equal(tokens, torch.tensor([[3, 5, 9]]))
    assert corrected_logits.shape == (1, 3, 32)
    seen = torch.cat(recording_head.seen_token_ids, dim=1)
    assert torch.equal(seen, torch.tensor([[2, 3, 5]]))
