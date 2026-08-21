"""Focused unit tests for DSpark correction integration helpers."""

from types import SimpleNamespace

import torch
from torch import nn

from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.model_definitions import MarkovHead


class _RecordingCorrectionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.previous_embeddings: list[torch.Tensor] = []
        self.current_embeddings: list[torch.Tensor] = []
        self.block_positions: list[torch.Tensor] = []
        self.previous_logits: list[torch.Tensor] = []
        self.previous_logits_masks: list[torch.Tensor] = []

    def forward(
        self,
        previous_embeddings,
        dflash_hidden,
        block_positions,
        previous_logits=None,
        previous_logits_mask=None,
        current_token_embeddings=None,
        cache=None,
        *,
        use_cache=False,
    ):
        del cache
        self.previous_embeddings.append(previous_embeddings.detach().clone())
        if current_token_embeddings is not None:
            self.current_embeddings.append(current_token_embeddings.detach().clone())
        self.block_positions.append(block_positions.detach().clone())
        if previous_logits is not None:
            assert previous_logits_mask is not None
            self.previous_logits.append(previous_logits.detach().clone())
            self.previous_logits_masks.append(previous_logits_mask.detach().clone())
        states = dflash_hidden.new_zeros(*dflash_hidden.shape[:-1], 4)
        delta_hidden = (
            torch.nn.functional.one_hot(
                block_positions, num_classes=dflash_hidden.shape[-1]
            ).to(dflash_hidden.dtype)
            * self.scale
        )
        next_cache = [] if use_cache else None
        return delta_hidden, states, next_cache


class _FusedRecordingCorrectionHead(_RecordingCorrectionHead):
    def forward(self, *args, **kwargs):
        delta_hidden, _, next_cache = super().forward(*args, **kwargs)
        return delta_hidden, delta_hidden, next_cache

    @staticmethod
    def fused_lm_head_residual(
        causal_states,
        lm_head_weight,
    ):
        return torch.nn.functional.linear(causal_states, lm_head_weight)


class _RecordingLogitCorrectionHead(nn.Module):
    output_mode = "logits"

    def __init__(self, draft_vocab_size: int) -> None:
        super().__init__()
        self.draft_vocab_size = draft_vocab_size
        self.previous_logits: list[torch.Tensor] = []
        self.previous_logits_masks: list[torch.Tensor] = []

    def forward(
        self,
        previous_embeddings,
        dflash_hidden,
        block_positions,
        previous_logits=None,
        previous_logits_mask=None,
        previous_rank_features=None,
        current_token_embeddings=None,
        cache=None,
        *,
        use_cache=False,
    ):
        del previous_embeddings, current_token_embeddings, cache
        assert previous_logits is not None or previous_rank_features is not None
        assert previous_logits_mask is not None
        recorded = (
            previous_logits if previous_logits is not None else previous_rank_features
        )
        self.previous_logits.append(recorded.detach().clone())
        self.previous_logits_masks.append(previous_logits_mask.detach().clone())
        states = dflash_hidden.new_zeros(*dflash_hidden.shape[:-1], 4)
        token_ids = block_positions + 1
        delta_logits = 10.0 * torch.nn.functional.one_hot(
            token_ids, num_classes=self.draft_vocab_size
        ).to(dflash_hidden.dtype)
        next_cache = [] if use_cache else None
        return delta_logits, states, next_cache

    @staticmethod
    def auxiliary_hidden_residual(
        causal_states,
    ):
        delta_hidden = causal_states.new_zeros(*causal_states.shape[:-1], 4)
        delta_hidden[..., 0] = 2.0
        return delta_hidden

    def fused_lm_head_residual(self, causal_states, lm_head_weight):
        return torch.nn.functional.linear(
            self.auxiliary_hidden_residual(causal_states),
            lm_head_weight,
        )

    def encode_previous_distribution(
        self,
        previous_logits_mask,
        *,
        previous_logits=None,
        candidate_ids=None,
        candidate_logits=None,
    ):
        del candidate_ids
        source = previous_logits if previous_logits is not None else candidate_logits
        assert source is not None
        rank = source.new_zeros(*source.shape[:-1], 1)
        rank = rank * previous_logits_mask.to(rank.dtype).unsqueeze(-1)
        return rank


class _RecordingHiddenFeedbackCorrectionHead(nn.Module):
    output_mode = "hidden"

    def __init__(self) -> None:
        super().__init__()
        self.previous_corrected_hidden: list[torch.Tensor] = []
        self.previous_corrected_hidden_masks: list[torch.Tensor] = []

    def forward(
        self,
        previous_embeddings,
        dflash_hidden,
        block_positions,
        previous_corrected_hidden=None,
        previous_corrected_hidden_mask=None,
        cache=None,
        *,
        use_cache=False,
    ):
        del previous_embeddings, cache
        assert previous_corrected_hidden is not None
        assert previous_corrected_hidden_mask is not None
        self.previous_corrected_hidden.append(
            previous_corrected_hidden.detach().clone()
        )
        self.previous_corrected_hidden_masks.append(
            previous_corrected_hidden_mask.detach().clone()
        )
        states = dflash_hidden.new_zeros(*dflash_hidden.shape[:-1], 4)
        delta_hidden = torch.nn.functional.one_hot(
            block_positions, num_classes=dflash_hidden.shape[-1]
        ).to(dflash_hidden.dtype)
        next_cache = [] if use_cache else None
        return delta_hidden, states, next_cache


class _RolloutHarness:
    candidate_selector = None
    _draft_ids_to_verifier = DSparkDraftModel._draft_ids_to_verifier
    _dflash2_proposal_logits = DSparkDraftModel._dflash2_proposal_logits
    dflash2_select_candidates = DSparkDraftModel.dflash2_select_candidates
    _replace_compact_feature_position = staticmethod(
        DSparkDraftModel._replace_compact_feature_position
    )
    _selector_correction_inputs = DSparkDraftModel._selector_correction_inputs
    _rollout_correction_steps = DSparkDraftModel._rollout_correction_steps
    rollout_correction = DSparkDraftModel.rollout_correction


class _CollaborationHarness:
    _apply_collaborative_markov = DSparkDraftModel._apply_collaborative_markov


class _CollaborativeRolloutHarness:
    candidate_selector = None
    _apply_collaborative_markov = DSparkDraftModel._apply_collaborative_markov
    _rollout_correction_steps = DSparkDraftModel._rollout_correction_steps
    rollout_correction = DSparkDraftModel.rollout_correction


class _CountingLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(value)


def test_rollout_uses_generated_token_as_next_feedback():
    harness = _RolloutHarness()
    harness.block_size = 3
    harness.config = SimpleNamespace(sample_from_anchor=True)
    harness.correction_head = _RecordingCorrectionHead()
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, 8)
    harness.d2t = None
    with torch.no_grad():
        harness.embed_tokens.weight.copy_(
            torch.arange(8, dtype=torch.float32).unsqueeze(-1).expand(-1, 4)
        )
        harness.lm_head.weight.zero_()
        harness.lm_head.weight[1, 0] = 10.0
        harness.lm_head.weight[2, 1] = 10.0
        harness.lm_head.weight[3, 2] = 10.0

    chosen_ids = torch.tensor([[1, 2, 3]])
    hidden = torch.zeros(1, 3, 4)
    tokens, _ = harness.rollout_correction(
        hidden,
        anchor_token_ids=torch.tensor([7]),
    )

    assert torch.equal(tokens, chosen_ids)
    feedback = torch.cat(harness.correction_head.previous_embeddings, dim=1)
    assert torch.equal(feedback[:, :, 0], torch.tensor([[7.0, 1.0, 2.0]]))
    positions = torch.cat(harness.correction_head.block_positions, dim=1)
    assert torch.equal(positions, torch.tensor([[0, 1, 2]]))
    assert harness.lm_head.calls == harness.block_size


def test_rollout_lm_head_fusion_projects_base_block_once():
    harness = _RolloutHarness()
    harness.block_size = 3
    harness.config = SimpleNamespace(
        sample_from_anchor=True,
        correction_lm_head_fusion=True,
    )
    harness.correction_head = _FusedRecordingCorrectionHead()
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, 8)
    harness.d2t = None
    with torch.no_grad():
        harness.embed_tokens.weight.copy_(
            torch.arange(8, dtype=torch.float32).unsqueeze(-1).expand(-1, 4)
        )
        harness.lm_head.weight.zero_()
        harness.lm_head.weight[1, 0] = 10.0
        harness.lm_head.weight[2, 1] = 10.0
        harness.lm_head.weight[3, 2] = 10.0

    tokens, logits = harness.rollout_correction(
        torch.zeros(1, 3, 4),
        anchor_token_ids=torch.tensor([7]),
    )

    assert torch.equal(tokens, torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, 8)
    assert harness.lm_head.calls == 1


def test_logit_residual_rollout_feeds_back_previous_final_logits():
    harness = _RolloutHarness()
    harness.block_size = 3
    harness.draft_vocab_size = 8
    harness.config = SimpleNamespace(sample_from_anchor=True)
    harness.correction_head = _RecordingLogitCorrectionHead(harness.draft_vocab_size)
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, harness.draft_vocab_size)
    harness.d2t = None
    with torch.no_grad():
        harness.lm_head.weight.zero_()

    tokens, logits = harness.rollout_correction(
        torch.zeros(1, 3, 4),
        anchor_token_ids=torch.tensor([7]),
    )

    assert torch.equal(tokens, torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, harness.draft_vocab_size)
    masks = torch.cat(harness.correction_head.previous_logits_masks, dim=1)
    assert torch.equal(masks, torch.tensor([[False, True, True]]))
    feedback = torch.cat(harness.correction_head.previous_logits, dim=1)
    assert torch.count_nonzero(feedback[:, 0]) == 0
    assert torch.equal(feedback[:, 1:], logits[:, :-1].detach())


def test_correction_rollout_uses_upstream_selector_path_and_keeps_dense_logits():
    harness = _RolloutHarness()
    harness.block_size = 3
    harness.config = SimpleNamespace(
        sample_from_anchor=True,
        correction_hidden_size=4,
    )
    harness.correction_head = _RecordingCorrectionHead()
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, 8)
    harness.draft_vocab_size = 8
    harness.d2t = None
    harness.candidate_selector = object()

    def select_path(_self, logits, hidden_states, anchor_token_ids):
        del hidden_states, anchor_token_ids
        candidate_ids = torch.tensor([[[4, 0], [5, 0], [6, 0]]], device=logits.device)
        candidate_logits = logits.new_tensor([[[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]]])
        selected_ids = torch.tensor([[4, 5, 6]], device=logits.device)
        return candidate_ids, candidate_logits, selected_ids

    harness.dflash2_select_path = select_path.__get__(harness)
    with torch.no_grad():
        harness.embed_tokens.weight.copy_(
            torch.arange(8, dtype=torch.float32).unsqueeze(-1).expand(-1, 4)
        )
        harness.lm_head.weight.zero_()
        harness.lm_head.weight[1, 0] = 10.0
        harness.lm_head.weight[2, 1] = 10.0
        harness.lm_head.weight[3, 2] = 10.0

    tokens, logits = harness.rollout_correction(
        torch.zeros(1, 3, 4),
        anchor_token_ids=torch.tensor([7]),
        base_logits=torch.zeros(1, 3, 8),
    )

    assert torch.equal(tokens, torch.tensor([[1, 2, 3]]))
    feedback = torch.cat(harness.correction_head.previous_embeddings, dim=1)
    assert torch.equal(feedback[:, :, 0], torch.tensor([[7.0, 4.0, 5.0]]))
    current = torch.cat(harness.correction_head.current_embeddings, dim=1)
    assert torch.equal(current[:, :, 0], torch.tensor([[4.0, 5.0, 6.0]]))
    assert not torch.isneginf(logits).any()
    assert harness.lm_head.calls == harness.block_size


def test_corrected_selector_feedback_uses_final_correction_token_next():
    harness = _RolloutHarness()
    harness.block_size = 3
    harness.config = SimpleNamespace(
        sample_from_anchor=True,
        selector_correction_feedback="corrected",
        correction_hidden_size=4,
    )
    harness.correction_head = _RecordingCorrectionHead()
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, 8)
    harness.draft_vocab_size = 8
    harness.d2t = None
    harness.candidate_selector = object()
    seen_previous = []

    def select_candidates(_self, logits, hidden_states, previous_token_ids):
        del hidden_states
        seen_previous.append(int(previous_token_ids.item()))
        candidate_ids = torch.tensor([[[4, 5]]], device=logits.device)
        candidate_logits = logits.new_tensor([[[10.0, 0.0]]])
        return candidate_ids, candidate_logits

    harness.dflash2_select_candidates = select_candidates.__get__(harness)
    with torch.no_grad():
        harness.embed_tokens.weight.copy_(
            torch.arange(8, dtype=torch.float32).unsqueeze(-1).expand(-1, 4)
        )
        harness.lm_head.weight.zero_()
        harness.lm_head.weight[1, 0] = 10.0
        harness.lm_head.weight[2, 1] = 10.0
        harness.lm_head.weight[3, 2] = 10.0

    tokens, _ = harness.rollout_correction(
        torch.zeros(1, 3, 4),
        anchor_token_ids=torch.tensor([7]),
        base_logits=torch.zeros(1, 3, 8),
    )

    assert torch.equal(tokens, torch.tensor([[1, 2, 3]]))
    assert seen_previous == [7, 1, 2]


def test_selector_logit_correction_reuses_precomputed_base_projection():
    harness = _RolloutHarness()
    harness.block_size = 2
    harness.draft_vocab_size = 8
    harness.config = SimpleNamespace(
        sample_from_anchor=True,
        correction_project_corrected_hidden=False,
    )
    harness.correction_head = _RecordingLogitCorrectionHead(harness.draft_vocab_size)
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, harness.draft_vocab_size)
    harness.d2t = None
    harness.candidate_selector = object()

    def select_path(_self, logits, hidden_states, anchor_token_ids):
        del hidden_states, anchor_token_ids
        candidate_ids = torch.tensor([[[4, 0], [5, 0]]], device=logits.device)
        candidate_logits = logits.new_tensor([[[10.0, 0.0], [10.0, 0.0]]])
        selected_ids = torch.tensor([[4, 5]], device=logits.device)
        return candidate_ids, candidate_logits, selected_ids

    harness.dflash2_select_path = select_path.__get__(harness)
    base_logits = torch.zeros(1, harness.block_size, harness.draft_vocab_size)
    tokens, _ = harness.rollout_correction(
        torch.zeros(1, harness.block_size, 4),
        anchor_token_ids=torch.tensor([7]),
        base_logits=base_logits,
    )

    assert torch.equal(tokens, torch.tensor([[1, 2]]))
    assert harness.lm_head.calls == 0


def test_no_anchor_sampling_starts_logit_feedback_from_verifier_logits():
    harness = _RolloutHarness()
    harness.block_size = 3
    harness.draft_vocab_size = 8
    harness.config = SimpleNamespace(
        sample_from_anchor=False,
        correction_hidden_size=4,
    )
    harness.correction_head = _RecordingLogitCorrectionHead(harness.draft_vocab_size)
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, harness.draft_vocab_size)
    harness.d2t = None
    initial_logits = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        harness.lm_head.weight.zero_()

    _, logits = harness.rollout_correction(
        torch.zeros(1, 3, 4),
        anchor_token_ids=torch.tensor([7]),
        initial_previous_logits=initial_logits,
    )

    masks = torch.cat(harness.correction_head.previous_logits_masks, dim=1)
    feedback = torch.cat(harness.correction_head.previous_logits, dim=1)
    assert torch.equal(masks, torch.tensor([[True, True]]))
    assert torch.equal(feedback[:, 0], initial_logits)
    assert torch.equal(feedback[:, 1], logits[:, 1].detach())


def test_logit_mode_can_project_corrected_hidden_before_adding_delta_logits():
    harness = _RolloutHarness()
    harness.block_size = 2
    harness.draft_vocab_size = 8
    harness.config = SimpleNamespace(
        sample_from_anchor=True,
        correction_hidden_size=4,
        correction_hidden_aux_loss=False,
        correction_hidden_feedback=False,
        correction_project_corrected_hidden=True,
    )
    harness.correction_head = _RecordingLogitCorrectionHead(harness.draft_vocab_size)
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, harness.draft_vocab_size)
    harness.d2t = None
    with torch.no_grad():
        harness.lm_head.weight.zero_()
        harness.lm_head.weight[7, 0] = 3.0

    _, logits, _, corrected_hidden = harness._rollout_correction_steps(
        torch.zeros(1, 2, 4),
        anchor_token_ids=torch.tensor([6]),
    )

    assert torch.equal(corrected_hidden[..., 0], torch.full((1, 2), 2.0))
    assert torch.equal(logits[..., 7], torch.full((1, 2), 6.0))
    assert harness.lm_head.calls == harness.block_size


def test_logit_dual_lm_head_fusion_projects_base_block_once():
    harness = _RolloutHarness()
    harness.block_size = 2
    harness.draft_vocab_size = 8
    harness.config = SimpleNamespace(
        sample_from_anchor=True,
        correction_hidden_size=4,
        correction_hidden_aux_loss=False,
        correction_hidden_feedback=False,
        correction_project_corrected_hidden=True,
        correction_lm_head_fusion=True,
    )
    harness.correction_head = _RecordingLogitCorrectionHead(
        harness.draft_vocab_size
    )
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, harness.draft_vocab_size)
    harness.d2t = None
    with torch.no_grad():
        harness.lm_head.weight.zero_()
        harness.lm_head.weight[7, 0] = 3.0

    _, logits = harness.rollout_correction(
        torch.zeros(1, 2, 4),
        anchor_token_ids=torch.tensor([6]),
    )

    assert torch.equal(logits[..., 7], torch.full((1, 2), 6.0))
    assert harness.lm_head.calls == 1


def test_logit_dual_lm_head_fusion_reuses_base_for_reserved_anchor_slot():
    harness = _RolloutHarness()
    harness.block_size = 2
    harness.draft_vocab_size = 8
    harness.config = SimpleNamespace(
        sample_from_anchor=False,
        correction_hidden_size=4,
        correction_hidden_aux_loss=False,
        correction_hidden_feedback=False,
        correction_project_corrected_hidden=True,
        correction_lm_head_fusion=True,
    )
    harness.correction_head = _RecordingLogitCorrectionHead(
        harness.draft_vocab_size
    )
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, harness.draft_vocab_size)
    harness.d2t = None
    with torch.no_grad():
        harness.lm_head.weight.zero_()
        harness.lm_head.weight[7, 0] = 3.0

    _, logits = harness.rollout_correction(
        torch.zeros(1, 2, 4),
        anchor_token_ids=torch.tensor([6]),
        initial_previous_logits=torch.zeros(1, harness.draft_vocab_size),
    )

    assert logits[0, 0, 7] == 0.0
    assert logits[0, 1, 7] == 6.0
    assert harness.lm_head.calls == 1


def test_corrected_hidden_is_fed_to_the_next_slot_when_enabled():
    harness = _RolloutHarness()
    harness.block_size = 3
    harness.config = SimpleNamespace(
        sample_from_anchor=True,
        correction_hidden_size=4,
        correction_hidden_aux_loss=False,
        correction_hidden_feedback=True,
    )
    harness.correction_head = _RecordingHiddenFeedbackCorrectionHead()
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, 8)
    harness.d2t = None

    _, _, _, corrected_hidden = harness._rollout_correction_steps(
        torch.zeros(1, 3, 4),
        anchor_token_ids=torch.tensor([7]),
    )

    masks = torch.cat(harness.correction_head.previous_corrected_hidden_masks, dim=1)
    assert torch.equal(masks, torch.tensor([[False, True, True]]))
    feedback = torch.cat(harness.correction_head.previous_corrected_hidden, dim=1)
    assert torch.count_nonzero(feedback[:, 0]) == 0
    assert torch.equal(feedback[:, 1:], corrected_hidden[:, :-1].detach())


def test_confidence_feature_detach_switch_controls_both_inputs():
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    sequential = torch.randn(2, 3, 2, requires_grad=True)

    coupled = DSparkDraftModel._confidence_features(hidden, sequential, detach=False)
    coupled.sum().backward()
    assert hidden.grad is not None
    assert sequential.grad is not None

    detached = DSparkDraftModel._confidence_features(hidden, sequential, detach=True)
    assert not detached.requires_grad
    assert torch.equal(detached[..., :4], hidden.detach())
    assert torch.equal(detached[..., 4:], sequential.detach())


def test_hidden_alignment_loss_is_masked_and_zero_for_matching_states():
    corrected = torch.tensor([[[1.0, 2.0], [100.0, 100.0]]])
    verifier = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0]])

    loss = DSparkDraftModel._hidden_alignment_loss(corrected, verifier, mask)

    assert torch.equal(loss, torch.zeros_like(loss))


def test_collaborative_markov_is_gated_from_correction_state():
    harness = _CollaborationHarness()
    harness.markov_head = MarkovHead(
        verifier_vocab_size=8,
        draft_vocab_size=8,
        markov_rank=2,
        hidden_size=4,
    )
    harness.correction_markov_gate = nn.Linear(4, 1)
    harness.correction_markov_scale = nn.Parameter(torch.zeros(()))
    with torch.no_grad():
        harness.correction_markov_gate.weight.zero_()
        harness.correction_markov_gate.bias.zero_()

    logits = torch.zeros(1, 2, 8)
    states = torch.randn(1, 2, 4, requires_grad=True)
    prev_ids = torch.tensor([[1, 2]])
    hidden = torch.randn(1, 2, 4)
    baseline_output, baseline_gate, _ = harness._apply_collaborative_markov(
        logits,
        states,
        prev_ids,
        hidden,
    )
    assert torch.equal(baseline_output, logits)
    assert torch.count_nonzero(baseline_gate) == 0

    with torch.no_grad():
        harness.correction_markov_scale.copy_(torch.atanh(torch.tensor(0.5)))
    output, gate, prev_emb = harness._apply_collaborative_markov(
        logits,
        states,
        prev_ids,
        hidden,
    )
    expected_bias = harness.markov_head.block_bias(
        prev_token_ids=prev_ids,
        hidden_states=hidden,
        prev_emb=prev_emb,
    )
    assert torch.allclose(gate, torch.full_like(gate, 0.25))
    assert torch.allclose(output, 0.25 * expected_bias)

    output.sum().backward()
    assert harness.correction_markov_gate.weight.grad is not None
    assert harness.correction_markov_scale.grad is not None
    assert harness.markov_head.markov_w2.weight.grad is not None


def test_generated_rollout_applies_collaborative_markov_each_step():
    harness = _CollaborativeRolloutHarness()
    harness.block_size = 2
    harness.config = SimpleNamespace(
        sample_from_anchor=True,
        correction_hidden_size=4,
    )
    harness.correction_head = _RecordingCorrectionHead()
    harness.embed_tokens = nn.Embedding(8, 4)
    harness.lm_head = _CountingLinear(4, 8)
    harness.markov_head = MarkovHead(
        verifier_vocab_size=8,
        draft_vocab_size=8,
        markov_rank=1,
        hidden_size=4,
    )
    harness.correction_markov_gate = nn.Linear(4, 1)
    harness.correction_markov_scale = nn.Parameter(torch.tensor(10.0))
    harness.d2t = None
    with torch.no_grad():
        harness.lm_head.weight.zero_()
        harness.markov_head.markov_w1.weight.fill_(1.0)
        harness.markov_head.markov_w2.weight.zero_()
        harness.markov_head.markov_w2.weight[5, 0] = 10.0
        harness.correction_markov_gate.weight.zero_()
        harness.correction_markov_gate.bias.fill_(10.0)

    tokens, _ = harness.rollout_correction(
        torch.zeros(1, 2, 4),
        anchor_token_ids=torch.tensor([3]),
    )
    assert torch.equal(tokens, torch.tensor([[5, 5]]))
    assert harness.lm_head.calls == harness.block_size
