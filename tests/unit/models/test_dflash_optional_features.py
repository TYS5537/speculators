"""Tests for opt-in DFlash backbone features and baseline parity."""

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.model_definitions import DFlash2GroupedConv
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _make_model(*, num_draft_layers: int = 1, **feature_flags) -> DFlashDraftModel:
    transformer_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_draft_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        layer_types=["full_attention"] * num_draft_layers,
    )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=32,
        block_size=3,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm="dflash",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=2)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
        **feature_flags,
    )
    return DFlashDraftModel(config).eval()


def test_optional_features_default_off_preserves_original_helpers():
    torch.manual_seed(0)
    model = _make_model()
    hidden = torch.randn(1, 5, 32)
    expected = model.hidden_norm(model.fc(hidden))
    actual = model._fuse_target_hidden(hidden)
    assert torch.equal(actual, expected)

    noise = torch.randn(1, 6, 16)
    document_ids = torch.zeros(1, 5, dtype=torch.long)
    conditioned = model._condition_noise_embedding(
        noise,
        actual,
        torch.tensor([2, 4]),
        document_ids,
    )
    assert conditioned is noise
    assert model.layer_fusion_norms is None
    assert model.layer_fusion_score is None
    assert model.context_hidden_proj is None
    assert model.block_position_embedding is None
    assert model.candidate_selector is None
    assert model.layers[0].attention_conv is None
    assert model.layers[0].mlp_conv is None
    optional_prefixes = (
        "layer_fusion_",
        "context_hidden_",
        "block_position_embedding",
        "candidate_selector",
    )
    assert not any(key.startswith(optional_prefixes) for key in model.state_dict())


def test_explicitly_disabled_dflash2_has_exact_baseline_state_dict():
    torch.manual_seed(11)
    baseline = _make_model()
    torch.manual_seed(11)
    disabled = _make_model(
        dflash2_dynamic_conv=False,
        dflash2_candidate_selector=False,
    )

    assert baseline.state_dict().keys() == disabled.state_dict().keys()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(
            value,
            disabled.state_dict()[key],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
            msg=key,
        )


def test_gated_layer_fusion_returns_draft_hidden_shape():
    torch.manual_seed(1)
    model = _make_model(dflash_gated_layer_fusion=True)
    hidden = torch.randn(1, 5, 32)
    fused = model._fuse_target_hidden(hidden)
    baseline = model.hidden_norm(model.fc(hidden))
    assert fused.shape == (1, 5, 16)
    assert torch.isfinite(fused).all()
    assert torch.equal(fused, baseline)
    assert model.fc.in_features == 32

    assert model.layer_fusion_gate is not None
    with torch.no_grad():
        model.layer_fusion_gate.fill_(1.0)
    assert not torch.equal(model._fuse_target_hidden(hidden), baseline)


def test_context_and_slot_residuals_start_at_exact_zero():
    torch.manual_seed(2)
    model = _make_model(
        dflash_context_residual=True,
        dflash_block_position_embedding=True,
    )
    noise = torch.randn(1, 6, 16)
    fused = torch.randn(1, 5, 16)
    document_ids = torch.zeros(1, 5, dtype=torch.long)
    initial = model._condition_noise_embedding(
        noise,
        fused,
        torch.tensor([2, 4]),
        document_ids,
    )
    assert torch.equal(initial, noise)

    assert model.context_hidden_proj is not None
    assert model.context_hidden_gate is not None
    assert model.block_position_embedding is not None
    with torch.no_grad():
        model.context_hidden_proj.weight.copy_(torch.eye(16))
        model.context_hidden_gate.fill_(1.0)
        model.block_position_embedding.weight[1].fill_(0.5)
    conditioned = model._condition_noise_embedding(
        noise,
        fused,
        torch.tensor([2, 4]),
        document_ids,
    )
    assert not torch.equal(conditioned, noise)


def test_context_residual_does_not_cross_document_boundary():
    model = _make_model(dflash_context_residual=True)
    assert model.context_hidden_proj is not None
    assert model.context_hidden_gate is not None
    with torch.no_grad():
        model.context_hidden_proj.weight.copy_(torch.eye(16))
        model.context_hidden_gate.fill_(1.0)

    noise = torch.zeros(1, 3, 16)
    fused = torch.ones(1, 4, 16)
    document_ids = torch.tensor([[0, 0, 1, 1]])
    conditioned = model._condition_noise_embedding(
        noise,
        fused,
        torch.tensor([2]),
        document_ids,
    )
    assert torch.equal(conditioned, noise)


def test_dflash2_dynamic_conv_starts_as_identity_and_is_block_local():
    model = _make_model(dflash2_dynamic_conv=True)
    conv = model.layers[0].attention_conv
    assert isinstance(conv, DFlash2GroupedConv)

    hidden = torch.randn(1, 6, 16)
    prepared, coefficients = conv.prepare(hidden)
    assert torch.equal(prepared, hidden)
    assert torch.equal(conv.finish(hidden, coefficients), hidden)

    with torch.no_grad():
        conv.base_kernel.zero_()
        conv.base_kernel[:, 1].fill_(1.0)
        conv.kernel_projection.weight.zero_()
    impulse = torch.zeros(1, 6, 16)
    impulse[:, 2].fill_(1.0)
    convolved, _ = conv.prepare(impulse)
    # Position 3 starts a new block and must not see position 2's impulse.
    assert torch.equal(convolved[:, 3], torch.zeros_like(convolved[:, 3]))


def test_dflash2_selector_starts_as_unary_topk_and_builds_lattice():
    model = _make_model(
        dflash2_candidate_selector=True,
        dflash2_selector_rank=8,
        dflash2_selector_top_k=4,
    )
    selector = model.candidate_selector
    assert selector is not None

    logits = torch.randn(2, 3, 32)
    hidden = torch.randn(2, 3, 16)
    previous_ids = torch.randint(0, 32, (2, 3))
    candidate_ids, candidate_logits = model.dflash2_select_candidates(
        logits, hidden, previous_ids
    )
    unary_logits = logits.gather(-1, candidate_ids)
    assert torch.equal(candidate_logits, unary_logits)

    predecessor_ids = torch.randint(0, 32, candidate_ids.shape)
    lattice = selector.score_lattice(
        candidate_ids,
        unary_logits,
        hidden,
        predecessor_ids,
    )
    assert lattice.shape == (2, 3, 4, 4)
    assert torch.equal(lattice, unary_logits.unsqueeze(-2).expand_as(lattice))


def test_dflash2_selector_block_loss_is_finite_and_backward_safe():
    model = _make_model(
        sample_from_anchor=True,
        dflash2_candidate_selector=True,
        dflash2_selector_rank=8,
        dflash2_selector_top_k=4,
    )
    logits = torch.randn(1, 6, 32, requires_grad=True)
    targets = torch.randn_like(logits)
    hidden = torch.randn(2, 3, 16, requires_grad=True)
    anchors = torch.tensor([2, 5])
    teacher_previous_ids = torch.randint(0, 32, (2, 3))
    teacher_previous_ids[:, 0] = anchors
    loss_mask = torch.ones(1, 6)

    candidate_ids, candidate_logits, selector_loss, selected_ids, teacher_rows = (
        model._dflash2_block_outputs(
            logits,
            targets,
            hidden,
            anchors,
            loss_mask,
            teacher_previous_ids,
        )
    )

    assert candidate_ids.shape == (2, 3, 4)
    assert candidate_logits.shape == candidate_ids.shape
    assert selected_ids.shape == candidate_ids.shape[:-1]
    assert teacher_rows.shape == candidate_ids.shape
    assert torch.isfinite(selector_loss)
    selector_loss.backward()
    assert logits.grad is not None
    assert model.candidate_selector is not None
    assert model.candidate_selector.hidden_projection.weight.grad is not None


def test_dflash2_global_search_can_outperform_local_greedy_path():
    model = _make_model(
        sample_from_anchor=True,
        dflash2_candidate_selector=True,
        dflash2_selector_rank=1,
        dflash2_selector_top_k=2,
    )
    selector = model.candidate_selector
    assert selector is not None
    with torch.no_grad():
        selector.hidden_projection.weight.zero_()
        selector.hidden_projection.weight[0, 0] = 1.0
        selector.predecessor_codebook.zero_()
        selector.predecessor_codebook[2, 0] = 1.0
        selector.successor_codebook.zero_()
        selector.successor_codebook[3, 0] = 8.0

    candidate_ids = torch.tensor([[[1, 2], [3, 4]]])
    unary_logits = torch.tensor([[[10.0, 9.5], [0.0, 0.0]]])
    hidden = torch.zeros(1, 2, 16)
    hidden[..., 0] = 1.0
    anchors = torch.tensor([0])

    model.config.dflash2_selector_search_mode = "greedy"
    greedy_ids, greedy_rows = model._dflash2_select_topk_path(
        candidate_ids, unary_logits, hidden, anchors
    )
    model.config.dflash2_selector_search_mode = "global"
    global_ids, global_rows = model._dflash2_select_topk_path(
        candidate_ids, unary_logits, hidden, anchors
    )

    assert torch.equal(greedy_ids, torch.tensor([[1, 3]]))
    assert torch.equal(global_ids, torch.tensor([[2, 3]]))
    assert torch.equal(greedy_rows[:, 1], torch.tensor([[0.0, 0.0]]))
    assert torch.equal(global_rows[:, 1], torch.tensor([[8.0, 0.0]]))
    global_proposal_rows = model._dflash2_proposal_logits(
        candidate_ids,
        global_rows,
        global_ids,
    )
    proposed_ids = candidate_ids.gather(
        -1, global_proposal_rows.argmax(dim=-1, keepdim=True)
    ).squeeze(-1)
    assert torch.equal(proposed_ids, global_ids)


def test_dflash2_global_search_is_invariant_to_predecessor_row_offsets():
    model = _make_model(
        sample_from_anchor=True,
        dflash2_candidate_selector=True,
        dflash2_selector_rank=1,
        dflash2_selector_top_k=2,
    )

    class OffsetSelector(torch.nn.Module):
        @staticmethod
        def forward(candidate_ids, unary_logits, hidden_states, previous_token_ids):
            del candidate_ids, hidden_states
            neutral = unary_logits.new_tensor([0.0, 0.0])
            offset = unary_logits.new_tensor([100.0, 90.0])
            return torch.where(
                (previous_token_ids == 2).unsqueeze(-1),
                offset.expand(*previous_token_ids.shape, 2),
                neutral.expand(*previous_token_ids.shape, 2),
            )

        @staticmethod
        def score_lattice(
            candidate_ids,
            unary_logits,
            hidden_states,
            predecessor_ids,
        ):
            del candidate_ids, unary_logits, hidden_states, predecessor_ids
            return torch.tensor([[[[0.0, 0.0], [100.0, 90.0]]]])

    model.candidate_selector = OffsetSelector()
    candidate_ids = torch.tensor([[[1, 2], [3, 4]]])
    unary_logits = torch.zeros(1, 2, 2)
    hidden = torch.zeros(1, 2, 16)
    anchors = torch.tensor([0])

    # Override the first selector row: token 1 is more likely than token 2.
    original_forward = model.candidate_selector.forward

    def forward(candidate_ids, unary_logits, hidden_states, previous_token_ids):
        if torch.equal(previous_token_ids, anchors):
            return unary_logits.new_tensor([[0.0, -1.0]])
        return original_forward(
            candidate_ids, unary_logits, hidden_states, previous_token_ids
        )

    model.candidate_selector.forward = forward
    model.config.dflash2_selector_search_mode = "global"
    selected_ids, _ = model._dflash2_select_topk_path(
        candidate_ids,
        unary_logits,
        hidden,
        anchors,
    )

    # Raw-energy Viterbi would choose predecessor 2 because of its +100 row
    # offset. Conditional log-probability Viterbi correctly ignores that offset.
    assert selected_ids[0, 0].item() == 1


@pytest.mark.parametrize(
    ("feature_flags", "missing_key"),
    [
        (
            {"dflash2_dynamic_conv": True},
            "layers.0.attention_conv.base_kernel",
        ),
        (
            {
                "dflash2_candidate_selector": True,
                "dflash2_selector_rank": 8,
                "dflash2_selector_top_k": 4,
            },
            "candidate_selector.hidden_projection.weight",
        ),
    ],
)
def test_dflash2_rejects_checkpoint_with_missing_trained_weights(
    feature_flags, missing_key
):
    model = _make_model(**feature_flags)
    with pytest.raises(RuntimeError, match="does not contain its trained weights"):
        model._prepare_missing_checkpoint_weights({"missing_keys": [missing_key]})
