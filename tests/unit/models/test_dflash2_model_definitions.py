"""Unit tests for DFlash2 convolution and candidate selection."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorModelConfig, SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from speculators.models.dflash2.metrics import (
    compute_metrics as compute_dflash2_metrics,
)
from speculators.models.dflash2.metrics import (
    compute_selector_loss,
    selector_greedy_path,
    selector_training_candidates,
)
from speculators.models.dflash2.model_definitions import (
    CandidateSelector,
    GroupedDynamicCausalConv,
    grouped_dynamic_conv,
)
from speculators.models.dspark.metrics import compute_metrics as compute_unary_metrics
from speculators.proposals import GreedyTokenProposalConfig
from speculators.train.optimizers import build_optimizers, split_named_params_for_muon


def _reference_grouped_conv(
    hidden_states,
    delta_kernel,
    base_kernel,
    *,
    block_size,
    group_size,
):
    output = torch.zeros_like(hidden_states)
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_delta = delta_kernel.reshape(
        -1, base_kernel.shape[0], hidden_states.shape[-1] // group_size
    )
    flat_output = output.reshape_as(flat_hidden)
    for token_idx in range(flat_hidden.shape[0]):
        block_position = token_idx % block_size
        for tap in range(min(base_kernel.shape[0], block_position + 1)):
            for channel_idx in range(flat_hidden.shape[-1]):
                group_idx = channel_idx // group_size
                coefficient = (
                    base_kernel[tap, channel_idx]
                    + flat_delta[token_idx, tap, group_idx]
                )
                flat_output[token_idx, channel_idx] += (
                    flat_hidden[token_idx - tap, channel_idx] * coefficient
                )
    return output


def test_grouped_dynamic_conv_matches_scalar_reference():
    """The vectorized kernel must preserve the public grouped-convolution formula."""
    torch.manual_seed(0)
    hidden = torch.randn(1, 8, 6)
    delta = torch.randn(1, 8, 3, 3)
    base = torch.randn(3, 6)

    actual = grouped_dynamic_conv(
        hidden,
        delta,
        base,
        block_size=4,
        group_size=2,
    )
    expected = _reference_grouped_conv(
        hidden,
        delta,
        base,
        block_size=4,
        group_size=2,
    )

    torch.testing.assert_close(actual, expected)


def test_grouped_dynamic_conv_does_not_cross_block_boundaries():
    """A causal tap at a new block must not read the prior block's final token."""
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


def test_grouped_dynamic_conv_preserves_bfloat16_activation_dtype():
    """FP32 master kernels must not upcast the BF16 convolution activation."""
    hidden = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    delta = torch.randn(1, 4, 2, 2, dtype=torch.bfloat16)
    base = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)

    output = grouped_dynamic_conv(
        hidden,
        delta,
        base,
        block_size=4,
        group_size=4,
    )

    assert output.dtype == torch.bfloat16
    output.float().sum().backward()
    assert base.grad is not None
    assert torch.isfinite(base.grad).all()


def test_grouped_dynamic_conv_validates_shapes():
    hidden = torch.zeros(1, 4, 6)
    delta = torch.zeros(1, 4, 2, 3)
    base = torch.zeros(2, 6)

    with pytest.raises(ValueError, match="divisible"):
        grouped_dynamic_conv(
            hidden,
            delta,
            base,
            block_size=4,
            group_size=4,
        )
    with pytest.raises(ValueError, match="delta_kernel"):
        grouped_dynamic_conv(
            hidden,
            delta[..., :2],
            base,
            block_size=4,
            group_size=2,
        )


def test_convolution_identity_initialization_and_gradients():
    """Identity warm-start must still allow both kernels and projection to learn."""
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


def test_candidate_selector_matches_loop_reference():
    """Vectorized transition scores must equal the bilinear edge definition."""
    torch.manual_seed(0)
    selector = CandidateSelector(
        vocab_size=11,
        hidden_size=6,
        rank=4,
        top_k=3,
    )
    hidden = torch.randn(2, 3, 6)
    predecessor_ids = torch.randint(0, 11, (2, 3))
    candidate_ids = torch.randint(0, 11, (2, 3, 3))

    actual = selector.transition_scores(
        hidden,
        predecessor_ids,
        candidate_ids,
    )
    expected = torch.empty_like(actual)
    projected = selector.hidden_projection(hidden)
    for block_idx in range(2):
        for position_idx in range(3):
            context = (
                selector.predecessor_codebook[predecessor_ids[block_idx, position_idx]]
                * projected[block_idx, position_idx]
            )
            for candidate_idx in range(3):
                token_id = candidate_ids[block_idx, position_idx, candidate_idx]
                expected[block_idx, position_idx, candidate_idx] = (
                    context * selector.successor_codebook[token_id]
                ).sum()

    torch.testing.assert_close(actual, expected)


def test_candidate_selector_selects_only_from_unary_top_k():
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


def test_candidate_codebooks_use_adamw_under_muon_optimizer():
    selector = CandidateSelector(
        vocab_size=11,
        hidden_size=6,
        rank=4,
        top_k=3,
    )

    muon, adamw = split_named_params_for_muon(selector)
    muon_names = {name for name, _ in muon}
    adamw_names = {name for name, _ in adamw}

    assert muon_names == {"hidden_projection.weight"}
    assert adamw_names == {
        "predecessor_codebook",
        "successor_codebook",
    }


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


def test_model_uses_canonical_checkpoint_keys():
    """Saved parameter names must load directly in the public vLLM implementation."""
    model = DFlash2DraftModel(_tiny_config())
    state_keys = set(model.state_dict())

    assert "layers.0.attention_conv.base_kernel" in state_keys
    assert "layers.0.attention_conv.kernel_projection.weight" in state_keys
    assert "layers.0.mlp_conv.base_kernel" in state_keys
    assert "layers.0.mlp_conv.kernel_projection.weight" in state_keys
    assert "candidate_selector.predecessor_codebook" in state_keys
    assert "candidate_selector.successor_codebook" in state_keys
    assert "candidate_selector.hidden_projection.weight" in state_keys
    assert "candidate_selector.predecessor_codebook.weight" not in state_keys


def test_config_round_trip_preserves_dflash2_contract(tmp_path):
    config = _tiny_config(
        speculators_config=SpeculatorsConfig(
            algorithm="dflash2",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="Qwen/Qwen3-4B",
                architectures=["Qwen3ForCausalLM"],
            ),
        )
    )

    config.save_pretrained(tmp_path)
    loaded = SpeculatorModelConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, DFlash2SpeculatorConfig)
    assert loaded.speculators_model_type == "dflash2"
    assert loaded.speculators_config.algorithm == "dflash2"
    assert loaded.architectures == ["DFlash2DraftModel"]
    assert loaded.sliding_window_non_causal is True
    assert loaded.conv_kernel_size == config.conv_kernel_size
    assert loaded.conv_group_size == config.conv_group_size
    assert loaded.selector_rank == config.selector_rank
    assert loaded.selector_top_k == config.selector_top_k


def test_model_rejects_pruned_draft_vocabulary():
    with pytest.raises(ValueError, match="full verifier vocabulary"):
        DFlash2DraftModel(_tiny_config(draft_vocab_size=32))


@pytest.mark.parametrize("swa_value", [None, False, True])
def test_training_factory_preserves_dflash2_swa_default(monkeypatch, swa_value):
    monkeypatch.setattr(
        VerifierConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: VerifierConfig(
            name_or_path="local-verifier", architectures=["Qwen3ForCausalLM"]
        ),
    )
    monkeypatch.setattr(DFlash2DraftModel, "load_verifier_weights", lambda _self: None)
    config = _tiny_config()
    monkeypatch.setattr(
        "speculators.models.utils.get_verifier_config",
        lambda *_args, **_kwargs: config.transformer_layer_config,
    )
    kwargs = {} if swa_value is None else {"sliding_window_non_causal": swa_value}
    model = DFlash2DraftModel.from_training_args(
        config.transformer_layer_config,
        draft_vocab_size=64,
        block_size=4,
        target_layer_ids=[0, 1],
        verifier_name_or_path="local-verifier",
        conv_group_size=4,
        selector_rank=8,
        selector_top_k=4,
        **kwargs,
    )

    assert model.config.sliding_window_non_causal is (swa_value is not False)


@pytest.mark.parametrize(
    "missing_key",
    [
        "layers.0.attention_conv.base_kernel",
        "layers.0.mlp_conv.kernel_projection.weight",
        "candidate_selector.predecessor_codebook",
        "candidate_selector.hidden_projection.weight",
    ],
)
def test_pretrained_load_rejects_missing_trained_dflash2_weights(
    monkeypatch, missing_key
):
    model = DFlash2DraftModel(_tiny_config())
    monkeypatch.setattr(model, "load_verifier_weights", lambda: None)

    with pytest.raises(RuntimeError, match="missing trained convolution/selector"):
        SpeculatorModel._finalize_pretrained_load(
            (model, {"missing_keys": [missing_key]}), None, None
        )


def test_pretrained_load_allows_reconstructed_verifier_weights(monkeypatch):
    model = DFlash2DraftModel(_tiny_config())
    monkeypatch.setattr(model, "load_verifier_weights", lambda: None)
    loading_info = {"missing_keys": ["embed_tokens.weight", "verifier_norm.weight"]}

    loaded, diagnostics = SpeculatorModel._finalize_pretrained_load(
        (model, loading_info), None, None
    )

    assert loaded is model
    assert diagnostics is loading_info


def test_standalone_selector_serving_adapter_matches_canonical_scores():
    model = DFlash2DraftModel(_tiny_config())
    unary_logits = torch.randn(1, 2, 64)
    hidden = torch.randn(1, 2, 16)
    previous_ids = torch.tensor([[3, 7]])

    actual_ids, actual_scores = model.dflash2_select_candidates(
        unary_logits, hidden, previous_ids
    )
    expected_ids, expected_scores = model.candidate_selector.select(
        unary_logits, hidden, previous_ids
    )

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_scores, expected_scores)


def test_predecessor_ids_shift_each_block_from_its_anchor():
    model = DFlash2DraftModel(_tiny_config())
    input_ids = torch.arange(16).unsqueeze(0)
    anchored_block_indices = torch.tensor([2, 3, 4, 5, 8, 9, 10, 11])

    predecessor_ids = model._predecessor_ids(
        input_ids,
        anchored_block_indices,
    )

    expected = torch.tensor([[2, 2, 3, 4], [8, 8, 9, 10]])
    torch.testing.assert_close(predecessor_ids, expected)


def test_predecessor_ids_sample_from_anchor():
    model = DFlash2DraftModel(_tiny_config(sample_from_anchor=True))
    input_ids = torch.arange(16).unsqueeze(0)
    anchored_block_indices = torch.tensor([2, 3, 4, 5, 8, 9, 10, 11])

    predecessor_ids = model._predecessor_ids(
        input_ids,
        anchored_block_indices,
    )

    expected = torch.tensor([[2, 3, 4, 5], [8, 9, 10, 11]])
    torch.testing.assert_close(predecessor_ids, expected)


def test_selector_training_candidates_keep_strict_top_k_and_inject_missing_target():
    unary_logits = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]]
    )
    strict_candidate_ids = unary_logits.topk(3, dim=-1).indices
    original_candidate_ids = strict_candidate_ids.clone()
    target_ids = torch.tensor([[4, 0]])

    training_candidate_ids, target_positions, contains_target = (
        selector_training_candidates(
            strict_candidate_ids,
            target_ids,
        )
    )

    assert strict_candidate_ids.tolist() == [[[5, 4, 3], [5, 4, 3]]]
    assert training_candidate_ids.tolist() == [[[5, 4, 3], [5, 4, 0]]]
    assert target_positions.tolist() == [[1, 2]]
    assert contains_target.tolist() == [[True, False]]
    torch.testing.assert_close(strict_candidate_ids, original_candidate_ids)


def _selector_objective_inputs():
    torch.manual_seed(7)
    selector = CandidateSelector(
        vocab_size=7,
        hidden_size=4,
        rank=3,
        top_k=2,
    )
    unary_logits = torch.randn(1, 4, 7, requires_grad=True)
    hidden_states = torch.randn(1, 4, 4)
    predecessor_ids = torch.tensor([[2, 2, 6, 5]])
    target_ids = torch.tensor([[2, 6, 5, 4]])
    targets = torch.full_like(unary_logits, -4.0)
    targets.scatter_(-1, target_ids.unsqueeze(-1), 4.0)
    loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    return (
        selector,
        unary_logits,
        hidden_states,
        predecessor_ids,
        target_ids,
        targets,
        loss_mask,
    )


def test_selector_loss_alpha_zero_preserves_unary_objective():
    (
        selector,
        _unary_logits,
        _hidden_states,
        _predecessor_ids,
        target_ids,
        targets,
        loss_mask,
    ) = _selector_objective_inputs()
    # The selector corrects the first unary argmax. At the second draft slot the
    # target is outside unary top-k but injected for training; choosing that
    # injected target must still end the serving path.
    unary_logits = torch.full_like(targets, -10.0)
    unary_logits.scatter_(-1, target_ids.unsqueeze(-1), 5.0)
    unary_logits.scatter_(-1, ((target_ids + 1) % 7).unsqueeze(-1), 6.0)
    unary_logits[0, 2, target_ids[0, 2]] = -10.0
    unary_logits[0, 2, (target_ids[0, 2] + 2) % 7] = 5.0
    unary_logits.requires_grad_()
    loss_config = resolve_loss_config("ce", "eager")
    tv_loss_fn = resolve_loss_config("tv", "eager")["tv"][0]

    expected, unary_metrics = compute_unary_metrics(
        unary_logits,
        targets,
        None,
        loss_mask,
        4,
        loss_config=loss_config,
        tv_loss_fn=tv_loss_fn,
        gamma=4.0,
        confidence_head_alpha=0.0,
        per_position_loss_weight="fixed-exp-decay",
        dpace_alpha=0.5,
        sample_from_anchor=False,
    )
    target_ids = targets.argmax(dim=-1)
    candidate_ids = unary_logits.topk(selector.top_k, dim=-1).indices
    training_candidate_ids, target_positions, contains_target = (
        selector_training_candidates(candidate_ids, target_ids)
    )
    assert contains_target.tolist() == [[True, True, False, True]]
    candidate_logits = torch.zeros_like(training_candidate_ids, dtype=torch.float32)
    candidate_logits.scatter_(-1, target_positions.unsqueeze(-1), 1.0)
    actual, metrics = compute_dflash2_metrics(
        unary_logits=unary_logits,
        targets=targets,
        training_candidate_ids=training_candidate_ids,
        candidate_logits=candidate_logits,
        target_positions=target_positions,
        contains_target=contains_target,
        loss_mask=loss_mask,
        block_size=4,
        top_k=selector.top_k,
        candidate_ids=candidate_ids,
        greedy_path_ids=torch.where(contains_target, target_ids, candidate_ids[..., 0]),
        loss_config=loss_config,
        tv_loss_fn=tv_loss_fn,
        selector_loss_alpha=0.0,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(metrics["unary_loss_sum"], expected.detach())
    assert metrics["selector_loss_sum"] > 0
    # One selected draft token plus the verifier's bonus token.
    assert metrics["eal_sum"].item() == 2.0
    assert metrics["eal_total"].item() == 1.0
    # The inherited unary EAL would stop at the first drafted position.
    assert unary_metrics["eal_sum"].item() == 1.0
    torch.testing.assert_close(
        metrics["accept_len_sum"], unary_metrics["accept_len_sum"]
    )
    target_probs = targets.softmax(dim=-1)
    strict_mass = target_probs.gather(-1, candidate_ids).sum(dim=-1)
    injected_mass = target_probs.gather(-1, training_candidate_ids).sum(dim=-1)
    expected_mass = (strict_mass * loss_mask).sum()
    torch.testing.assert_close(
        metrics["unary_candidate_target_mass_at_2_sum"], expected_mass
    )
    assert expected_mass < (injected_mass * loss_mask).sum()


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_selector_eal_uses_selected_predecessors_on_off_policy_data(
    monkeypatch, sample_from_anchor
):
    """A recorded token can differ from a correctly selected teacher argmax."""
    model = DFlash2DraftModel(_tiny_config(sample_from_anchor=sample_from_anchor))
    selector = CandidateSelector(vocab_size=5, hidden_size=1, rank=1, top_k=2)
    with torch.no_grad():
        selector.predecessor_codebook[:, 0] = torch.tensor([0.0, 1.0, -1.0, 0.0, 1.0])
        selector.successor_codebook[:, 0] = torch.tensor([0.0, 0.0, 0.0, 1.0, -1.0])
        selector.hidden_projection.weight.fill_(1.0)
    model.candidate_selector = selector
    start = 0 if sample_from_anchor else 1
    target_ids = torch.full((1, 4), 3, dtype=torch.long)
    target_ids[:, start] = 2
    if not sample_from_anchor:
        target_ids[:, 0] = 0
    unary_logits = torch.full((1, 4, 5), -20.0)
    unary_logits[..., 3] = 0.5
    unary_logits[..., 4] = 0.0
    unary_logits[:, start, 2] = 5.0
    unary_logits[:, start, 1] = 4.0
    hidden = torch.ones(1, 4, 1)
    targets = torch.full_like(unary_logits, -5.0)
    targets.scatter_(-1, target_ids.unsqueeze(-1), 5.0)
    loss_mask = torch.ones(1, 4)
    loss_mask[:, :start] = 0
    block_indices = torch.arange(4)
    monkeypatch.setattr(
        model,
        "_backbone_forward",
        lambda *_args, **_kwargs: (
            hidden,
            unary_logits,
            targets,
            loss_mask,
            block_indices,
            None,
            None,
        ),
    )
    # The recorded first draft token is 1. The selector chooses the teacher's 2.
    input_ids = torch.tensor([[0, 1, 3, 3]])
    _, loss, metrics = model(
        hidden_states=hidden,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=hidden,
        document_ids=torch.zeros_like(input_ids),
        loss_config=resolve_loss_config("ce", "eager"),
        tv_loss_fn=resolve_loss_config("tv", "eager")["tv"][0],
    )

    assert torch.isfinite(loss)
    assert (
        metrics["teacher_forced_selector_acc_sum"]
        == metrics["teacher_forced_selector_acc_total"]
    )
    # At the second slot, predecessor 1 selects target 3, but the actual
    # predecessor 2 selects 4. Only the first draft token is accepted.
    assert metrics["eal_sum"].item() == 2.0
    assert metrics["eal_total"].item() == 1.0
    assert metrics["unary_top_2_oracle_accepted_length_sum"].item() == 5 - start


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_selector_greedy_path_restarts_from_each_blocks_anchor(sample_from_anchor):
    selector = CandidateSelector(vocab_size=3, hidden_size=1, rank=1, top_k=2)
    with torch.no_grad():
        selector.predecessor_codebook[:, 0] = torch.tensor([1.0, -1.0, 1.0])
        selector.successor_codebook[:, 0] = torch.tensor([0.0, 1.0, -1.0])
        selector.hidden_projection.weight.fill_(1.0)
    unary = torch.tensor([[[-20.0, 0.0, 0.0]]]).expand(1, 8, 3)
    candidates = unary.topk(2, dim=-1).indices
    path = selector_greedy_path(
        selector,
        unary,
        torch.ones(1, 8, 1),
        candidates,
        torch.tensor([0, 1]),
        4,
        sample_from_anchor=sample_from_anchor,
    ).view(2, 4)

    expected = (
        [[1, 2, 1, 2], [2, 1, 2, 1]]
        if sample_from_anchor
        else [[0, 1, 2, 1], [1, 2, 1, 2]]
    )
    assert path.tolist() == expected


@pytest.mark.parametrize(
    ("first_target_present", "valid_slots", "expected_length"),
    [(True, [1, 1, 1, 1], 5), (False, [1, 1, 1, 1], 1), (True, [1, 0, 0, 0], 2)],
)
def test_sample_from_anchor_oracle_includes_slot_zero(
    first_target_present, valid_slots, expected_length
):
    unary = torch.tensor([[[2.0, 1.0, -2.0]]]).repeat(1, 4, 1)
    target_ids = torch.zeros(1, 4, dtype=torch.long)
    if not first_target_present:
        target_ids[:, 0] = 2
    targets = torch.full_like(unary, -5.0)
    targets.scatter_(-1, target_ids.unsqueeze(-1), 5.0)
    candidates = unary.topk(2, dim=-1).indices
    training_candidates, target_positions, contains_target = (
        selector_training_candidates(candidates, target_ids)
    )
    _, metrics = compute_dflash2_metrics(
        unary_logits=unary,
        targets=targets,
        training_candidate_ids=training_candidates,
        candidate_logits=unary.gather(-1, training_candidates),
        target_positions=target_positions,
        contains_target=contains_target,
        loss_mask=torch.tensor([valid_slots], dtype=torch.float32),
        block_size=4,
        top_k=2,
        sample_from_anchor=True,
        candidate_ids=candidates,
        greedy_path_ids=unary.argmax(dim=-1),
        loss_config=resolve_loss_config("ce", "eager"),
        tv_loss_fn=resolve_loss_config("tv", "eager")["tv"][0],
    )

    assert metrics["unary_top_2_oracle_accepted_length_sum"].item() == expected_length
    assert metrics["unary_top_2_oracle_accepted_length_total"].item() == 1.0


def test_selector_loss_reaches_every_selector_parameter():
    (
        selector,
        unary_logits,
        hidden_states,
        predecessor_ids,
        target_ids,
        _targets,
        loss_mask,
    ) = _selector_objective_inputs()
    candidate_ids = unary_logits.topk(selector.top_k, dim=-1).indices
    training_candidate_ids, target_positions, _contains_target = (
        selector_training_candidates(candidate_ids, target_ids)
    )
    candidate_logits = selector.score_candidates(
        unary_logits, hidden_states, predecessor_ids, training_candidate_ids
    )

    loss = compute_selector_loss(
        candidate_logits,
        target_positions,
        loss_mask,
        4,
        gamma=4.0,
        per_position_loss_weight="fixed-exp-decay",
        dpace_alpha=0.5,
    )
    loss.backward()

    for name, parameter in selector.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"
        assert torch.count_nonzero(parameter.grad), f"zero gradient for {name}"


def test_bf16_candidate_metrics_preserve_large_batch_counts():
    """512 blocks must not round 3583 valid prediction slots to 3584 in BF16."""
    unary = torch.tensor([[[2.0, 1.0]]], dtype=torch.bfloat16).repeat(1, 4096, 1)
    targets = unary.clone()
    targets[:, 1] = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    candidates = unary.topk(1, dim=-1).indices
    training_candidates, target_positions, contains_target = (
        selector_training_candidates(candidates, targets.argmax(dim=-1))
    )
    mask = torch.ones(1, 4096)
    mask[:, ::8] = 0
    mask[:, -1] = 0
    _, metrics = compute_dflash2_metrics(
        unary_logits=unary,
        targets=targets,
        training_candidate_ids=training_candidates,
        candidate_logits=unary.gather(-1, training_candidates),
        target_positions=target_positions,
        contains_target=contains_target,
        loss_mask=mask,
        block_size=8,
        top_k=1,
        candidate_ids=candidates,
        greedy_path_ids=unary.argmax(dim=-1),
        loss_config=resolve_loss_config("ce", "eager"),
        tv_loss_fn=resolve_loss_config("tv", "eager")["tv"][0],
    )
    assert metrics["unary_candidate_recall_at_1_total"].item() == 3583
    assert metrics["unary_candidate_recall_at_1_sum"].item() == 3582
    assert metrics["teacher_forced_selector_acc_total"].item() == 3582
    assert metrics["teacher_forced_selector_acc_sum"].item() == 3582


def test_trainer_kwargs_include_selector_loss_alpha():
    train_kwargs, val_kwargs = DFlash2DraftModel.get_trainer_kwargs(
        loss_fn="ce",
        loss_implementation="eager",
        selector_loss_alpha=0.25,
    )

    assert train_kwargs["selector_loss_alpha"] == pytest.approx(0.25)
    assert val_kwargs["selector_loss_alpha"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", torch.float32),
        pytest.param(
            "cuda",
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_tiny_forward_backward_reaches_all_new_parameters(device, dtype):
    """A real training step must update every convolution and selector tensor."""
    torch.manual_seed(0)
    model = DFlash2DraftModel(_tiny_config()).to(  # type: ignore[call-arg]
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.isnan().any():
                torch.nn.init.normal_(parameter, std=0.02)

    seq_len = 32
    hidden_size = model.config.transformer_layer_config.hidden_size
    inputs = {
        "hidden_states": torch.randn(
            1,
            seq_len,
            2 * hidden_size,
            device=device,
            dtype=dtype,
        ),
        "input_ids": torch.randint(
            0,
            model.verifier_vocab_size,
            (1, seq_len),
            device=device,
        ),
        "loss_mask": torch.ones(1, seq_len, device=device),
        "verifier_last_hidden_states": torch.randn(
            1,
            seq_len,
            hidden_size,
            device=device,
            dtype=dtype,
        ),
        "document_ids": torch.zeros(1, seq_len, device=device, dtype=torch.long),
    }
    eager_kl = resolve_loss_config("kl_div", "eager")
    eager_tv = resolve_loss_config("tv", "eager")["tv"][0]

    _, loss, _ = model(  # type: ignore[call-arg]
        **inputs,
        max_anchors=4,
        loss_config=eager_kl,
        tv_loss_fn=eager_tv,
    )
    assert torch.isfinite(loss)
    loss.backward()

    new_parameter_fragments = (
        "attention_conv",
        "mlp_conv",
        "candidate_selector",
    )
    new_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if any(fragment in name for fragment in new_parameter_fragments)
    }
    assert new_parameters
    for name, parameter in new_parameters.items():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"
        assert torch.count_nonzero(parameter.grad), f"zero gradient for {name}"


def test_qwen4b_recipe_bf16_autocast_adamw_step():
    """Exercise the launcher's SDPA and compound objective with FP32 master weights."""
    torch.manual_seed(42)
    config = _tiny_config(
        block_size=8,
        aux_hidden_state_layer_ids=[0, 1, 2, 3, 4],
        conv_group_size=16,
        selector_rank=256,
        selector_top_k=16,
    )
    config.transformer_layer_config.num_hidden_layers = 5
    config.transformer_layer_config.layer_types = ["full_attention"] * 5
    config.transformer_layer_config._attn_implementation = "sdpa"
    model = DFlash2DraftModel(config)
    # Supply local frozen verifier weights without downloading the target model.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.isnan().any():
                if "norm" in name:
                    parameter.fill_(1)
                else:
                    parameter.normal_(std=0.02)

    optimizer = build_optimizers(
        model, SimpleNamespace(optimizer="adamw", lr=6e-4, weight_decay=0.0)
    )[0]
    train_kwargs, _ = model.get_trainer_kwargs(
        loss_fn='{"ce": 0.1, "tv": 0.9}',
        loss_implementation="legacy",
        dflash_decay_gamma=7.0,
        selector_loss_alpha=0.1,
        max_anchors=4,
    )
    new_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if any(
            part in name
            for part in ("attention_conv", "mlp_conv", "candidate_selector")
        )
    }
    before = {
        name: parameter.detach().clone() for name, parameter in new_parameters.items()
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, loss, metrics = model(
            hidden_states=torch.randn(1, 32, 80, dtype=torch.bfloat16),
            input_ids=torch.randint(0, 64, (1, 32)),
            loss_mask=torch.ones(1, 32),
            verifier_last_hidden_states=torch.randn(1, 32, 16, dtype=torch.bfloat16),
            document_ids=torch.zeros(1, 32, dtype=torch.long),
            **train_kwargs,
        )
    assert torch.isfinite(loss)
    torch.testing.assert_close(
        loss.detach(), metrics["unary_loss_sum"] + 0.1 * metrics["selector_loss_sum"]
    )
    loss.backward()
    for name, parameter in new_parameters.items():
        assert parameter.dtype == torch.float32, name
        assert parameter.grad is not None, name
        assert parameter.grad.dtype == torch.float32, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    for name, parameter in new_parameters.items():
        assert torch.isfinite(parameter).all(), name
        assert not torch.equal(parameter, before[name]), name
        assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32, name
        assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32, name
