"""Numerical parity and recomputation boundaries for draft-layer checkpointing."""

import copy
from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash import core as dflash_core
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.mmuse import MMuseDraftModel, MMuseSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _make_model(
    *,
    features=False,
    dropout=0.0,
    correction=False,
    attention_impl="eager",
    correction_output_mode="hidden",
):
    transformer_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=dropout,
        layer_types=["full_attention", "full_attention"],
    )
    transformer_config._attn_implementation = attention_impl
    enhanced = features or correction
    algorithm = "mmuse" if enhanced else "dflash"
    config_class = MMuseSpeculatorConfig if enhanced else DFlashSpeculatorConfig
    feature_kwargs = (
        {
            "markov_rank": 0,
            "enable_confidence_head": False,
            "sample_from_anchor": correction,
            "dflash_gated_layer_fusion": features,
            "dflash_context_residual": features,
            "dflash_block_position_embedding": features,
            "dflash2_dynamic_conv": features,
            "dflash2_conv_group_size": 4,
        }
        if enhanced
        else {}
    )
    correction_kwargs = (
        {
            "enable_correction_head": True,
            "correction_hidden_size": 16,
            "correction_rank": 4,
            "correction_num_heads": 4,
            "correction_hidden_feedback": True,
            "correction_output_mode": correction_output_mode,
        }
        if correction
        else {}
    )
    config = config_class(
        transformer_layer_config=transformer_config,
        draft_vocab_size=32,
        block_size=3,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm=algorithm,
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=3 if correction else 2)
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None, architectures=["Qwen3ForCausalLM"]
            ),
        ),
        **feature_kwargs,
        **correction_kwargs,
    )
    model_class = MMuseDraftModel if enhanced else DFlashDraftModel
    model = model_class(config).train()
    with torch.no_grad():
        # These normally come from the target; an unloaded model uses NaN sentinels.
        for module in (
            model.embed_tokens,
            model.lm_head,
            model.verifier_lm_head,
        ):
            torch.nn.init.normal_(module.weight, std=0.1)
        if features:
            model.layer_fusion_gate.fill_(0.7)
            model.context_hidden_gate.fill_(0.4)
            torch.nn.init.normal_(model.layer_fusion_score.weight, std=0.1)
            torch.nn.init.normal_(model.block_position_embedding.weight, std=0.02)
            for layer in model.layers:
                for conv in (layer.attention_conv, layer.mlp_conv):
                    conv.base_kernel[:, 1].fill_(0.1)
                    torch.nn.init.normal_(conv.kernel_projection.weight, std=0.01)
    return model


def _inputs():
    generator = torch.Generator().manual_seed(123)
    return {
        "hidden_states": torch.randn(1, 8, 32, generator=generator),
        "input_ids": torch.arange(1, 9).unsqueeze(0),
        "loss_mask": torch.ones(1, 8),
        "verifier_last_hidden_states": torch.randn(1, 8, 16, generator=generator),
        "document_ids": torch.zeros(1, 8, dtype=torch.long),
        "max_anchors": 2,
    }


@pytest.fixture
def fixed_anchors(monkeypatch):
    selector = Mock(
        return_value=(
            torch.tensor([1, 4]),
            torch.tensor([True, True]),
        )
    )
    monkeypatch.setattr(dflash_core, "select_anchors", selector)
    return selector


def _record_layer_calls(model):
    counts = [0] * len(model.layers)
    handles = []
    for index, layer in enumerate(model.layers):

        def record(module, args, *, index=index):
            counts[index] += 1

        handles.append(layer.register_forward_pre_hook(record))
    return counts, handles


def _assert_gradients_match(plain, checked):
    checked_parameters = dict(checked.named_parameters())
    for name, parameter in plain.named_parameters():
        other = checked_parameters[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(
                parameter.grad, other.grad, rtol=1e-5, atol=1e-7, msg=name
            )


@pytest.mark.parametrize("attention_impl", ["eager", "sdpa"])
@pytest.mark.parametrize(
    ("features", "dropout", "freeze_projection", "autocast"),
    [
        (False, 0.0, False, False),
        (False, 0.0, True, False),
        (True, 0.0, False, False),
        (True, 0.2, False, False),
        (True, 0.2, False, True),
    ],
)
def test_checkpoint_matches_forward_gradients_and_optimizer_step(
    fixed_anchors, *, features, dropout, freeze_projection, autocast, attention_impl
):
    torch.manual_seed(17)
    plain = _make_model(
        features=features, dropout=dropout, attention_impl=attention_impl
    )
    if freeze_projection:
        # Exercise trainable decoder weights with no grad-requiring layer inputs.
        plain.fc.requires_grad_(False)
        plain.hidden_norm.requires_grad_(False)
    checked = copy.deepcopy(plain)
    checked.set_activation_checkpointing(True)
    inputs = _inputs()
    assert not inputs["hidden_states"].requires_grad
    assert not checked.embed_tokens.weight.requires_grad
    plain_counts, plain_handles = _record_layer_calls(plain)
    checked_counts, checked_handles = _record_layer_calls(checked)
    try:
        torch.manual_seed(29)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            plain_outputs = plain._backbone_forward(**inputs)
        torch.manual_seed(29)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            checked_outputs = checked._backbone_forward(**inputs)
        for actual, expected in zip(checked_outputs, plain_outputs, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        assert checked_counts == plain_counts == [1, 1]
        # Backward runs outside autocast; checkpoint must restore its forward state.
        plain_loss = (
            (plain_outputs[1].float() - plain_outputs[2].float()).square().mean()
        )
        checked_loss = (
            (checked_outputs[1].float() - checked_outputs[2].float()).square().mean()
        )
        plain_loss.backward()
        checked_loss.backward()
        _assert_gradients_match(plain, checked)
        assert plain_counts == [1, 1]
        assert checked_counts == [2, 2]
        # Anchor selection is outside the checkpointed region.
        assert fixed_anchors.call_count == 2
        if not freeze_projection:
            assert checked.fc.weight.grad.abs().sum() > 0
        if features:
            assert checked.layer_fusion_score.weight.grad.abs().sum() > 0
            for layer in checked.layers:
                for conv in (layer.attention_conv, layer.mlp_conv):
                    assert conv.kernel_projection.weight.grad.abs().sum() > 0
        before = plain.layers[0].self_attn.q_proj.weight.detach().clone()
        for model in (plain, checked):
            torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9).step()
        assert not torch.equal(before, plain.layers[0].self_attn.q_proj.weight)
        for name, parameter in plain.named_parameters():
            torch.testing.assert_close(
                parameter,
                dict(checked.named_parameters())[name],
                rtol=1e-5,
                atol=1e-7,
                msg=name,
            )
    finally:
        for handle in plain_handles + checked_handles:
            handle.remove()


@pytest.mark.parametrize(
    ("enabled", "training", "grad_enabled"),
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_disabled_eval_and_no_grad_bypass_checkpoint(
    monkeypatch, fixed_anchors, enabled, training, grad_enabled
):
    model = _make_model()
    model.set_activation_checkpointing(enabled)
    model.train(training)
    checkpoint = Mock(side_effect=AssertionError("checkpoint must be bypassed"))
    monkeypatch.setattr(dflash_core, "checkpoint", checkpoint)
    with nullcontext() if grad_enabled else torch.no_grad():
        outputs = model._backbone_forward(**_inputs())
    assert torch.isfinite(outputs[0]).all()
    checkpoint.assert_not_called()


@pytest.mark.parametrize("correction", [False, True])
def test_switch_is_runtime_only_and_inherited(correction):
    model = _make_model(correction=correction)
    assert model.activation_checkpointing is False
    original_config = copy.deepcopy(model.config.to_dict())
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    for enabled in (True, False):
        model.set_activation_checkpointing(enabled)
        assert model.activation_checkpointing is enabled
        assert model.config.to_dict() == original_config
        assert model.state_dict().keys() == original_state.keys()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
    assert DSparkDraftModel._backbone_forward is DFlashDraftModel._backbone_forward
    assert MMuseDraftModel._backbone_forward is DFlashDraftModel._backbone_forward


@pytest.mark.parametrize("attention_impl", ["eager", "sdpa"])
@pytest.mark.parametrize("correction_output_mode", ["hidden", "logits"])
def test_dspark_hidden_feedback_is_not_recomputed(
    fixed_anchors, attention_impl, correction_output_mode
):
    torch.manual_seed(71)
    plain = _make_model(
        features=True,
        correction=True,
        attention_impl=attention_impl,
        correction_output_mode=correction_output_mode,
    )
    checked = copy.deepcopy(plain)
    checked.set_activation_checkpointing(True)
    decoder_counts, decoder_handles = _record_layer_calls(checked)
    correction_calls = []

    def record_feedback(module, args, kwargs):
        correction_calls.append(kwargs["previous_corrected_hidden"].detach().clone())

    handle = checked.correction_head.register_forward_pre_hook(
        record_feedback, with_kwargs=True
    )

    def run(model):
        uses_logits = correction_output_mode == "logits"
        hidden, base_logits, targets, _, indices, _, _ = model._backbone_forward(
            **_inputs(), project_logits=uses_logits
        )
        hidden_blocks = hidden.view(2, 3, 16)
        token_ids = _inputs()["input_ids"][0, indices].view(2, 3)
        with torch.no_grad():
            previous_embeddings = model.embed_tokens(token_ids)
        block_positions = torch.arange(3).expand(2, -1)
        previous_target_logits = None
        previous_target_mask = None
        if uses_logits:
            target_blocks = targets.view(2, 3, -1)
            previous_target_logits = torch.cat(
                [torch.zeros_like(target_blocks[:, :1]), target_blocks[:, :-1]], dim=1
            )
            previous_target_mask = block_positions > 0
        return model._teacher_forced_hidden_feedback_correction(
            hidden_blocks,
            previous_embeddings,
            block_positions,
            base_logits.view(2, 3, -1) if uses_logits else None,
            previous_target_logits,
            previous_target_mask,
        )

    try:
        expected = run(plain)
        actual = run(checked)
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right)
        assert len(correction_calls) == 3
        assert torch.count_nonzero(correction_calls[0]) == 0
        assert torch.count_nonzero(correction_calls[1]) > 0
        expected[0].square().mean().backward()
        actual[0].square().mean().backward()
        _assert_gradients_match(plain, checked)
        assert decoder_counts == [2, 2]
        assert len(correction_calls) == 3
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in checked.correction_head.parameters()
        )
    finally:
        handle.remove()
        for decoder_handle in decoder_handles:
            decoder_handle.remove()
