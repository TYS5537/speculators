"""Active-slot, anchor restoration and gradient contracts for parallel Correction."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import core as dflash_core
from speculators.models.mmuse import MMuseDraftModel, MMuseSpeculatorConfig
from speculators.models.mmuse import core as mmuse_core
from speculators.proposals.greedy import GreedyTokenProposalConfig


class _RecordingHead(nn.Module):
    """Pointwise, nonzero residuals make a per-slot reference independent of packing."""

    def __init__(self, output_mode):
        super().__init__()
        self.output_mode = output_mode
        self.state_scale = nn.Parameter(torch.tensor(0.7))
        self.residual_scale = nn.Parameter(torch.tensor(0.3))
        self.auxiliary_scale = nn.Parameter(torch.tensor(0.2))
        self.register_buffer("projection", torch.arange(24).reshape(6, 4) / 29)
        self.calls = []
        self.auxiliary_calls = 0

    def forward(self, previous, hidden, positions, **kwargs):
        self.calls.append((previous, hidden, positions, kwargs))
        states = hidden * self.state_scale + previous * 0.1
        states = states + positions.unsqueeze(-1) * 0.01
        current = kwargs.get("current_token_embeddings")
        if current is not None:
            states = states + current * 0.11
        if self.output_mode == "logits":
            source = kwargs.get("previous_logits")
            if source is None:
                source = kwargs["previous_rank_features"]
            feature = source.mean(dim=-1) * kwargs["previous_logits_mask"]
            states = states + feature.unsqueeze(-1) * 0.13
            residual = functional.linear(states, self.projection) * self.residual_scale
        else:
            residual = states * self.residual_scale
        return residual, states, None

    def auxiliary_hidden_residual(self, states):
        self.auxiliary_calls += 1
        return states * self.auxiliary_scale


class _ParallelHarness(nn.Module):
    _teacher_forced_parallel_correction = (
        MMuseDraftModel._teacher_forced_parallel_correction
    )

    def __init__(self, sample_from_anchor, mode):
        super().__init__()
        self.config = SimpleNamespace(
            sample_from_anchor=sample_from_anchor,
            correction_project_corrected_hidden=mode.startswith("dual"),
            correction_hidden_aux_loss=mode.endswith("aux"),
        )
        self.correction_head = _RecordingHead(
            "hidden" if mode == "hidden" else "logits"
        )
        self.embed_tokens = nn.Embedding(6, 4)
        self.lm_head = nn.Linear(4, 6, bias=False)


def _reference_slots(model, hidden, previous_ids, positions, *, targets, selector):
    """Assemble each slot explicitly; never call the production packing helper."""
    states, corrected, logits = [], [], []
    output_mode = model.correction_head.output_mode
    use_hidden = (
        output_mode == "hidden"
        or model.config.correction_hidden_aux_loss
        or model.config.correction_project_corrected_hidden
    )
    for position in range(hidden.shape[1]):
        original = hidden[:, position : position + 1]
        if position == 0 and not model.config.sample_from_anchor:
            slot_state = torch.zeros_like(original)
            slot_hidden = original
            slot_logits = functional.linear(original, model.lm_head.weight)
        else:
            kwargs = {}
            if selector:
                kwargs["current_token_embeddings"] = selector["current"][
                    :, position : position + 1
                ]
            if output_mode == "logits":
                if selector:
                    kwargs["previous_rank_features"] = selector["rank"][
                        :, position : position + 1
                    ]
                    kwargs["previous_logits_mask"] = selector["mask"][
                        :, position : position + 1
                    ]
                else:
                    kwargs["previous_logits"] = (
                        targets[:, position - 1 : position]
                        if position
                        else torch.zeros_like(targets[:, :1])
                    )
                    kwargs["previous_logits_mask"] = torch.full_like(
                        positions[:, position : position + 1],
                        position > 0,
                        dtype=torch.bool,
                    )
            with torch.no_grad():
                previous = model.embed_tokens(previous_ids[:, position : position + 1])
            residual, slot_state, _ = model.correction_head(
                previous, original, positions[:, position : position + 1], **kwargs
            )
            slot_hidden = original
            if use_hidden:
                delta = (
                    residual
                    if output_mode == "hidden"
                    else model.correction_head.auxiliary_hidden_residual(slot_state)
                )
                slot_hidden = original + delta
            projection_input = (
                slot_hidden
                if output_mode == "hidden"
                or model.config.correction_project_corrected_hidden
                else original
            )
            slot_logits = functional.linear(projection_input, model.lm_head.weight)
            if output_mode == "logits":
                slot_logits = slot_logits + residual
        states.append(slot_state)
        corrected.append(slot_hidden)
        logits.append(slot_logits)
    return (
        torch.cat(logits, dim=1).reshape(1, -1, 6),
        torch.cat(states, dim=1),
        torch.cat(corrected, dim=1) if use_hidden else None,
    )


def _objective(logits, states, corrected):
    result = logits.square().mean() + states.square().mean() * 0.3
    if corrected is not None:
        result = result + corrected.square().mean() * 0.2
    return result


def _assert_gradient_contract(model, reference, selector, reference_selector, *, start):
    for name, parameter in model.named_parameters():
        other = dict(reference.named_parameters())[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(
                parameter.grad, other.grad, rtol=1e-12, atol=1e-12, msg=name
            )
    assert model.embed_tokens.weight.grad is None
    assert model.correction_head.state_scale.grad.abs() > 0
    for name in ("current", "rank"):
        if name in selector:
            actual_grad, expected_grad = (
                selector[name].grad,
                reference_selector[name].grad,
            )
            assert (actual_grad is None) == (expected_grad is None)
            if actual_grad is not None:
                torch.testing.assert_close(
                    actual_grad, expected_grad, rtol=1e-12, atol=1e-12
                )
                if start:
                    assert torch.count_nonzero(actual_grad[:, 0]) == 0


@pytest.mark.parametrize("sample_from_anchor", [False, True])
@pytest.mark.parametrize("mode", ["hidden", "logits", "logits_aux", "dual", "dual_aux"])
@pytest.mark.parametrize("with_selector", [False, True])
def test_parallel_slots_match_independent_values_and_gradients(
    sample_from_anchor, mode, with_selector
):
    torch.manual_seed(205)
    model = _ParallelHarness(sample_from_anchor, mode).double()
    reference = copy.deepcopy(model)
    hidden = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    previous_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    positions = torch.arange(3).expand(2, -1)
    targets = torch.arange(36, dtype=torch.float64).reshape(2, 3, 6) / 43
    selector = {}
    if with_selector:
        selector = {
            "current": torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True),
            "rank": torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True),
            "mask": torch.tensor([[False, True, False], [False, True, True]]),
        }
    reference_selector = {
        name: value.detach().clone().requires_grad_(value.requires_grad)
        for name, value in selector.items()
    }
    projections = []
    handle = model.lm_head.register_forward_hook(
        lambda _module, _args, output: projections.append(output)
    )
    try:
        logits, states, corrected = model._teacher_forced_parallel_correction(
            hidden,
            previous_ids,
            positions,
            targets=targets.reshape(1, 6, 6),
            base_logits_blocks=(
                functional.linear(hidden, model.lm_head.weight)
                if mode.startswith("logits")
                else None
            ),
            selector_current_embeddings=selector.get("current"),
            selector_previous_rank_features=selector.get("rank"),
            selector_previous_logits_mask=selector.get("mask"),
        )
    finally:
        handle.remove()
    assert len(projections) == int(mode.startswith("dual"))
    assert model.correction_head.auxiliary_calls == int(
        "aux" in mode or mode.startswith("dual")
    )
    assert len(model.correction_head.calls) == 1
    start = 0 if sample_from_anchor else 1
    previous, active_hidden, active_positions, kwargs = model.correction_head.calls[0]
    torch.testing.assert_close(previous, model.embed_tokens(previous_ids[:, start:]))
    assert not previous.requires_grad
    torch.testing.assert_close(active_hidden, hidden[:, start:])
    torch.testing.assert_close(active_positions, positions[:, start:])
    if with_selector:
        torch.testing.assert_close(
            kwargs["current_token_embeddings"], selector["current"][:, start:]
        )
    if mode != "hidden":
        if with_selector:
            assert kwargs["previous_logits"] is None
            torch.testing.assert_close(
                kwargs["previous_rank_features"], selector["rank"][:, start:]
            )
            torch.testing.assert_close(
                kwargs["previous_logits_mask"], selector["mask"][:, start:]
            )
        else:
            assert "previous_rank_features" not in kwargs
            expected_targets = torch.cat(
                [torch.zeros_like(targets[:, :1]), targets[:, :-1]], dim=1
            )
            torch.testing.assert_close(
                kwargs["previous_logits"], expected_targets[:, start:]
            )
            torch.testing.assert_close(
                kwargs["previous_logits_mask"], (positions > 0)[:, start:]
            )
    else:
        assert "previous_logits" not in kwargs
        assert "previous_logits_mask" not in kwargs
        assert "previous_rank_features" not in kwargs
        assert logits is None
        logits = functional.linear(corrected, model.lm_head.weight).reshape(1, 6, 6)

    expected = _reference_slots(
        reference,
        reference_hidden,
        previous_ids,
        positions,
        targets=targets,
        selector=reference_selector,
    )
    for actual_value, expected_value in zip(
        (logits, states, corrected), expected, strict=True
    ):
        if expected_value is None:
            assert actual_value is None
        else:
            torch.testing.assert_close(
                actual_value, expected_value, rtol=1e-12, atol=1e-12
            )
    if not sample_from_anchor:
        assert torch.count_nonzero(states[:, 0]) == 0
        if corrected is not None:
            torch.testing.assert_close(corrected[:, 0], hidden[:, 0], rtol=0, atol=0)
    assert torch.count_nonzero(states[:, start:]) > 0
    _objective(logits, states, corrected).backward()
    _objective(*expected).backward()
    torch.testing.assert_close(
        hidden.grad, reference_hidden.grad, rtol=1e-12, atol=1e-12
    )
    _assert_gradient_contract(
        model, reference, selector, reference_selector, start=start
    )


def _tiny_model(sample_from_anchor, mode, with_selector):
    transformer = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        layer_types=["full_attention"],
    )
    transformer._attn_implementation = "eager"
    config = MMuseSpeculatorConfig(
        transformer_layer_config=transformer,
        draft_vocab_size=32,
        block_size=3,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=0,
        sample_from_anchor=sample_from_anchor,
        enable_correction_head=True,
        correction_hidden_size=16,
        correction_num_heads=4,
        correction_rank=4,
        correction_output_mode="hidden" if mode == "hidden" else "logits",
        correction_project_corrected_hidden=mode == "dual",
        correction_hidden_aux_loss=True,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        dflash2_candidate_selector=with_selector,
        dflash2_selector_rank=4,
        dflash2_selector_top_k=4,
        selector_correction_feedback="corrected"
        if mode == "logits" and with_selector
        else "static",
        speculators_config=SpeculatorsConfig(
            algorithm="mmuse",
            proposal_methods=[
                GreedyTokenProposalConfig(
                    speculative_tokens=3 if sample_from_anchor else 2
                )
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None, architectures=["Qwen3ForCausalLM"]
            ),
        ),
    )
    model = MMuseDraftModel(config).train()
    with torch.no_grad():
        for module in (model.embed_tokens, model.lm_head, model.verifier_lm_head):
            module.weight.normal_(std=0.1)
        model.verifier_norm.weight.fill_(1)
        model.correction_head.correction_up.weight.normal_(std=0.1)
        if model.correction_head.auxiliary_hidden_up is not None:
            model.correction_head.auxiliary_hidden_up.weight.normal_(std=0.1)
    return model


@pytest.mark.parametrize(
    ("sample_from_anchor", "mode", "with_selector"),
    [
        (True, "hidden", False),
        (False, "hidden", True),
        (False, "logits", False),
        (True, "logits", True),
        (False, "dual", True),
        (True, "dual", False),
    ],
)
def test_real_forward_keeps_anchor_and_projection_contract(
    monkeypatch, sample_from_anchor, mode, with_selector
):
    torch.manual_seed(711)
    model = _tiny_model(sample_from_anchor, mode, with_selector)
    monkeypatch.setattr(
        dflash_core,
        "select_anchors",
        lambda *_args, **_kwargs: (torch.tensor([1, 4]), torch.tensor([True, True])),
    )
    recorded = {}
    original_backbone = model._backbone_forward
    original_metrics = mmuse_core.compute_metrics

    def record_backbone(*args, **kwargs):
        result = original_backbone(*args, **kwargs)
        recorded["hidden"] = result[0].view(2, 3, 16)
        return result

    def record_metrics(*args, **kwargs):
        recorded["logits"] = args[0].view(2, 3, 32)
        return original_metrics(*args, **kwargs)

    monkeypatch.setattr(model, "_backbone_forward", record_backbone)
    monkeypatch.setattr(mmuse_core, "compute_metrics", record_metrics)
    projections, correction_calls, confidence_inputs = [], [], []
    handles = [
        model.lm_head.register_forward_hook(
            lambda _module, _args, output: projections.append(output)
        ),
        model.correction_head.register_forward_hook(
            lambda _module, args, output: correction_calls.append((args, output))
        ),
        model.confidence_head.register_forward_pre_hook(
            lambda _module, args: confidence_inputs.append(args[0])
        ),
    ]
    try:
        _, loss, metrics = model(
            hidden_states=torch.randn(1, 8, 32),
            input_ids=torch.arange(1, 9).unsqueeze(0),
            loss_mask=torch.ones(1, 8),
            verifier_last_hidden_states=torch.randn(1, 8, 16),
            document_ids=torch.zeros(1, 8, dtype=torch.long),
            max_anchors=2,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert len(projections) == 1 + int(with_selector and mode != "logits")
    assert len(correction_calls) == 1
    assert len(confidence_inputs) == 1
    start = 0 if sample_from_anchor else 1
    args, output = correction_calls[0]
    torch.testing.assert_close(args[1], recorded["hidden"][:, start:])
    torch.testing.assert_close(args[2], torch.arange(start, 3).expand(2, -1))
    confidence_states = confidence_inputs[0][..., 16:]
    torch.testing.assert_close(confidence_states[:, start:], output[1])
    if not sample_from_anchor:
        assert torch.count_nonzero(confidence_states[:, 0]) == 0
        expected_anchor = functional.linear(
            recorded["hidden"][:, 0], model.lm_head.weight
        )
        torch.testing.assert_close(
            recorded["logits"][:, 0], expected_anchor, rtol=1e-6, atol=1e-7
        )
    assert torch.isfinite(loss)
    assert metrics["correction_hidden_aux_loss_sum"] > 0
    loss.backward()
    for parameter in (model.fc.weight, model.correction_head.correction_up.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
