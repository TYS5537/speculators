"""Joint Selector/Correction preparation, teacher forcing and gradient boundaries."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional

from speculators.models.mmuse.core import MMuseDraftModel


class _RecordingEmbedding(nn.Embedding):
    def __init__(self):
        super().__init__(12, 4, dtype=torch.float64)
        self.calls = []

    def forward(self, ids):
        self.calls.append((ids, torch.is_grad_enabled()))
        return super().forward(ids)


class _RecordingEncoder(nn.Module):
    def __init__(self, output_mode):
        super().__init__()
        self.output_mode = output_mode
        self.projection = nn.Linear(6, 2, bias=False, dtype=torch.float64)
        self.calls = []

    def encode_previous_distribution(self, mask, **kwargs):
        self.calls.append((mask, kwargs, torch.is_grad_enabled()))
        if "previous_logits" in kwargs:
            features = self.projection(kwargs["previous_logits"].softmax(dim=-1))
        else:
            codes = self.projection.weight.transpose(0, 1)[kwargs["candidate_ids"]]
            probabilities = kwargs["candidate_logits"].softmax(dim=-1)
            features = (codes * probabilities.unsqueeze(-1)).sum(dim=-2)
        return features * mask.unsqueeze(-1)


class _ConditioningHarness(nn.Module):
    _prepare_selector_conditioning = MMuseDraftModel._prepare_selector_conditioning
    _draft_ids_to_verifier = MMuseDraftModel._draft_ids_to_verifier
    _dflash2_proposal_logits = MMuseDraftModel._dflash2_proposal_logits
    _prepend_zero_compact_feature = staticmethod(
        MMuseDraftModel._prepend_zero_compact_feature
    )
    _replace_compact_feature_position = staticmethod(
        MMuseDraftModel._replace_compact_feature_position
    )

    def __init__(
        self, *, mode="logits", feedback="static", search="greedy", sample=True, block=3
    ):
        super().__init__()
        self.config = SimpleNamespace(
            sample_from_anchor=sample,
            selector_correction_feedback=feedback,
            dflash2_selector_search_mode=search,
        )
        self.draft_vocab_size = 6
        self.candidate_selector = object()
        self.correction_head = _RecordingEncoder(mode)
        self.embed_tokens = _RecordingEmbedding()
        self.d2t = torch.ones(6, dtype=torch.long)
        self.selector_loss = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.candidate_ids = (
            (torch.arange(block * 2).reshape(1, block, 2) % 6).expand(2, -1, -1).clone()
        )
        self.realized_logits = (
            torch.tensor([2.0, -1.0], dtype=torch.float64)
            .expand(2, block, 2)
            .clone()
            .requires_grad_()
        )
        self.teacher_logits = (
            torch.tensor([-3.0, 4.0], dtype=torch.float64)
            .expand(2, block, 2)
            .clone()
            .requires_grad_()
        )
        self.selected_ids = self.candidate_ids[..., 1 if search == "global" else 0]
        self.block_calls, self.static_calls = [], []

    def _dflash2_block_outputs(self, *args, **kwargs):
        self.block_calls.append((args, kwargs, torch.is_grad_enabled()))
        return (
            self.candidate_ids,
            self.realized_logits,
            self.selector_loss,
            self.selected_ids,
            self.teacher_logits,
        )

    def _selector_correction_inputs(self, *args, **kwargs):
        result = MMuseDraftModel._selector_correction_inputs(self, *args, **kwargs)
        self.static_calls.append((args, kwargs, result))
        return result


def _inputs(block=3):
    return {
        "base_logits": torch.randn(
            1, 2 * block, 6, dtype=torch.float64, requires_grad=True
        ),
        "targets": torch.randn(
            1, 2 * block, 6, dtype=torch.float64, requires_grad=True
        ),
        "hidden_blocks": torch.randn(
            2, block, 4, dtype=torch.float64, requires_grad=True
        ),
        "block_tokens": torch.tensor([[9], [10]]).expand(2, block),
        "aligned_loss_mask": torch.ones(1, 2 * block),
        "prev_token_ids": torch.arange(2 * block).reshape(2, block) + 3,
        "block_positions": torch.arange(block).expand(2, -1),
    }


def _assert_block_call(model, inputs):
    assert len(model.block_calls) == 1
    args, kwargs, grad_enabled = model.block_calls[0]
    for index, name in enumerate(("base_logits", "targets", "hidden_blocks")):
        assert args[index] is inputs[name]
    torch.testing.assert_close(args[3], inputs["block_tokens"][:, 0])
    assert args[4] is inputs["aligned_loss_mask"]
    assert kwargs["teacher_previous_token_ids"] is inputs["prev_token_ids"]
    assert grad_enabled


def _assert_embeddings(model, embeddings, expected_ids):
    assert len(model.embed_tokens.calls) == 1
    ids, grad_enabled = model.embed_tokens.calls[0]
    torch.testing.assert_close(ids, expected_ids)
    assert not grad_enabled
    assert not embeddings.requires_grad
    expected = functional.embedding(expected_ids, model.embed_tokens.weight)
    torch.testing.assert_close(embeddings, expected)


@pytest.mark.parametrize(
    ("selector", "correction"), [(False, False), (True, False), (False, True)]
)
def test_non_joint_configuration_is_a_true_noop(selector, correction):
    model = _ConditioningHarness()
    if not selector:
        model.candidate_selector = None
    if not correction:
        model.correction_head = None
    inputs = _inputs()
    inputs["base_logits"] = None
    loss, previous, embeddings, rank, mask = model._prepare_selector_conditioning(
        **inputs,
        correction_output_mode="logits" if correction else None,
    )
    assert previous is inputs["prev_token_ids"]
    assert (loss, embeddings, rank, mask) == (None, None, None, None)
    assert model.block_calls == []
    assert model.static_calls == []
    assert model.embed_tokens.calls == []
    if correction:
        assert model.correction_head.calls == []


def test_joint_conditioning_requires_base_before_any_selector_work():
    model = _ConditioningHarness()
    inputs = _inputs()
    inputs["base_logits"] = None
    with pytest.raises(RuntimeError, match="pure DFlash base logits"):
        model._prepare_selector_conditioning(**inputs, correction_output_mode="logits")
    assert model.block_calls == []
    assert model.static_calls == []
    assert model.correction_head.calls == []
    assert model.embed_tokens.calls == []


@pytest.mark.parametrize("mode", ["hidden", "logits"])
@pytest.mark.parametrize("search", ["greedy", "global"])
@pytest.mark.parametrize("sample", [False, True])
def test_static_conditioning_delegates_path_and_preserves_sparse_gradient_boundary(
    mode, search, sample
):
    model = _ConditioningHarness(mode=mode, search=search, sample=sample)
    inputs = _inputs()
    loss, previous, embeddings, rank, mask = model._prepare_selector_conditioning(
        **inputs,
        correction_output_mode=mode,
    )
    _assert_block_call(model, inputs)
    assert len(model.static_calls) == 1
    args, kwargs, delegated = model.static_calls[0]
    assert args[0] is model.candidate_ids
    assert args[1] is model.realized_logits
    assert args[2] is model.selected_ids
    torch.testing.assert_close(args[3], inputs["block_tokens"][:, 0])
    if sample:
        assert kwargs["initial_previous_logits"] is None
    else:
        torch.testing.assert_close(
            kwargs["initial_previous_logits"], inputs["targets"].view(2, 3, 6)[:, 0]
        )
    assert loss is model.selector_loss
    assert previous is delegated[1]
    assert rank is delegated[2]
    assert mask is delegated[3]
    _assert_embeddings(model, embeddings, model.selected_ids + 1)
    expected_previous = inputs["block_tokens"][:, :1].expand(2, 3).clone()
    start = 0 if sample else 1
    expected_previous[:, start + 1 :] = model.selected_ids[:, start:-1] + 1
    torch.testing.assert_close(previous, expected_previous)
    objective = loss
    if mode == "hidden":
        assert rank is None
        assert mask is None
        assert model.correction_head.calls == []
    else:
        assert rank.requires_grad
        assert len(model.correction_head.calls) == 1 + int(not sample)
        sparse_mask, sparse_kwargs, grad_enabled = model.correction_head.calls[0]
        assert grad_enabled
        assert not sparse_kwargs["candidate_logits"].requires_grad
        source = model.realized_logits[:, start:-1].detach()
        if search == "global":
            source = torch.where(
                model.candidate_ids[:, start:-1]
                == model.selected_ids[:, start:-1].unsqueeze(-1),
                torch.zeros_like(source),
                torch.full_like(source, -torch.inf),
            )
        torch.testing.assert_close(
            sparse_kwargs["candidate_logits"][:, start + 1 :], source
        )
        assert not sparse_mask[:, : start + 1].any()
        torch.testing.assert_close(mask, inputs["block_positions"] > 0)
        if not sample:
            initial_mask, initial_kwargs, initial_grad_enabled = (
                model.correction_head.calls[1]
            )
            assert initial_mask.all()
            assert initial_grad_enabled
            torch.testing.assert_close(
                initial_kwargs["previous_logits"][:, 0],
                inputs["targets"].view(2, 3, 6)[:, 0],
            )
        objective = objective + rank.sum()
    objective.backward()
    assert model.selector_loss.grad == 1
    assert model.embed_tokens.weight.grad is None
    assert model.realized_logits.grad is None
    if mode == "logits":
        gradient = model.correction_head.projection.weight.grad
        assert gradient is not None
        assert torch.count_nonzero(gradient) > 0
        if not sample:
            assert inputs["targets"].grad is not None
            assert torch.count_nonzero(inputs["targets"].grad) > 0


@pytest.mark.parametrize("mode", ["hidden", "logits"])
@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_corrected_conditioning_is_teacher_forced_in_train_and_eval(
    mode, sample, training
):
    model = _ConditioningHarness(mode=mode, feedback="corrected", sample=sample).train(
        training
    )
    inputs = _inputs()
    loss, previous, embeddings, rank, mask = model._prepare_selector_conditioning(
        **inputs,
        correction_output_mode=mode,
    )
    _assert_block_call(model, inputs)
    assert model.static_calls == []
    assert previous is inputs["prev_token_ids"]
    assert loss is model.selector_loss
    teacher_ids = model.candidate_ids[..., 1]
    assert not torch.equal(teacher_ids, model.selected_ids)
    _assert_embeddings(model, embeddings, teacher_ids + 1)
    objective = loss
    if mode == "hidden":
        assert rank is None
        assert mask is None
        assert model.correction_head.calls == []
    else:
        assert len(model.correction_head.calls) == 1
        source_mask, kwargs, grad_enabled = model.correction_head.calls[0]
        assert grad_enabled
        assert source_mask.shape == (2, 2)
        assert source_mask.dtype == torch.bool
        assert source_mask.device == inputs["hidden_blocks"].device
        assert source_mask.all()
        source = inputs["targets"].view(2, 3, 6)[:, :-1]
        torch.testing.assert_close(kwargs["previous_logits"], source)
        assert kwargs.keys() == {"previous_logits"}
        expected = model.correction_head.projection(source.softmax(dim=-1))
        expected = torch.cat([expected.new_zeros(2, 1, 2), expected], dim=1)
        torch.testing.assert_close(rank, expected)
        torch.testing.assert_close(mask, inputs["block_positions"] > 0)
        assert torch.count_nonzero(rank[:, 0]) == 0
        assert rank.requires_grad
        objective = objective + rank.sum()
    objective.backward()
    assert model.selector_loss.grad == 1
    assert model.embed_tokens.weight.grad is None
    assert model.realized_logits.grad is None
    if mode == "logits":
        assert torch.count_nonzero(model.correction_head.projection.weight.grad) > 0
        assert torch.count_nonzero(inputs["targets"].grad) > 0
        assert torch.count_nonzero(inputs["targets"].grad.view(2, 3, 6)[:, -1]) == 0


@pytest.mark.parametrize("sample", [False, True])
def test_corrected_single_slot_prepends_zero_to_empty_teacher_features(sample):
    model = _ConditioningHarness(feedback="corrected", sample=sample, block=1)
    inputs = _inputs(block=1)
    loss, previous, embeddings, rank, mask = model._prepare_selector_conditioning(
        **inputs,
        correction_output_mode="logits",
    )
    assert previous is inputs["prev_token_ids"]
    _assert_embeddings(model, embeddings, model.candidate_ids[..., 1] + 1)
    assert len(model.correction_head.calls) == 1
    source_mask, kwargs, grad_enabled = model.correction_head.calls[0]
    assert grad_enabled
    assert source_mask.shape == (2, 0)
    assert kwargs["previous_logits"].shape == (2, 0, 6)
    assert rank.shape == (2, 1, 2)
    assert mask.shape == (2, 1)
    assert not mask.any()
    assert torch.count_nonzero(rank) == 0
    (loss + rank.sum()).backward()
    assert model.selector_loss.grad == 1
    assert model.correction_head.projection.weight.grad is not None
    assert torch.count_nonzero(model.correction_head.projection.weight.grad) == 0
    assert model.embed_tokens.weight.grad is None
