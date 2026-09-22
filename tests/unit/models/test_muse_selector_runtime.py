"""Selector runtime ownership and its minimal host contract."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from speculators.models.dflash import DFlashDraftModel
from speculators.models.dspark import DSparkDraftModel
from speculators.models.muse import MuseDraftModel
from speculators.models.muse.backbone import MuseBackboneMixin
from speculators.models.muse.selector import DFlash2CandidateSelector
from speculators.models.muse.selector_runtime import MuseSelectorMixin

_RUNTIME_METHODS = (
    "dflash2_select_candidates",
    "dflash2_sparse_logits",
    "_dflash2_select_topk_path",
    "dflash2_select_path",
    "_dflash2_proposal_logits",
    "_dflash2_block_outputs",
)


class _SelectorHost(MuseSelectorMixin, nn.Module):
    _draft_ids_to_verifier = MuseBackboneMixin._draft_ids_to_verifier

    def __init__(self, *, search="greedy", sample=True, block=3, top_k=3):
        super().__init__()
        self.config = SimpleNamespace(
            dflash2_selector_search_mode=search,
            sample_from_anchor=sample,
        )
        self.block_size, self.draft_vocab_size = block, 7
        self.verifier_ids = torch.tensor([11, 8, 14, 9, 12, 10, 13])
        self.d2t = self.verifier_ids - torch.arange(7)
        self.candidate_selector = DFlash2CandidateSelector(
            hidden_size=4,
            verifier_vocab_size=16,
            draft_vocab_size=7,
            rank=3,
            top_k=top_k,
        )
        with torch.no_grad():
            for parameter in self.candidate_selector.parameters():
                parameter.normal_(std=0.4)


def _inputs(block=3):
    logits = torch.randn(1, 2 * block, 7, requires_grad=True)
    teacher_ids = torch.tensor([[7], [15]]).expand(2, block).clone()
    return {
        "logits": logits,
        "blocks": logits.view(2, block, 7),
        "targets": torch.randn(1, 2 * block, 7, requires_grad=True),
        "hidden": torch.randn(2, block, 4, requires_grad=True),
        "anchors": torch.tensor([7, 15]),
        "teacher_ids": teacher_ids,
        "mask": torch.ones(1, 2 * block),
    }


def _invoke(host, method, inputs):
    function = getattr(host, method)
    if method in {"dflash2_select_candidates", "dflash2_sparse_logits"}:
        return function(inputs["blocks"], inputs["hidden"], inputs["teacher_ids"])
    if method == "dflash2_select_path":
        return function(inputs["blocks"], inputs["hidden"], inputs["anchors"])
    if method == "_dflash2_select_topk_path":
        unary, ids = inputs["blocks"].topk(3, dim=-1)
        return function(ids, unary, inputs["hidden"], inputs["anchors"])
    return function(
        inputs["logits"],
        inputs["targets"],
        inputs["hidden"],
        inputs["anchors"],
        inputs["mask"],
        inputs["teacher_ids"],
    )


def test_runtime_is_stateless_and_legacy_method_access_still_resolves():
    assert issubclass(MuseBackboneMixin, MuseSelectorMixin)
    assert not issubclass(MuseSelectorMixin, nn.Module)
    assert "__init__" not in MuseSelectorMixin.__dict__
    for name in _RUNTIME_METHODS:
        implementation = MuseSelectorMixin.__dict__[name]
        assert name not in MuseBackboneMixin.__dict__
        assert getattr(MuseBackboneMixin, name) is implementation
        assert getattr(MuseDraftModel, name) is implementation
        assert not hasattr(DSparkDraftModel, name)
        assert not hasattr(DFlashDraftModel, name)
    assert "_draft_ids_to_verifier" in MuseBackboneMixin.__dict__
    assert "_draft_ids_to_verifier" not in MuseSelectorMixin.__dict__
    assert MuseDraftModel.__mro__.index(
        MuseSelectorMixin
    ) < MuseDraftModel.__mro__.index(DSparkDraftModel)
    host = _SelectorHost()
    direct = nn.Module()
    direct.candidate_selector = copy.deepcopy(host.candidate_selector)
    assert host.state_dict().keys() == direct.state_dict().keys()
    for name, value in direct.state_dict().items():
        torch.testing.assert_close(host.state_dict()[name], value, rtol=0, atol=0)
    assert set(dict(host.named_modules())) == {
        "",
        "candidate_selector",
        "candidate_selector.hidden_projection",
    }


@pytest.mark.parametrize("search", ["greedy", "global"])
@pytest.mark.parametrize("sample", [False, True])
def test_minimal_host_keeps_public_paths_mapping_and_training_gradient_contract(
    monkeypatch, search, sample
):
    torch.manual_seed(432)
    host = _SelectorHost(search=search, sample=sample)
    inputs = _inputs()
    ids, scores = _invoke(host, "dflash2_select_candidates", inputs)
    sparse, sparse_ids = _invoke(host, "dflash2_sparse_logits", inputs)
    torch.testing.assert_close(ids, sparse_ids)
    torch.testing.assert_close(sparse.gather(-1, ids), scores)
    assert torch.equal(torch.isfinite(sparse).sum(dim=-1), torch.full((2, 3), 3))
    assert scores.requires_grad

    predecessors, lattice_predecessors = [], []
    handle = host.candidate_selector.register_forward_pre_hook(
        lambda _module, args: predecessors.append(
            (args[3].clone(), torch.is_grad_enabled())
        )
    )
    original_lattice = host.candidate_selector.score_lattice

    def record_lattice(candidate_ids, unary, hidden, previous_ids):
        lattice_predecessors.append((previous_ids.clone(), torch.is_grad_enabled()))
        return original_lattice(candidate_ids, unary, hidden, previous_ids)

    monkeypatch.setattr(host.candidate_selector, "score_lattice", record_lattice)
    try:
        path_ids, rows, selected = _invoke(host, "dflash2_select_path", inputs)
    finally:
        handle.remove()
    torch.testing.assert_close(path_ids, ids)
    assert not rows.requires_grad
    assert rows.grad_fn is None
    assert not selected.requires_grad
    start = 0 if sample else 1
    active_count = 3 - start
    realized_predecessors = predecessors[-active_count:]
    torch.testing.assert_close(realized_predecessors[0][0], inputs["anchors"])
    for offset, (previous_ids, grad_enabled) in enumerate(realized_predecessors):
        assert not grad_enabled
        if offset:
            expected_ids = host.verifier_ids[selected[:, start + offset - 1]]
            torch.testing.assert_close(previous_ids, expected_ids)
    assert len(lattice_predecessors) == (active_count - 1 if search == "global" else 0)
    for offset, (previous_ids, grad_enabled) in enumerate(lattice_predecessors):
        assert not grad_enabled
        torch.testing.assert_close(
            previous_ids[:, 0], host.verifier_ids[path_ids[:, start + offset]]
        )
    if not sample:
        torch.testing.assert_close(
            rows[:, 0], inputs["blocks"][:, 0].gather(-1, path_ids[:, 0])
        )
        torch.testing.assert_close(selected[:, 0], path_ids[:, 0, 0])
    proposal = host._dflash2_proposal_logits(path_ids, rows, selected)
    proposed = path_ids.gather(-1, proposal.argmax(dim=-1, keepdim=True)).squeeze(-1)
    torch.testing.assert_close(proposed, selected)
    if search == "greedy":
        assert proposal is rows
    else:
        assert torch.isfinite(proposal).sum(dim=-1).eq(1).all()

    teacher_ids, teacher_realized, loss, teacher_selected, teacher_rows = _invoke(
        host, "_dflash2_block_outputs", inputs
    )
    torch.testing.assert_close(teacher_ids, path_ids)
    torch.testing.assert_close(teacher_realized, rows)
    torch.testing.assert_close(teacher_selected, selected)
    assert not teacher_realized.requires_grad
    assert teacher_rows.requires_grad
    assert torch.isfinite(loss)
    if not sample:
        torch.testing.assert_close(
            teacher_rows[:, 0], inputs["blocks"][:, 0].gather(-1, ids[:, 0])
        )
    loss.backward()
    assert inputs["targets"].grad is None
    for value in (
        inputs["logits"],
        inputs["hidden"],
        *host.candidate_selector.parameters(),
    ):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0
    if not sample:
        assert torch.count_nonzero(inputs["hidden"].grad[:, 0]) == 0
        assert torch.count_nonzero(inputs["logits"].grad.view(2, 3, 7)[:, 0]) == 0


@pytest.mark.parametrize("search", ["greedy", "global"])
@pytest.mark.parametrize("sample", [False, True])
def test_single_slot_top1_preserves_reserved_anchor_and_empty_active_loss(
    search, sample
):
    torch.manual_seed(193)
    host = _SelectorHost(search=search, sample=sample, block=1, top_k=1)
    inputs = _inputs(block=1)
    calls = []
    handle = host.candidate_selector.register_forward_pre_hook(
        lambda _module, args: calls.append(args)
    )
    try:
        ids, rows, selected = _invoke(host, "dflash2_select_path", inputs)
    finally:
        handle.remove()
    assert len(calls) == (1 + int(search == "global") if sample else 0)
    torch.testing.assert_close(selected, inputs["blocks"].argmax(dim=-1))
    if not sample:
        torch.testing.assert_close(rows, inputs["blocks"].gather(-1, ids))
    _, _, loss, _, teacher_rows = _invoke(host, "_dflash2_block_outputs", inputs)
    assert loss == 0
    assert torch.isfinite(loss)
    assert teacher_rows.shape == (2, 1, 1)
    loss.backward()
    assert inputs["targets"].grad is None
    assert host.candidate_selector.hidden_projection.weight.grad is not None
    assert (
        torch.count_nonzero(host.candidate_selector.hidden_projection.weight.grad) == 0
    )


@pytest.mark.parametrize(
    "method", [name for name in _RUNTIME_METHODS if name != "_dflash2_proposal_logits"]
)
def test_disabled_selector_fails_through_all_runtime_entrypoints(method):
    host = _SelectorHost()
    host.candidate_selector = None
    with pytest.raises(RuntimeError, match="selector is not enabled"):
        _invoke(host, method, _inputs())


@pytest.mark.parametrize(
    ("method", "field", "replacement"),
    [
        ("dflash2_select_candidates", "hidden", torch.zeros(2, 2, 4)),
        ("dflash2_sparse_logits", "teacher_ids", torch.zeros(2, 2, dtype=torch.long)),
        ("dflash2_select_path", "hidden", torch.zeros(2, 2, 4)),
        ("_dflash2_select_topk_path", "anchors", torch.zeros(2, 1, dtype=torch.long)),
        ("_dflash2_block_outputs", "targets", torch.zeros(1, 5, 7)),
        ("_dflash2_block_outputs", "mask", torch.zeros(2, 3)),
    ],
)
def test_shape_guards_remain_before_selector_execution(method, field, replacement):
    host = _SelectorHost()
    inputs = _inputs()
    inputs[field] = replacement
    calls = []
    handle = host.candidate_selector.register_forward_pre_hook(
        lambda _module, args: calls.append(args)
    )
    try:
        with pytest.raises(ValueError):
            _invoke(host, method, inputs)
    finally:
        handle.remove()
    assert calls == []
