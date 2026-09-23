"""Greedy/Viterbi path ordering, search boundaries and teacher-loss contracts."""

import itertools

import pytest
import torch
from torch import nn

from tests.unit.models.test_mmuse_selector_runtime import _inputs, _SelectorHost


def _fixed_candidates(block, top_k=3, dtype=torch.float32):
    ids = torch.tensor([6, 1, 4, 0, 5, 2, 3], dtype=torch.int32)[:top_k]
    ids = ids.expand(2, block, top_k).clone()
    unary = torch.randn(2, block, 2 * top_k, dtype=dtype)[..., ::2].requires_grad_()
    hidden = torch.randn(2, block, 8, dtype=dtype)[..., ::2].requires_grad_()
    return ids, unary, hidden, torch.tensor([7, 15], dtype=torch.int32)


def _conditional_path_scores(host, ids, unary, hidden, anchors):
    """Enumerate tiny paths independently of the streaming max/backtracking code."""
    first = host.candidate_selector(
        ids[:, 0], unary[:, 0], hidden[:, 0], anchors.long()
    )
    first = first.float().log_softmax(dim=-1)
    edges = []
    for position in range(1, ids.shape[1]):
        lattice = host.candidate_selector.score_lattice(
            ids[:, position : position + 1],
            unary[:, position : position + 1],
            hidden[:, position : position + 1],
            host._draft_ids_to_verifier(ids[:, position - 1]).unsqueeze(1),
        )[:, 0]
        edges.append(lattice.float().log_softmax(dim=-1))
    scores = []
    for path in itertools.product(range(ids.shape[-1]), repeat=ids.shape[1]):
        score = first[:, path[0]]
        for position, edge in enumerate(edges):
            score = score + edge[:, path[position], path[position + 1]]
        scores.append(score)
    return first, edges, torch.stack(scores).max(dim=0).values


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("block", [1, 3, 4])
@pytest.mark.parametrize("top_k", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_global_path_maximizes_normalized_probability_and_returns_realized_rows(
    sample, block, top_k, dtype
):
    torch.manual_seed(49)
    host = _SelectorHost(search="global", sample=sample, block=block, top_k=top_k).to(
        dtype
    )
    ids, unary, hidden, anchors = _fixed_candidates(block, top_k, dtype)
    snapshots = [value.detach().clone() for value in (ids, unary, hidden, anchors)]
    selected, rows = host._dflash2_select_topk_path(ids, unary, hidden, anchors)
    start = 0 if sample else 1
    assert selected.dtype == ids.dtype
    assert rows.dtype == unary.dtype
    assert not selected.requires_grad
    assert not rows.requires_grad
    torch.testing.assert_close(rows[:, :start], unary[:, :start], rtol=0, atol=0)
    torch.testing.assert_close(selected[:, :start], ids[:, :start, 0], rtol=0, atol=0)
    if start < block:
        _assert_optimal_realized_path(
            host,
            ids[:, start:],
            unary[:, start:],
            hidden[:, start:],
            anchors,
            selected[:, start:],
            rows[:, start:],
        )
    for value, snapshot in zip((ids, unary, hidden, anchors), snapshots, strict=True):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)


@torch.no_grad()
def _assert_optimal_realized_path(host, ids, unary, hidden, anchors, selected, rows):
    first, edges, optimum = _conditional_path_scores(host, ids, unary, hidden, anchors)
    indices = (ids == selected.unsqueeze(-1)).long().argmax(dim=-1)
    batch = torch.arange(ids.shape[0])
    score = first[batch, indices[:, 0]]
    for position, edge in enumerate(edges):
        score = score + edge[batch, indices[:, position], indices[:, position + 1]]
    torch.testing.assert_close(score, optimum, rtol=0, atol=0)
    previous = anchors.long()
    for position in range(ids.shape[1]):
        expected = host.candidate_selector(
            ids[:, position], unary[:, position], hidden[:, position], previous
        )
        torch.testing.assert_close(rows[:, position], expected, rtol=0, atol=0)
        previous = host._draft_ids_to_verifier(selected[:, position])


@pytest.mark.parametrize("search", ["greedy", "global"])
@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("top_k", [1, 3, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_equal_rows_keep_first_supplied_candidate_without_sorting(
    search, sample, top_k, dtype
):
    host = _SelectorHost(search=search, sample=sample, top_k=top_k).to(dtype)
    with torch.no_grad():
        host.candidate_selector.hidden_projection.weight.zero_()
    ids, unary, hidden, anchors = _fixed_candidates(3, top_k, dtype)
    unary = torch.zeros_like(unary, requires_grad=True)
    selected, rows = host._dflash2_select_topk_path(ids, unary, hidden, anchors)
    torch.testing.assert_close(selected, ids[..., 0], rtol=0, atol=0)
    torch.testing.assert_close(rows, unary.detach(), rtol=0, atol=0)
    assert selected.data_ptr() != ids.data_ptr()
    assert rows.data_ptr() != unary.data_ptr()


class _CrossTieSelector(nn.Module):
    """Two equally probable paths: 6 -> 1 and 1 -> 6, with unsorted token IDs."""

    def forward(self, candidate_ids, unary_logits, hidden_states, previous_token_ids):
        # The host maps draft token 6 to verifier 13, and draft 1 to verifier 8.
        row = torch.where(
            (previous_token_ids == 13).unsqueeze(-1),
            unary_logits.new_tensor([-torch.inf, 0.0]),
            unary_logits,
        )
        return torch.where(
            (previous_token_ids == 8).unsqueeze(-1),
            unary_logits.new_tensor([0.0, -torch.inf]),
            row,
        )

    def score_lattice(
        self, candidate_ids, unary_logits, hidden_states, predecessor_ids
    ):
        return unary_logits.new_tensor([[-torch.inf, 0.0], [0.0, -torch.inf]]).expand(
            2, 1, 2, 2
        )


@pytest.mark.parametrize("sample", [False, True])
def test_global_ties_choose_terminal_index_before_predecessor_indices(sample):
    block = 2 if sample else 3
    host = _SelectorHost(search="global", sample=sample, block=block, top_k=2)
    host.candidate_selector = _CrossTieSelector()
    ids, unary, hidden, anchors = _fixed_candidates(block, top_k=2)
    unary = torch.zeros_like(unary)
    selected, _ = host._dflash2_select_topk_path(ids, unary, hidden, anchors)
    expected = torch.tensor([1, 6] if sample else [6, 1, 6], dtype=ids.dtype).expand(
        2, -1
    )
    torch.testing.assert_close(selected, expected)
    host.config.dflash2_selector_search_mode = "greedy"
    selected, _ = host._dflash2_select_topk_path(ids, unary, hidden, anchors)
    expected = torch.tensor([6, 1] if sample else [6, 6, 1], dtype=ids.dtype).expand(
        2, -1
    )
    torch.testing.assert_close(selected, expected)


class _CountingSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, candidate_ids, unary_logits, hidden_states, previous_token_ids):
        self.calls.append(
            ("row", candidate_ids, previous_token_ids, torch.is_grad_enabled())
        )
        return unary_logits + len(self.calls)

    def score_lattice(
        self, candidate_ids, unary_logits, hidden_states, predecessor_ids
    ):
        self.calls.append(
            ("lattice", candidate_ids, predecessor_ids, torch.is_grad_enabled())
        )
        return unary_logits.unsqueeze(-2).expand(
            *unary_logits.shape[:-1], predecessor_ids.shape[-1], unary_logits.shape[-1]
        ) + len(self.calls)


@pytest.mark.parametrize("search", ["greedy", "global"])
@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("block", [1, 3, 7])
def test_search_preserves_streaming_call_order_rescoring_and_last_mapping(
    monkeypatch, search, sample, block
):
    host = _SelectorHost(search=search, sample=sample, block=block)
    host.candidate_selector = _CountingSelector()
    ids, unary, hidden, anchors = _fixed_candidates(block)
    mappings = []
    original_map = host._draft_ids_to_verifier

    def map_ids(value):
        mappings.append((value.clone(), torch.is_grad_enabled()))
        return original_map(value)

    monkeypatch.setattr(host, "_draft_ids_to_verifier", map_ids)
    selected, rows = host._dflash2_select_topk_path(ids, unary, hidden, anchors)
    start = 0 if sample else 1
    count = block - start
    prefix = ["row"] + ["lattice"] * (count - 1) if search == "global" and count else []
    calls = host.candidate_selector.calls
    assert [entry[0] for entry in calls] == prefix + ["row"] * count
    assert all(not entry[3] for entry in calls)
    assert all(not entry[1] for entry in mappings)
    assert len(mappings) == count + (count - 1 if prefix else 0)
    for index, (kind, candidates, previous, _) in enumerate(calls):
        if kind == "lattice":
            assert candidates.shape == previous.shape == (2, 1, 3)
            torch.testing.assert_close(
                previous[:, 0], original_map(ids[:, start + index - 1])
            )
    _assert_rescored_rows(
        calls[-count:] if count else [],
        selected,
        rows,
        unary,
        anchors,
        original_map,
        start,
        len(prefix),
    )


def _assert_rescored_rows(
    calls, selected, rows, unary, anchors, map_ids, start, offset
):
    for index, (_, _, previous, _) in enumerate(calls):
        expected_previous = (
            anchors.long() if index == 0 else map_ids(selected[:, start + index - 1])
        )
        torch.testing.assert_close(previous, expected_previous)
        torch.testing.assert_close(
            rows[:, start + index],
            unary[:, start + index].detach() + (offset + index + 1),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("first_failure", range(4))
def test_validation_priority_precedes_search_or_empty_block_dispatch(first_failure):
    host = _SelectorHost(search="unknown", sample=False, block=1)
    ids, unary, hidden, _ = _fixed_candidates(1)
    anchors = torch.empty(2, 1)
    errors = [
        (RuntimeError, "selector is not enabled"),
        (ValueError, "candidate IDs and unary logits must align"),
        (ValueError, "candidates and hidden blocks must align"),
        (ValueError, "one anchor per block"),
    ]
    if first_failure == 0:
        host.candidate_selector = None
    if first_failure <= 1:
        unary = torch.empty(0)
    if first_failure <= 2:
        hidden = torch.empty(0)
    error, message = errors[first_failure]
    with pytest.raises(error, match=message):
        host._dflash2_select_topk_path(ids, unary, hidden, anchors)


@pytest.mark.parametrize(
    ("sample", "block"), [(False, 0), (True, 0), (False, 1), (True, 1)]
)
@pytest.mark.parametrize("missing_mode", [False, True])
def test_empty_active_block_returns_before_search_mode_access(
    sample, block, missing_mode
):
    host = _SelectorHost(search="unknown", sample=sample, block=block)
    if missing_mode:
        del host.config.dflash2_selector_search_mode
    ids, unary, hidden, anchors = _fixed_candidates(block)
    if sample and block:
        error = AttributeError if missing_mode else ValueError
        with pytest.raises(
            error, match="selector_search_mode|Unsupported DFlash2 selector"
        ):
            host._dflash2_select_topk_path(ids, unary, hidden, anchors)
        assert torch.is_grad_enabled()
    else:
        selected, rows = host._dflash2_select_topk_path(ids, unary, hidden, anchors)
        torch.testing.assert_close(selected, ids[..., 0])
        torch.testing.assert_close(rows, unary.detach())
        assert not rows.requires_grad


@pytest.mark.parametrize("search", ["greedy", "global"])
@pytest.mark.parametrize("sample", [False, True])
def test_path_dispatch_compiles_with_exact_teacher_loss_and_gradients(search, sample):
    host = _SelectorHost(search=search, sample=sample).double()
    inputs = _inputs()
    for name in ("logits", "targets", "hidden", "mask"):
        value = inputs[name]
        inputs[name] = value.double().detach().requires_grad_(value.requires_grad)
    args = tuple(
        inputs[name]
        for name in ("logits", "targets", "hidden", "anchors", "mask", "teacher_ids")
    )
    compiled = torch.compile(
        host._dflash2_block_outputs, backend="eager", fullgraph=True
    )
    actual, expected = compiled(*args), host._dflash2_block_outputs(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual[1].requires_grad
    assert actual[4].requires_grad
    leaves = (
        inputs["logits"],
        inputs["targets"],
        inputs["hidden"],
        *host.candidate_selector.parameters(),
    )
    actual_grads = torch.autograd.grad(actual[2], leaves, allow_unused=True)
    expected_grads = torch.autograd.grad(expected[2], leaves, allow_unused=True)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=0, atol=0)
