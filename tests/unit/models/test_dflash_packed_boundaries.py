"""Packed-document boundaries retain valid draft prefixes without cross-doc loss."""

import ast
import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def utils():
    root = Path(__file__).parents[3]
    return _load_file(
        "packed_dflash_utils", root / "src/speculators/models/dflash/utils.py"
    )


@pytest.mark.parametrize(
    ("sample_from_anchor", "expected"),
    [(True, [1, 1, 0, 0]), (False, [0, 1, 1, 0])],
)
def test_boundary_checks_the_predicted_token_not_only_its_hidden_position(
    utils, sample_from_anchor, expected
):
    result = utils.build_anchored_loss_mask(
        torch.ones(1, 8),
        torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1]]),
        torch.tensor([0]),
        torch.tensor([True]),
        4,
        sample_from_anchor=sample_from_anchor,
    )

    assert result.tolist() == [expected]


@pytest.mark.parametrize(
    ("sample_from_anchor", "expected"),
    [(True, [1, 0, 0, 0]), (False, [0, 1, 0, 0])],
)
def test_short_document_keeps_its_partial_block(utils, sample_from_anchor, expected):
    result = utils.build_anchored_loss_mask(
        torch.ones(1, 8),
        torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]]),
        torch.tensor([0]),
        torch.tensor([True]),
        4,
        sample_from_anchor=sample_from_anchor,
    )

    assert result.tolist() == [expected]


@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_invalid_anchor_padding_and_out_of_range_tail_have_no_loss(
    utils, sample_from_anchor
):
    result = utils.build_anchored_loss_mask(
        torch.ones(1, 8),
        torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1]]),
        torch.tensor([0, 3, 7]),
        torch.tensor([True, False, True]),
        4,
        sample_from_anchor=sample_from_anchor,
    )

    assert result.shape == (1, 12)
    assert result[:, :4].sum() == 2
    assert torch.count_nonzero(result[:, 4:]) == 0


@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.int64, torch.float16, torch.float32]
)
@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_single_document_preserves_existing_mask_semantics_and_dtype(
    utils, dtype, sample_from_anchor
):
    mask = torch.tensor([[0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0]], dtype=dtype)
    anchors = torch.tensor([1, 5])
    indices = utils.get_base_indices_for_anchored_blocks(anchors, 4)
    expected = mask[:, indices].clone()
    if not sample_from_anchor:
        expected[:, ::4] = 0

    result = utils.build_anchored_loss_mask(
        mask,
        torch.zeros_like(mask, dtype=torch.long),
        anchors,
        torch.tensor([True, True]),
        4,
        sample_from_anchor=sample_from_anchor,
    )

    assert result.dtype == dtype
    torch.testing.assert_close(result, expected)


def test_anchor_selection_excludes_padding_and_document_final_tokens(utils):
    docs = torch.tensor([[0, 0, 0, 1, 1, -1, -1, -1, -1, -1, -1, -1]])
    anchors, valid = utils.select_anchors(
        torch.ones_like(docs), 8, 4, document_ids=docs
    )

    assert anchors[valid].tolist() == [0, 1, 3]
    assert valid.sum() == 3


@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_all_padding_has_no_valid_anchors_or_supervision(utils, sample_from_anchor):
    docs = torch.full((1, 12), -1)
    mask = torch.ones_like(docs)
    anchors, valid = utils.select_anchors(mask, 3, 4, document_ids=docs)
    aligned = utils.build_anchored_loss_mask(
        mask, docs, anchors, valid, 4, sample_from_anchor=sample_from_anchor
    )

    assert not valid.any()
    assert torch.count_nonzero(aligned) == 0


@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_real_backbone_wires_document_aware_selection_and_loss_mask(
    utils, sample_from_anchor
):
    root = Path(__file__).parents[3]
    path = root / "src/speculators/models/dflash/core.py"
    model = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "DFlashDraftModel"
    )
    methods = [
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_build_attention_mask", "_backbone_forward"}
    ]
    for method in methods:
        method.decorator_list = []
    namespace = {
        "torch": torch,
        "select_anchors": utils.select_anchors,
        "get_base_indices_for_anchored_blocks": (
            utils.get_base_indices_for_anchored_blocks
        ),
        "build_anchored_loss_mask": utils.build_anchored_loss_mask,
    }
    exec(  # noqa: S102 -- Execute the current methods without importing HF model classes.
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    harness = SimpleNamespace(
        block_size=4,
        use_draft_vocab=False,
        mask_token_id=0,
        uses_full_attn=False,
        uses_sliding_window_attn=False,
        config=SimpleNamespace(sample_from_anchor=sample_from_anchor),
        embed_tokens=torch.nn.Embedding(16, 3),
        verifier_norm=torch.nn.Linear(3, 3, bias=False),
        verifier_lm_head=torch.nn.Linear(3, 5, bias=False),
        norm=torch.nn.Identity(),
        lm_head=torch.nn.Linear(3, 5, bias=False),
        layers=[],
        _fuse_target_hidden=lambda value: value,
        _condition_noise_embedding=lambda noise, *_args: noise,
        rotary_emb=lambda *_args: (None, None),
    )
    for name in ("_build_attention_mask", "_backbone_forward"):
        setattr(harness, name, MethodType(namespace[name], harness))
    docs = torch.tensor([[0, 0, 0, 1, 1, -1, -1, -1, -1, -1, -1, -1]])
    _, _, _, aligned, indices, _, _ = harness._backbone_forward(
        torch.zeros(1, 12, 3),
        torch.arange(12).unsqueeze(0),
        torch.ones(1, 12),
        torch.zeros(1, 12, 3),
        docs,
        max_anchors=8,
    )

    assert indices.reshape(8, 4)[:3, 0].tolist() == [0, 1, 3]
    expected = (
        [[1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
        if sample_from_anchor
        else [[0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]]
    )
    assert aligned.reshape(8, 4)[:3].tolist() == expected
    assert torch.count_nonzero(aligned.reshape(8, 4)[3:]) == 0


@pytest.mark.parametrize("sliding_window", [None, 2])
def test_invalid_tail_queries_cannot_relay_another_documents_hidden_states(
    sliding_window,
):
    root = Path(__file__).parents[3]
    attention = _load_file(
        "packed_dflash_attention", root / "src/speculators/models/dflash/attention.py"
    )
    docs = torch.tensor([0, 0, 0, 1, 1, 1, -1, -1])
    mask_mod, q_len, kv_len = attention.create_anchor_block_mask_mod(
        docs,
        total_seq_len=8,
        anchor_positions=torch.tensor([1, 4]),
        block_size=4,
        sliding_window=sliding_window,
        sliding_window_non_causal=True,
    )
    mask = mask_mod(
        torch.tensor(0),
        torch.tensor(0),
        torch.arange(q_len)[:, None],
        torch.arange(kv_len)[None, :],
    )
    assert not mask[:4, 3:8].any()
    assert not mask[:4, 12:].any()
    torch.manual_seed(23)
    context = torch.randn(1, 1, 8, 4)
    queries = torch.randn(1, 1, 8, 4)

    def two_layers(base, synthetic):
        for _ in range(2):
            values = torch.cat([base, synthetic], dim=-2)
            synthetic = torch.nn.functional.scaled_dot_product_attention(
                synthetic, values, values, attn_mask=mask
            )
        return synthetic

    expected = two_layers(context, queries)
    changed_context = context.clone()
    changed_context[:, :, 3:] += 100
    changed_queries = queries.clone()
    changed_queries[:, :, 4:] -= 100
    actual = two_layers(changed_context, changed_queries)

    torch.testing.assert_close(actual[:, :, :4], expected[:, :, :4])
