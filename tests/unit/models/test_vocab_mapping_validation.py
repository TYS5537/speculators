"""Vocabulary mappings must preserve every trained output column's meaning."""

import ast
from pathlib import Path

import pytest
import torch


@pytest.fixture
def vocab_model():
    path = Path(__file__).parents[3] / "src/speculators/model.py"
    model_class = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "DraftVocabMixin"
    )
    method = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_vocab_mappings"
    )
    namespace = {"torch": torch}
    exec(  # noqa: S102 -- Exercise the real method without Transformers dependencies.
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )

    class VocabModel(torch.nn.Module):
        load_vocab_mappings = namespace["load_vocab_mappings"]

        def __init__(self, *, full_vocab=False):
            super().__init__()
            self.verifier_vocab_size = 6
            self.draft_vocab_size = 6 if full_vocab else 3
            self.use_draft_vocab = not full_vocab
            self.register_buffer(
                "t2d", None if full_vocab else torch.zeros(6, dtype=torch.bool)
            )
            self.register_buffer(
                "d2t", None if full_vocab else torch.zeros(3, dtype=torch.long)
            )
            self.lm_head = torch.nn.Linear(2, self.draft_vocab_size, bias=False)
            # Stand in for other trained, vocabulary-indexed parameters too.
            self.selector_codebook = torch.nn.Parameter(
                torch.arange(self.draft_vocab_size, dtype=torch.float32)
            )

    return VocabModel


def _mapping():
    return (
        torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.bool),
        torch.tensor([1, 2, 3], dtype=torch.long),
    )


def _assert_unchanged(model, before):
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name])


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.uint8, torch.int32])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_first_load_accepts_valid_integer_mappings(
    vocab_model, mask_dtype, offset_dtype
):
    model = vocab_model()
    t2d, d2t = _mapping()
    model.load_vocab_mappings(t2d.to(mask_dtype), d2t.to(offset_dtype))
    torch.testing.assert_close(model.t2d, t2d)
    torch.testing.assert_close(model.d2t, d2t)
    assert (torch.arange(3) + model.d2t).tolist() == [1, 3, 5]


def test_checkpoint_mapping_is_preserved_and_identical_reload_is_a_noop(vocab_model):
    model = vocab_model()
    t2d, d2t = _mapping()
    # HF restores checkpoint buffers before calling load_vocab_mappings.
    model.load_state_dict({"t2d": t2d, "d2t": d2t}, strict=False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_vocab_mappings(t2d.to(torch.int32), d2t.to(torch.int32))
    model.load_vocab_mappings(None, None)
    _assert_unchanged(model, before)


def test_same_size_different_checkpoint_mapping_is_rejected_without_mutation(
    vocab_model,
):
    model = vocab_model()
    t2d, d2t = _mapping()
    model.load_state_dict({"t2d": t2d, "d2t": d2t}, strict=False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="Cannot replace an initialized"):
        model.load_vocab_mappings(~t2d, torch.tensor([0, 1, 2]))
    _assert_unchanged(model, before)


def test_zero_offsets_do_not_make_an_existing_mapping_uninitialized(vocab_model):
    model = vocab_model()
    model.load_vocab_mappings(
        torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool),
        torch.zeros(3, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="Cannot replace an initialized"):
        model.load_vocab_mappings(*_mapping())


@pytest.mark.parametrize(
    ("t2d", "d2t", "message"),
    [
        (torch.tensor(True), torch.tensor([1, 2, 3]), "one-dimensional"),
        (torch.ones(6, 1), torch.tensor([1, 2, 3]), "one-dimensional"),
        (torch.ones(6, dtype=torch.bool), torch.tensor([[1, 2, 3]]), "one-dimensional"),
        (torch.tensor([0.0, 1, 0, 1, 0, 1]), torch.tensor([1, 2, 3]), "t2d must"),
        (torch.tensor([0, 1, 0, 1, 0, 1]), torch.tensor([1.0, 2, 3]), "integer dtype"),
        (
            torch.tensor([0, 1, 0, 1, 0, 1]),
            torch.ones(3, dtype=torch.bool),
            "integer dtype",
        ),
        (torch.tensor([0, 2, 0, 1, 0, 0]), torch.tensor([1, 2, 3]), "binary"),
        (torch.tensor([0, 1, 0, 1, 1]), torch.tensor([1, 2, 3]), "verifier_vocab_size"),
        (torch.tensor([0, 1, 0, 1, 0, 0]), torch.tensor([1, 2, 3]), "expected 3"),
        (torch.tensor([0, 1, 0, 1, 0, 1]), torch.tensor([1, 2]), "draft_vocab_size"),
        (torch.tensor([0, 1, 0, 1, 0, 1]), torch.tensor([0, 1, 2]), "vocabulary order"),
        (torch.tensor([0, 1, 0, 1, 0, 1]), torch.tensor([1, 0, 3]), "vocabulary order"),
        (torch.tensor([0, 1, 0, 1, 0, 1]), torch.tensor([3, 0, 3]), "vocabulary order"),
        (
            torch.tensor([0, 1, 0, 1, 0, 1]),
            torch.tensor([-1, 2, 3]),
            "vocabulary order",
        ),
        (torch.tensor([0, 1, 0, 1, 0, 1]), torch.tensor([1, 2, 4]), "vocabulary order"),
    ],
    ids=[
        "scalar-mask",
        "matrix-mask",
        "matrix-offsets",
        "float-mask",
        "float-offsets",
        "bool-offsets",
        "nonbinary-mask",
        "wrong-target-size",
        "wrong-selected-count",
        "wrong-draft-size",
        "inconsistent-pair",
        "duplicate-target-id",
        "reordered-target-ids",
        "negative-target-id",
        "out-of-range-target-id",
    ],
)
def test_invalid_mapping_is_rejected_before_changing_buffers(
    vocab_model, t2d, d2t, message
):
    model = vocab_model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match=message):
        model.load_vocab_mappings(t2d, d2t)
    _assert_unchanged(model, before)


@pytest.mark.parametrize("missing_mask", [False, True])
def test_both_mappings_are_required(vocab_model, missing_mask):
    t2d, d2t = _mapping()
    with pytest.raises(ValueError, match="Both t2d and d2t"):
        vocab_model().load_vocab_mappings(
            None if missing_mask else t2d, d2t if missing_mask else None
        )


def test_none_preserves_new_placeholders(vocab_model):
    model = vocab_model()
    model.load_vocab_mappings(None, None)
    assert not model.t2d.any()
    assert not model.d2t.any()


def test_full_vocab_identity_files_are_still_ignored(vocab_model):
    model = vocab_model(full_vocab=True)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_vocab_mappings(
        torch.ones(6, dtype=torch.bool), torch.zeros(6, dtype=torch.long)
    )
    assert model.t2d is None
    assert model.d2t is None
    _assert_unchanged(model, before)
