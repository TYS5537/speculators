"""Full-target calibration must not change conditional pruned-vocabulary KD."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[3]


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def projection():
    return _load_file(
        "pruned_distribution_projection",
        ROOT / "src/speculators/models/dflash/target_distribution.py",
    ).project_target_distribution


@pytest.fixture
def metrics_modules(monkeypatch):
    common = _load_file(
        "pruned_distribution_common", ROOT / "src/speculators/models/metrics.py"
    )
    monkeypatch.setitem(sys.modules, "speculators.models.metrics", common)
    dflash = _load_file(
        "pruned_distribution_dflash", ROOT / "src/speculators/models/dflash/metrics.py"
    )
    dspark = _load_file(
        "pruned_distribution_dspark", ROOT / "src/speculators/models/dspark/metrics.py"
    )
    return common, dflash, dspark


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [1, 3, 20])
def test_chunked_projection_matches_dense_and_calls_head_hooks(
    projection, dtype, chunk_size
):
    torch.manual_seed(32)
    head = torch.nn.Linear(4, 9, bias=False, dtype=dtype)
    head.requires_grad_(False)
    hidden = torch.randn(1, 7, 4, dtype=dtype, requires_grad=True)
    selected = torch.tensor([0, 2, 5, 8])
    full_logits = head(hidden).detach()
    inverse = torch.full((9,), -1, dtype=torch.long)
    inverse[selected] = torch.arange(4)
    calls = []
    handle = head.register_forward_pre_hook(
        lambda _module, args: calls.append(args[0].shape[-2])
    )
    try:
        subset, log_z, greedy = projection(
            hidden, head, selected, token_chunk_size=chunk_size
        )
    finally:
        handle.remove()

    torch.testing.assert_close(subset, full_logits[..., selected])
    torch.testing.assert_close(log_z, full_logits.float().logsumexp(-1))
    assert torch.equal(greedy, inverse[full_logits.argmax(-1)])
    assert subset.dtype == dtype
    assert log_z.dtype == torch.float32
    assert greedy.dtype == torch.long
    assert not subset.requires_grad
    assert not log_z.requires_grad
    assert sum(calls) == 7
    assert max(calls) <= chunk_size
    assert len(calls) == (7 + chunk_size - 1) // chunk_size


@pytest.mark.parametrize(("selected", "expected"), [([1, 3], -1), ([0, 3], 0)])
def test_global_argmax_ties_use_first_target_id_not_first_subset_id(
    projection, selected, expected
):
    head = torch.nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        head.weight.zero_()
    _, _, greedy = projection(torch.ones(1, 3, 2), head, torch.tensor(selected))
    assert greedy.tolist() == [[expected] * 3]


def test_empty_projection_keeps_explicit_vocabulary_shape(projection):
    head = torch.nn.Linear(2, 4, bias=False)
    subset, log_z, greedy = projection(torch.empty(1, 0, 2), head, torch.tensor([0, 2]))
    assert subset.shape == (1, 0, 2)
    assert log_z.shape == greedy.shape == (1, 0)


def _backbone_harness(projection, sample_from_anchor, *, pruned):
    utils = _load_file(
        "pruned_distribution_utils", ROOT / "src/speculators/models/dflash/utils.py"
    )
    path = ROOT / "src/speculators/models/dflash/core.py"
    model = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "DFlashDraftModel"
    )
    method = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == "_backbone_forward"
    )
    method.decorator_list = []
    namespace = {
        "torch": torch,
        "project_target_distribution": projection,
        "get_base_indices_for_anchored_blocks": (
            utils.get_base_indices_for_anchored_blocks
        ),
        "build_anchored_loss_mask": utils.build_anchored_loss_mask,
    }
    exec(  # noqa: S102 -- Execute the real method without importing Transformers.
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    width = 3 if pruned else 6
    norm = torch.nn.Linear(3, 3, bias=False)
    norm.weight.data.copy_(torch.diag(torch.tensor([0.5, 1.0, 2.0])))
    head = torch.nn.Linear(3, 6, bias=False)
    head.weight.data.copy_(
        torch.tensor(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=torch.float32,
        )
    )
    harness = SimpleNamespace(
        block_size=3,
        mask_token_id=0,
        config=SimpleNamespace(sample_from_anchor=sample_from_anchor),
        use_draft_vocab=pruned,
        draft_vocab_size=width,
        d2t=torch.tensor([0, 1, 2]) if pruned else None,
        embed_tokens=torch.nn.Embedding(16, 3),
        verifier_norm=norm,
        verifier_lm_head=head,
        norm=torch.nn.Identity(),
        lm_head=torch.nn.Linear(3, width, bias=False),
        layers=[],
        _fuse_target_hidden=lambda value: value,
        _condition_noise_embedding=lambda noise, *_args: noise,
        rotary_emb=lambda *_args: (None, None),
        _build_attention_mask=lambda *_args: (
            None,
            None,
            torch.tensor([1, 4]),
            torch.tensor([True, True]),
        ),
    )
    harness._backbone_forward = MethodType(namespace["_backbone_forward"], harness)
    return harness


@pytest.mark.parametrize("sample_from_anchor", [False, True])
@pytest.mark.parametrize("pruned", [False, True])
def test_real_backbone_aligns_all_target_statistics_with_teacher_logits(
    projection, sample_from_anchor, pruned
):
    harness = _backbone_harness(projection, sample_from_anchor, pruned=pruned)
    teacher = torch.tensor(
        [
            [
                [2, 0, 0],
                [-2, 0, 0],
                [0, 2, 0],
                [0, -2, 0],
                [0, 0, 2],
                [0, 0, -2],
                [1, 1, 1],
                [-1, -1, -1],
            ]
        ],
        dtype=torch.float32,
    )
    outputs = harness._backbone_forward(
        torch.zeros_like(teacher),
        torch.arange(8).unsqueeze(0),
        torch.ones(1, 8),
        teacher,
        torch.zeros(1, 8, dtype=torch.long),
        max_anchors=2,
    )
    _, _, targets, _, indices, log_z, greedy = outputs
    full = harness.verifier_lm_head(harness.verifier_norm(teacher))
    if not sample_from_anchor:
        full = full.roll(1, dims=1)
    full = full[:, indices]
    selected = torch.tensor([0, 2, 4]) if pruned else torch.arange(6)
    torch.testing.assert_close(targets, full[..., selected])
    if pruned:
        inverse = torch.tensor([0, -1, 1, -1, 2, -1])
        torch.testing.assert_close(log_z, full.logsumexp(-1))
        assert torch.equal(greedy, inverse[full.argmax(-1)])
        torch.testing.assert_close(
            (targets.float() - log_z.unsqueeze(-1)).exp(),
            full.softmax(-1)[..., selected],
        )
    else:
        assert log_z is None
        assert greedy is None


def test_confidence_uses_fifteen_percent_absolute_probability(metrics_modules):
    common, _, dspark = metrics_modules
    # Target puts 85% outside the draft vocabulary; conditional KD is [2/3, 1/3].
    targets = torch.tensor([[[0.10, 0.05]]]).log()
    logits = targets.clone().requires_grad_()
    confidence = torch.zeros(1, 1, requires_grad=True)
    loss, metrics = dspark.compute_metrics(
        logits,
        targets,
        confidence,
        torch.ones(1, 1),
        1,
        loss_config={"tv": (common.tv_loss, 1.0)},
        target_log_normalizer=torch.zeros(1, 1),
        target_argmax_ids=torch.full((1, 1), -1),
        base_logits=logits.detach(),
        rollout_logits=logits.detach(),
        collaboration_base_logits=logits.detach(),
    )
    loss.backward()
    assert confidence.grad.item() == pytest.approx(0.35, abs=1e-5)
    for prefix in ("accept", "base_accept", "rollout_accept", "correction_only_accept"):
        rate = metrics[f"{prefix}_rate_sum"] / metrics[f"{prefix}_rate_total"]
        assert rate.item() == pytest.approx(0.15)
    assert metrics["full_acc_sum"].item() == 0
    assert metrics["rollout_full_acc_sum"].item() == 0


@pytest.mark.parametrize("adaptive_loss", ["none", "cat"])
def test_full_target_statistics_leave_conditional_kd_values_and_gradients_unchanged(
    metrics_modules, adaptive_loss
):
    common, _, dspark = metrics_modules
    torch.manual_seed(76)
    values = torch.randn(1, 6, 3)
    targets = torch.randn(1, 6, 3)
    results = []
    for calibrated in (False, True):
        logits = values.clone().requires_grad_()
        kwargs = (
            {
                "target_log_normalizer": targets.logsumexp(-1) + 2.0,
                "target_argmax_ids": torch.full((1, 6), -1),
            }
            if calibrated
            else {}
        )
        loss, _ = dspark.compute_metrics(
            logits,
            targets,
            None,
            torch.ones(1, 6),
            3,
            loss_config={"ce": (common.ce_loss, 0.1), "tv": (common.tv_loss, 0.9)},
            adaptive_loss=adaptive_loss,
            first_error_focal_alpha=0.2,
            **kwargs,
        )
        loss.backward()
        results.append((loss.detach(), logits.grad))
    for actual, expected in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
def test_sparse_candidates_cannot_match_an_out_of_subset_target(
    metrics_modules, algorithm
):
    common, dflash, dspark = metrics_modules
    targets = torch.tensor([[[0.10, 0.05]]]).log()
    kwargs = {
        "loss_config": {"tv": (common.tv_loss, 1.0)},
        "sample_from_anchor": True,
        "proposal_candidate_ids": torch.tensor([[[0]]]),
        "proposal_candidate_logits": torch.zeros(1, 1, 1),
        "target_argmax_ids": torch.full((1, 1), -1),
    }
    if algorithm == "dflash":
        _, metrics = dflash.compute_metrics(
            targets, targets, torch.ones(1, 1), 1, **kwargs
        )
    else:
        _, metrics = dspark.compute_metrics(
            targets,
            targets,
            None,
            torch.ones(1, 1),
            1,
            target_log_normalizer=torch.zeros(1, 1),
            **kwargs,
        )
        assert metrics["accept_rate_sum"].item() == pytest.approx(0.10)
    assert metrics["full_acc_sum"].item() == 0
    assert metrics["position_0_acc_sum"].item() == 0


def _load_vocab_mixin():
    path = ROOT / "src/speculators/model.py"
    mixin = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "DraftVocabMixin"
    )
    namespace = {"torch": torch, "nn": torch.nn}
    exec(  # noqa: S102 -- Exercise the real loader without requiring HF dependencies.
        compile(ast.Module(body=[mixin], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["DraftVocabMixin"]


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("needs_full", [False, True])
def test_real_weight_loader_preserves_full_untied_teacher_without_trainable_additions(
    monkeypatch, tied, needs_full
):
    mixin = _load_vocab_mixin()

    class TinyModel(mixin):
        def __init__(self, full):
            super().__init__()
            self._needs_full_verifier_distribution = full
            self.config = SimpleNamespace(
                draft_vocab_size=3,
                transformer_layer_config=SimpleNamespace(vocab_size=6, hidden_size=2),
                speculators_config=SimpleNamespace(
                    verifier=SimpleNamespace(name_or_path="local-test-target")
                ),
            )
            self._init_vocab(self.config)
            self.fc = torch.nn.Linear(2, 2, bias=False)
            self.load_vocab_mappings(
                torch.tensor([True, False, True, False, True, False]),
                torch.tensor([0, 1, 2]),
            )

    embedding = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    teacher = embedding if tied else -embedding - 20
    weights = {"embed_tokens.weight": embedding}
    if not tied:
        weights["lm_head.weight"] = teacher
    loading = ModuleType("speculators.utils.loading")
    loading.load_model_layers = lambda *_args: weights
    monkeypatch.setitem(sys.modules, "speculators.utils.loading", loading)
    legacy = TinyModel(False)
    model = TinyModel(needs_full)
    assert set(model.state_dict()) == set(legacy.state_dict())
    # Reproduce HF resetting requires_grad during loading: the loader must refreeze
    # both the existing teacher and draft vocabulary heads, not the trainable FC.
    model.requires_grad_(True)
    model.load_verifier_weights()
    torch.testing.assert_close(model.embed_tokens.weight, embedding)
    torch.testing.assert_close(model.lm_head.weight, teacher[[0, 2, 4]])
    torch.testing.assert_close(
        model.verifier_lm_head.weight, teacher if needs_full else teacher[[0, 2, 4]]
    )
    trainable = {
        name: tuple(value.shape)
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    assert trainable == {"fc.weight": (2, 2)}
    assert trainable == {
        name: tuple(value.shape)
        for name, value in legacy.named_parameters()
        if value.requires_grad
    }

    legacy.load_verifier_weights()
    # Existing DFlash checkpoints omit this borrowed teacher weight. Loading such
    # a state must preserve the freshly reconstructed full head despite its size.
    old_state = {
        key: value
        for key, value in legacy.state_dict().items()
        if key != "verifier_lm_head.weight"
    }
    result = model.load_state_dict(old_state, strict=False)
    assert result.missing_keys == ["verifier_lm_head.weight"]
    assert not result.unexpected_keys
    torch.testing.assert_close(
        model.verifier_lm_head.weight, teacher if needs_full else teacher[[0, 2, 4]]
    )
    model.to(dtype=torch.bfloat16)
    assert model.verifier_lm_head.weight.dtype == torch.bfloat16
    assert not model.verifier_lm_head.weight.requires_grad
