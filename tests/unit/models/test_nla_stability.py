"""NLA must keep useful gradients below probability-space precision/floors."""

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).parents[3]


@pytest.fixture
def metrics(monkeypatch):
    name = "nla_stability_metrics"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "src/speculators/models/metrics.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "speculators.models.metrics", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gap", [13.0, 1000.0])
def test_low_overlap_retains_cross_entropy_value_and_gradient(metrics, dtype, gap):
    logits = torch.tensor([[[-gap, 0.0]]], dtype=dtype, requires_grad=True)
    targets = torch.tensor([[[10000.0, -10000.0]]], dtype=dtype)
    actual = metrics.neg_log_acceptance_loss(logits, targets)
    expected = torch.nn.functional.cross_entropy(
        logits.float().reshape(1, 2), torch.tensor([0]), reduction="none"
    ).reshape(1, 1)
    actual_gradient = torch.autograd.grad(actual.sum(), logits)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), logits)[0]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert actual_gradient[0, 0, 0] < -0.99
    assert targets.grad is None


def test_large_vocabulary_uniform_draft_is_not_a_constant_loss(metrics):
    vocab = 151936
    logits = torch.zeros(1, 1, vocab, requires_grad=True)
    targets = torch.full_like(logits, -100.0)
    targets[..., 7] = 100.0
    loss = metrics.neg_log_acceptance_loss(logits, targets)
    loss.sum().backward()
    assert loss.item() == pytest.approx(math.log(vocab))
    expected_gradient = torch.full_like(logits, 1 / vocab)
    expected_gradient[..., 7] -= 1
    torch.testing.assert_close(logits.grad, expected_gradient)


def test_regular_overlap_matches_probability_formula_and_both_gradients(metrics):
    torch.manual_seed(11)
    logits = torch.randn(2, 4, 17, requires_grad=True)
    targets = torch.randn(2, 4, 17, requires_grad=True)
    actual = metrics.neg_log_acceptance_loss(logits, targets)
    expected = -torch.log(
        torch.minimum(logits.softmax(-1), targets.softmax(-1)).sum(-1)
    )
    actual_gradients = torch.autograd.grad(actual.sum(), (logits, targets))
    expected_gradients = torch.autograd.grad(expected.sum(), (logits, targets))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_partial_probability_tie_uses_half_derivative_for_each_side(metrics):
    logits = torch.tensor([[[0.25, 0.25, 0.5]]]).log().requires_grad_()
    targets = torch.tensor([[[0.25, 0.5, 0.25]]]).log().requires_grad_()
    loss = metrics.neg_log_acceptance_loss(logits, targets)
    draft_grad, target_grad = torch.autograd.grad(loss.sum(), (logits, targets))
    torch.testing.assert_close(loss, torch.tensor([[-math.log(0.75)]]))
    torch.testing.assert_close(draft_grad, torch.tensor([[[-1 / 24, -5 / 24, 0.25]]]))
    torch.testing.assert_close(target_grad, torch.tensor([[[-1 / 24, 0.25, -5 / 24]]]))


def test_identical_distributions_have_zero_loss_and_gradients(metrics):
    torch.manual_seed(12)
    logits = torch.randn(1, 3, 19, requires_grad=True)
    targets = logits.detach().clone().requires_grad_()
    loss = metrics.neg_log_acceptance_loss(logits, targets)
    gradients = torch.autograd.grad(loss.sum(), (logits, targets))
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=2e-7, rtol=0)
    for gradient in gradients:
        torch.testing.assert_close(
            gradient, torch.zeros_like(gradient), atol=2e-7, rtol=0
        )


def test_overlap_below_probability_underflow_remains_finite_and_trainable(metrics):
    logits = torch.tensor([[[-1000.0, 0.0]]], requires_grad=True)
    targets = torch.tensor([[[0.0, -1000.0]]], requires_grad=True)
    assert torch.minimum(logits.softmax(-1), targets.softmax(-1)).sum() == 0
    loss = metrics.neg_log_acceptance_loss(logits, targets)
    gradients = torch.autograd.grad(loss.sum(), (logits, targets))
    torch.testing.assert_close(loss, torch.tensor([[1000.0 - math.log(2)]]))
    torch.testing.assert_close(gradients[0], torch.tensor([[[-0.5, 0.5]]]))
    torch.testing.assert_close(gradients[1], torch.tensor([[[0.5, -0.5]]]))


@pytest.mark.parametrize("chunk_size", [1, 5, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_recomputation_matches_eager_loss_and_both_gradients(
    metrics, chunk_size, dtype
):
    torch.manual_seed(13)
    logits = torch.randn(2, 7, 11, dtype=dtype, requires_grad=True)
    targets = torch.randn(2, 7, 11, dtype=dtype, requires_grad=True)
    chunked = metrics.chunked_neg_log_acceptance_loss(
        logits, targets, token_chunk_size=chunk_size
    )
    eager = metrics.neg_log_acceptance_loss(logits, targets)
    chunked_gradients = torch.autograd.grad(chunked.sum(), (logits, targets))
    eager_gradients = torch.autograd.grad(eager.sum(), (logits, targets))
    torch.testing.assert_close(chunked, eager, atol=0, rtol=0)
    for actual, expected in zip(chunked_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_checkpoint_keeps_input_references_not_vocabulary_activations(
    metrics, monkeypatch
):
    logits = torch.randn(2, 7, 11, requires_grad=True)
    targets = torch.randn_like(logits)
    input_storages = {
        logits.untyped_storage().data_ptr(),
        targets.untyped_storage().data_ptr(),
    }
    saved_tensors = []
    chunk_rows = []
    original = metrics.neg_log_acceptance_loss

    def observed(draft, teacher):
        chunk_rows.append(draft.shape[0])
        return original(draft, teacher)

    def pack(tensor):
        saved_tensors.append(tensor)
        return tensor

    monkeypatch.setattr(metrics, "neg_log_acceptance_loss", observed)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = metrics.chunked_neg_log_acceptance_loss(
            logits, targets, token_chunk_size=3
        )
    assert len(chunk_rows) == 5
    assert max(chunk_rows) <= 3
    assert saved_tensors
    for tensor in saved_tensors:
        assert tensor.numel() == 0 or (
            tensor.untyped_storage().data_ptr() in input_storages
        )
    loss.sum().backward()
    assert len(chunk_rows) == 10  # Every chunk was recomputed for backward.
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert targets.grad is None


def test_chunked_no_grad_and_empty_inputs(metrics, monkeypatch):
    monkeypatch.setattr(
        metrics, "checkpoint", Mock(side_effect=AssertionError("no backward needed"))
    )
    logits = torch.randn(1, 5, 3, requires_grad=True)
    targets = torch.randn_like(logits)
    with torch.no_grad():
        actual = metrics.chunked_neg_log_acceptance_loss(
            logits, targets, token_chunk_size=2
        )
        expected = metrics.neg_log_acceptance_loss(logits, targets)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    empty = torch.empty(1, 0, 3)
    assert metrics.chunked_neg_log_acceptance_loss(empty, empty).shape == (1, 0)


def test_dispatcher_and_legacy_fused_entry_use_stable_loss_without_tv(
    metrics, monkeypatch
):
    monkeypatch.setattr(
        metrics, "_fused_kernel", Mock(side_effect=AssertionError("no TV subtraction"))
    )
    # Load the actual compatibility entry point without requiring Triton on CPU.
    path = ROOT / "src/speculators/models/fused_tv_loss.py"
    definitions = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "fused_nla_loss"
    ]
    legacy = ModuleType("legacy_fused_nla_cpu")
    exec(  # noqa: S102 -- Execute the production entry point, excluding GPU kernels.
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        legacy.__dict__,
    )
    logits = torch.tensor([[[-13.0, 0.0]]], requires_grad=True)
    targets = torch.tensor([[[100.0, -100.0]]])
    expected = metrics.neg_log_acceptance_loss(logits, targets)
    configured, _ = metrics.resolve_loss_config("nla")["nla"]
    for loss_fn in (configured, legacy.fused_nla_loss):
        actual = loss_fn(logits, targets)
        gradient = torch.autograd.grad(actual.sum(), logits)[0]
        torch.testing.assert_close(actual, expected)
        assert gradient[..., 0] < -0.99


def test_chunked_compile_eager_preserves_loss_and_gradients(metrics):
    logits = torch.tensor([[[-13.0, 0.0], [0.0, -1000.0]]], requires_grad=True)
    targets = torch.tensor([[[100.0, -100.0], [-10000.0, 10000.0]]])

    def objective(draft, teacher):
        return metrics.chunked_neg_log_acceptance_loss(
            draft, teacher, token_chunk_size=1
        ).sum()

    compiled = torch.compile(objective, backend="eager", fullgraph=True)
    eager = objective(logits, targets)
    eager_gradient = torch.autograd.grad(eager, logits)[0]
    actual = compiled(logits, targets)
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual, eager)
    torch.testing.assert_close(actual_gradient, eager_gradient)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_dispatcher_and_legacy_entry_match_stable_eager(metrics):
    pytest.importorskip("triton")
    path = ROOT / "src/speculators/models/fused_tv_loss.py"
    spec = importlib.util.spec_from_file_location("nla_cuda_legacy", path)
    legacy = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(legacy)
    logits = torch.zeros(1, 257, 5, device="cuda", requires_grad=True)
    with torch.no_grad():
        logits[..., 0] = -13
    targets = torch.full_like(logits, -100)
    targets[..., 0] = 100
    eager = metrics.neg_log_acceptance_loss(logits, targets)
    eager_gradient = torch.autograd.grad(eager.sum(), logits)[0]
    for loss_fn in (metrics.nla_loss_fused_or_eager, legacy.fused_nla_loss):
        actual = loss_fn(logits, targets)
        actual_gradient = torch.autograd.grad(actual.sum(), logits)[0]
        torch.testing.assert_close(actual, eager)
        torch.testing.assert_close(actual_gradient, eager_gradient)
