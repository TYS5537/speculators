"""Raw supervision counts must survive model filtering but stay out of logs."""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def metrics_modules(monkeypatch):
    root = Path(__file__).parents[3]
    common = _load_file(
        "supervision_common", root / "src/speculators/models/metrics.py"
    )
    monkeypatch.setitem(sys.modules, "speculators.models.metrics", common)
    dflash = _load_file(
        "supervision_dflash", root / "src/speculators/models/dflash/metrics.py"
    )
    dspark = _load_file(
        "supervision_dspark", root / "src/speculators/models/dspark/metrics.py"
    )
    path = root / "src/speculators/train/utils.py"
    normalize = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "normalize_counted_metrics"
    )
    namespace = {}
    exec(  # noqa: S102 -- Load the pure normalization function without tokenizer imports.
        compile(ast.Module(body=[normalize], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return common, dflash, dspark, namespace["normalize_counted_metrics"]


@pytest.mark.parametrize("algorithm", ["dflash", "dspark"])
@pytest.mark.parametrize("sample_from_anchor", [True, False])
@pytest.mark.parametrize(
    "mask_values",
    [[0] * 8, [1, 1, 0, 0, 1, 0, 0, 0], [1] * 8],
    ids=["empty", "partial", "full"],
)
def test_supervision_total_is_the_unclamped_post_anchor_count(
    metrics_modules, algorithm, sample_from_anchor, mask_values
):
    common, dflash, dspark, normalize = metrics_modules
    torch.manual_seed(31)
    logits = torch.randn(1, 8, 5, requires_grad=True)
    targets = torch.randn(1, 8, 5)
    mask = torch.tensor([mask_values], dtype=torch.float32)
    if not sample_from_anchor:
        mask[:, ::4] = 0
    expected_count = mask.sum().item()
    kwargs = {
        "block_size": 4,
        "sample_from_anchor": sample_from_anchor,
        "loss_config": {"kl_div": (common.kl_div_loss, 1.0)},
    }
    if algorithm == "dflash":
        loss, raw = dflash.compute_metrics(logits, targets, mask, **kwargs)
    else:
        confidence = torch.zeros(1, 8, requires_grad=True)
        loss, raw = dspark.compute_metrics(logits, targets, confidence, mask, **kwargs)
        for diagnostics in (False, True):
            filtered = dspark.select_logged_metrics(
                raw, include_diagnostics=diagnostics
            )
            assert filtered["supervision_total"].item() == expected_count
        raw = dspark.select_logged_metrics(raw)

    assert torch.isfinite(loss)
    assert raw["supervision_total"].item() == expected_count
    assert raw["supervision_total"].dtype == torch.float32
    assert not raw["supervision_total"].requires_grad
    assert raw["loss_total"].item() == 1
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    if expected_count == 0:
        assert loss.item() == 0
        assert torch.count_nonzero(logits.grad) == 0

    logged = normalize({key: value.item() for key, value in raw.items()}, world_size=2)
    assert "supervision_total" not in logged
    assert "supervision" not in logged
    assert "loss" in logged
