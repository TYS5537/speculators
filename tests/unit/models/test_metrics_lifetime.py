"""Keep metric memory optimizations independent of training and logging semantics."""

import weakref

import pytest
import torch

from speculators.models.dspark import metrics as dspark_metrics
from speculators.models.metrics import resolve_loss_config
from speculators.models.mmuse.metrics import (
    select_logged_metrics as select_mmuse_metrics,
)

_LOSSES = resolve_loss_config('{"ce": 0.1, "tv": 0.9}')
_DIAGNOSTIC_KEYS = {
    f"confidence_{name}_{suffix}"
    for name in ("abs_error", "pred_mean", "cumprod_bias")
    for suffix in ("sum", "total")
}


def _inputs(dtype=torch.float32):
    generator = torch.Generator().manual_seed(37)
    return {
        "logits": torch.randn(1, 6, 7, generator=generator).to(dtype),
        "targets": torch.randn(1, 6, 7, generator=generator).to(dtype),
        "confidence_logits": torch.randn(1, 6, generator=generator).to(dtype),
        "loss_mask": torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=dtype),
        "block_size": 3,
        "loss_config": _LOSSES,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length_alpha", [0.0, 0.1])
@pytest.mark.parametrize("base_diagnostics", [False, True])
def test_skipping_hidden_diagnostics_preserves_loss_gradients_update_and_rng(
    dtype, length_alpha, base_diagnostics
):
    inputs = _inputs(dtype)
    inputs.update(
        confidence_length_alpha=length_alpha,
        rollout_logits=inputs["targets"].flip(-1),
        collaboration_base_logits=inputs["targets"].roll(1, dims=-1),
        collaboration_gate=torch.full((2, 3, 1), 0.2, dtype=dtype),
        base_logits=inputs["targets"].clone() if base_diagnostics else None,
    )
    results = []
    for diagnostics in (True, False):
        logits = torch.nn.Parameter(inputs["logits"].clone())
        confidence = torch.nn.Parameter(inputs["confidence_logits"].clone())
        optimizer = torch.optim.AdamW([logits, confidence], lr=1e-3)
        torch.manual_seed(41)
        loss, metrics = dspark_metrics.compute_metrics(
            **{**inputs, "logits": logits, "confidence_logits": confidence},
            confidence_diagnostics=diagnostics,
        )
        loss.backward()
        gradients = (logits.grad.clone(), confidence.grad.clone())
        optimizer.step()
        results.append(
            (loss, metrics, gradients, logits, confidence, torch.get_rng_state())
        )
    full, compact = results
    assert set(full[1]) - set(compact[1]) == _DIAGNOSTIC_KEYS
    for key, value in compact[1].items():
        torch.testing.assert_close(value, full[1][key], rtol=0, atol=0)
    for index in (0, 2, 3, 4, 5):
        torch.testing.assert_close(full[index], compact[index], rtol=0, atol=0)
    for selector in (dspark_metrics.select_logged_metrics, select_mmuse_metrics):
        expected, actual = selector(full[1]), selector(compact[1])
        assert list(expected) == list(actual)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


@pytest.mark.parametrize("base_diagnostics", [False, True])
@pytest.mark.parametrize("proposal", [False, True])
def test_overlap_probabilities_are_released_before_loss(
    monkeypatch, base_diagnostics, proposal
):
    inputs = _inputs()
    inputs.update(
        rollout_logits=inputs["targets"].flip(-1),
        collaboration_base_logits=inputs["targets"].roll(1, dims=-1),
        base_logits=inputs["targets"].clone() if base_diagnostics else None,
    )
    if proposal:
        scores, ids = inputs["logits"].topk(3, dim=-1)
        inputs.update(proposal_candidate_ids=ids, proposal_candidate_logits=scores)
    probabilities = []
    original_softmax = dspark_metrics.softmax
    original_loss = dspark_metrics.compound_loss
    loss_calls = []

    def track_probabilities(*args, **kwargs):
        result = original_softmax(*args, **kwargs)
        probabilities.append(weakref.ref(result))
        return result

    def check_lifetime(*args, **kwargs):
        assert len(probabilities) == 4  # Target, draft, rollout, collaboration.
        assert [reference() is not None for reference in probabilities] == [
            base_diagnostics,
            False,
            False,
            False,
        ]
        loss_calls.append(True)
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(dspark_metrics, "softmax", track_probabilities)
    monkeypatch.setattr(dspark_metrics, "compound_loss", check_lifetime)
    dspark_metrics.compute_metrics(**inputs)
    assert len(loss_calls) == (2 if base_diagnostics else 1)
    assert all(reference() is None for reference in probabilities)


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("length_alpha", [0.0, 0.1])
def test_confidence_probability_is_only_computed_when_consumed(
    monkeypatch, diagnostics, length_alpha
):
    original_sigmoid = torch.Tensor.sigmoid
    calls = []

    def record_sigmoid(tensor):
        calls.append(True)
        return original_sigmoid(tensor)

    monkeypatch.setattr(torch.Tensor, "sigmoid", record_sigmoid)
    dspark_metrics.compute_metrics(
        **_inputs(),
        confidence_length_alpha=length_alpha,
        confidence_diagnostics=diagnostics,
    )
    assert len(calls) == int(diagnostics or length_alpha > 0)
