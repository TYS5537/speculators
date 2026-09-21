"""Regression tests for exact acceptance ratios of rare proposed tokens."""

# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_evaluator():
    path = Path(__file__).parents[3] / "scripts/evaluate/dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("offline_probability_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.torch = torch
    return module


def _verify(module, draft_probability, target_probability):
    logits = torch.tensor(
        [[[1.0 - target_probability, target_probability], [1.0, 0.0]]]
    ).log()

    class Target:
        @staticmethod
        def __call__(**_kwargs):
            return SimpleNamespace(logits=logits)

    return module.verify_draft_tokens(
        target_model=Target(),
        proposal=module.DraftProposal(
            draft_token_count=1,
            verify_input_ids=torch.tensor([[0, 1]]),
            draft_probs=torch.tensor([[[1.0 - draft_probability, draft_probability]]]),
        ),
        position_ids=torch.arange(2).unsqueeze(0),
        start=0,
        past_key_values_target=None,
        temperature=1.0,
        max_proposal_tokens=1,
        current_token_ids=torch.tensor([[0]]),
    )


@pytest.mark.parametrize("draft_probability", [1e-9, 1e-20, 1e-40])
@pytest.mark.parametrize("ratio", [0.2, 1.0, 2.0])
def test_rare_proposal_uses_actual_q_without_probability_floor(
    monkeypatch, draft_probability, ratio
):
    module = _load_evaluator()
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.full_like(value, 0.5))
    monkeypatch.setattr(module, "sample_from_probs", lambda probs: probs.argmax(dim=-1))

    result = _verify(module, draft_probability, draft_probability * ratio)

    assert result.accept_probs.item() == pytest.approx(min(ratio, 1.0), rel=1e-4)
    assert result.accepted_draft_tokens == int(ratio >= 1.0)
    assert result.effective_proposal_length == 1
    # Rejection falls back to the other token; an accepted proposal gets a bonus.
    assert result.next_token.tolist() == [0]
    assert result.committed_tokens.tolist() == ([[1, 0]] if ratio >= 1.0 else [[0]])


@pytest.mark.parametrize("draft_probability", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_selected_draft_probability_fails_explicitly(draft_probability):
    module = _load_evaluator()

    with pytest.raises(ValueError, match="finite, positive draft probability"):
        _verify(module, draft_probability, 0.1)
