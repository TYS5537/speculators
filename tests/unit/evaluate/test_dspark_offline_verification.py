"""Verifier branch, stop-token and RNG contracts on real CPU tensors."""

# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def evaluator(monkeypatch):
    path = Path(__file__).parents[3] / "scripts/evaluate/dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("offline_verification_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.torch = torch
    return module


def _case(module, *, count=3, temperature=1.0, dtype=torch.float32):
    # Exact binary probabilities: selected acceptance ratios are [1, 1/2, 1].
    probabilities = torch.tensor(
        [
            [
                [0.125, 0.5, 0.25, 0.125],
                [0.5, 0.25, 0.125, 0.125],
                [0.125, 0.25, 0.125, 0.5],
            ]
        ],
    )[:, :count]
    bonus = torch.tensor([[[0.125, 0.125, 0.25, 0.5]]])
    probabilities = torch.cat([probabilities, bonus], dim=1)
    output = SimpleNamespace(
        logits=probabilities.log().to(dtype),
        hidden_states=object(),
        past_key_values=object(),
    )
    target = Mock(return_value=output)
    proposal = module.DraftProposal(
        draft_token_count=count,
        verify_input_ids=torch.tensor([[0, 1, 2, 3]])[:, : count + 1],
        draft_probs=torch.full((1, count, 4), 0.25) if count else None,
    )
    return SimpleNamespace(
        output=output,
        target=target,
        proposal=proposal,
        probabilities=probabilities,
        kwargs={
            "target_model": target,
            "proposal": proposal,
            "position_ids": torch.arange(10, 22).unsqueeze(0),
            "start": 3,
            "past_key_values_target": object(),
            "temperature": temperature,
            "max_proposal_tokens": 3,
            "current_token_ids": torch.tensor([[0]]),
        },
    )


def _observe_sampling(monkeypatch, module, *, uniforms=None):
    events = []
    rand_like = torch.rand_like
    residual = module.sample_residual
    sample = module.sample_from_probs
    multinomial = torch.multinomial

    def record_uniform(probs):
        events.append(("uniform", probs.clone()))
        return rand_like(probs) if uniforms is None else probs.new_tensor([uniforms])

    def record_residual(target_probs, draft_probs):
        events.append(("residual", target_probs.clone(), draft_probs.clone()))
        return residual(target_probs, draft_probs)

    def record_sample(probs):
        events.append(("sample", probs.clone()))
        return sample(probs)

    def record_multinomial(probs, *, num_samples):
        events.append(("multinomial", probs.clone(), num_samples))
        return multinomial(probs, num_samples=num_samples)

    monkeypatch.setattr(torch, "rand_like", record_uniform)
    monkeypatch.setattr(torch, "multinomial", record_multinomial)
    monkeypatch.setattr(module, "sample_residual", record_residual)
    monkeypatch.setattr(module, "sample_from_probs", record_sample)
    return events


@pytest.mark.parametrize(
    ("uniforms", "stops", "accepted", "effective", "terminated", "prefix"),
    [
        ([0.1, 0.1, 0.1], None, 3, 3, False, [1, 1, 1]),
        ([0.1, 0.1, 0.1], [], 3, 3, False, [1, 1, 1]),
        ([0.1, 0.75, 0.1], None, 1, 3, False, [1, 0, 0]),
        ([0.1, 0.1, 0.1], [1], 1, 1, True, [1, 1, 1]),
        ([0.1, 0.1, 0.1], [2], 2, 2, True, [1, 1, 1]),
        ([0.1, 0.1, 0.1], [3], 3, 3, True, [1, 1, 1]),
        ([0.1, 0.1, 0.1], [3, 1, 1], 1, 1, True, [1, 1, 1]),
        ([0.1, 0.1, 0.1], [0], 3, 3, False, [1, 1, 1]),
        ([0.1, 0.75, 0.1], [1], 1, 1, True, [1, 0, 0]),
        ([0.1, 0.75, 0.1], [2], 1, 3, False, [1, 0, 0]),
        ([0.1, 0.75, 0.1], [3], 1, 3, False, [1, 0, 0]),
        ([0.1, 0.75, 0.1], [0], 1, 3, False, [1, 0, 0]),
    ],
    ids=[
        "all",
        "empty-stop-list",
        "reject-middle",
        "stop-first",
        "stop-middle",
        "stop-last",
        "earliest-stop",
        "anchor-is-stop",
        "stop-before-rejection",
        "rejected-stop",
        "stop-after-rejection",
        "replacement-is-stop",
    ],
)
def test_acceptance_stop_truncation_and_sampling_order(
    evaluator, monkeypatch, uniforms, stops, accepted, effective, terminated, prefix
):
    case = _case(evaluator)
    case.kwargs["stop_token_ids"] = stops
    original_ids = case.proposal.verify_input_ids.clone()
    original_q = case.proposal.draft_probs.clone()
    original_logits = case.output.logits.clone()
    events = _observe_sampling(monkeypatch, evaluator, uniforms=uniforms)

    result = evaluator.verify_draft_tokens(**case.kwargs)

    assert result.target_output is case.output
    kwargs = case.target.call_args.kwargs
    assert case.target.call_count == 1
    assert kwargs["input_ids"] is case.proposal.verify_input_ids
    assert kwargs["past_key_values"] is case.kwargs["past_key_values_target"]
    assert kwargs["use_cache"] is kwargs["output_hidden_states"] is True
    torch.testing.assert_close(kwargs["position_ids"], torch.tensor([[13, 14, 15, 16]]))
    torch.testing.assert_close(result.target_probs, case.probabilities)
    assert result.accepted_draft_tokens == accepted
    assert result.effective_proposal_length == effective
    assert result.terminated_by_stop_token is terminated
    # Prefix mask remains full-width; only probability diagnostics are truncated.
    assert result.accept_prefix_mask.tolist() == [prefix]
    assert result.accept_prefix_mask.dtype == torch.int64
    torch.testing.assert_close(
        result.accept_probs, torch.tensor([[1.0, 0.5, 1.0]])[:, :effective]
    )
    torch.testing.assert_close(
        result.support_accept_rates, torch.full((1, effective), 0.75)
    )
    assert [event[0] for event in events] == (
        ["uniform", "residual", "sample", "multinomial"]
        if accepted < 3
        else ["uniform", "sample", "multinomial"]
    )
    assert events[0][1].shape == (1, 3)
    sampled = events[-2][1]
    if accepted < 3:
        torch.testing.assert_close(events[1][1], result.target_probs[:, accepted, :])
        torch.testing.assert_close(events[1][2], original_q[:, accepted, :])
        expected = [1.0, 0.0, 0.0, 0.0] if accepted == 1 else [0.0, 0.0, 0.0, 1.0]
        torch.testing.assert_close(sampled, torch.tensor([expected]))
    else:
        torch.testing.assert_close(sampled, result.target_probs[:, -1:, :])
    assert events[-1][1].shape == (1, 4)
    assert events[-1][2] == 1
    assert result.next_token.shape == (1,)
    torch.testing.assert_close(
        result.committed_tokens,
        torch.cat(
            [original_ids[:, 1 : accepted + 1], result.next_token[:, None]], dim=1
        ),
    )
    torch.testing.assert_close(case.proposal.verify_input_ids, original_ids)
    torch.testing.assert_close(case.proposal.draft_probs, original_q)
    torch.testing.assert_close(case.output.logits, original_logits)


@pytest.mark.parametrize("rejected_index", [0, 2])
def test_first_or_last_rejection_uses_its_own_residual(
    evaluator, monkeypatch, rejected_index
):
    case = _case(evaluator)
    row = torch.tensor([0.5, 0.125, 0.25, 0.125])
    case.output.logits[:, rejected_index, :] = row.log()
    uniforms = [0.1, 0.1, 0.1]
    uniforms[rejected_index] = 0.75
    events = _observe_sampling(monkeypatch, evaluator, uniforms=uniforms)

    result = evaluator.verify_draft_tokens(**case.kwargs)

    assert result.accepted_draft_tokens == rejected_index
    assert result.effective_proposal_length == 3
    assert result.accept_prefix_mask.tolist() == [
        [1] * rejected_index + [0] * (3 - rejected_index)
    ]
    assert result.next_token.tolist() == [0]
    torch.testing.assert_close(events[1][1], result.target_probs[:, rejected_index, :])
    torch.testing.assert_close(events[-2][1], torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("stop_length", [1, 2])
def test_equal_distributions_keep_residual_fallback_after_an_accepted_stop(
    evaluator, monkeypatch, stop_length
):
    case = _case(evaluator)
    case.proposal.draft_probs = evaluator.logits_to_probs(case.output.logits, 1.0)[
        :, :3, :
    ].clone()
    case.kwargs["stop_token_ids"] = [stop_length]
    events = _observe_sampling(monkeypatch, evaluator, uniforms=[0.1, 0.1, 0.1])

    result = evaluator.verify_draft_tokens(**case.kwargs)

    assert result.terminated_by_stop_token
    assert result.accepted_draft_tokens == stop_length
    assert [event[0] for event in events] == [
        "uniform",
        "residual",
        "sample",
        "multinomial",
    ]
    torch.testing.assert_close(events[-2][1], result.target_probs[:, stop_length, :])
    torch.testing.assert_close(result.accept_probs, torch.ones(1, stop_length))
    torch.testing.assert_close(result.support_accept_rates, torch.ones(1, stop_length))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("temperature", [-1.0, 0.0, 0.7, 1.0])
def test_temperature_and_logits_dtype_keep_probability_and_draw_contracts(
    evaluator, monkeypatch, dtype, temperature
):
    case = _case(evaluator, dtype=dtype, temperature=temperature)
    events = _observe_sampling(monkeypatch, evaluator, uniforms=[0.1, 0.9, 0.1])

    result = evaluator.verify_draft_tokens(**case.kwargs)

    expected_dtype = dtype if temperature <= 0 else torch.float32
    assert result.target_probs.dtype == expected_dtype
    assert result.accept_probs.dtype == torch.float32
    assert result.support_accept_rates.dtype == torch.float32
    assert result.accepted_draft_tokens == 1
    assert [event[0] for event in events] == [
        "uniform",
        "residual",
        "sample",
        "multinomial",
    ]
    expected = (
        torch.nn.functional.one_hot(torch.tensor([[1, 0, 3, 3]]), num_classes=4).to(
            dtype
        )
        if temperature <= 0
        else torch.softmax(case.output.logits.float() / temperature, dim=-1)
    )
    torch.testing.assert_close(result.target_probs, expected)


@pytest.mark.parametrize("seed", [0, 7, 41])
@pytest.mark.parametrize("stop_token_ids", [None, [1], [2], [3]])
@pytest.mark.parametrize("count", [0, 3])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_rng_state_matches_one_full_acceptance_draw_then_one_continuation_draw(
    evaluator, seed, stop_token_ids, count, temperature
):
    case = _case(evaluator, count=count, temperature=temperature)
    case.kwargs["stop_token_ids"] = stop_token_ids
    target_probs = evaluator.logits_to_probs(case.output.logits, temperature)
    # Independent oracle for this fixed proposal: target-greedy accepts [1,0,1].
    ratios = torch.tensor([[1.0, 0.0 if temperature == 0.0 else 0.5, 1.0]])
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        accepted = count
        if count:
            draws = torch.rand_like(ratios)[0].tolist()
            accepted = next(
                (i for i, value in enumerate(draws) if value >= ratios[0, i]), count
            )
        if stop_token_ids and stop_token_ids[0] <= accepted:
            accepted = stop_token_ids[0]
        if accepted < count:
            distribution = (target_probs[:, accepted, :] - 0.25).clamp_min(0)
            distribution /= distribution.sum(dim=-1, keepdim=True)
        else:
            distribution = target_probs[:, -1, :]
        expected_token = torch.multinomial(distribution, num_samples=1).squeeze(1)
        expected_state = torch.get_rng_state().clone()

        torch.manual_seed(seed)
        result = evaluator.verify_draft_tokens(**case.kwargs)

        assert result.accepted_draft_tokens == accepted
        torch.testing.assert_close(result.next_token, expected_token)
        assert torch.equal(torch.get_rng_state(), expected_state)
        if count == 0:
            assert result.accept_prefix_mask is None
            assert result.accept_probs is None
            assert result.support_accept_rates is None
            assert result.effective_proposal_length == 0
            assert not result.terminated_by_stop_token
            torch.testing.assert_close(result.committed_tokens, expected_token[:, None])


@pytest.mark.parametrize(
    "invalid", ["oversize", "anchor", "missing", 0.0, -0.1, float("nan"), float("inf")]
)
def test_validation_failure_order_preserves_target_side_effects_without_sampling(
    evaluator, monkeypatch, invalid
):
    case = _case(evaluator)
    if invalid == "oversize":
        case.kwargs["max_proposal_tokens"] = 2
        message = "exceeds max_proposal_tokens"
    elif invalid == "anchor":
        case.kwargs["current_token_ids"] = torch.tensor([[1]])
        message = "must start with current token"
    elif invalid == "missing":
        case.proposal.draft_probs = None
        message = "draft_probs is required"
    else:
        case.proposal.draft_probs[0, 1, 2] = invalid
        message = "finite, positive draft probability"
    events = _observe_sampling(monkeypatch, evaluator)
    state = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match=message):
        evaluator.verify_draft_tokens(**case.kwargs)

    assert case.target.call_count == int(invalid not in ("oversize", "anchor"))
    assert events == []
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_target_failure_propagates_before_acceptance_or_continuation(
    evaluator, monkeypatch, failure_type
):
    case = _case(evaluator)
    failure = failure_type("injected target failure")
    case.target.side_effect = failure
    events = _observe_sampling(monkeypatch, evaluator)
    state = torch.get_rng_state().clone()

    with pytest.raises(failure_type) as caught:
        evaluator.verify_draft_tokens(**case.kwargs)

    assert caught.value is failure
    assert case.target.call_count == 1
    assert events == []
    assert torch.equal(torch.get_rng_state(), state)
