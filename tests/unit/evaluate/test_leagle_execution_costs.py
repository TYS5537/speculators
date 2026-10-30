"""Deferred diagnostics and score-free greedy proposals retain decoding results."""

from types import SimpleNamespace

import pytest
import torch

from speculators_eval import leagle_decoding as decoding

from .test_leagle_offline_eval import make_real_models


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(418)
    torch.set_num_threads(1)


@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("static", [False, True])
def test_greedy_can_discard_scores_without_changing_tokens(candidate, reduced, static):
    _, draft = make_real_models(reduced=reduced, candidate_head=candidate)
    draft.set_inference_optimizations(sparse_candidates=candidate, static_kv=static)
    states, anchor = torch.randn(1, 3, 32), torch.tensor([[5]])
    before = torch.get_rng_state()
    expected, scores = draft.propose(states, anchor, num_tokens=7)
    actual, discarded = draft.propose(states, anchor, num_tokens=7, return_logits=False)
    assert scores is not None
    assert discarded is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="retain logits"):
        draft.propose(states, anchor, temperature=1, return_logits=False)


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("candidate", [False, True])
def test_generation_matches_eager_diagnostics_and_retained_scores(
    reuse, temperature, candidate, monkeypatch
):
    target, draft = make_real_models(candidate_head=candidate)
    draft.set_inference_optimizations(sparse_candidates=candidate)
    inputs = torch.tensor([[5, 9, 2]])
    arguments = {
        "max_new_tokens": 25,
        "spec_tokens": 7,
        "temperature": temperature,
        "stop_token_ids": [],
        "draft_kv_reuse": reuse,
    }
    original_propose, original_verify = draft.propose, decoding._verify_sampled
    score_requests = []

    def record(*args, **kwargs):
        score_requests.append(kwargs.get("return_logits", True))
        return original_propose(*args, **kwargs)

    monkeypatch.setattr(draft, "propose", record)
    torch.manual_seed(57)
    actual = decoding.generate_speculative(target, draft, inputs, **arguments)
    actual_rng = torch.get_rng_state()

    def retain_scores(*args, **kwargs):
        kwargs["return_logits"] = True
        return original_propose(*args, **kwargs)

    def eager_stats(*args, **kwargs):
        kwargs["defer_statistics"] = False
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(draft, "propose", retain_scores)
    monkeypatch.setattr(decoding, "_verify_sampled", eager_stats)
    torch.manual_seed(57)
    expected = decoding.generate_speculative(target, draft, inputs, **arguments)
    torch.testing.assert_close(vars(actual), vars(expected), rtol=0, atol=0)
    torch.testing.assert_close(actual_rng, torch.get_rng_state(), rtol=0, atol=0)
    assert score_requests
    assert all(request == (temperature > 0) for request in score_requests)


@pytest.mark.parametrize("stop_ids", [[], [0], [0, 1]])
def test_diagnostics_transfer_once_and_keep_empty_rounds(stop_ids, monkeypatch):
    original = torch.Tensor.tolist
    transfers = []

    def record(tensor):
        transfers.append(tensor.shape)
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", record)
    probs, support = [], []
    for count in (3, 0):
        result = decoding._verify_sampled(
            SimpleNamespace(use_draft_vocab=False),
            torch.zeros(1, count, dtype=torch.long),
            torch.zeros(1, count, 4),
            torch.zeros(1, count + 1, 4),
            1,
            stop_ids,
            defer_statistics=True,
        )
        probs.append(result[3])
        support.append(result[4])
    assert transfers == []
    actual = decoding._statistics_to_host(probs, support)
    assert len(transfers) == 1
    assert actual[0][-1] == actual[1][-1] == []
    assert actual[0][0] == [1.0] * (1 if stop_ids else 3)
    assert actual[1][0] == actual[0][0]


def test_stop_ids_are_reused_and_work_with_multiple_eos():
    tokens = torch.tensor([[5, 7, 3]])
    stops = decoding._stop_tensor(tokens, [3, 7])
    assert decoding._stop_tensor(tokens, stops) is stops
    assert decoding._contains_stop(tokens, stops)
    assert decoding._truncate_accepted_stop(tokens, 3, stops) == (2, True)
    assert decoding._truncate_accepted_stop(tokens, 1, stops) == (1, False)
