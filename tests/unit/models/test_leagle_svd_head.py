"""Runtime SVD retrieval against independent SVD and unchanged training weights."""

import copy

import pytest
import torch

from speculators.losses import resolve_loss_config
from speculators.models.leagle import svd_head
from speculators.models.leagle.svd_head import SVDCandidateHead

from .test_leagle import batch, make_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


@pytest.mark.parametrize("shape", [(31, 12), (6, 12)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_chunked_factors_match_independent_truncated_svd(shape, dtype, monkeypatch):
    weight = (torch.randn(*shape) * 0.2).to(dtype)
    original, rng = weight.clone(), torch.get_rng_state()
    monkeypatch.setattr(svd_head, "_VOCAB_CHUNK_SIZE", 4)
    head = SVDCandidateHead(weight, 3)
    u, singular, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    expected = (u[:, :3] * singular[:3]) @ vh[:3]
    actual = head.up_weight.double() @ head.down_weight.double()
    tolerance = 0.004 if dtype == torch.bfloat16 else 1e-7
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(weight, original, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not list(head.parameters())
    assert not head.state_dict()
    assert all(not b.requires_grad for b in head.buffers())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sparse", [False, True])
def test_rescoring_uses_original_hidden_and_dense_rows(dtype, sparse):
    weight = torch.randn(23, 12).to(dtype)
    hidden = torch.randn(2, 3, 12).to(dtype)
    head = SVDCandidateHead(weight, 4)
    candidates = head.coarse_logits(hidden).topk(5, sorted=False).indices
    result = head(hidden, weight, 5, sparse=sparse)
    full = torch.nn.functional.linear(hidden, weight)
    if sparse:
        assert torch.equal(result.indices, candidates)
        selected = result.values
    else:
        assert torch.all(torch.isfinite(result).sum(-1) == 5)
        selected = result.gather(-1, candidates)
    torch.testing.assert_close(
        selected, full.gather(-1, candidates), atol=0.03, rtol=0.02
    )


def test_full_rank_reconstructs_dense_scores_and_zero_weights_are_valid():
    hidden = torch.randn(2, 7)
    for weight in (torch.randn(11, 7), torch.zeros(11, 7)):
        head = SVDCandidateHead(weight, 7)
        torch.testing.assert_close(
            head.coarse_logits(hidden), torch.nn.functional.linear(hidden, weight)
        )
        torch.testing.assert_close(
            head(hidden, weight, 11), torch.nn.functional.linear(hidden, weight)
        )


@pytest.mark.parametrize("rank", [0, -1, 8, 2.5, True])
def test_invalid_rank_rejected(rank):
    with pytest.raises(ValueError, match="SVD rank"):
        SVDCandidateHead(torch.ones(5, 7), rank)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_weights_rejected(value):
    weight = torch.ones(5, 7)
    weight[-1, -1] = value
    with pytest.raises(ValueError, match="finite"):
        SVDCandidateHead(weight, 3)


@pytest.mark.parametrize(
    ("candidate", "rerank"), [(False, True), (True, True), (True, False)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_runtime_head_does_not_change_config_parameters_or_training(
    candidate, rerank, dtype
):
    baseline = make_model(candidate_head=candidate, candidate_rerank=rerank).to(dtype)
    draft = copy.deepcopy(baseline).eval()
    config_before = copy.deepcopy(draft.config.to_dict())
    original_keys = set(draft.state_dict())
    original_parameters = list(draft.parameters())
    rng = torch.get_rng_state()
    draft.set_svd_candidate_head(5, top_k=2)
    assert torch.equal(torch.get_rng_state(), rng)
    assert draft.config.to_dict() == config_before
    assert set(draft.state_dict()) == original_keys
    assert all(
        a is b for a, b in zip(original_parameters, draft.parameters(), strict=True)
    )
    assert draft.candidate_head_enabled
    assert draft.candidate_rerank
    assert draft.candidate_top_k == 2
    inputs = batch()
    inputs["hidden_states"] = inputs["hidden_states"].to(dtype)
    teacher = torch.randn(1, 6, 16).to(dtype)
    losses = []
    for model in (baseline, draft):
        model.train()
        _, loss, _ = model(
            **inputs,
            verifier_last_hidden_states=teacher,
            ttt_steps=3,
            loss_config=resolve_loss_config("kl_div", implementation="eager"),
        )
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(*losses, atol=0, rtol=0)
    for expected, actual in zip(baseline.parameters(), draft.parameters(), strict=True):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(expected.grad, actual.grad, atol=0, rtol=0)


def test_runtime_head_requires_eval_and_can_be_disabled():
    draft = make_model()
    with pytest.raises(ValueError, match="inference only"):
        draft.set_svd_candidate_head(3, top_k=2)
    draft.eval()
    with pytest.raises(ValueError, match="Candidate top_k"):
        draft.set_svd_candidate_head(3, top_k=0)
    draft.set_svd_candidate_head(3, top_k=2)
    draft.set_candidate_head(False)
    hidden = torch.randn(1, 1, 16)
    torch.testing.assert_close(
        draft._output_logits(hidden, False), draft.lm_head(hidden)
    )
    draft.set_candidate_head(True)
    assert draft.candidate_top_k == 2
