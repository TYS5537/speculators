"""Training acceptance proxies share DSpark's definitions without changing loss."""

import copy
import weakref

import pytest
import torch
from transformers import LlamaConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.dspark.metrics import compute_metrics as dspark_metrics
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.models.eagle3 import metrics as eagle_metrics
from speculators.models.eagle3.metrics import EagleTrainingMetrics
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.utils import normalize_counted_metrics
from tests.unit.models.test_leagle import batch, make_model


def normalize(counts):
    return normalize_counted_metrics(
        {key: value.item() for key, value in counts.items()}
    )


def collect(logits, targets, docs, mask):
    tracker = EagleTrainingMetrics(docs, mask, len(logits))
    for step, scores in enumerate(logits):
        tracker.update(scores, targets, step)
    return tracker.finish(logits.new_zeros(()))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_matches_dspark_on_the_same_trajectories(dtype):
    torch.manual_seed(37)
    steps, length, vocab = 4, 8, 11
    scores = torch.randn(steps, 1, length, vocab).to(dtype)
    targets = torch.randn(1, length, vocab).to(dtype)
    docs = torch.tensor([[0, 0, 0, 0, 0, 1, 1, -1]])
    mask = torch.tensor([[1, 0, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    actual = normalize(collect(scores, targets, docs, mask))

    # Independently materialize each start as a DSpark block, including tails,
    # a masked gap and a document boundary. Slot 0 is a real draft prediction.
    block_scores = torch.zeros(length, steps, vocab, dtype=dtype)
    block_targets = torch.zeros_like(block_scores)
    block_mask = torch.zeros(length, steps, dtype=torch.bool)
    for start in range(length):
        for step in range(steps):
            target = start + step
            if target >= length:
                continue
            block_scores[start, step] = scores[step, 0, start]
            block_targets[start, step] = targets[0, target]
            block_mask[start, step] = (
                mask[0, start]
                and mask[0, target]
                and docs[0, start] >= 0
                and docs[0, start] == docs[0, target]
            )
    _, reference = dspark_metrics(
        block_scores.reshape(1, -1, vocab),
        block_targets.reshape(1, -1, vocab),
        None,
        block_mask.reshape(1, -1),
        block_size=steps,
        sample_from_anchor=True,
        loss_config=resolve_loss_config("kl_div", "eager"),
    )
    expected = normalize(reference)
    for name, value in actual.items():
        if name != "loss":
            assert value == pytest.approx(expected[name], abs=1e-6), name


def test_prefix_stops_at_first_error_and_marginal_accuracy_stays_independent():
    targets = torch.tensor([[[100.0, 0.0]]]).expand(1, 4, 2).clone()
    scores = targets.unsqueeze(0).expand(3, 1, 4, 2).clone()
    # At start 0: wrong then right; at start 1: right then wrong then right.
    scores[0, 0, 0] = targets[0, 0].flip(0)
    scores[1, 0, 1] = targets[0, 0].flip(0)
    result = normalize(collect(scores, targets, torch.zeros(1, 4), None))
    # Accepted draft counts per start: [0, 1, 2, 1], plus one verifier token.
    assert result["eal"] == 2.0
    assert result["accept_len"] == pytest.approx(2.0)
    assert result["position_1_acc"] == pytest.approx(2 / 3)
    assert result["full_acc"] == pytest.approx(7 / 9)


def test_document_boundaries_padding_and_masked_gaps_truncate_lengths():
    targets = torch.zeros(1, 8, 3)
    scores = targets.unsqueeze(0).expand(4, -1, -1, -1)
    docs = torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1]])
    mask = torch.tensor([[1, 1, 0, 1, 1, 1, 0, 0]])
    raw = collect(scores, targets, docs, mask)
    result = normalize(raw)
    assert raw["eal_total"].item() == 5
    assert [raw[f"position_{i}_acc_total"].item() for i in range(4)] == [5, 3, 1, 0]
    assert result["eal"] == result["accept_len"] == pytest.approx(14 / 5)


@pytest.mark.parametrize("steps", [1, 7])
def test_complete_trajectory_reaches_steps_plus_one(steps):
    targets = torch.zeros(1, steps, 3)
    tracker = EagleTrainingMetrics(torch.zeros(1, steps), None, steps)
    for step in range(steps):
        tracker.update(targets, targets, step)
    # Start 0 has all K slots; shorter starts at the sequence tail are truncated.
    assert tracker.correct_prefix[0, 0]
    assert tracker.counts["eal_sum"].item() == steps + steps * (steps + 1) / 2
    assert tracker.counts["accept_len_sum"] == tracker.counts["eal_sum"]


def test_empty_rank_and_unequal_lengths_reduce_counts_without_mean_of_means():
    steps = 7
    counts = []
    for length, active in ((4, True), (2, True), (3, False)):
        targets = torch.zeros(1, length, 3)
        counts.append(
            collect(
                targets.unsqueeze(0).expand(steps, -1, -1, -1),
                targets,
                torch.zeros(1, length),
                torch.full((1, length), active),
            )
        )
    assert all(set(part) == set(counts[0]) for part in counts)
    assert all(value == 0 for value in normalize(counts[-1]).values())
    reduced = {key: sum(part[key] for part in counts) for key in counts[0]}
    result = normalize(reduced)
    assert result["eal"] == pytest.approx((14 + 5) / 6)
    assert result["accept_len"] == result["eal"]
    assert result["full_acc"] == 1.0
    assert result["position_6_acc"] == 0.0


def test_overlap_chunks_release_probabilities_and_do_not_build_a_graph(monkeypatch):
    logits = torch.randn(1, 11, 19, requires_grad=True)
    targets = torch.randn_like(logits, requires_grad=True)
    expected = torch.minimum(logits.softmax(-1), targets.softmax(-1)).sum(-1)
    softmax = torch.softmax
    references, sizes = [], []

    def track(value, *args, **kwargs):
        result = softmax(value, *args, **kwargs)
        references.append(weakref.ref(result))
        sizes.append(result.numel())
        assert not result.requires_grad
        return result

    monkeypatch.setattr(eagle_metrics, "_OVERLAP_CHUNK_ELEMENTS", 2 * 19)
    monkeypatch.setattr(torch, "softmax", track)
    actual = eagle_metrics._distribution_overlap(logits, targets)
    torch.testing.assert_close(actual, expected)
    assert max(sizes) <= 2 * 19
    assert all(ref() is None for ref in references)
    assert not actual.requires_grad


def make_draft(algorithm, reduced):
    leagle = make_model(layers=1, loops=2, attention="sdpa", reduced=reduced)
    if algorithm == "leagle":
        return leagle
    decoder = LlamaConfig(**leagle.config.transformer_layer_config.to_dict())
    decoder._attn_implementation = "sdpa"
    model = Eagle3DraftModel(
        Eagle3SpeculatorConfig(
            transformer_layer_config=decoder,
            draft_vocab_size=leagle.draft_vocab_size,
            eagle_aux_hidden_state_layer_ids=[0, 1],
            norm_before_residual=True,
            norm_before_fc=True,
            norm_output=True,
            speculators_config=SpeculatorsConfig(
                algorithm="eagle3",
                proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=7)],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=None, architectures=["LlamaForCausalLM"]
                ),
            ),
        )
    )
    model.load_vocab_mappings(leagle.t2d, leagle.d2t)
    with torch.no_grad():
        for name in ("embed_tokens", "lm_head", "verifier_lm_head"):
            getattr(model, name).weight.copy_(getattr(leagle, name).weight)
    return model


@pytest.mark.parametrize("algorithm", ["eagle3", "leagle"])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("loss_fn", ["kl_div", '{"ce": 0.1, "tv": 0.9}'])
def test_model_logging_preserves_loss_gradients_updates_and_rng(
    algorithm, reduced, loss_fn, monkeypatch
):
    torch.manual_seed(19)
    torch.set_num_threads(1)
    model = make_draft(algorithm, reduced)
    inputs = batch()
    inputs["loss_mask"] = torch.tensor([[1, 1, 0, 1, 1, 1]])
    inputs["verifier_last_hidden_states"] = torch.randn(1, 6, 16)
    results = []
    for enabled in (True, False):
        draft = copy.deepcopy(model).train()
        optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-4)
        torch.manual_seed(29)
        with monkeypatch.context() as patch:
            if not enabled:
                patch.setattr(EagleTrainingMetrics, "update", lambda *args: None)
            _, loss, counts = draft(
                **inputs,
                ttt_steps=3,
                loss_config=resolve_loss_config(loss_fn, "eager"),
            )
        loss.backward()
        gradients = [p.grad.clone() for p in draft.parameters() if p.grad is not None]
        optimizer.step()
        results.append((loss, gradients, draft.state_dict(), torch.get_rng_state()))
        expected = {"loss", "full_acc", "accept_len", "eal"} | {
            f"position_{i}_acc" for i in range(3)
        }
        assert set(normalize(counts)) == expected
        assert all(
            value.ndim == 0 and not value.requires_grad for value in counts.values()
        )
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
