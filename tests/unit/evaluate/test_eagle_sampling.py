"""Sampled AR proposals, rejection correction, rollback and eval entrypoints."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speculators.models.leagle import core as leagle_core
from speculators.proposals.sampling import logits_to_probs, sample_from_logits
from speculators_eval import eagle3_decoding, leagle, leagle_decoding, parallel
from speculators_eval.leagle_decoding import generate_speculative
from speculators_eval.reporting import EvalStats

from .test_eagle3_offline_eval import make_eagle
from .test_leagle_offline_eval import (
    ControlledDraft,
    CountingTarget,
    make_real_models,
    save_fixture,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def models(algorithm, reduced):
    if algorithm == "eagle3":
        return make_eagle(layers=1, reduced=reduced)
    return make_real_models(layers=1, loops=2, reduced=reduced)


@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_sampling_probabilities_use_fp32_and_greedy_does_not_draw_rng(temperature):
    logits = torch.tensor([[[2, -1, 0], [-2, 1, 1]]], dtype=torch.bfloat16)
    probs = logits_to_probs(logits, temperature)
    assert probs.dtype == torch.float32
    torch.testing.assert_close(probs, (logits.float() / temperature).softmax(-1))
    state = torch.get_rng_state()
    torch.testing.assert_close(sample_from_logits(logits, 0), logits.argmax(-1))
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("algorithm", ["eagle3", "leagle"])
@pytest.mark.parametrize("reduced", [False, True])
def test_proposals_feed_sampled_target_ids_back_and_return_conditional_logits(
    algorithm, reduced, monkeypatch
):
    _, draft = models(algorithm, reduced)
    module = eagle3_decoding if algorithm == "eagle3" else leagle_core
    chosen, distributions, inputs = [], [], []

    def choose_non_greedy(logits, temperature):
        assert temperature == 0.7
        ids = (logits.argmax(-1) + 1) % logits.shape[-1]
        chosen.append(ids)
        distributions.append(logits)
        return ids

    monkeypatch.setattr(module, "sample_from_logits", choose_non_greedy)
    handle = draft.embed_tokens.register_forward_pre_hook(
        lambda _module, args: inputs.append(args[0].clone())
    )
    hidden = torch.randn(1, 3, 32)
    positions = torch.arange(3).unsqueeze(0)
    try:
        if algorithm == "eagle3":
            ids, logits = eagle3_decoding.propose_eagle3(
                draft,
                hidden_states=hidden,
                input_ids=torch.tensor([[5, 9, 2, 4]]),
                position_ids=positions,
                num_tokens=3,
                temperature=0.7,
            )
        else:
            ids, logits = draft.propose(
                hidden,
                torch.tensor([[4]]),
                position_ids=positions,
                num_tokens=3,
                temperature=0.7,
            )
    finally:
        handle.remove()
    expected = torch.cat(chosen, dim=1)
    if reduced:
        expected = expected + draft.d2t[expected]
    torch.testing.assert_close(ids, expected)
    torch.testing.assert_close(logits, torch.cat(distributions, dim=1))
    for step in (1, 2):
        torch.testing.assert_close(inputs[step], ids[:, step - 1 : step])
    expanded = leagle_decoding._draft_probabilities(draft, logits, 0.7, 32)
    expected_probs = torch.zeros(1, 3, 32)
    target_ids = torch.arange(logits.shape[-1])
    if reduced:
        target_ids = target_ids + draft.d2t
    expected_probs[:, :, target_ids] = (logits.float() / 0.7).softmax(-1)
    torch.testing.assert_close(expanded, expected_probs)


@pytest.mark.parametrize("reject", [False, True])
def test_verification_accepts_prefix_then_samples_residual_or_bonus(
    reject, monkeypatch
):
    target_p = torch.tensor(
        [
            [
                [0.125, 0.5, 0.25, 0.125],
                [0.5, 0.25, 0.125, 0.125],
                [0.125, 0.25, 0.125, 0.5],
                [0, 0, 1.0, 0],
            ]
        ]
    )
    uniforms = torch.tensor([[0.0, 0.75 if reject else 0.0, 0.0]])
    monkeypatch.setattr(torch, "rand_like", lambda _: uniforms)
    result = leagle_decoding._verify_sampled(
        SimpleNamespace(use_draft_vocab=False),
        torch.tensor([[1, 2, 3]]),
        torch.full((1, 3, 4), 0.25).log(),
        target_p.log(),
        1.0,
        [],
    )
    accepted, stopped, continuation, probabilities, overlap = result
    assert accepted == (1 if reject else 3)
    assert not stopped
    assert continuation.item() == (0 if reject else 2)
    assert probabilities == pytest.approx([1, 0.5, 1])
    assert overlap == pytest.approx([0.75, 0.75, 0.75])


@pytest.mark.parametrize("reduced", [False, True])
def test_acceptance_plus_residual_recovers_target_distribution(reduced, monkeypatch):
    # Enumerate the possible q proposals; weight accepted tokens and rejected
    # continuations independently to verify the final marginal is exactly p.
    target_p = torch.tensor([0.1, 0.2, 0.4, 0.3])
    q = torch.tensor([0.0, 0.4, 0.0, 0.6])
    draft = SimpleNamespace(use_draft_vocab=reduced, d2t=torch.tensor([1, 2]))
    draft_logits = (q[[1, 3]] if reduced else q).log().reshape(1, 1, -1)
    draws = []

    def record_draw(weights):
        draws.append(weights / weights.sum(-1, keepdim=True))
        return weights.argmax(-1)

    monkeypatch.setattr(leagle_decoding, "sample_from_probs", record_draw)
    monkeypatch.setattr(torch, "rand_like", lambda p: torch.full_like(p, 0.99))
    marginal = torch.zeros(4)
    for token in (1, 3):
        accepted, _, _, probabilities, _ = leagle_decoding._verify_sampled(
            draft,
            torch.tensor([[token]]),
            draft_logits,
            target_p.log().reshape(1, 1, -1).expand(1, 2, -1),
            1.0,
            [],
        )
        assert accepted == 0
        alpha = probabilities[0]
        marginal[token] += q[token] * alpha
        marginal += q[token] * (1 - alpha) * draws[-1][0]
    torch.testing.assert_close(marginal, target_p)
    assert draws[-1][0, 0] > 0  # Outside q's support.
    assert draws[-1][0, 2] > 0


class SampledControlledDraft(ControlledDraft):
    use_draft_vocab = False

    def propose(self, *args, temperature, **kwargs):
        assert temperature == 1.0
        ids, _ = super().propose(*args, **kwargs)
        logits = torch.full((*ids.shape, 64), -torch.inf)
        logits.scatter_(-1, ids.unsqueeze(-1), 0.0)
        return ids, logits  # A legitimate, deterministic proposal distribution.


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 9, 17])
@pytest.mark.parametrize("reject_at", [None, 0, 2])
@pytest.mark.parametrize("stop", [None, 2, 7, 63])
def test_sampled_rounds_rollback_eos_short_tails_and_statistics(
    budget, reject_at, stop
):
    response = generate_speculative(
        CountingTarget().eval(),
        SampledControlledDraft(reject_at, rejected_id=63).eval(),
        torch.tensor([[0, 1]]),
        temperature=1.0,
        max_new_tokens=budget,
        stop_token_ids=[] if stop is None else [stop],
    )
    end = min(2 + budget, stop + 1) if stop is not None else 2 + budget
    torch.testing.assert_close(response.output_ids, torch.arange(end).unsqueeze(0))
    for count, accepted, probs, overlaps in zip(
        response.proposal_lengths,
        response.accepted_draft_lengths,
        response.accept_prob_lists,
        response.support_accept_rate_lists,
        strict=True,
    ):
        assert 0 <= accepted <= count <= 7
        assert len(probs) == len(overlaps) == count
    stats = EvalStats()
    stats.add_response(response)
    assert 1 <= stats.acceptance_length <= 8


@pytest.mark.parametrize("algorithm", ["eagle3", "leagle"])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_sampled_decoding_is_seed_reproducible(algorithm, reduced, dtype):
    target, draft = models(algorithm, reduced)
    target, draft = target.to(dtype), draft.to(dtype)
    kwargs = (
        {"propose": eagle3_decoding.propose_eagle3} if algorithm == "eagle3" else {}
    )
    outputs = []
    for _ in range(2):
        torch.manual_seed(53)
        outputs.append(
            generate_speculative(
                target,
                draft,
                torch.tensor([[5, 9, 2]]),
                temperature=1.0,
                max_new_tokens=19,
                **kwargs,
            )
        )
    torch.testing.assert_close(outputs[0].output_ids, outputs[1].output_ids)
    assert outputs[0].accepted_draft_lengths == outputs[1].accepted_draft_lengths
    assert outputs[0].accept_prob_lists == outputs[1].accept_prob_lists
    assert outputs[0].num_output_tokens == 19
    stats = EvalStats()
    stats.add_response(outputs[0])
    assert all(0 <= p <= 1 for p in stats.position_accept_prob_means)


@pytest.mark.parametrize("algorithm", ["eagle3", "leagle"])
@pytest.mark.parametrize("reuse", [False, True])
def test_temperature_one_checkpoint_pipeline_workers_and_provenance(
    tmp_path, algorithm, reuse
):
    args = save_fixture(tmp_path)
    args.no_draft_kv_reuse = not reuse
    if algorithm == "eagle3":
        target, draft = make_eagle(layers=1)
        target.save_pretrained(args.verifier_model)
        draft.save_pretrained(args.draft_model)
    args.temperature, args.check_greedy = 1.0, False
    entrypoint = Path(f"scripts/evaluate/{algorithm}_offline_eval.py")
    leagle.run(args, entrypoint=entrypoint)
    metadata = json.loads((args.output_dir / "eval_config.json").read_text())
    assert metadata["arguments"]["temperature"] == 1.0
    assert metadata["proposal"] == "single-chain-sampling"
    assert metadata["probability_diagnostics_available"] is True
    assert metadata["draft_kv_reuse_between_rounds"] is reuse
    assert metadata["draft_cache_policy"] == (
        "verified-prefix-only" if reuse else "rebuild-each-round"
    )
    rows = json.loads((args.output_dir / "summary.json").read_text())
    assert rows[0]["probability_diagnostics_available"] is True
    assert rows[0]["num_requests"] == 2
    records = [
        json.loads(line)
        for line in (args.output_dir / "artifacts/toy.jsonl").read_text().splitlines()
    ]
    assert all(
        len(r["accept_prob_lists"]) == len(r["proposal_lengths"]) for r in records
    )
    command = parallel.worker_command(
        args,
        entrypoint=entrypoint,
        dataset_path=args.datasets_root,
        shard_index=1,
        num_shards=2,
        output_dir=tmp_path / "worker",
    )
    child_args = leagle.build_parser().parse_args(command[2:])
    assert child_args.temperature == 1.0
    assert not child_args.check_greedy
    assert child_args.no_draft_kv_reuse is not reuse


@pytest.mark.parametrize(
    "temperature", [-1.0, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_temperature_rejected_before_model_loading(tmp_path, temperature):
    args = save_fixture(tmp_path)
    args.temperature = temperature
    with pytest.raises(ValueError, match="finite and >= 0"):
        leagle.validate_args(args)
    with pytest.raises(ValueError, match="finite and >= 0"):
        sample_from_logits(torch.zeros(1, 3), temperature)
