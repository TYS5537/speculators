"""Trained candidate checkpoints through sampling, caching and the eval CLI."""

import json
from pathlib import Path

import pytest
import torch

from speculators import SpeculatorModel
from speculators.losses import resolve_loss_config
from speculators_eval import leagle, leagle_decoding, parallel
from speculators_eval.leagle_decoding import check_greedy_output, generate_speculative

from .test_eagle3_offline_eval import make_eagle
from .test_leagle_offline_eval import make_real_models, save_fixture


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_candidate_head_with_cache_matches_rebuild_and_target_greedy(
    temperature, reduced, dtype
):
    target, draft = make_real_models(
        layers=2, loops=3, reduced=reduced, candidate_head=True
    )
    target, draft = target.to(dtype), draft.to(dtype)
    draft.set_candidate_head(True)
    inputs = torch.tensor([[5, 9, 2]])
    responses = []
    for reuse in (False, True):
        torch.manual_seed(53)
        response = generate_speculative(
            target,
            draft,
            inputs,
            max_new_tokens=19,
            temperature=temperature,
            draft_kv_reuse=reuse,
        )
        responses.append(response)
        if temperature == 0:
            check_greedy_output(target, inputs, response, 19, [])
        assert response.num_output_tokens == 19
    torch.testing.assert_close(responses[0].output_ids, responses[1].output_ids)
    assert responses[0].accepted_draft_lengths == responses[1].accepted_draft_lengths


@pytest.mark.parametrize("reduced", [False, True])
def test_real_candidate_q_and_residual_recover_tokens_outside_candidates(
    reduced, monkeypatch
):
    _, draft = make_real_models(reduced=reduced, candidate_head=True)
    draft.set_candidate_head(True)
    _, logits = draft.propose(
        torch.randn(1, 3, 32), torch.tensor([[5]]), num_tokens=1, temperature=1.0
    )
    q = leagle_decoding._draft_probabilities(draft, logits, 1.0, 32)[0, 0]
    assert (q > 0).sum() == 3
    outside = (q == 0).float()
    outside /= outside.sum()
    target_p = (q + outside) / 2
    draws = []

    def record_draw(weights):
        draws.append(weights / weights.sum(-1, keepdim=True))
        return weights.argmax(-1)

    monkeypatch.setattr(leagle_decoding, "sample_from_probs", record_draw)
    monkeypatch.setattr(torch, "rand_like", lambda p: torch.full_like(p, 0.99))
    marginal = torch.zeros(32)
    for token in (q > 0).nonzero().flatten():
        accepted, _, _, probabilities, _ = leagle_decoding._verify_sampled(
            draft,
            token.reshape(1, 1),
            logits,
            target_p.log().reshape(1, 1, -1).expand(1, 2, -1),
            1.0,
            [],
        )
        assert accepted == 0
        alpha = probabilities[0]
        marginal[token] += q[token] * alpha
        marginal += q[token] * (1 - alpha) * draws[-1][0]
    torch.testing.assert_close(marginal, target_p)
    assert torch.all(marginal[q == 0] > 0)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("switch", [None, False, True])
def test_training_checkpoint_and_eval_workers_and_provenance(
    tmp_path, temperature, switch
):
    args = save_fixture(tmp_path)
    _, draft = make_real_models(layers=2, loops=3, candidate_head=True)
    draft.train()
    _, loss, _ = draft(
        hidden_states=torch.randn(1, 4, 32),
        input_ids=torch.tensor([[5, 9, 2, 4]]),
        document_ids=torch.zeros(1, 4, dtype=torch.long),
        verifier_last_hidden_states=torch.randn(1, 4, 16),
        ttt_steps=2,
        loss_config=resolve_loss_config("kl_div", implementation="eager"),
    )
    loss.backward()
    torch.optim.AdamW(draft.parameters(), lr=0.01).step()
    draft.save_pretrained(args.draft_model)
    assert SpeculatorModel.from_pretrained(args.draft_model).candidate_head_enabled
    args.candidate_head = switch
    args.candidate_top_k = 2 if switch is not False else None
    args.temperature, args.check_greedy = temperature, temperature == 0
    entrypoint = Path("scripts/evaluate/leagle_offline_eval.py")
    worker = parallel.worker_command(
        args,
        entrypoint=entrypoint,
        dataset_path=args.datasets_root,
        shard_index=1,
        num_shards=2,
        output_dir=tmp_path / "worker",
    )
    child = leagle.build_parser().parse_args(worker[2:])
    assert child.candidate_head is switch
    assert child.candidate_top_k == args.candidate_top_k
    leagle.run(args, entrypoint=entrypoint)
    metadata = json.loads((args.output_dir / "eval_config.json").read_text())
    assert metadata["candidate_head"]["enabled"] is (switch is not False)
    assert metadata["candidate_head"]["rank"] == 4
    assert metadata["candidate_head"]["top_k"] == args.candidate_top_k
    assert metadata["candidate_head"]["proposal_support"] == (
        "retrieved-candidates" if switch is not False else "draft-vocabulary"
    )
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary[0]["num_requests"] == 2
    assert 1 <= summary[0]["acceptance_length"] <= 8


def test_flags_fail_for_dense_checkpoints_eagle_and_disabled_head(tmp_path):
    args = save_fixture(tmp_path)
    args.candidate_head = True
    with pytest.raises(ValueError, match="Train LEAGLE"):
        leagle.load_models(args)
    _, eagle = make_eagle()
    eagle.save_pretrained(args.draft_model)
    with pytest.raises(ValueError, match="only for LEAGLE"):
        leagle.load_models(args)
    args.candidate_head, args.candidate_top_k = False, 2
    with pytest.raises(ValueError, match="requires --candidate-head"):
        leagle.validate_args(args)
    args.candidate_head, args.candidate_top_k = True, 0
    with pytest.raises(ValueError, match="must be positive"):
        leagle.validate_args(args)


def test_auto_switch_follows_dense_checkpoint_and_rejects_candidate_count(tmp_path):
    args = save_fixture(tmp_path)
    assert args.candidate_head is None
    config = json.loads((Path(args.draft_model) / "config.json").read_text())
    assert leagle.candidate_head_settings(args, config)["enabled"] is False
    args.candidate_top_k = 2
    with pytest.raises(ValueError, match="requires an enabled"):
        leagle.candidate_head_settings(args, config)
