"""Optional compact proposals and static caches retain exact verification rules."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speculators.models.leagle.cache import ReusableDraftKV, StaticKVBuffer
from speculators.models.leagle.sparse_candidates import SparseCandidateLogits
from speculators.proposals.sampling import logits_to_probs
from speculators_eval import leagle, leagle_decoding, parallel
from speculators_eval.leagle_decoding import check_greedy_output, generate_speculative

from .test_leagle_offline_eval import make_real_models, save_fixture


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reuse_buffers", [False, True])
def test_static_kv_matches_dynamic_for_dense_and_compact_proposals(
    temperature, reduced, sparse, dtype, reuse_buffers
):
    target, draft = make_real_models(
        layers=2, loops=2, reduced=reduced, candidate_head=True
    )
    target, draft = target.to(dtype), draft.to(dtype)
    inputs = torch.tensor([[5, 9, 2]])
    responses = []
    for static in (False, True):
        draft.set_inference_optimizations(
            sparse_candidates=sparse,
            static_kv=static,
            reuse_rollout_buffers=static and reuse_buffers,
        )
        torch.manual_seed(53)
        result = generate_speculative(
            target, draft, inputs, max_new_tokens=19, temperature=temperature
        )
        responses.append(result)
        if temperature == 0:
            check_greedy_output(target, inputs, result, 19, [])
    torch.testing.assert_close(responses[0].output_ids, responses[1].output_ids)
    assert responses[0].accepted_draft_lengths == responses[1].accepted_draft_lengths
    torch.testing.assert_close(
        responses[0].accept_prob_lists, responses[1].accept_prob_lists
    )


@pytest.mark.parametrize("reduced", [False, True])
def test_compact_rejection_matches_dense_and_recovers_full_target_distribution(
    reduced, monkeypatch
):
    _, draft = make_real_models(reduced=reduced, candidate_head=True)
    draft.set_inference_optimizations(sparse_candidates=True)
    _, compact = draft.propose(
        torch.randn(1, 3, 32), torch.tensor([[5]]), num_tokens=1, temperature=1
    )
    assert isinstance(compact, SparseCandidateLogits)
    assert compact.values.shape == (1, 1, 3)
    dense = compact.values.new_full((1, 1, 32), -torch.inf).scatter_(
        -1, compact.indices, compact.values
    )
    q = logits_to_probs(dense, 1)[0, 0]
    outside = (q == 0).float()
    outside /= outside.sum()
    p = (q + outside) / 2
    draws = []

    def draw(weights):
        draws.append(weights / weights.sum(-1, keepdim=True))
        return weights.argmax(-1)

    monkeypatch.setattr(leagle_decoding, "sample_from_probs", draw)
    monkeypatch.setattr(torch, "rand_like", lambda p: torch.full_like(p, 0.99))
    marginal = torch.zeros(32)
    for token in compact.indices.flatten():
        results = []
        for scores in (dense, compact):
            results.append(
                leagle_decoding._verify_sampled(
                    SimpleNamespace(use_draft_vocab=False),
                    token.reshape(1, 1),
                    scores,
                    p.log().reshape(1, 1, -1).expand(1, 2, -1),
                    1.0,
                    [],
                )
            )
        torch.testing.assert_close(results[0], results[1])
        torch.testing.assert_close(draws[-2], draws[-1])
        accepted, _, _, probabilities, _ = results[1]
        assert accepted == 0
        marginal[token] += q[token] * probabilities[0]
        marginal += q[token] * (1 - probabilities[0]) * draws[-1][0]
    torch.testing.assert_close(marginal, p)
    assert torch.all(marginal[q == 0] > 0)


def test_sparse_greedy_ties_use_lowest_vocabulary_id():
    scores = SparseCandidateLogits(
        torch.tensor([[[7, 2, 5]]]), torch.tensor([[[3.0, 3.0, 1.0]]]), 10
    )
    assert scores.sample(0).item() == 2


def test_static_buffer_bounds_storage_and_prefix_integrity():
    buffer = StaticKVBuffer(4)
    with pytest.raises(ValueError, match="inference-only"):
        buffer.append(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 4))
    with torch.inference_mode():
        key = torch.randn(1, 2, 3, 4)
        first, _ = buffer.append(key, key + 1)
        pointer = first.data_ptr()
        second, values = buffer.append(key[..., :1, :], key[..., :1, :] + 2)
        assert second.data_ptr() == pointer
        torch.testing.assert_close(second[..., :3, :], key)
        torch.testing.assert_close(values[..., :3, :], key + 1)
        with pytest.raises(ValueError, match="capacity"):
            buffer.append(key, key)


def test_eval_flags_reach_workers_and_provenance(tmp_path):
    args = save_fixture(tmp_path)
    _, draft = make_real_models(candidate_head=True)
    draft.save_pretrained(args.draft_model)
    args.leagle_sparse_candidates = args.leagle_static_kv = True
    args.leagle_reuse_rollout_buffers = True
    args.temperature, args.check_greedy = 1.0, False
    entrypoint = Path("scripts/evaluate/leagle_offline_eval.py")
    command = parallel.worker_command(
        args,
        entrypoint=entrypoint,
        dataset_path=args.datasets_root,
        shard_index=0,
        num_shards=2,
        output_dir=tmp_path / "worker",
    )
    child = leagle.build_parser().parse_args(command[2:])
    assert child.leagle_sparse_candidates
    assert child.leagle_static_kv
    assert child.leagle_reuse_rollout_buffers
    leagle.run(args, entrypoint=entrypoint)
    metadata = json.loads((args.output_dir / "eval_config.json").read_text())
    assert metadata["experimental_optimizations"] == {
        "sparse_candidates": True,
        "static_kv": True,
        "reuse_rollout_buffers": True,
        "static_target_kv": False,
        "selective_target_hidden_states": False,
        "share_output_head": False,
    }


@pytest.mark.parametrize("reuse", [False, True])
def test_static_caches_are_reset_per_request(reuse):
    target, draft = make_real_models(candidate_head=False)
    draft.set_inference_optimizations(static_kv=True)
    results = [
        generate_speculative(
            target,
            draft,
            torch.tensor([[5, 9, 2]]),
            max_new_tokens=13,
            draft_kv_reuse=reuse,
        )
        for _ in range(2)
    ]
    torch.testing.assert_close(results[0].output_ids, results[1].output_ids)
    check_greedy_output(target, torch.tensor([[5, 9, 2]]), results[0], 13, [])


def test_static_full_attention_avoids_masks_but_irregular_positions_fall_back(
    monkeypatch,
):
    _, draft = make_real_models(candidate_head=True)
    hidden, anchor = torch.randn(1, 3, 32), torch.tensor([[5]])
    for positions in (torch.tensor([[0, 1, 2]]), torch.tensor([[0, 8, 2]])):
        draft.set_inference_optimizations()
        reference = draft.propose(hidden, anchor, position_ids=positions, num_tokens=4)
        draft.set_inference_optimizations(static_kv=True)
        calls = []
        original = draft._create_mask

        def record(*args, _calls=calls, _original=original, **kwargs):
            _calls.append(1)
            return _original(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(draft, "_create_mask", record)
            result = draft.propose(hidden, anchor, position_ids=positions, num_tokens=4)
        torch.testing.assert_close(result, reference)
        assert bool(calls) is (positions[0, 1].item() == 8)


def test_inference_flags_reject_missing_head_or_dynamic_rope():
    _, draft = make_real_models()
    with pytest.raises(ValueError, match="static KV"):
        draft.set_inference_optimizations(reuse_rollout_buffers=True)
    with pytest.raises(ValueError, match="enabled candidate"):
        draft.set_inference_optimizations(sparse_candidates=True)
    draft.rotary_emb.rope_type = "dynamic"
    with pytest.raises(ValueError, match="position-independent"):
        draft.set_inference_optimizations(static_kv=True)


def test_reusable_buffers_replace_synthetic_states_and_keep_storage():
    buffer = ReusableDraftKV(24)
    context = torch.randn(1, 2, 18, 4)
    with torch.inference_mode():
        buffer.begin_round(context[..., :3, :], context[..., :3, :] + 1)
        pointer = buffer.key.data_ptr()
        for length in (4, 8, 16):
            synthetic = torch.full((1, 2, 7, 4), 999.0)
            buffer.append_draft(
                context[..., : buffer.verified_length, :],
                context[..., : buffer.verified_length, :] + 1,
                synthetic,
                synthetic,
            )
            previous = buffer.verified_length
            buffer.begin_round(context[..., :length, :], context[..., :length, :] + 1)
            assert buffer.length == buffer.verified_length == length
            assert buffer.key.data_ptr() == pointer
            torch.testing.assert_close(
                buffer.key[..., :length, :], context[..., :length, :]
            )
            torch.testing.assert_close(
                buffer.value[..., previous:length, :],
                context[..., previous:length, :] + 1,
            )
        with pytest.raises(ValueError, match="growing"):
            buffer.begin_round(context[..., :3, :], context[..., :3, :])


def test_rollout_reuse_is_request_local_and_requires_prefix_cache():
    target, draft = make_real_models(layers=2, loops=2)
    draft.set_inference_optimizations(static_kv=True, reuse_rollout_buffers=True)
    inputs = torch.tensor([[5, 9, 2]])
    outputs = [
        generate_speculative(target, draft, inputs, max_new_tokens=19) for _ in range(2)
    ]
    torch.testing.assert_close(vars(outputs[0]), vars(outputs[1]))
    check_greedy_output(target, inputs, outputs[0], 19, [])
    with pytest.raises(ValueError, match="native cached"):
        generate_speculative(target, draft, inputs, draft_kv_reuse=False)


@pytest.mark.parametrize(("static", "reuse_prefix"), [(False, True), (True, False)])
def test_cli_rejects_incompatible_rollout_storage_flags(static, reuse_prefix):
    args = SimpleNamespace(
        leagle_reuse_rollout_buffers=True,
        leagle_static_kv=static,
        no_draft_kv_reuse=not reuse_prefix,
    )
    with pytest.raises(ValueError, match="requires"):
        leagle.optimization_settings(args)
