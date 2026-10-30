"""Target buffers must hide rejected KV; selected states keep HF slot numbering."""

import itertools
import json
from pathlib import Path

import pytest
import torch

from speculators_eval import leagle, parallel
from speculators_eval.leagle_decoding import check_greedy_output, generate_speculative
from speculators_eval.leagle_target import VerifiedStaticCache, selected_hidden_target

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


@pytest.mark.parametrize("reject_at", [None, 0, 2])
@pytest.mark.parametrize("budget", [0, 1, 2, 9, 17])
@pytest.mark.parametrize("stop", [None, 3, 7, 10])
def test_static_target_rejection_eos_and_token_budgets(reject_at, budget, stop):
    target, draft = CountingTarget().eval(), ControlledDraft(reject_at).eval()
    draft.static_target_kv = True
    result = generate_speculative(
        target,
        draft,
        torch.tensor([[0, 1]]),
        max_new_tokens=budget,
        stop_token_ids=[] if stop is None else [stop],
    )
    end = budget + 2 if stop is None else min(budget + 2, stop + 1)
    torch.testing.assert_close(result.output_ids, torch.arange(end).unsqueeze(0))


@pytest.mark.parametrize(
    ("static", "selected", "shared"), list(itertools.product([False, True], repeat=3))
)
@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("candidate", [False, True])
def test_real_qwen_target_options_match_baseline_and_rng(
    static,
    selected,
    shared,
    temperature,
    dtype,
    candidate,
):
    target, draft = make_real_models(
        layers=2, loops=2, reduced=candidate, candidate_head=candidate
    )
    target, draft = target.to(dtype), draft.to(dtype)
    inputs, results = torch.tensor([[5, 9, 2]]), []
    for optimized in (False, True):
        draft.set_inference_optimizations(
            static_target_kv=optimized and static,
            selective_target_hidden_states=optimized and selected,
            share_output_head=optimized and shared,
            static_kv=True,
            reuse_rollout_buffers=True,
            sparse_candidates=candidate,
        )
        torch.manual_seed(393)
        result = generate_speculative(
            target, draft, inputs, max_new_tokens=19, temperature=temperature
        )
        results.append((result, torch.get_rng_state()))
        if temperature == 0:
            check_greedy_output(target, inputs, result, 19, [])
    left, right = results[0][0], results[1][0]
    torch.testing.assert_close(left.output_ids, right.output_ids, atol=0, rtol=0)
    assert left.accepted_draft_lengths == right.accepted_draft_lengths
    torch.testing.assert_close(
        left.accept_prob_lists, right.accept_prob_lists, atol=0, rtol=0
    )
    torch.testing.assert_close(results[0][1], results[1][1], atol=0, rtol=0)
    draft.set_inference_optimizations()
    assert draft.lm_head.weight is not draft.verifier_lm_head.weight


@torch.inference_mode()
def test_static_target_storage_rollback_replacement_and_reset():
    target, _ = make_real_models()
    cache = VerifiedStaticCache(target.config, capacity=6)
    initial = torch.randn(1, 2, 5, 4)
    keys, _ = cache.update(initial, initial + 1, 0)
    pointer = keys.data_ptr()
    cache.crop(2)
    replacement = torch.randn(1, 2, 3, 4)
    keys, values = cache.update(replacement, replacement - 1, 0)
    assert keys.data_ptr() == pointer
    assert cache.get_seq_length() == 5
    torch.testing.assert_close(keys[..., :2, :], initial[..., :2, :])
    torch.testing.assert_close(keys[..., 2:, :], replacement)
    torch.testing.assert_close(values[..., 2:, :], replacement - 1)
    cache.crop(-1)
    assert cache.get_seq_length() == 4
    cache.reset()
    assert cache.get_seq_length() == 0
    fresh, _ = cache.update(replacement, replacement, 0)
    assert fresh.data_ptr() == pointer
    torch.testing.assert_close(fresh, replacement)
    with pytest.raises(ValueError, match="capacity"):
        cache.update(initial, initial, 0)


@torch.inference_mode()
def test_selected_hidden_slots_match_hf_and_remove_hooks_on_failure():
    target, _ = make_real_models()
    ids = [0, 1, 3]
    inputs = torch.tensor([[5, 9, 2]])
    expected = target(input_ids=inputs, use_cache=False, output_hidden_states=True)
    counts = [
        (len(layer._forward_pre_hooks), len(layer._forward_hooks))
        for layer in target.modules()
    ]
    with selected_hidden_target(target, ids, enabled=True) as selected:
        for _ in range(2):
            actual = selected(input_ids=inputs, use_cache=False)
            assert list(actual.hidden_states) == ids
            torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
            for slot in ids:
                torch.testing.assert_close(
                    actual.hidden_states[slot],
                    expected.hidden_states[slot],
                    atol=0,
                    rtol=0,
                )
    with (
        pytest.raises(RuntimeError, match="injected"),
        selected_hidden_target(target, ids, enabled=True),
    ):
        raise RuntimeError("injected")
    assert counts == [
        (len(layer._forward_pre_hooks), len(layer._forward_hooks))
        for layer in target.modules()
    ]


def test_target_flags_survive_worker_forwarding_and_default_off(tmp_path):
    flags = [
        "--leagle-static-target-kv",
        "--leagle-selective-target-hidden-states",
        "--leagle-share-output-head",
    ]
    base = [
        "--verifier-model",
        "target",
        "--draft-model",
        "draft",
        "--datasets-root",
        str(tmp_path),
    ]
    default = leagle.build_parser().parse_args(base)
    assert all(not getattr(default, flag[2:].replace("-", "_")) for flag in flags)
    args = leagle.build_parser().parse_args([*base, *flags])
    command = parallel.worker_command(
        args,
        entrypoint=Path("scripts/evaluate/leagle_offline_eval.py"),
        dataset_path=tmp_path,
        shard_index=0,
        num_shards=2,
        output_dir=tmp_path,
    )
    child = leagle.build_parser().parse_args(command[2:])
    assert all(getattr(child, flag[2:].replace("-", "_")) for flag in flags)


def test_target_options_reject_unsupported_backbone():
    target, _ = make_real_models()
    target.config.model_type = "unsupported"
    with pytest.raises(ValueError, match="Qwen3"):
        VerifiedStaticCache(target.config, capacity=8)
    with (
        pytest.raises(ValueError, match="Qwen3"),
        selected_hidden_target(target, [1], enabled=True),
    ):
        pass


def test_disabled_target_options_preserve_input_validation_and_zero_budget():
    target, draft = torch.nn.Identity().eval(), torch.nn.Identity().eval()
    with pytest.raises(ValueError, match="nonempty"):
        generate_speculative(target, draft, torch.empty((1, 0), dtype=torch.long))
    ids = torch.tensor([[1, 2]])
    result = generate_speculative(target, draft, ids, max_new_tokens=0)
    torch.testing.assert_close(result.output_ids, ids)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_checkpoint_eval_records_and_applies_target_options(tmp_path, temperature):
    args = save_fixture(tmp_path)
    args.temperature, args.check_greedy = temperature, temperature == 0
    results = []
    for enabled in (False, True):
        args.leagle_static_target_kv = enabled
        args.leagle_selective_target_hidden_states = enabled
        args.leagle_share_output_head = enabled
        args.output_dir = tmp_path / str(enabled)
        leagle.run(args, entrypoint=Path("scripts/evaluate/leagle_offline_eval.py"))
        metadata = json.loads((args.output_dir / "eval_config.json").read_text())
        for key in (
            "static_target_kv",
            "selective_target_hidden_states",
            "share_output_head",
        ):
            assert metadata["experimental_optimizations"][key] is enabled
        samples = [
            json.loads(line)
            for line in (args.output_dir / "artifacts/toy.jsonl")
            .read_text()
            .splitlines()
        ]
        results.append(
            [
                (sample["output_token_ids"], sample["accepted_draft_lengths"])
                for sample in samples
            ]
        )
    assert results[0] == results[1]
