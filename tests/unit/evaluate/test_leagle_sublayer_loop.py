"""Each sublayer attention visit retains the correct cache across proposal rounds."""

import copy
import json
from pathlib import Path

import pytest
import torch

from speculators_eval import provenance
from speculators_eval.eagle_cache import LEagleCachedProposer
from speculators_eval.leagle_decoding import check_greedy_output, generate_speculative

from .test_leagle_offline_eval import make_real_models
from .test_offline_checkpoint_provenance import fixture_args


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


@pytest.mark.parametrize("mode", ["attn", "ffn"])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("static", "reuse"), [(False, False), (True, False), (True, True)]
)
def test_prefix_and_rollout_cache_matches_rebuilt_context(
    mode, temperature, dtype, static, reuse
):
    _, reference = make_real_models(layers=2, loops=3, loop_mode=mode)
    reference.to(dtype)
    draft = copy.deepcopy(reference)
    draft.set_inference_optimizations(static_kv=static, reuse_rollout_buffers=reuse)
    cached = LEagleCachedProposer(draft, capacity=32)
    hidden = torch.randn(1, 5, 32).to(dtype)
    tokens = torch.tensor([[5, 9, 2, 4, 7, 8]])
    for accepted in (0, 2, 4, 1):
        length = hidden.shape[1]
        positions = torch.arange(length).unsqueeze(0)
        rng = torch.get_rng_state()
        expected_ids, expected_logits = reference.propose(
            hidden,
            tokens[:, -1:],
            position_ids=positions,
            num_tokens=4,
            temperature=temperature,
        )
        torch.set_rng_state(rng)
        actual_ids, actual_logits = cached(
            draft,
            hidden_states=hidden,
            input_ids=tokens,
            position_ids=positions,
            num_tokens=4,
            temperature=temperature,
        )
        torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(actual_logits, expected_logits)
        if reuse:
            assert len(cached.rollout_buffers) == (6 if mode == "attn" else 2)
            for owner, buffer in zip(
                draft.config.attention_layer_indices,
                cached.rollout_buffers,
                strict=True,
            ):
                assert buffer.length == length + 4
                assert buffer.verified_length == length
                torch.testing.assert_close(
                    buffer.key[..., :length, :], cached.context_kv[owner][0]
                )
                torch.testing.assert_close(
                    buffer.value[..., :length, :], cached.context_kv[owner][1]
                )
        tokens = torch.cat(
            [tokens, expected_ids[:, :accepted], torch.tensor([[11]])], -1
        )
        hidden = torch.cat(
            [hidden, (torch.randn(1, accepted + 1, 32) * 3).to(dtype)], 1
        )


@pytest.mark.parametrize("mode", ["attn", "ffn"])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_end_to_end_verification_uses_saved_sublayer_architecture(mode, temperature):
    target, draft = make_real_models(layers=2, loops=3, loop_mode=mode, reduced=True)
    inputs = torch.tensor([[5, 9, 2]])
    torch.manual_seed(53)
    reference = generate_speculative(
        target,
        draft,
        inputs,
        draft_kv_reuse=False,
        max_new_tokens=19,
        temperature=temperature,
    )
    draft.set_inference_optimizations(static_kv=True, reuse_rollout_buffers=True)
    torch.manual_seed(53)
    actual = generate_speculative(
        target, draft, inputs, max_new_tokens=19, temperature=temperature
    )
    torch.testing.assert_close(actual.output_ids, reference.output_ids, rtol=0, atol=0)
    assert actual.accepted_draft_lengths == reference.accepted_draft_lengths
    if temperature == 0:
        check_greedy_output(target, inputs, actual, 19, [])


@pytest.mark.parametrize("mode", ["full", "attn", "ffn"])
def test_provenance_records_attention_and_ffn_depths(mode, tmp_path):
    args = fixture_args(tmp_path)
    path = Path(args.draft_model) / "config.json"
    config = json.loads(path.read_text())
    config["loop_mode"] = mode
    path.write_text(json.dumps(config))
    provenance.prepare_checkpoints(args)
    provenance.write_eval_provenance(args)
    recorded = json.loads((args.output_dir / "eval_config.json").read_text())
    assert recorded["loop_mode"] == mode
    assert recorded["attention_depth"] == (2 if mode == "ffn" else 6)
    assert recorded["ffn_depth"] == (2 if mode == "attn" else 6)
