"""Cached inference matches full-prefix reconstruction without reusing predictions."""

from contextlib import ExitStack

import pytest
import torch

from speculators_eval.eagle3_decoding import propose_eagle3
from speculators_eval.eagle_cache import cached_proposal
from speculators_eval.leagle_decoding import generate_speculative

from .test_eagle3_offline_eval import make_eagle
from .test_leagle_offline_eval import make_real_models


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def make_models(variant, reduced=False):
    if variant in {"llama", "qwen3"}:
        target, draft = make_eagle(
            layers=2, arch=variant, reduced=reduced, fc_norm=True, norm_output=False
        )
        return target, draft, propose_eagle3
    layers, loops = (1, 1) if variant == "leagle" else (2, 3)
    target, draft = make_real_models(layers=layers, loops=loops, reduced=reduced)
    return target, draft, None


def reference_proposal(draft, propose, kwargs):
    if propose is not None:
        return propose(draft, **kwargs)
    return draft.propose(
        kwargs["hidden_states"],
        kwargs["input_ids"][:, -1:],
        position_ids=kwargs["position_ids"],
        num_tokens=kwargs["num_tokens"],
        temperature=kwargs["temperature"],
    )


@pytest.mark.parametrize("variant", ["llama", "qwen3", "leagle", "looped"])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_cached_logits_match_reference_after_all_rejection_cases(
    variant, reduced, temperature
):
    _, draft, propose = make_models(variant, reduced)
    cached = cached_proposal(draft, propose)
    hidden = torch.randn(1, 5, 32)
    inputs = torch.tensor([[5, 9, 2, 4, 7, 8]])
    # Consume the round AFTER full acceptance too. Accepted synthetic states
    # deliberately differ from the replacement target features below.
    for accepted in (0, 3, 7, 1):
        length = hidden.shape[1]
        kwargs = {
            "hidden_states": hidden,
            "input_ids": inputs,
            "position_ids": torch.arange(length).unsqueeze(0),
            "num_tokens": 7,
            "temperature": temperature,
        }
        rng = torch.get_rng_state()
        expected_ids, expected_logits = reference_proposal(draft, propose, kwargs)
        torch.set_rng_state(rng)
        actual_ids, actual_logits = cached(draft, **kwargs)
        torch.testing.assert_close(actual_ids, expected_ids)
        torch.testing.assert_close(actual_logits, expected_logits, atol=2e-6, rtol=1e-5)
        if propose is not None:
            assert cached.keywords["prefix_cache"].get_seq_length() == length
        else:
            assert len(cached.context_kv) == len(draft.layers)
            assert all(
                k.shape[-2] == v.shape[-2] == length for k, v in cached.context_kv
            )
        inputs = torch.cat(
            [inputs, expected_ids[:, :accepted], torch.tensor([[11]])], dim=1
        )
        hidden = torch.cat([hidden, torch.randn(1, accepted + 1, 32) * 3], dim=1)


@pytest.mark.parametrize("variant", ["llama", "qwen3", "leagle", "looped"])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_generation_cache_preserves_outputs_and_is_request_local(
    variant, temperature, dtype
):
    target, draft, propose = make_models(variant, reduced=True)
    target, draft = target.to(dtype), draft.to(dtype)
    kwargs = {"propose": propose, "temperature": temperature, "max_new_tokens": 19}
    inputs = torch.tensor([[5, 9, 2]])
    torch.manual_seed(53)
    expected = generate_speculative(
        target, draft, inputs, draft_kv_reuse=False, **kwargs
    )
    # An unrelated, longer request must not leave a model/runner-owned cache.
    generate_speculative(target, draft, torch.tensor([[4, 8, 1, 3, 2]]), **kwargs)
    torch.manual_seed(53)
    actual = generate_speculative(target, draft, inputs, **kwargs)
    torch.testing.assert_close(actual.output_ids, expected.output_ids)
    assert actual.proposal_lengths == expected.proposal_lengths
    assert actual.accepted_draft_lengths == expected.accepted_draft_lengths
    for probabilities, reference in zip(
        actual.accept_prob_lists, expected.accept_prob_lists, strict=True
    ):
        torch.testing.assert_close(
            torch.tensor(probabilities), torch.tensor(reference), atol=0.002, rtol=0.02
        )


@pytest.mark.parametrize("variant", ["llama", "qwen3", "looped"])
def test_only_new_confirmed_positions_are_fused_and_projected(variant):
    _, draft, propose = make_models(variant)
    cached = cached_proposal(draft, propose)
    fused_lengths, layer_queries = [], []
    key_lengths = [[] for _ in draft.layers]
    hidden = torch.randn(1, 8, 32)
    tokens = torch.randint(0, 32, (1, 9))
    with ExitStack() as hooks:
        hooks.callback(
            draft.fc.register_forward_pre_hook(
                lambda _m, args: fused_lengths.append(args[0].shape[1])
            ).remove
        )
        for index, layer in enumerate(draft.layers):
            hooks.callback(
                layer.self_attn.k_proj.register_forward_pre_hook(
                    lambda _m, args, index=index: key_lengths[index].append(
                        args[0].shape[1]
                    )
                ).remove
            )
        # The MLP sees exactly the query rows (not LEAGLE's target context).
        hooks.callback(
            draft.layers[0]
            .mlp.register_forward_pre_hook(
                lambda _m, args: layer_queries.append(args[0].shape[1])
            )
            .remove
        )
        for length in (5, 8):
            cached(
                draft,
                hidden_states=hidden[:, :length],
                input_ids=tokens[:, : length + 1],
                position_ids=torch.arange(length).unsqueeze(0),
                num_tokens=4,
            )
    assert fused_lengths == [5, 3]
    if propose is not None:
        assert layer_queries == [5, 1, 1, 1, 3, 1, 1, 1]
    else:
        assert layer_queries == [1] * (2 * 4 * draft.config.num_loops)
        for lengths in key_lengths:
            # One context projection per physical layer, not per shared loop.
            assert lengths == [5, *([1] * 12), 3, *([1] * 12)]


@pytest.mark.parametrize("variant", ["llama", "looped"])
def test_cache_refuses_length_dependent_rope_and_repeated_prefix(variant):
    _, draft, propose = make_models(variant)
    cached = cached_proposal(draft, propose)
    kwargs = {
        "hidden_states": torch.randn(1, 5, 32),
        "input_ids": torch.tensor([[5, 9, 2, 4, 7, 8]]),
        "position_ids": torch.arange(5).unsqueeze(0),
        "num_tokens": 3,
    }
    cached(draft, **kwargs)
    with pytest.raises(ValueError, match="growing verified prefix"):
        cached(draft, **kwargs)
    draft.rotary_emb.rope_type = "dynamic"
    with pytest.raises(ValueError, match="no-draft-kv-reuse"):
        cached_proposal(draft, propose)
