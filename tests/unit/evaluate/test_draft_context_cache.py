# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.
"""Real draft-layer cache/reference parity, including MMuse and reject boundaries."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.models.mmuse import MMuseDraftModel, MMuseSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators_eval.draft_cache import DraftContextCache, validate_draft_cache


@pytest.fixture
def evaluator():
    path = Path(__file__).parents[3] / "scripts/evaluate/dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("draft_cache_eval_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.torch = torch
    yield module
    sys.modules.pop(spec.name, None)


def make_draft(*, enhanced=False, attention="sdpa", window="full", anchor=True):
    torch.manual_seed(71)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        layer_types=["full_attention", "full_attention"]
        if window == "full"
        else ["sliding_attention", "full_attention"],
        sliding_window=None if window == "full" else 5,
    )
    config._attn_implementation = attention
    options = {}
    if enhanced:
        options = {
            "dflash_gated_layer_fusion": True,
            "dflash_context_residual": True,
            "dflash_block_position_embedding": True,
            "dflash2_dynamic_conv": True,
            "dflash2_conv_group_size": 4,
            "dflash2_candidate_selector": True,
            "dflash2_selector_rank": 4,
            "dflash2_selector_top_k": 4,
            "enable_correction_head": True,
            "correction_output_mode": "logits",
            "correction_hidden_size": 16,
            "correction_num_heads": 4,
            "correction_rank": 4,
            "correction_hidden_feedback": True,
            "correction_project_corrected_hidden": True,
            "correction_lm_head_fusion": True,
        }
    algorithm = "mmuse" if enhanced else "dspark"
    config_class = MMuseSpeculatorConfig if enhanced else DSparkSpeculatorConfig
    model_class = MMuseDraftModel if enhanced else DSparkDraftModel
    model = model_class(
        config_class(
            transformer_layer_config=config,
            draft_vocab_size=32,
            block_size=4,
            aux_hidden_state_layer_ids=[0, 1],
            mask_token_id=0,
            markov_rank=0 if enhanced else 4,
            enable_confidence_head=False,
            sample_from_anchor=anchor,
            sliding_window_non_causal=window == "noncausal",
            speculators_config=SpeculatorsConfig(
                algorithm=algorithm,
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=4 if anchor else 3)
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=None, architectures=["Qwen3ForCausalLM"]
                ),
            ),
            **options,
        )
    ).eval()
    # Exercise trained/non-identity extensions, not only their zero-init bypass.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.fill_(1)
            elif "base_kernel" in name:
                parameter.add_(torch.randn_like(parameter) * 0.05)
            else:
                parameter.normal_(std=0.15)
    return model


def make_runner(evaluator, model, *, reuse=True, temperature=0.0):
    return evaluator.DSparkOfflineRunner(
        target_model=torch.nn.Linear(1, 1),
        draft_model=model,
        tokenizer=None,
        args=SimpleNamespace(
            device="cpu", draft_kv_reuse=reuse, temperature=temperature
        ),
    )


@pytest.mark.parametrize("enhanced", [False, True])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("window", ["full", "causal", "noncausal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reused_context_matches_fresh_native_layers(
    evaluator, enhanced, attention, window, dtype
):
    draft = make_draft(enhanced=enhanced, attention=attention, window=window).to(dtype)
    runner = make_runner(evaluator, draft)
    validate_draft_cache(draft)
    cache = DraftContextCache()
    context = torch.randn(1, 25, 32).to(dtype)
    tokens = torch.randint(0, 32, (1, 25))
    # Full accept, first-token rejection and partial accepts only append REAL
    # target states. Cached synthetic query states must never survive a round.
    lengths = [3, 8, 9, 12, 17, 18]
    expected_projection_lengths = []
    previous = 0
    tolerance = 0.04 if dtype == torch.bfloat16 else 2e-5
    for start in lengths:
        projected = []
        hook = draft.fc.register_forward_pre_hook(
            lambda _module, inputs, rows=projected: rows.append(inputs[0].shape[1])
        )
        key_rows = [[] for _ in draft.layers]
        key_hooks = [
            layer.self_attn.k_proj.register_forward_pre_hook(
                lambda _module, inputs, rows=rows: rows.append(inputs[0].shape[1])
            )
            for layer, rows in zip(draft.layers, key_rows, strict=True)
        ]
        with torch.inference_mode():
            cached_hidden, cached_logits = runner._single_anchor_backbone(
                context[:, :start], tokens, start, cache=cache
            )
        hook.remove()
        for key_hook in key_hooks:
            key_hook.remove()
        assert projected == [start + 1 - previous]
        assert all(
            rows == [start + 1 - previous, draft.block_size] for rows in key_rows
        )
        expected_projection_lengths.extend(projected)
        with torch.inference_mode():
            reference_hidden, reference_logits = runner._single_anchor_backbone(
                context[:, :start], tokens, start
            )
            fresh = DraftContextCache()
            runner._single_anchor_backbone(
                context[:, :start], tokens, start, cache=fresh
            )
        torch.testing.assert_close(
            cached_hidden, reference_hidden, atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            cached_logits, reference_logits, atol=tolerance, rtol=tolerance
        )
        assert cache.length == cache.kv.get_seq_length() == start
        assert cache.fused.shape[1] == start
        for index in range(len(draft.layers)):
            assert cache.kv.get_seq_length(index) == start
            torch.testing.assert_close(
                cache.kv.layers[index].keys,
                fresh.kv.layers[index].keys,
                atol=tolerance,
                rtol=tolerance,
            )
            torch.testing.assert_close(
                cache.kv.layers[index].values,
                fresh.kv.layers[index].values,
                atol=tolerance,
                rtol=tolerance,
            )
        previous = start
    assert sum(expected_projection_lengths) == lengths[-1] + len(lengths)
    assert sum(expected_projection_lengths) < sum(length + 1 for length in lengths)


@pytest.mark.parametrize("enhanced", [False, True])
@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_proposals_probabilities_and_rng_match_with_native_heads(
    evaluator, enhanced, anchor, temperature, dtype
):
    draft = make_draft(enhanced=enhanced, anchor=anchor).to(dtype)
    runner = make_runner(evaluator, draft, temperature=temperature)
    states = torch.randn(1, 25, 32).to(dtype)
    tokens = torch.randint(0, 32, (1, 25))
    cache = DraftContextCache()
    for start in [5, 6, 9, 14]:
        context = SimpleNamespace(
            target_hidden_states=states[:, :start],
            correction_previous_logits=torch.randn(1, 32).to(dtype)
            if runner.uses_initial_correction_logits
            else None,
            draft_cache=cache,
        )
        with torch.inference_mode():
            torch.manual_seed(927)
            actual = runner._propose(
                context=context, output_ids=tokens, position_ids=None, start=start
            )
            actual_rng = torch.get_rng_state()
            context.draft_cache = None
            torch.manual_seed(927)
            expected = runner._propose(
                context=context, output_ids=tokens, position_ids=None, start=start
            )
            expected_rng = torch.get_rng_state()
        assert torch.equal(actual.verify_input_ids, expected.verify_input_ids)
        tolerance = 0.002 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(
            actual.draft_probs, expected.draft_probs, atol=tolerance, rtol=tolerance
        )
        assert torch.equal(actual_rng, expected_rng)


@pytest.mark.parametrize("enhanced", [False, True])
@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_complete_decoding_matches_with_real_target_and_fresh_sample_cache(
    evaluator, enhanced, anchor, temperature
):
    draft = make_draft(enhanced=enhanced, anchor=anchor)
    target = Qwen3ForCausalLM(draft.config.transformer_layer_config).eval()
    runner = make_runner(evaluator, draft, temperature=temperature)
    runner.target_model = target
    evaluator.DynamicCache = DynamicCache
    weights = {key: value.clone() for key, value in draft.state_dict().items()}

    def generate(budget):
        with torch.inference_mode():
            return evaluator.generate_decoding_sample(
                target_model=target,
                input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                max_new_tokens=budget,
                max_proposal_tokens=runner.max_proposal_tokens,
                temperature=temperature,
                stop_token_ids=[31],
                init_context=runner._init_context,
                propose=runner._propose,
                update=runner._update,
            )

    for budget in [0, 1, 2, 19]:
        runner.args.draft_kv_reuse = True
        torch.manual_seed(99)
        actual = generate(budget)
        actual_rng = torch.get_rng_state()
        runner.args.draft_kv_reuse = False
        torch.manual_seed(99)
        expected = generate(budget)
        expected_rng = torch.get_rng_state()
        assert torch.equal(actual.output_ids, expected.output_ids)
        assert actual.num_output_tokens == expected.num_output_tokens
        assert actual.proposal_lengths == expected.proposal_lengths
        assert actual.accepted_draft_lengths == expected.accepted_draft_lengths
        for name in ("accept_prob_lists", "support_accept_rate_lists"):
            for row, reference in zip(
                getattr(actual, name), getattr(expected, name), strict=True
            ):
                assert row == pytest.approx(reference, abs=2e-5)
        assert torch.equal(actual_rng, expected_rng)
    assert all(
        torch.equal(value, weights[key]) for key, value in draft.state_dict().items()
    )


def test_sample_cache_is_private_and_disabled_path_allocates_none(evaluator):
    draft = make_draft()
    runner = make_runner(evaluator, draft)
    output = SimpleNamespace(hidden_states=[torch.randn(1, 5, 16)] * 2)
    first = runner._init_context(initial_output=output)
    second = runner._init_context(initial_output=output)
    assert first.draft_cache is not second.draft_cache
    assert first.draft_cache.kv is not second.draft_cache.kv
    runner.args.draft_kv_reuse = False
    assert runner._init_context(initial_output=output).draft_cache is None


@pytest.mark.parametrize("rope_type", ["dynamic", "longrope", "unknown", None])
def test_context_cache_rejects_position_dependent_or_unknown_rope(rope_type):
    draft = make_draft()
    draft.rotary_emb.rope_type = rope_type
    with pytest.raises(ValueError, match="RoPE type"):
        validate_draft_cache(draft)


def test_cache_requires_eval_and_append_only_confirmed_context(evaluator):
    draft = make_draft()
    with pytest.raises(ValueError, match="eval-mode"):
        validate_draft_cache(draft.train())
    draft.eval()
    cache = DraftContextCache()
    with pytest.raises(ValueError, match="inference-only"):
        cache.prepare(draft, torch.randn(1, 5, 32), 4)
    runner = make_runner(evaluator, draft)
    with torch.inference_mode():
        runner._single_anchor_backbone(
            torch.randn(1, 5, 32), torch.ones(1, 12, dtype=torch.long), 5, cache=cache
        )
        with pytest.raises(ValueError, match="append-only"):
            cache.prepare(draft, torch.randn(1, 5, 32), 4)


def test_diagnostics_are_transferred_once_and_preserve_empty_rounds(
    evaluator, monkeypatch
):
    rows = [torch.tensor([0.2, 0.7]), None, torch.tensor([1.0]), torch.tensor([])]
    calls = []
    original = torch.Tensor.tolist

    def track(value):
        calls.append(value.numel())
        return original(value)

    monkeypatch.setattr(torch.Tensor, "tolist", track)
    result = evaluator._probability_rows_to_lists(rows)
    assert result[0] == pytest.approx([0.2, 0.7])
    assert result[1:] == [[], [1.0], []]
    assert calls == [3]
