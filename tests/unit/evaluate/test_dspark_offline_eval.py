# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_module():
    path = Path(__file__).parents[3] / "scripts" / "evaluate" / "dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("dspark_offline_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.torch = torch
    return module


class Tokenizer:
    @staticmethod
    def apply_chat_template(messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is True
        return json.dumps({"messages": messages, "kwargs": kwargs})


def test_prompt_from_turns_wraps_plain_text_by_default():
    module = _load_module()
    args = SimpleNamespace(enable_thinking="false", raw_prompt_mode="auto")

    prompt = module._prompt_from_record(
        {"turns": ["Solve this."]},
        Tokenizer(),
        source="sample.jsonl:1",
        args=args,
    )

    parsed = json.loads(prompt)
    assert parsed["messages"] == [{"role": "user", "content": "Solve this."}]
    assert parsed["kwargs"] == {"enable_thinking": False}


def test_prompt_from_turns_uses_first_turn_like_deepspec():
    module = _load_module()
    args = SimpleNamespace(enable_thinking="false", raw_prompt_mode="auto")

    prompt = module._prompt_from_record(
        {"turns": ["First user turn.", "Second user turn."]},
        Tokenizer(),
        source="mt_bench.jsonl:1",
        args=args,
    )

    parsed = json.loads(prompt)
    assert parsed["messages"] == [{"role": "user", "content": "First user turn."}]
    assert "Second user turn." not in prompt


def test_prompt_from_chatml_turns_stays_raw_in_auto_mode():
    module = _load_module()
    text = "<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n"
    args = SimpleNamespace(enable_thinking="false", raw_prompt_mode="auto")

    prompt = module._prompt_from_record(
        {"prompt": text},
        Tokenizer(),
        source="sample.jsonl:1",
        args=args,
    )

    assert prompt == text


def test_prompt_from_sharegpt_stops_before_answer():
    module = _load_module()
    args = SimpleNamespace(enable_thinking="default", raw_prompt_mode="auto")

    prompt = module._prompt_from_record(
        {
            "conversations": [
                {"from": "human", "value": "Question?"},
                {"from": "gpt", "value": "Answer."},
            ],
        },
        Tokenizer(),
        source="sample.jsonl:1",
        args=args,
    )

    parsed = json.loads(prompt)
    assert parsed["messages"] == [{"role": "user", "content": "Question?"}]
    assert "Answer." not in prompt
    assert parsed["kwargs"] == {}


def test_discover_datasets_filters_by_stem(tmp_path: Path):
    module = _load_module()
    keep = tmp_path / "aime24.jsonl"
    drop = tmp_path / "humaneval.jsonl"
    keep.write_text(json.dumps({"prompt": "a"}) + "\n", encoding="utf-8")
    drop.write_text(json.dumps({"prompt": "b"}) + "\n", encoding="utf-8")

    paths = module._discover_datasets(tmp_path, ["aime24"])

    assert paths == [keep]


def test_deepspec_dataset_sample_limits_match_eval_py():
    module = _load_module()

    assert module.DEEPSPEC_EVAL_SAMPLE_LIMITS == {
        "gsm8k": 500,
        "math500": 500,
        "aime25": 30,
        "humaneval": 164,
        "mbpp": 256,
        "livecodebench": 500,
        "mt-bench": 80,
        "alpaca": 500,
        "arena-hard-v2": 500,
    }


def test_deepspec_sample_selection_is_seeded_before_truncation():
    module = _load_module()
    records = [{"id": idx} for idx in range(600)]

    selected = module._select_eval_records(
        records,
        dataset_name="gsm8k",
        max_samples=None,
        seed=980406,
    )
    repeated = module._select_eval_records(
        records,
        dataset_name="gsm8k",
        max_samples=None,
        seed=980406,
    )

    assert len(selected) == 500
    assert selected == repeated
    assert selected != records[:500]


def test_explicit_max_samples_overrides_deepspec_limit():
    module = _load_module()
    records = [{"id": idx} for idx in range(600)]

    selected = module._select_eval_records(
        records,
        dataset_name="gsm8k",
        max_samples=17,
        seed=980406,
    )

    assert len(selected) == 17


def test_sample_from_anchor_slot_target_positions():
    module = _load_module()
    draft = SimpleNamespace(
        block_size=4,
        config=SimpleNamespace(sample_from_anchor=True),
    )

    assert module.first_draft_slot_for_draft(draft) == 0
    assert module.speculative_slots_for_draft(draft) == 4
    assert [module.target_position_for_slot(draft, 10, slot) for slot in range(4)] == [
        11,
        12,
        13,
        14,
    ]


def test_no_sample_from_anchor_slot_target_positions():
    module = _load_module()
    draft = SimpleNamespace(
        block_size=4,
        config=SimpleNamespace(sample_from_anchor=False),
    )

    assert module.first_draft_slot_for_draft(draft) == 1
    assert module.speculative_slots_for_draft(draft) == 3
    assert [module.target_position_for_slot(draft, 10, slot) for slot in range(4)] == [
        10,
        11,
        12,
        13,
    ]


def test_no_sample_from_anchor_rejects_zero_proposal_block():
    module = _load_module()
    draft = SimpleNamespace(
        block_size=1,
        config=SimpleNamespace(sample_from_anchor=False),
    )

    with pytest.raises(ValueError, match="block_size >= 2"):
        module.speculative_slots_for_draft(draft)


def test_stop_token_truncates_probability_stats_with_effective_proposal():
    module = _load_module()

    class TargetModel:
        @staticmethod
        def __call__(**_kwargs):
            logits = torch.full((1, 4, 5), -10.0)
            logits[0, 0, 1] = 10.0
            logits[0, 1, 2] = 10.0
            logits[0, 2, 3] = 10.0
            logits[0, 3, 4] = 10.0
            return SimpleNamespace(logits=logits)

    draft_ids = torch.tensor([[1, 2, 3]])
    proposal = module.DraftProposal(
        draft_token_count=3,
        verify_input_ids=torch.tensor([[0, 1, 2, 3]]),
        draft_probs=torch.nn.functional.one_hot(draft_ids, num_classes=5).float(),
    )

    result = module.verify_draft_tokens(
        target_model=TargetModel(),
        proposal=proposal,
        position_ids=torch.arange(4).unsqueeze(0),
        start=0,
        past_key_values_target=None,
        temperature=0.0,
        max_proposal_tokens=3,
        current_token_ids=torch.tensor([[0]]),
        stop_token_ids=[2],
    )

    assert result.terminated_by_stop_token
    assert result.accepted_draft_tokens == 2
    assert result.effective_proposal_length == 2
    assert result.accept_probs.shape == (1, 2)
    assert result.support_accept_rates.shape == (1, 2)


def _run_budgeted_decoding(
    *, max_new_tokens, sample_from_anchor=True, drafts=None, stop_token_ids=None
):
    """Exercise the real verifier with deterministic CPU tensors and a KV cache."""
    module = _load_module()
    events = SimpleNamespace(target_inputs=[], starts=[], updates=[], initializations=0)
    vocab_size = 16

    class Cache:
        length = 0

        def __init__(self):
            self.crops = []

        def crop(self, length):
            assert 0 <= length <= self.length
            self.length = length
            self.crops.append(length)

    cache = Cache()

    class Target:
        @staticmethod
        def new_cache():
            return cache

        @staticmethod
        def __call__(*, input_ids, position_ids, past_key_values, **_kwargs):
            assert past_key_values is cache
            assert position_ids.shape == input_ids.shape
            assert int(position_ids[0, 0]) == cache.length
            events.target_inputs.append(input_ids[0].tolist())
            cache.length += input_ids.shape[1]
            # Position p deterministically predicts token p+1, including on reject.
            next_ids = position_ids + 1
            logits = torch.full((*input_ids.shape, vocab_size), -torch.inf)
            logits.scatter_(-1, next_ids.unsqueeze(-1), 0.0)
            return SimpleNamespace(logits=logits)

    draft = SimpleNamespace(
        block_size=4,
        config=SimpleNamespace(sample_from_anchor=sample_from_anchor),
    )
    first_slot = module.first_draft_slot_for_draft(draft)
    max_proposal_tokens = module.speculative_slots_for_draft(draft)

    def init_context(**_kwargs):
        events.initializations += 1
        return events

    def propose(*, context, output_ids, start, **_kwargs):
        assert context is events
        proposal_index = len(events.starts)
        events.starts.append(start)
        slots = torch.tensor(
            [
                [
                    module.target_position_for_slot(draft, start, slot)
                    for slot in range(draft.block_size)
                ]
            ]
        )
        proposed = slots[:, first_slot:]
        if drafts is not None and proposal_index < len(drafts):
            proposed = torch.tensor([drafts[proposal_index]])
        assert proposed.shape[1] == max_proposal_tokens
        return module.DraftProposal(
            draft_token_count=max_proposal_tokens,
            verify_input_ids=torch.cat([output_ids[:, start : start + 1], proposed], 1),
            draft_probs=torch.nn.functional.one_hot(proposed, vocab_size).float(),
        )

    def update(context, verification):
        assert context is events
        events.updates.append(verification.accepted_draft_tokens)

    response = module.generate_decoding_sample(
        target_model=Target(),
        input_ids=torch.tensor([[0, 1]]),
        max_new_tokens=max_new_tokens,
        max_proposal_tokens=max_proposal_tokens,
        temperature=0.0,
        stop_token_ids=stop_token_ids,
        init_context=init_context,
        propose=propose,
        update=update,
    )
    assert response.num_output_tokens <= max(max_new_tokens, 0)
    assert len(response.proposal_lengths) == len(response.accepted_draft_lengths)
    for length, accepted, accept_probs, support_rates in zip(
        response.proposal_lengths,
        response.accepted_draft_lengths,
        response.accept_prob_lists,
        response.support_accept_rate_lists,
        strict=True,
    ):
        assert 0 <= accepted <= length
        assert len(accept_probs) == len(support_rates) == length
    return response, events, cache


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_nonpositive_generation_budget_skips_target_and_drafter(max_new_tokens):
    response, events, _ = _run_budgeted_decoding(max_new_tokens=max_new_tokens)

    assert response.output_ids.tolist() == [[0, 1]]
    assert response.num_output_tokens == 0
    assert response.proposal_lengths == response.accepted_draft_lengths == []
    assert response.accept_prob_lists == response.support_accept_rate_lists == []
    assert events.target_inputs == events.starts == events.updates == []
    assert events.initializations == 0


def test_single_token_generation_budget_only_prefills():
    response, events, _ = _run_budgeted_decoding(max_new_tokens=1)

    assert response.output_ids.tolist() == [[0, 1, 2]]
    assert events.target_inputs == [[0, 1]]
    assert events.initializations == 0
    assert events.starts == events.updates == response.proposal_lengths == []


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_runner_nonpositive_budget_skips_target_budget_validation(max_new_tokens):
    module = _load_module()
    runner = module.DSparkOfflineRunner.__new__(module.DSparkOfflineRunner)
    runner.args = SimpleNamespace(max_new_tokens=max_new_tokens, temperature=0.0)
    runner.device = torch.device("cpu")
    runner.max_proposal_tokens = 3
    runner.tokenizer = lambda *_args, **_kwargs: SimpleNamespace(
        input_ids=torch.tensor([[0, 1]])
    )

    def unexpected_validation(*_args, **_kwargs):
        pytest.fail("An empty generation must not request a target-side budget")

    runner.target_model = SimpleNamespace(validate_request_budget=unexpected_validation)

    response = runner.generate_one("prompt", stop_token_ids=None)

    assert response.output_ids.tolist() == [[0, 1]]
    assert response.num_output_tokens == 0
    assert response.proposal_lengths == response.accepted_draft_lengths == []


def test_one_remaining_token_skips_drafter_and_records_zero_draft_round():
    response, events, cache = _run_budgeted_decoding(max_new_tokens=2)

    assert response.output_ids.tolist() == [[0, 1, 2, 3]]
    assert events.target_inputs == [[0, 1], [2]]
    assert events.starts == events.updates == []
    assert response.proposal_lengths == response.accepted_draft_lengths == [0]
    assert response.accept_prob_lists == response.support_accept_rate_lists == [[]]
    assert cache.crops == [3]


@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_short_tail_clips_verification_and_statistics_before_target(sample_from_anchor):
    response, events, cache = _run_budgeted_decoding(
        max_new_tokens=4, sample_from_anchor=sample_from_anchor
    )

    assert response.output_ids.tolist() == [[0, 1, 2, 3, 4, 5]]
    assert events.target_inputs == [[0, 1], [2, 3, 4]]
    assert events.starts == [2]
    assert events.updates == []
    assert response.proposal_lengths == response.accepted_draft_lengths == [2]
    assert response.accept_prob_lists == response.support_accept_rate_lists == [[1, 1]]
    assert cache.crops == [5]


def test_full_round_fills_budget_without_an_extra_round_or_context_update():
    response, events, cache = _run_budgeted_decoding(
        max_new_tokens=5, sample_from_anchor=False
    )

    assert response.output_ids.tolist() == [list(range(7))]
    assert events.target_inputs == [[0, 1], [2, 3, 4, 5]]
    assert events.starts == [2]
    assert events.updates == []
    assert response.proposal_lengths == response.accepted_draft_lengths == [3]
    assert cache.crops == [6]


def test_rejection_updates_context_then_clips_the_next_round_to_budget():
    response, events, cache = _run_budgeted_decoding(
        max_new_tokens=6,
        sample_from_anchor=False,
        drafts=[[3, 9, 5], [5, 6, 7]],
    )

    assert response.output_ids.tolist() == [list(range(8))]
    assert events.target_inputs == [[0, 1], [2, 3, 9, 5], [4, 5, 6]]
    assert events.starts == [2, 4]
    assert events.updates == [1]
    assert cache.crops == [4, 7]
    assert response.proposal_lengths == [3, 2]
    assert response.accepted_draft_lengths == [1, 2]
    assert response.accept_prob_lists == [[1, 0, 1], [1, 1]]
    assert response.support_accept_rate_lists == [[1, 0, 1], [1, 1]]


@pytest.mark.parametrize(
    ("max_new_tokens", "eos_id", "expected_ids", "verify_ids", "proposal_length"),
    [
        (5, 4, [0, 1, 2, 3, 4], [2, 3, 4, 5], 2),
        (4, 5, [0, 1, 2, 3, 4, 5], [2, 3, 4], 2),
        (4, 6, [0, 1, 2, 3, 4, 5], [2, 3, 4], 2),
    ],
    ids=["accepted-eos-inside-tail", "bonus-eos", "eos-outside-budget"],
)
def test_eos_and_token_budget_keep_committed_tokens_and_metrics_aligned(
    max_new_tokens, eos_id, expected_ids, verify_ids, proposal_length
):
    response, events, _ = _run_budgeted_decoding(
        max_new_tokens=max_new_tokens, stop_token_ids=[eos_id]
    )

    assert response.output_ids.tolist() == [expected_ids]
    assert events.target_inputs == [[0, 1], verify_ids]
    assert events.updates == []
    assert response.proposal_lengths == [proposal_length]
    assert response.accepted_draft_lengths == [proposal_length]
    assert response.accept_prob_lists == [[1] * proposal_length]
    assert response.support_accept_rate_lists == [[1] * proposal_length]


def test_detects_preprojection_correction():
    module = _load_module()

    draft = SimpleNamespace(
        correction_head=SimpleNamespace(position_embedding=object()),
    )

    assert module._is_preprojection_correction(draft)
    assert not module._is_preprojection_correction(
        SimpleNamespace(correction_head=None)
    )


def test_preprojection_rollout_receives_hidden_states_without_base_logits():
    module = _load_module()
    calls = []

    class Draft:
        correction_head = SimpleNamespace(position_embedding=object())

        @staticmethod
        def rollout_correction(*args, **kwargs):
            calls.append((args, kwargs))
            return "tokens", "logits"

    hidden_states = object()
    anchor_token_ids = object()
    result = module._run_preprojection_correction_rollout(
        Draft(),
        hidden_states=hidden_states,
        anchor_token_ids=anchor_token_ids,
        temperature=0.7,
    )

    assert result == ("tokens", "logits")
    assert calls == [
        (
            (hidden_states,),
            {
                "anchor_token_ids": anchor_token_ids,
                "temperature": 0.7,
            },
        ),
    ]


def test_preprojection_rollout_builds_one_base_block_for_lm_head_fusion():
    module = _load_module()
    calls = []

    class LMHead:
        def __init__(self):
            self.weight = module.torch.zeros(8, 4)
            self.calls = 0

        def __call__(self, hidden):
            self.calls += 1
            return hidden.new_zeros(*hidden.shape[:-1], 8)

    class Draft:
        correction_head = SimpleNamespace(position_embedding=object())
        candidate_selector = None
        config = SimpleNamespace(correction_lm_head_fusion=True)
        lm_head = LMHead()

        @staticmethod
        def rollout_correction(*args, **kwargs):
            calls.append((args, kwargs))
            return "tokens", "logits"

    hidden_states = module.torch.zeros(1, 3, 4)
    result = module._run_preprojection_correction_rollout(
        Draft(),
        hidden_states=hidden_states,
        anchor_token_ids=module.torch.tensor([1]),
        temperature=0.0,
    )

    assert result == ("tokens", "logits")
    assert Draft.lm_head.calls == 1
    assert calls[0][1]["base_logits"].shape == (1, 3, 8)


def test_correction_selector_positive_temperature_returns_sparse_q():
    module = _load_module()

    class Draft:
        correction_head = SimpleNamespace(position_embedding=object())
        use_draft_vocab = False
        d2t = None

        @staticmethod
        def rollout_correction(*args, **kwargs):
            del args, kwargs
            tokens = module.torch.tensor([[2]])
            logits = module.torch.tensor(
                [[[-module.torch.inf, 0.0, math.log(3.0), -module.torch.inf]]]
            )
            return tokens, logits

    runner = module.DSparkOfflineRunner.__new__(module.DSparkOfflineRunner)
    runner.draft_model = Draft()
    runner.args = SimpleNamespace(temperature=1.0)
    runner.first_draft_slot = 0
    runner.max_proposal_tokens = 1

    proposed, probabilities = runner._sample_correction_tokens(
        None,
        module.torch.zeros(1, 1, 3),
        module.torch.tensor([5]),
        None,
    )

    assert proposed == [2]
    assert probabilities[0, 0].tolist() == pytest.approx([0.0, 0.25, 0.75, 0.0])


def test_target_logits_are_selected_in_draft_vocab_order():
    module = _load_module()
    runner = module.DSparkOfflineRunner.__new__(module.DSparkOfflineRunner)
    runner.draft_model = SimpleNamespace(
        use_draft_vocab=True,
        draft_vocab_size=3,
        d2t=module.torch.tensor([0, 1, 2]),
    )
    target_logits = module.torch.arange(6, dtype=module.torch.float32).unsqueeze(0)

    selected = runner._target_logits_to_draft_vocab(target_logits)

    assert module.torch.equal(
        selected,
        module.torch.tensor([[0.0, 2.0, 4.0]]),
    )


def test_offline_selector_returns_realized_topk_q_and_feedback_token():
    module = _load_module()
    seen_previous = []

    class Draft:
        correction_head = None
        markov_head = None
        candidate_selector = object()
        use_draft_vocab = False
        d2t = None

        @staticmethod
        def dflash2_select_candidates(logits, hidden_states, previous_token_ids):
            del hidden_states
            seen_previous.append(int(previous_token_ids.item()))
            if len(seen_previous) == 1:
                ids = module.torch.tensor([[[1, 2]]], device=logits.device)
            else:
                ids = module.torch.tensor([[[0, 3]]], device=logits.device)
            scores = module.torch.tensor(
                [[[0.0, 10.0]]], device=logits.device, dtype=logits.dtype
            )
            return ids, scores

    runner = module.DSparkOfflineRunner.__new__(module.DSparkOfflineRunner)
    runner.draft_model = Draft()
    runner.args = SimpleNamespace(temperature=0.0)
    runner.device = module.torch.device("cpu")
    runner.first_draft_slot = 0
    runner.max_proposal_tokens = 2
    base_logits = module.torch.zeros(1, 2, 4)
    hidden_states = module.torch.zeros(1, 2, 3)

    proposed, probabilities = runner._sample_dspark_tokens(
        base_logits,
        hidden_states,
        module.torch.tensor([5]),
        None,
    )

    assert proposed == [2, 3]
    assert seen_previous == [5, 2]
    assert module.torch.equal(
        probabilities,
        module.torch.tensor([[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]),
    )


def test_offline_selector_positive_temperature_q_and_vocab_mapping():
    module = _load_module()
    module.sample_from_probs = lambda probs: probs.argmax(dim=-1)

    class Draft:
        correction_head = None
        markov_head = None
        candidate_selector = object()
        use_draft_vocab = True
        draft_vocab_size = 4
        verifier_vocab_size = 7
        d2t = module.torch.tensor([0, 1, 2, 3])
        t2d = module.torch.zeros(7, dtype=module.torch.long)

        @staticmethod
        def dflash2_select_candidates(logits, hidden_states, previous_token_ids):
            del hidden_states, previous_token_ids
            ids = module.torch.tensor([[[1, 3]]], device=logits.device)
            scores = module.torch.tensor(
                [[[0.0, math.log(3.0)]]],
                device=logits.device,
                dtype=logits.dtype,
            )
            return ids, scores

    runner = module.DSparkOfflineRunner.__new__(module.DSparkOfflineRunner)
    runner.draft_model = Draft()
    runner.args = SimpleNamespace(temperature=1.0)
    runner.device = module.torch.device("cpu")
    runner.first_draft_slot = 0
    runner.max_proposal_tokens = 1

    proposed, draft_q = runner._sample_dspark_tokens(
        module.torch.zeros(1, 1, 4),
        module.torch.zeros(1, 1, 3),
        module.torch.tensor([5]),
        None,
    )
    target_q = runner._expand_draft_probs_to_target_vocab(draft_q)

    assert proposed == [6]
    assert draft_q.sum().item() == pytest.approx(1.0)
    assert draft_q[0, 0].tolist() == pytest.approx([0.0, 0.25, 0.0, 0.75])
    assert target_q[0, 0].tolist() == pytest.approx(
        [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.75]
    )
    assert target_q[0, 0, proposed[0]].item() > 0.0


def test_offline_global_selector_returns_viterbi_path_as_exact_q():
    module = _load_module()

    class Draft:
        correction_head = None
        markov_head = None
        candidate_selector = object()
        config = SimpleNamespace(dflash2_selector_search_mode="global")
        use_draft_vocab = False
        d2t = None

        @staticmethod
        def dflash2_select_path(logits, hidden_states, anchor_token_ids):
            del logits, hidden_states, anchor_token_ids
            candidate_ids = module.torch.tensor([[[1, 2], [3, 4]]])
            realized_rows = module.torch.tensor([[[10.0, 9.0], [8.0, 0.0]]])
            # The first Viterbi choice is intentionally not its realized-row argmax.
            selected_ids = module.torch.tensor([[2, 3]])
            return candidate_ids, realized_rows, selected_ids

    runner = module.DSparkOfflineRunner.__new__(module.DSparkOfflineRunner)
    runner.draft_model = Draft()
    runner.args = SimpleNamespace(temperature=1.0)
    runner.device = module.torch.device("cpu")
    runner.first_draft_slot = 0
    runner.max_proposal_tokens = 2

    proposed, draft_q = runner._sample_dspark_tokens(
        module.torch.zeros(1, 2, 5),
        module.torch.zeros(1, 2, 3),
        module.torch.tensor([7]),
        None,
    )

    assert proposed == [2, 3]
    assert module.torch.equal(
        draft_q,
        module.torch.tensor([[[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]]]),
    )


def test_shard_records_round_robin():
    module = _load_module()
    records = [{"prompt": str(i)} for i in range(7)]

    shard = module._shard_records(records, shard_index=1, num_shards=3)

    assert shard == [(2, records[1]), (5, records[4])]


def test_eval_stats_position_probability_means():
    module = _load_module()
    stats = module.EvalStats()
    stats.add_response(
        SimpleNamespace(
            num_output_tokens=0,
            proposal_lengths=[2, 2],
            accepted_draft_lengths=[1, 0],
            accept_prob_lists=[[0.8, 0.2], [0.4, 0.1]],
            support_accept_rate_lists=[[0.9, 0.3], [0.7, 0.5]],
        ),
    )

    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(
            stats.position_accept_prob_means,
            [0.6, 0.15],
            strict=True,
        )
    )
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(
            stats.position_support_accept_rate_means,
            [0.8, 0.4],
            strict=True,
        )
    )


def test_aggregate_rows_recomputes_weighted_lengths():
    module = _load_module()

    row = module._aggregate_rows(
        "sample",
        [
            {
                "num_requests": 2,
                "elapsed_s": 4.0,
                "total_output_tokens": 20,
                "num_proposals": 2,
                "num_proposed_draft_tokens": 8,
                "num_accepted_draft_tokens": 4,
            },
            {
                "num_requests": 3,
                "elapsed_s": 5.0,
                "total_output_tokens": 40,
                "num_proposals": 3,
                "num_proposed_draft_tokens": 18,
                "num_accepted_draft_tokens": 9,
            },
        ],
    )

    assert row["dataset"] == "sample"
    assert row["num_requests"] == 5
    assert row["elapsed_s"] == 5.0
    assert row["output_tokens_per_second"] == 12.0
    assert row["draft_length"] == 5.2
    assert row["acceptance_length"] == 3.6
    assert row["accepted_draft_length"] == 2.6


def test_aggregate_rows_recomputes_measured_base_speedup():
    module = _load_module()

    row = module._aggregate_rows(
        "sample",
        [
            {
                "num_requests": 2,
                "elapsed_s": 4.0,
                "total_output_tokens": 20,
                "base_elapsed_s": 7.0,
                "base_total_output_tokens": 20,
                "num_proposals": 2,
                "num_proposed_draft_tokens": 8,
                "num_accepted_draft_tokens": 4,
            },
            {
                "num_requests": 3,
                "elapsed_s": 5.0,
                "total_output_tokens": 40,
                "base_elapsed_s": 8.0,
                "base_total_output_tokens": 28,
                "num_proposals": 3,
                "num_proposed_draft_tokens": 18,
                "num_accepted_draft_tokens": 9,
            },
        ],
    )

    assert row["output_tokens_per_second"] == 12.0
    assert row["base_elapsed_s"] == 8.0
    assert row["base_output_tokens_per_second"] == 6.0
    assert row["speedup_vs_base"] == 2.0
