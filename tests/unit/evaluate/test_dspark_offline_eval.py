import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = Path(__file__).parents[3] / "scripts" / "evaluate" / "dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("dspark_offline_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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
