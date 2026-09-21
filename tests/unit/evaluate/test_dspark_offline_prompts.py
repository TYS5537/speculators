"""Prompt-field selection and Alpaca instruction/input composition."""

# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def evaluator():
    path = Path(__file__).parents[3] / "scripts/evaluate/dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("dspark_offline_prompts_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return json.dumps(messages)


def render(evaluator, record, *, mode="raw"):
    tokenizer = Tokenizer()
    prompt = evaluator._prompt_from_record(
        record,
        tokenizer,
        source="alpaca.jsonl:1",
        args=SimpleNamespace(enable_thinking="false", raw_prompt_mode=mode),
    )
    return prompt, tokenizer.calls


@pytest.mark.parametrize("mode", ["raw", "auto", "chat_template"])
def test_instruction_and_input_are_combined_before_formatting(evaluator, mode):
    prompt, calls = render(
        evaluator,
        {
            "instruction": "Translate into English.",
            "input": "Bonjour.",
            "output": "REFERENCE ANSWER MUST NOT LEAK",
            "answer": "ANOTHER REFERENCE ANSWER",
        },
        mode=mode,
    )
    combined = "Translate into English.\n\nBonjour."
    if mode == "raw":
        assert prompt == combined
        assert calls == []
    else:
        messages = [{"role": "user", "content": combined}]
        assert json.loads(prompt) == messages
        assert calls == [
            (
                messages,
                {
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                },
            )
        ]
    assert "REFERENCE ANSWER" not in prompt


@pytest.mark.parametrize("input_value", [None, "", "  ", [], ["", " "]])
def test_empty_input_keeps_instruction_only(evaluator, input_value):
    prompt, _ = render(
        evaluator, {"instruction": "Explain this.", "input": input_value}
    )
    assert prompt == "Explain this."


def test_instruction_without_input_does_not_include_output(evaluator):
    prompt, _ = render(evaluator, {"instruction": "Question", "output": "Answer"})
    assert prompt == "Question"


def test_instruction_and_input_support_string_lists(evaluator):
    prompt, _ = render(
        evaluator,
        {"instruction": ["Read this.", "", "Summarize."], "input": ["A.", "B."]},
    )
    assert prompt == "Read this.\n\nSummarize.\n\nA.\n\nB."


@pytest.mark.parametrize("instruction", [None, "", [], [" "]])
def test_input_without_instruction_preserves_existing_list_behavior(
    evaluator, instruction
):
    prompt, _ = render(evaluator, {"instruction": instruction, "input": ["A", "B"]})
    assert prompt == "A\n\nB"


@pytest.mark.parametrize("full_prompt", ["Complete request", ["Complete", "request"]])
def test_explicit_prompt_precedes_instruction_and_input(evaluator, full_prompt):
    prompt, _ = render(
        evaluator,
        {"prompt": full_prompt, "instruction": "Ignored", "input": "Also ignored"},
    )
    assert prompt == (
        full_prompt if isinstance(full_prompt, str) else "\n\n".join(full_prompt)
    )


@pytest.mark.parametrize(
    "conversation",
    [
        {"turns": ["Conversation request", "Second turn"]},
        {"messages": [{"role": "user", "content": "Conversation request"}]},
        {
            "conversations": [
                {"from": "human", "value": "Conversation request"},
                {"from": "gpt", "value": "Reference response"},
            ]
        },
    ],
)
def test_conversation_fields_keep_priority(evaluator, conversation):
    prompt, calls = render(
        evaluator,
        {**conversation, "prompt": "Ignored", "instruction": "Ignored", "input": "X"},
        mode="chat_template",
    )
    assert json.loads(prompt) == [{"role": "user", "content": "Conversation request"}]
    assert len(calls) == 1


def test_preformatted_explicit_prompt_is_not_wrapped_again(evaluator):
    text = "<|im_start|>user\nComplete request<|im_end|>\n<|im_start|>assistant\n"
    prompt, calls = render(
        evaluator,
        {"prompt": text, "instruction": "Ignored", "input": "Ignored"},
        mode="auto",
    )
    assert prompt == text
    assert calls == []
