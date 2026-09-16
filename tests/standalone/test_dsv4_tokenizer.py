"""Server tokenization contract tests; no model, torch, or NPU required."""

# ruff: noqa: PT009, PT027 -- Keep this suite runnable without pytest/torch.

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from speculators_dsv4.tokenizer import DSV4ServerTokenizer


class DSV4ServerTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.post.return_value = {"tokens": [1, 5, 9], "count": 3}
        self.tokenizer = DSV4ServerTokenizer(
            self.client, "target", eos_token_id=2, vocab_size=128
        )
        self.torch = SimpleNamespace(long=object(), tensor=Mock())

    def test_chat_uses_server_and_forwards_thinking_without_local_mapping(self):
        messages = [{"role": "user", "content": "hello"}]
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        self.assertEqual(ids, [1, 5, 9])
        self.client.post.assert_called_once_with(
            "/tokenize",
            cast_to=dict,
            body={
                "model": "target",
                "messages": messages,
                "add_generation_prompt": True,
                "add_special_tokens": False,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        self.assertEqual(messages, [{"role": "user", "content": "hello"}])

    def test_chat_default_leaves_thinking_decision_to_server(self):
        self.tokenizer.apply_chat_template([{"role": "user", "content": "hello"}])
        body = self.client.post.call_args.kwargs["body"]
        self.assertEqual(body["chat_template_kwargs"], {})

    def test_chat_explicit_false_is_not_replaced_by_a_truthy_default(self):
        self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            enable_thinking=False,
            chat_template_kwargs={"enable_thinking": True, "drop_thinking": False},
        )
        body = self.client.post.call_args.kwargs["body"]
        self.assertEqual(
            body["chat_template_kwargs"],
            {"enable_thinking": False, "drop_thinking": False},
        )

    def test_rendered_ids_are_not_decoded_or_tokenized_twice(self):
        ids = self.tokenizer.apply_chat_template([{"role": "user", "content": "hi"}])
        with patch.dict(sys.modules, {"torch": self.torch}):
            result = self.tokenizer(ids, return_tensors="pt")
        self.assertEqual(self.client.post.call_count, 1)
        self.torch.tensor.assert_called_once_with([[1, 5, 9]], dtype=self.torch.long)
        self.assertIs(result.input_ids, self.torch.tensor.return_value)
        self.assertEqual(ids, [1, 5, 9])

    def test_raw_prompt_does_not_add_a_second_bos_or_apply_chat_template(self):
        raw = "<special-bos>already rendered prompt"
        with patch.dict(sys.modules, {"torch": self.torch}):
            self.tokenizer(raw)
        self.client.post.assert_called_once_with(
            "/tokenize",
            cast_to=dict,
            body={"model": "target", "prompt": raw, "add_special_tokens": False},
        )

    def test_tool_and_continuation_fields_use_the_api_protocol(self):
        messages = [{"role": "assistant", "content": "prefix"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            tools=tools,
            reasoning_effort="none",
        )
        body = self.client.post.call_args.kwargs["body"]
        self.assertEqual(body["tools"], tools)
        self.assertTrue(body["continue_final_message"])
        self.assertFalse(body["add_generation_prompt"])
        self.assertEqual(body["chat_template_kwargs"], {"reasoning_effort": "none"})

    def test_invalid_response_ids_fail_before_tensor_creation(self):
        for invalid in (None, [], "1", [True], [-1], [128], [1.5], [[1]], [2**63]):
            with self.subTest(invalid=invalid):
                self.client.post.return_value = {"tokens": invalid}
                with self.assertRaises(ValueError):
                    self.tokenizer("hello")

    def test_invalid_pretokenized_prompt_does_not_call_server(self):
        for invalid in (None, [], [True], [-1], [128], [1.5], [[1]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.tokenizer(invalid)
        self.client.post.assert_not_called()

    def test_malformed_response_and_count_mismatch_are_rejected(self):
        for response in (
            [],
            {"tokens": [1], "count": 2},
            {"tokens": [1], "count": True},
        ):
            with self.subTest(response=response), self.assertRaises(ValueError):
                self.client.post.return_value = response
                self.tokenizer("hello")

    def test_invalid_template_controls_fail_without_request(self):
        messages = [{"role": "user", "content": "hi"}]
        for kwargs in (
            {"tokenize": "false"},
            {"add_generation_prompt": "true"},
            {"continue_final_message": True},
            {"chat_template": "untrusted template"},
            {"chat_template_kwargs": None},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.tokenizer.apply_chat_template(messages, **kwargs)
        self.client.post.assert_not_called()

    def test_only_pt_tensors_are_supported(self):
        with self.assertRaises(ValueError):
            self.tokenizer([1], return_tensors="np")
        self.client.post.assert_not_called()

    def test_configuration_is_checked_and_eos_is_exposed(self):
        self.assertEqual(self.tokenizer.eos_token_id, 2)
        self.assertEqual(
            DSV4ServerTokenizer(self.client, "target", [2, 3]).eos_token_id, [2, 3]
        )
        for vocab in (0, -1, True, "128"):
            with self.subTest(vocab=vocab), self.assertRaises(ValueError):
                DSV4ServerTokenizer(self.client, "target", 2, vocab_size=vocab)
        with self.assertRaises(ValueError):
            DSV4ServerTokenizer(self.client, " ", 2)
        with self.assertRaises(ValueError):
            DSV4ServerTokenizer(self.client, "target", True)


if __name__ == "__main__":
    unittest.main()
