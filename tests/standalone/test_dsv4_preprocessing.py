"""No-model tests for strict official-renderer token boundaries and contracts."""

# ruff: noqa: PT009, PT027 -- Deliberately runnable with stdlib unittest only.

import json
import tempfile
import unittest
from pathlib import Path

from speculators_dsv4 import preprocessing as prep


class TextRenderer:
    """Tiny token renderer that emulates history-dependent reasoning removal."""

    eos_token_id = 1

    def apply_chat_template(self, messages, **options):
        last_user = max(i for i, item in enumerate(messages) if item["role"] == "user")
        ids = [0]
        for index, item in enumerate(messages):
            role = item["role"]
            if role == "user":
                ids += [2] + [ord(char) for char in item["content"]] + [3]
                ids += [4 if options["enable_thinking"] and index >= last_user else 5]
            elif role == "assistant":
                if options["enable_thinking"] and index > last_user:
                    ids += [ord(char) for char in item.get("reasoning", "")] + [5]
                ids += [ord(char) for char in item["content"]] + [1]
            else:
                ids += [ord(char) for char in item["content"]]
        return ids


class EncoderTests(unittest.TestCase):
    def test_current_assistant_and_eos_only(self):
        encoder = prep.DSV4TrainingEncoder(TextRenderer())
        row = encoder.encode(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "Answer"},
            ]
        )[0]
        trained = [
            token
            for token, bit in zip(row["input_ids"], row["loss_mask"], strict=True)
            if bit
        ]
        self.assertEqual(trained, [ord(char) for char in "Answer"] + [1])
        self.assertEqual(row["seq_len"], len(row["input_ids"]))

    def test_thinking_history_rerender_uses_independent_rows(self):
        encoder = prep.DSV4TrainingEncoder(TextRenderer(), enable_thinking=True)
        rows = encoder.encode(
            [
                {"from": "human", "value": "Q"},
                {"from": "gpt", "value": "A", "thinking": "R"},
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": "a", "reasoning_content": "r"},
            ]
        )
        self.assertEqual(len(rows), 2)
        for row, text in zip(rows, ("RA", "ra"), strict=True):
            trained = [
                token
                for token, bit in zip(row["input_ids"], row["loss_mask"], strict=True)
                if bit
            ]
            self.assertEqual(trained, [ord(text[0]), 5, ord(text[1]), 1])
        self.assertIn(ord("R"), rows[0]["input_ids"])
        self.assertNotIn(ord("R"), rows[1]["input_ids"])

    def test_prefix_instability_is_fatal(self):
        class Unstable(TextRenderer):
            def apply_chat_template(self, messages, **options):
                ids = super().apply_chat_template(messages, **options)
                if messages[-1]["role"] == "assistant":
                    ids[0] = 88
                return ids

        with self.assertRaisesRegex(ValueError, "prompt token prefix"):
            prep.DSV4TrainingEncoder(Unstable()).encode(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ]
            )

    def test_missing_eos_is_fatal(self):
        renderer = TextRenderer()
        renderer.eos_token_id = 99
        with self.assertRaisesRegex(ValueError, "EOS"):
            prep.DSV4TrainingEncoder(renderer).encode(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ]
            )

    def test_unsupported_messages_fail_closed(self):
        bad_assistants = [
            {"role": "tool", "content": "A"},
            {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
            {"role": "assistant", "content": "A", "tool_calls": []},
            {"role": "assistant", "content": "<think>R</think>A"},
            {"role": "assistant", "content": "A", "reasoning": "R", "thinking": "S"},
        ]
        for bad in bad_assistants:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prep.DSV4TrainingEncoder(TextRenderer()).encode(
                    [{"role": "user", "content": "Q"}, bad]
                )

    def test_chat_mode_cannot_silently_drop_reasoning(self):
        with self.assertRaisesRegex(ValueError, "enable-thinking"):
            prep.DSV4TrainingEncoder(TextRenderer()).encode(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A", "reasoning": "R"},
                ]
            )

    def test_rows_validate_before_truncation(self):
        invalid = [
            {"input_ids": [0, 999], "loss_mask": [0, 1]},
            {"input_ids": [0, 2.0], "loss_mask": [0, 1]},
            {"input_ids": [0, 2], "loss_mask": [0, 0.5]},
            {"input_ids": [0, 2], "loss_mask": [0]},
            {"input_ids": [0, 2], "loss_mask": [1, 1]},
        ]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                prep._validated_row(row, 100, 1)

    def test_valid_truncation(self):
        self.assertEqual(
            prep._validated_row(
                {"input_ids": [0, 2, 3], "loss_mask": [0, 1, 1]}, 100, 2
            ),
            {"input_ids": [0, 2], "loss_mask": [0, 1], "seq_len": 2},
        )


class ContractTests(unittest.TestCase):
    def test_manifest_identity_and_tokenizer_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            report = {
                "model_path": str(root),
                "checkpoint_signature": "abc",
                "config": {"vocab_size": 100},
            }
            value = {
                **prep.data_identity(report),
                "encoding": prep.ENCODING,
                "enable_thinking": False,
                "mask_policy": "current-assistant-continuation-including-eos",
            }
            path = root / prep.DATA_MANIFEST
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(prep.validate_data_manifest(root, report), value)
            (root / "tokenizer.json").write_text('{"changed":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenizer_files"):
                prep.validate_data_manifest(path, report)

    def test_missing_contract_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing DSV4"):
            prep.validate_data_manifest("not-existing-dsv4-manifest.json", {})

    def test_endpoint_rejects_unsafe_urls(self):
        for endpoint in (
            "file:///tmp/model",
            "https://user:secret@example.com",
            "http://127.0.0.1/v1",
            "http://127.0.0.1?key=x",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                prep.TokenizationClient(endpoint)

    def test_redirects_are_not_followed(self):
        with self.assertRaisesRegex(ValueError, "redirect"):
            prep._NoRedirect().redirect_request(
                None, None, 302, "", {}, "http://elsewhere"
            )


if __name__ == "__main__":
    unittest.main()
