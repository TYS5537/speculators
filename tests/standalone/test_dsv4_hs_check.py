"""Dependency-free tests for the HS concurrency probe (not actual DP/NPU tests)."""

# ruff: noqa: PT009, PT027 -- Keep the suite runnable without pytest/torch.

import importlib.util
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SPEC = importlib.util.spec_from_file_location(
    "dsv4_hs_check", Path(__file__).resolve().parents[2] / "scripts/check_dsv4_hs.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HSCheckTests(unittest.TestCase):
    def test_concurrent_different_length_requests_and_unique_handles(self):
        barrier = threading.Barrier(2, timeout=5)

        def check_one(tokens):
            barrier.wait()
            return {"hidden_states_file": f"hs-{len(tokens)}", "tokens": tokens}

        result = MODULE.check_requests([1, 2, 3, 4], 4, 2, check_one)
        self.assertEqual(
            [item["tokens"] for item in result], [[1, 2, 3, 4], [1, 2, 3], [1, 2], [1]]
        )

    def test_single_request_keeps_original_tokens(self):
        check_one = Mock(return_value={"hidden_states_file": "hs-1"})
        MODULE.check_requests([1, 2], 1, 1, check_one)
        check_one.assert_called_once_with([1, 2])

    def test_duplicate_handles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same output file"):
            MODULE.check_requests(
                [1, 2], 2, 2, lambda _: {"hidden_states_file": "same"}
            )

    def test_request_failure_is_not_hidden(self):
        with self.assertRaisesRegex(RuntimeError, "engine failure"):
            MODULE.check_requests(
                [1, 2], 2, 2, Mock(side_effect=RuntimeError("engine failure"))
            )

    def test_invalid_probe_counts(self):
        for requests, concurrency in ((0, 1), (-1, 2), (2, 0), (1, -1)):
            with (
                self.subTest(requests=requests, concurrency=concurrency),
                self.assertRaises(ValueError),
            ):
                MODULE.check_requests([1], requests, concurrency, Mock())

    def test_decode_probe_uses_greedy_multiple_tokens_and_requires_completion(self):
        client = Mock()
        response = SimpleNamespace(usage=SimpleNamespace(completion_tokens=4))
        client.completions.create.return_value = response
        self.assertIs(
            MODULE.request_decode_probe(
                client, "target", [1, 2], max_tokens=4, timeout=120
            ),
            response,
        )
        client.completions.create.assert_called_once_with(
            model="target",
            prompt=[1, 2],
            max_tokens=4,
            temperature=0,
            extra_body={"return_token_ids": True, "ignore_eos": True},
            timeout=120,
        )
        for invalid in (None, SimpleNamespace(completion_tokens=1)):
            response.usage = invalid
            with self.assertRaisesRegex(ValueError, "Decode probe requested"):
                MODULE.request_decode_probe(
                    client, "target", [1, 2], max_tokens=4, timeout=120
                )

    def test_payload_checks_tokens_shape_dtype_and_finiteness(self):
        torch = SimpleNamespace(bfloat16="bf16", isfinite=Mock())
        torch.isfinite.return_value.all.return_value.item.return_value = True
        hidden = SimpleNamespace(shape=(2, 6, 4096), dtype="bf16")
        tokens = Mock()
        tokens.tolist.return_value = [1, 2]
        payload = {"token_ids": tokens, "hidden_states": hidden}
        self.assertIs(MODULE.validate_payload(payload, [1, 2], 6, torch), hidden)
        for invalid in (
            None,
            {**payload, "hidden_states": SimpleNamespace(shape=(1, 6, 4096))},
            {
                **payload,
                "hidden_states": SimpleNamespace(shape=(2, 6, 4096), dtype="fp32"),
            },
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                MODULE.validate_payload(invalid, [1, 2], 6, torch)
        with self.assertRaisesRegex(ValueError, "token IDs"):
            MODULE.validate_payload(payload, [2, 1], 6, torch)
        torch.isfinite.return_value.all.return_value.item.return_value = False
        with self.assertRaisesRegex(ValueError, "finite BF16"):
            MODULE.validate_payload(payload, [1, 2], 6, torch)


if __name__ == "__main__":
    unittest.main()
