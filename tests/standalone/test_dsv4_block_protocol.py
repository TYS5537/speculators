"""Validate the block request contract without loading torch or vLLM."""

# ruff: noqa: PT009, PT027 -- Keep standalone tests runnable without pytest.

import unittest

from speculators_dsv4.block_protocol import validate_block_request


class BlockProtocolTests(unittest.TestCase):
    def test_prefill_and_verification_offsets(self):
        for logits_start, hidden_start in ((7, 0), (4, 4), (4, 8), (0, 0)):
            with self.subTest(logits_start=logits_start, hidden_start=hidden_start):
                request = {
                    "version": 1,
                    "logits_start": logits_start,
                    "hidden_start": hidden_start,
                }
                self.assertEqual(
                    validate_block_request(request, 8), (logits_start, hidden_start)
                )

    def test_missing_extra_or_non_dict_fields_fail(self):
        for request in (
            None,
            [],
            {},
            {"version": 1, "logits_start": 0},
            {"version": 1, "logits_start": 0, "hidden_start": 0, "path": "/tmp/x"},
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                validate_block_request(request, 8)

    def test_versions_offsets_and_prompt_length_are_strict_integers(self):
        good = {"version": 1, "logits_start": 7, "hidden_start": 0}
        for field, bad_values in (
            ("version", (True, 1.0, "1", 0, 2)),
            ("logits_start", (False, 7.0, "7", -1, 8)),
            ("hidden_start", (False, 0.0, "0", -1, 9)),
        ):
            for value in bad_values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    validate_block_request({**good, field: value}, 8)
        for length in (True, 8.0, "8", 0, -1):
            with self.subTest(length=length), self.assertRaises(ValueError):
                validate_block_request(good, length)


if __name__ == "__main__":
    unittest.main()
