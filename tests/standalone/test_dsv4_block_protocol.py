"""Validate the block request contract without loading torch or vLLM."""

# ruff: noqa: PT009, PT027 -- Keep standalone tests runnable without pytest.

import unittest

from speculators_dsv4.block_protocol import validate_block_request


class BlockProtocolTests(unittest.TestCase):
    def test_snapshot_protocol_rejects_paths_duplicates_and_missing_keys(self):
        good = {
            "version": 3,
            "logits_start": 7,
            "hidden_start": 8,
            "output_mode": "greedy",
            "profile": False,
            "cache": {"read": "a" * 32, "write": "b" * 32, "release": ["a" * 32]},
        }
        self.assertEqual(validate_block_request(good, 8), (7, 8))
        for cache in (
            None,
            {},
            {**good["cache"], "read": "/tmp/private"},
            {**good["cache"], "write": "a" * 32},
            {**good["cache"], "release": ["a" * 32] * 2},
            {**good["cache"], "write": True},
            {**good["cache"], "release": [None]},
            {**good["cache"], "path": "/tmp/x"},
        ):
            with self.subTest(cache=cache), self.assertRaises(ValueError):
                validate_block_request({**good, "cache": cache}, 8)

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

    def test_version_two_requires_explicit_mode_and_profile(self):
        request = {
            "version": 2,
            "logits_start": 3,
            "hidden_start": 2,
            "output_mode": "greedy",
            "profile": False,
        }
        for mode in ("greedy", "logprobs"):
            for profile in (True, False):
                self.assertEqual(
                    validate_block_request(
                        {**request, "output_mode": mode, "profile": profile}, 8
                    ),
                    (3, 2),
                )
        for field, value in (
            ("profile", 1),
            ("profile", "false"),
            ("output_mode", "auto"),
            ("output_mode", None),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_block_request({**request, field: value}, 8)
        for field in ("profile", "output_mode"):
            invalid = dict(request)
            del invalid[field]
            with self.assertRaises(ValueError):
                validate_block_request(invalid, 8)
        with self.assertRaises(ValueError):
            validate_block_request({**request, "version": 1}, 8)


if __name__ == "__main__":
    unittest.main()
