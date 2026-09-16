"""DSV4 probability/cache contracts; no PyTorch or backend installation needed."""

# ruff: noqa: PT009, PT027 -- Keep this suite runnable with stdlib unittest.

import math
import tempfile
import unittest
from pathlib import Path

from speculators_dsv4.offline import (
    TokenHistoryCache,
    parse_full_logprobs,
    request_hidden_file,
)


class FullLogprobTests(unittest.TestCase):
    def test_full_distribution_is_returned_in_token_id_order(self):
        values = {
            "token_id:2": math.log(0.5),
            "token_id:0": math.log(0.2),
            "token_id:1": math.log(0.3),
        }
        parsed = parse_full_logprobs(values, 3)
        self.assertEqual(parsed, [math.log(0.2), math.log(0.3), math.log(0.5)])

    def test_zero_probability_tokens_are_preserved(self):
        parsed = parse_full_logprobs({"token_id:0": -math.inf, "token_id:1": 0.0}, 2)
        self.assertEqual(parsed, [-math.inf, 0.0])

    def test_rejects_truncated_or_out_of_vocabulary_distribution(self):
        for values in (
            {"token_id:0": 0.0},
            {"token_id:0": math.log(0.5), "token_id:2": math.log(0.5)},
            {
                "token_id:0": math.log(1 / 3),
                "token_id:1": math.log(1 / 3),
                "token_id:2": math.log(1 / 3),
            },
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_full_logprobs(values, 2)

    def test_rejects_text_tokens_and_duplicate_numeric_ids(self):
        for values in (
            {"hello": math.log(0.5), "world": math.log(0.5)},
            {"token_id:0": math.log(0.5), "token_id:00": math.log(0.5)},
            {"token_id:-1": math.log(0.5), "token_id:1": math.log(0.5)},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_full_logprobs(values, 2)

    def test_rejects_nonfinite_or_non_normalized_logprobs(self):
        for values in (
            {"token_id:0": math.nan, "token_id:1": 0.0},
            {"token_id:0": math.inf, "token_id:1": 0.0},
            {"token_id:0": -math.inf, "token_id:1": -math.inf},
            {"token_id:0": 0.0, "token_id:1": 0.0},
            {"token_id:0": math.log(0.2), "token_id:1": math.log(0.3)},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_full_logprobs(values, 2)


class TokenHistoryCacheTests(unittest.TestCase):
    def test_new_caches_do_not_share_token_lists(self):
        first = TokenHistoryCache()
        second = TokenHistoryCache()
        first.tokens.extend([4, 2, 7])
        self.assertEqual(first.get_seq_length(), 3)
        self.assertEqual(second.tokens, [])

    def test_crop_discards_rejected_candidates_only(self):
        cache = TokenHistoryCache()
        # Prefix [4, 2], anchor 7, one accepted candidate 3, rejected 9 and 8.
        cache.tokens.extend([4, 2, 7, 3, 9, 8])
        cache.crop(4)
        self.assertEqual(cache.tokens, [4, 2, 7, 3])
        self.assertEqual(cache.get_seq_length(), 4)
        cache.crop(4)
        self.assertEqual(cache.tokens, [4, 2, 7, 3])
        cache.crop(0)
        self.assertEqual(cache.tokens, [])

    def test_crop_rejects_negative_and_expanding_lengths(self):
        cache = TokenHistoryCache()
        cache.tokens.extend([1, 2, 3])
        for length in (-1, 4):
            with self.subTest(length=length), self.assertRaises(ValueError):
                cache.crop(length)
            self.assertEqual(cache.tokens, [1, 2, 3])


class RequestHiddenFileTests(unittest.TestCase):
    def test_accepts_only_current_request_direct_child(self):
        request_id = "0123456789abcdef0123456789abcdef"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            for name in (
                f"cmpl-{request_id}-0.safetensors",
                f"cmpl-{request_id}-0-deadbeef.safetensors",
            ):
                path = directory / name
                self.assertEqual(
                    request_hidden_file(str(path), directory, request_id), path
                )

    def test_rejects_unrelated_cached_or_outside_files(self):
        request_id = "0123456789abcdef0123456789abcdef"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            for path in (
                directory / "0.safetensors",
                directory / "cmpl-another-request-0.safetensors",
                directory / f"cmpl-{request_id}-1.safetensors",
                directory / "nested" / f"cmpl-{request_id}-0.safetensors",
                directory.parent / f"cmpl-{request_id}-0.safetensors",
                Path(f"cmpl-{request_id}-0.safetensors"),
            ):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    request_hidden_file(str(path), directory, request_id)

    def test_rejects_missing_or_non_string_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            for handle in (None, "", 123):
                with self.subTest(handle=handle), self.assertRaises(ValueError):
                    request_hidden_file(handle, directory, "request")


if __name__ == "__main__":
    unittest.main()
