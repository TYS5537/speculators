"""Dependency-free position/threshold checks for the teacher parity command."""

# ruff: noqa: PT009, PT027 -- Keep dependency-free unittest coverage.

import math
import unittest

from speculators_dsv4.parity import ParityThresholds, position_chunks, select_positions


class TeacherOptionsTests(unittest.TestCase):
    def test_positions_are_next_token_rows_not_shifted_labels(self):
        self.assertEqual(select_positions(7), [3, 4, 5, 6])
        self.assertEqual(select_positions(2, tail_positions=4), [0, 1])
        self.assertEqual(select_positions(7, positions=[0, 2, 6]), [0, 2, 6])

    def test_bad_positions(self):
        for positions in ([], [1, 0], [0, 0], [-1], [4], [True]):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                select_positions(4, positions=positions)

    def test_bounded_contiguous_chunks(self):
        self.assertEqual(
            list(position_chunks([0, 1, 2, 8, 9], 2)), [[0, 1], [2], [8, 9]]
        )
        with self.assertRaises(ValueError):
            list(position_chunks([0], 0))

    def test_threshold_validation(self):
        ParityThresholds().validate()
        for value in (-1, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ParityThresholds(max_tv=value).validate()
        with self.assertRaises(ValueError):
            ParityThresholds(min_argmax_agreement=1.1).validate()


if __name__ == "__main__":
    unittest.main()
