"""Read-only structural checks against real Hugging Face Arrow datasets."""

# ruff: noqa: PT009, PT027 -- Deliberately runnable with stdlib unittest only.

import hashlib
import importlib
import tempfile
import unittest
from pathlib import Path


class ExternalArrowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.datasets = importlib.import_module("datasets")
            cls.arrow = importlib.import_module("pyarrow")
        except ImportError as error:
            raise unittest.SkipTest(
                "Real Arrow tests require datasets and pyarrow"
            ) from error
        module = importlib.import_module("speculators_dsv4.external_data")
        cls.validate = staticmethod(module.validate_external_arrow)
        if not cls.datasets.are_progress_bars_disabled():
            cls.datasets.disable_progress_bars()
            cls.addClassCleanup(cls.datasets.enable_progress_bars)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.saved_count = 0

    @staticmethod
    def _valid_columns():
        return {
            "input_ids": [[0, 2, 3, 4, 99], [10, 11, 12], [20, 21], [42]],
            "loss_mask": [[1, 0, 1, 0, 1], [0, 1, 1], [0, 0], [1]],
            "seq_len": [5, 3, 2, 1],
        }

    def _save(self, columns=None, *, table=None):
        self.saved_count += 1
        path = self.root / f"data-{self.saved_count}"
        dataset = (
            self.datasets.Dataset(table)
            if table is not None
            else self.datasets.Dataset.from_dict(
                self._valid_columns() if columns is None else columns
            )
        )
        dataset.save_to_disk(str(path))
        return path

    @staticmethod
    def _file_hashes(path):
        return {
            str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in path.rglob("*")
            if file.is_file()
        }

    def _assert_invalid_column(self, field, values):
        columns = {
            "input_ids": [[1, 2]],
            "loss_mask": [[0, 1]],
            "seq_len": [2],
        }
        columns[field] = values
        path = self._save(columns)
        before = self._file_hashes(path)
        with self.assertRaisesRegex(ValueError, field):
            self.validate(path, 100, batch_size=1)
        self.assertEqual(self._file_hashes(path), before)

    def test_valid_default_list_format_and_report(self):
        path = self._save()
        dataset = self.datasets.load_from_disk(str(path))
        self.assertIsNone(dataset.format["type"])
        self.assertIsInstance(dataset[0]["input_ids"], list)
        report = self.validate(path, 100, batch_size=2)
        self.assertEqual(
            report,
            {
                "data_path": str(path.resolve()),
                "row_count": 4,
                "checked_rows": 4,
                "row_start": 0,
                "row_stop": 4,
                "provenance": "user-asserted",
                "validation": "structure-only",
            },
        )

    def test_tokens_masks_order_format_and_all_files_are_unchanged(self):
        path = self._save()
        before_hashes = self._file_hashes(path)
        before = self.datasets.load_from_disk(str(path))
        before_rows, before_format = before[:], before.format
        del before
        self.validate(path, 100, batch_size=1)
        after = self.datasets.load_from_disk(str(path))
        self.assertEqual(after[:], before_rows)
        self.assertEqual(after.format, before_format)
        self.assertEqual(self._file_hashes(path), before_hashes)

    def test_saved_numpy_format_is_not_rewritten_by_validation(self):
        path = self.root / "numpy-data"
        dataset = self.datasets.Dataset.from_dict(self._valid_columns())
        dataset.set_format("numpy")
        dataset.save_to_disk(str(path))
        before = self._file_hashes(path)
        self.assertEqual(self.validate(path, 100)["checked_rows"], 4)
        self.assertEqual(self._file_hashes(path), before)
        after = self.datasets.load_from_disk(str(path))
        self.assertEqual(after.format["type"], "numpy")
        self.assertEqual(after.with_format(None)[:], self._valid_columns())

    def test_extra_columns_are_accepted_without_rewriting_or_reencoding(self):
        columns = self._valid_columns()
        columns["messages"] = ["not a chat message"] * 4
        columns["source_id"] = ["z", "a", "x", "b"]
        path = self._save(columns)
        before = self._file_hashes(path)
        self.assertEqual(self.validate(path, 100)["checked_rows"], 4)
        self.assertEqual(self._file_hashes(path), before)
        self.assertEqual(self.datasets.load_from_disk(str(path))[:], columns)

    def test_all_binary_mask_numeric_types_are_accepted(self):
        for dtype, values in (
            (self.arrow.bool_(), [False, True]),
            (self.arrow.int8(), [0, 1]),
            (self.arrow.uint64(), [1, 0]),
            (self.arrow.float32(), [1.0, 0.0]),
            (self.arrow.float64(), [-0.0, 1.0]),
        ):
            with self.subTest(dtype=str(dtype)):
                table = self.arrow.table(
                    {
                        "input_ids": self.arrow.array([[0, 99]]),
                        "loss_mask": self.arrow.array(
                            [values], type=self.arrow.list_(dtype)
                        ),
                        "seq_len": self.arrow.array([2]),
                    }
                )
                report = self.validate(self._save(table=table), 100)
                self.assertEqual(report["checked_rows"], 1)

    def test_integer_id_widths_are_accepted(self):
        for dtype in (self.arrow.int32(), self.arrow.int64(), self.arrow.uint64()):
            with self.subTest(dtype=str(dtype)):
                table = self.arrow.table(
                    {
                        "input_ids": self.arrow.array(
                            [[0, 99]], type=self.arrow.list_(dtype)
                        ),
                        "loss_mask": self.arrow.array([[0, 1]]),
                        "seq_len": self.arrow.array([2], type=self.arrow.int32()),
                    }
                )
                self.assertEqual(
                    self.validate(self._save(table=table), 100)["checked_rows"], 1
                )

    def test_float16_masks_use_safe_temporary_comparison_values(self):
        for value in (1.0, 0.5, float("nan"), float("inf")):
            with self.subTest(value=value):
                table = self.arrow.table(
                    {
                        "input_ids": self.arrow.array([[0, 99]]),
                        "loss_mask": self.arrow.array(
                            [[0.0, value]], type=self.arrow.list_(self.arrow.float16())
                        ),
                        "seq_len": self.arrow.array([2]),
                    }
                )
                path = self._save(table=table)
                before = self._file_hashes(path)
                if value == 1.0:
                    self.assertEqual(self.validate(path, 100)["checked_rows"], 1)
                else:
                    with self.assertRaisesRegex(ValueError, "loss_mask"):
                        self.validate(path, 100)
                self.assertEqual(self._file_hashes(path), before)

    def test_fixed_size_lists_are_still_one_dimensional(self):
        table = self.arrow.table(
            {
                "input_ids": self.arrow.array(
                    [[0, 99]], type=self.arrow.list_(self.arrow.int64(), 2)
                ),
                "loss_mask": self.arrow.array(
                    [[0, 1]], type=self.arrow.list_(self.arrow.int8(), 2)
                ),
                "seq_len": self.arrow.array([2]),
            }
        )
        self.assertEqual(self.validate(self._save(table=table), 100)["checked_rows"], 1)

    def test_missing_required_columns(self):
        for field in ("input_ids", "loss_mask", "seq_len"):
            with self.subTest(field=field):
                columns = self._valid_columns()
                del columns[field]
                with self.assertRaisesRegex(ValueError, field):
                    self.validate(self._save(columns), 100)

    def test_ids_must_be_one_dimensional_integers_not_booleans(self):
        for values in ([1], [[True, False]], [[1.0, 2.0]], [["1", "2"]], [[[1, 2]]]):
            with self.subTest(values=values):
                self._assert_invalid_column("input_ids", values)

    def test_ids_must_be_within_vocabulary(self):
        for values in ([[0, -1]], [[0, 100]], [[1000, 1]]):
            with self.subTest(values=values):
                self._assert_invalid_column("input_ids", values)

    def test_large_unsigned_id_cannot_wrap_into_valid_range(self):
        table = self.arrow.table(
            {
                "input_ids": self.arrow.array(
                    [[0, 2**63 + 3]], type=self.arrow.list_(self.arrow.uint64())
                ),
                "loss_mask": self.arrow.array([[0, 1]]),
                "seq_len": self.arrow.array([2]),
            }
        )
        with self.assertRaisesRegex(ValueError, "input_ids"):
            self.validate(self._save(table=table), 100)

    def test_large_unsigned_mask_has_a_field_specific_error(self):
        table = self.arrow.table(
            {
                "input_ids": self.arrow.array([[0, 99]]),
                "loss_mask": self.arrow.array(
                    [[0, 2**63 + 3]], type=self.arrow.list_(self.arrow.uint64())
                ),
                "seq_len": self.arrow.array([2]),
            }
        )
        with self.assertRaisesRegex(ValueError, "loss_mask"):
            self.validate(self._save(table=table), 100)

    def test_unsigned_seq_len_rejects_overflow_with_field_specific_error(self):
        for length in (2, 2**63 + 3):
            with self.subTest(length=length):
                table = self.arrow.table(
                    {
                        "input_ids": self.arrow.array([[0, 99]]),
                        "loss_mask": self.arrow.array([[0, 1]]),
                        "seq_len": self.arrow.array([length], type=self.arrow.uint64()),
                    }
                )
                path = self._save(table=table)
                if length == 2:
                    self.assertEqual(self.validate(path, 100)["checked_rows"], 1)
                else:
                    with self.assertRaisesRegex(ValueError, "seq_len"):
                        self.validate(path, 100)

    def test_masks_must_be_one_dimensional_numeric_or_boolean(self):
        for values in ([1], [["0", "1"]], [[[0, 1]]]):
            with self.subTest(values=values):
                self._assert_invalid_column("loss_mask", values)

    def test_masks_reject_nonbinary_values_and_nonfinite_values(self):
        for value in (
            -1,
            2,
            0.5,
            1e-12,
            1.0000000000000002,
            float("nan"),
            float("inf"),
            -float("inf"),
        ):
            with self.subTest(value=value):
                self._assert_invalid_column("loss_mask", [[0, value]])

    def test_no_null_rows_or_elements(self):
        for field, values in (
            ("input_ids", [None]),
            ("input_ids", [[1, None]]),
            ("loss_mask", [None]),
            ("loss_mask", [[0, None]]),
            ("seq_len", [None]),
        ):
            with self.subTest(field=field, values=values):
                self._assert_invalid_column(field, values)

    def test_seq_len_must_be_positive_integer_not_boolean(self):
        for values in ([True], [2.0], ["2"], [[2]], [0], [-1]):
            with self.subTest(values=values):
                self._assert_invalid_column("seq_len", values)

    def test_seq_len_matches_actual_token_length(self):
        for length in (1, 3):
            with self.subTest(length=length):
                self._assert_invalid_column("seq_len", [length])

    def test_mask_length_matches_actual_token_length(self):
        for mask in ([], [1], [0, 1, 1]):
            with self.subTest(mask=mask):
                self._assert_invalid_column("loss_mask", [mask])

    def test_empty_sequence_is_rejected(self):
        table = self.arrow.table(
            {
                "input_ids": self.arrow.array(
                    [[]], type=self.arrow.list_(self.arrow.int64())
                ),
                "loss_mask": self.arrow.array(
                    [[]], type=self.arrow.list_(self.arrow.int64())
                ),
                "seq_len": self.arrow.array([0]),
            }
        )
        with self.assertRaises(ValueError):
            self.validate(self._save(table=table), 100)

    def test_empty_dataset_is_rejected(self):
        table = self.arrow.table(
            {
                "input_ids": self.arrow.array(
                    [], type=self.arrow.list_(self.arrow.int64())
                ),
                "loss_mask": self.arrow.array(
                    [], type=self.arrow.list_(self.arrow.int64())
                ),
                "seq_len": self.arrow.array([], type=self.arrow.int64()),
            }
        )
        with self.assertRaises(ValueError):
            self.validate(self._save(table=table), 100)

    def test_dataset_dict_is_not_silently_concatenated_or_split_selected(self):
        path = self.root / "splits"
        dataset = self.datasets.Dataset.from_dict(self._valid_columns())
        self.datasets.DatasetDict(
            {"train": dataset, "validation": dataset}
        ).save_to_disk(str(path))
        before = self._file_hashes(path)
        with self.assertRaises(ValueError):
            self.validate(path, 100)
        self.assertEqual(self._file_hashes(path), before)

    def test_missing_path_or_non_dataset_directory_is_rejected(self):
        for path in (self.root / "absent", self.root):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.validate(path, 100)

    def test_invalid_topology_and_batch_sizes(self):
        path = self._save()
        for options in (
            {"world_size": 0},
            {"world_size": -1},
            {"world_size": True},
            {"world_size": 1.0},
            {"rank": -1},
            {"rank": 1},
            {"rank": True},
            {"rank": False},
            {"rank": 0.0},
            {"rank": 2, "world_size": 2},
            {"batch_size": 0},
            {"batch_size": -1},
            {"batch_size": True},
            {"batch_size": 1.0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.validate(path, 100, **options)

    def test_all_batches_are_scanned_not_only_initial_rows(self):
        columns = {
            "input_ids": [[1, 2] for _ in range(521)],
            "loss_mask": [[0, 1] for _ in range(521)],
            "seq_len": [2] * 521,
        }
        for field, value in (
            ("input_ids", [1, 100]),
            ("loss_mask", [0, 0.25]),
            ("seq_len", 3),
        ):
            with self.subTest(field=field):
                invalid = {name: list(values) for name, values in columns.items()}
                invalid[field][-1] = value
                with self.assertRaisesRegex(ValueError, field):
                    self.validate(self._save(invalid), 100, batch_size=17)

    def test_contiguous_rank_ranges_cover_every_row_exactly_once(self):
        path = self._save()
        for world_size in (1, 2, 3, 4, 7):
            with self.subTest(world_size=world_size):
                coverage = []
                total_checked = 0
                for rank in range(world_size):
                    report = self.validate(
                        path, 100, rank=rank, world_size=world_size, batch_size=1
                    )
                    start, stop = 4 * rank // world_size, 4 * (rank + 1) // world_size
                    self.assertEqual(report["row_count"], 4)
                    self.assertEqual(
                        (report["row_start"], report["row_stop"]), (start, stop)
                    )
                    self.assertEqual(report["checked_rows"], stop - start)
                    coverage.extend(range(start, stop))
                    total_checked += report["checked_rows"]
                self.assertEqual(coverage, list(range(4)))
                self.assertEqual(total_checked, 4)

    def test_rank_only_scans_its_owned_rows(self):
        columns = self._valid_columns()
        columns["input_ids"][-1] = [100]
        path = self._save(columns)
        for rank in (0, 1):
            report = self.validate(path, 100, rank=rank, world_size=3, batch_size=1)
            self.assertEqual(report["checked_rows"], 1)
        with self.assertRaisesRegex(ValueError, "input_ids"):
            self.validate(path, 100, rank=2, world_size=3, batch_size=1)


if __name__ == "__main__":
    unittest.main()
