"""Exercise the real Arrow HS prefix path without importing torch or vLLM."""

# ruff: noqa: PT009, PT027 -- Deliberately runnable with stdlib unittest.

import ast
import copy
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class Tensor(np.ndarray):
    """Only the tensor operations used by these real dataset methods."""

    def __new__(cls, values, dtype=None):
        return np.asarray(values, dtype=dtype).view(cls)

    def flatten(self, start_dim=0, end_dim=-1):
        if end_dim < 0:
            end_dim += self.ndim
        shape = (
            *self.shape[:start_dim],
            -1,
            *self.shape[end_dim + 1 :],
        )
        return self.reshape(shape)

    def isnan(self):
        return np.isnan(self)

    def isfinite(self):
        return np.isfinite(self)

    def is_floating_point(self):
        return np.issubdtype(self.dtype, np.floating)

    def numel(self):
        return self.size


class DatasetView:
    def __init__(self, rows, format_type=None, columns=None):
        self.rows = rows
        self.format_type = format_type
        self.columns = columns

    def with_format(self, format_type, columns=None, output_all_columns=False):
        return DatasetView(self.rows, format_type, columns)

    def select(self, indices):
        return DatasetView(
            [self.rows[index] for index in indices], self.format_type, self.columns
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, str):
            return [row[index] for row in self.rows]
        row = self.rows[index]
        if self.format_type is None:
            return row
        columns = set(row) if self.columns is None else set(self.columns)
        return {
            key: Tensor(value) if key in {"input_ids", "loss_mask"} else value
            for key, value in row.items()
            if key in columns
        }


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *[
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names
            ],
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 -- Execute selected local repository definitions only.
        compile(ast.fix_missing_locations(module), str(path), "exec"), namespace
    )


class ArrowHSPrefixTests(unittest.TestCase):
    def setUp(self):
        self.rows = [self.make_row(5188, offset=index * 10) for index in range(4)]
        self.source = DatasetView(self.rows)
        self.loader = Mock(return_value=self.source)
        self.transfer = SimpleNamespace(
            get_cached=Mock(return_value=None),
            get_generated=Mock(),
            cache=Mock(),
            delete=Mock(),
            setup=Mock(),
        )
        self.generated_override = None
        self.generator = Mock(side_effect=self.generate)
        self.namespace = {
            "Dataset": object,
            "load_from_disk": self.loader,
            "FileTransfer": Mock(return_value=self.transfer),
            "Path": Path,
            "torch": SimpleNamespace(
                Tensor=Tensor,
                bfloat16="bf16",
                long=np.int64,
                bool=np.bool_,
                tensor=Tensor,
                arange=lambda size, dtype=None: Tensor(np.arange(size), dtype),
                equal=np.array_equal,
                isfinite=np.isfinite,
            ),
            "DEFAULT_REQUEST_TIMEOUT": 120,
            "DEFAULT_MAX_RETRIES": 2,
            "generate_hidden_states": self.generator,
            "InvalidResponseError": type("InvalidResponseError", (Exception,), {}),
            "warnings": warnings,
            "cast": cast,
        }
        load_definitions(
            ROOT / "src/speculators/data_generation/offline.py",
            {"check_hidden_states", "align_hidden_states"},
            self.namespace,
        )
        load_definitions(
            ROOT / "src/speculators/train/data.py",
            {
                "BaseDataset",
                "ArrowDataset",
                "build_client_item",
                "_has_multimodal_content",
            },
            self.namespace,
        )
        self.dataset_class = self.namespace["ArrowDataset"]

    @staticmethod
    def make_row(length, offset=0):
        return {
            "input_ids": [(index + offset) % 1000 for index in range(length)],
            "loss_mask": [int(index % 3 != 1) for index in range(length)],
            "seq_len": length,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "unused"}]}
            ],
        }

    @staticmethod
    def packet(tokens):
        count = len(tokens)
        return {
            "token_ids": Tensor(tokens),
            "hidden_states": Tensor(
                np.arange(count * 12, dtype=np.float32).reshape(count, 3, 4)
            ),
        }

    def generate(self, client, model, payload, **kwargs):
        packet = self.generated_override
        if packet is None:
            packet = self.packet(payload["input_ids"])
        self.transfer.get_generated.return_value = packet
        return "generated-hs-handle"

    def construct(self, *, external=True, max_len=3072, **kwargs):
        if not external:
            self.source = DatasetView(self.rows, format_type="torch")
            self.loader.return_value = self.source
        dataset = self.dataset_class(
            max_len=max_len,
            datapath="external-arrow",
            transfer=self.transfer,
            pretokenized_text_only=external,
            model="target-model",
            request_timeout=91,
            max_retries=3,
            **kwargs,
        )
        dataset.client = object()
        return dataset

    def assert_sample(self, sample, row, length):
        self.assertIsNotNone(sample)
        self.assertEqual(sample["input_ids"].tolist(), row["input_ids"][:length])
        self.assertEqual(sample["loss_mask"].tolist(), row["loss_mask"][:length])
        self.assertEqual(sample["hidden_states"].shape, (length, 8))
        self.assertEqual(sample["verifier_last_hidden_states"].shape, (length, 4))
        expected = self.packet(row["input_ids"][:length])["hidden_states"]
        np.testing.assert_array_equal(
            sample["hidden_states"], expected[:, :-1].reshape(length, 8)
        )
        np.testing.assert_array_equal(
            sample["verifier_last_hidden_states"], expected[:, -1]
        )

    def test_long_external_row_is_truncated_before_hs_request(self):
        before = copy.deepcopy(self.rows)
        dataset = self.construct()
        sample = dataset[0]
        self.assert_sample(sample, self.rows[0], 3072)
        args, kwargs = self.generator.call_args
        self.assertIs(args[0], dataset.client)
        self.assertEqual(args[1], "target-model")
        self.assertEqual(args[2], {"input_ids": self.rows[0]["input_ids"][:3072]})
        self.assertEqual(kwargs, {"timeout": 91, "max_retries": 3})
        self.assertEqual(sample["lengths"].tolist(), [3072])
        self.assertEqual(sample["position_ids"].tolist(), list(range(3072)))
        self.assertEqual(self.rows, before)
        self.assertEqual(len(dataset.data[0]["input_ids"]), 5188)

    def test_helper_returns_prefix_view_without_mutating_arrow_rows(self):
        before = copy.deepcopy(self.rows)
        dataset = self.construct()
        item = dataset._get_dataset_item(0)
        self.assertEqual(set(item), {"input_ids", "loss_mask"})
        self.assertEqual(item["input_ids"].tolist(), self.rows[0]["input_ids"][:3072])
        self.assertEqual(item["loss_mask"].tolist(), self.rows[0]["loss_mask"][:3072])
        self.assertEqual(self.rows, before)
        self.generator.assert_not_called()

    def test_external_prefix_requires_positive_max_len(self):
        for max_len in (0, -1):
            with self.subTest(max_len=max_len), self.assertRaises(ValueError):
                self.construct(max_len=max_len)
        self.generator.assert_not_called()

    def test_short_and_exact_boundary_rows_keep_their_lengths(self):
        for length in (1, 512, 3072):
            with self.subTest(length=length):
                self.rows[0] = self.make_row(length)
                dataset = self.construct()
                sample = dataset._get_raw_data(0)
                self.assert_sample(sample, self.rows[0], length)
                self.assertEqual(
                    self.generator.call_args.args[2],
                    {"input_ids": self.rows[0]["input_ids"]},
                )

    def test_direct_generation_also_uses_the_prefix_helper(self):
        dataset = self.construct()
        loaded = dataset._maybe_generate_hs(0)
        self.assertEqual(len(loaded["token_ids"]), 3072)
        self.assertEqual(loaded["hidden_states"].shape[0], 3072)
        self.assertEqual(
            self.generator.call_args.args[2]["input_ids"],
            self.rows[0]["input_ids"][:3072],
        )

    def test_default_path_keeps_original_full_request_and_raw_sample(self):
        self.rows[0].pop("messages")
        dataset = self.construct(external=False)
        sample = dataset._get_raw_data(0)
        self.assert_sample(sample, self.rows[0], 5188)
        self.assertEqual(
            self.generator.call_args.args[2], {"input_ids": self.rows[0]["input_ids"]}
        )
        self.assertEqual(len(dataset._get_dataset_item(0)["input_ids"]), 5188)

    def test_default_cached_token_mismatch_fails_closed(self):
        packet = self.packet(self.rows[0]["input_ids"])
        packet["token_ids"][0] = 9999
        self.transfer.get_cached.return_value = packet
        with self.assertRaisesRegex(ValueError, "Invalid hidden states for row 0"):
            self.construct(external=False)._get_raw_data(0)
        self.generator.assert_not_called()

    def test_on_missing_skip_never_generates_hidden_states(self):
        for external in (False, True):
            with self.subTest(external=external):
                result = self.construct(external=external, on_missing="skip")[0]
                self.assertIsNone(result)
        self.generator.assert_not_called()
        self.transfer.get_generated.assert_not_called()
        self.transfer.cache.assert_not_called()
        self.transfer.delete.assert_not_called()

    def test_long_cached_packet_reuses_only_matching_prefix(self):
        packet = self.packet(self.rows[0]["input_ids"])
        # The unused suffix does not need to match a separately encoded sample.
        packet["token_ids"][3072:] = 9999
        before_tokens = packet["token_ids"].copy()
        before_hidden = packet["hidden_states"].copy()
        self.transfer.get_cached.return_value = packet
        sample = self.construct()._get_raw_data(0)
        self.assert_sample(sample, self.rows[0], 3072)
        np.testing.assert_array_equal(packet["token_ids"], before_tokens)
        np.testing.assert_array_equal(packet["hidden_states"], before_hidden)
        self.generator.assert_not_called()
        self.transfer.cache.assert_not_called()
        self.transfer.delete.assert_not_called()

    def test_exact_prefix_cache_is_reused_without_regeneration(self):
        self.transfer.get_cached.return_value = self.packet(
            self.rows[0]["input_ids"][:3072]
        )
        sample = self.construct()._get_raw_data(0)
        self.assert_sample(sample, self.rows[0], 3072)
        self.generator.assert_not_called()

    def test_cached_packet_shorter_than_required_prefix_is_fatal(self):
        self.transfer.get_cached.return_value = self.packet(
            self.rows[0]["input_ids"][:3071]
        )
        with self.assertRaises(ValueError):
            self.construct()._get_raw_data(0)
        self.generator.assert_not_called()

    def test_cached_token_mismatch_within_required_prefix_is_fatal(self):
        for position in (0, 3071):
            with self.subTest(position=position):
                packet = self.packet(self.rows[0]["input_ids"])
                packet["token_ids"][position] = 9999
                self.transfer.get_cached.return_value = packet
                with self.assertRaises(ValueError):
                    self.construct()._get_raw_data(0)
        self.generator.assert_not_called()

    def test_cached_hs_token_length_mismatch_is_fatal_even_with_enough_prefix(self):
        for length in (3071, 5187, 5189):
            with self.subTest(hidden_length=length):
                packet = self.packet(self.rows[0]["input_ids"])
                packet["hidden_states"] = Tensor(np.zeros((length, 3, 4)))
                self.transfer.get_cached.return_value = packet
                with self.assertRaises(ValueError):
                    self.construct()._get_raw_data(0)

    def test_cached_hidden_shape_and_token_rank_must_be_valid(self):
        for shape in ((5188,), (5188, 12), (5188, 1, 4), (5188, 3, 0)):
            with self.subTest(hidden_shape=shape):
                packet = self.packet(self.rows[0]["input_ids"])
                packet["hidden_states"] = Tensor(np.zeros(shape))
                self.transfer.get_cached.return_value = packet
                with self.assertRaises(ValueError):
                    self.construct()._get_raw_data(0)
        packet = self.packet(self.rows[0]["input_ids"])
        packet["token_ids"] = packet["token_ids"].reshape(1, -1)
        self.transfer.get_cached.return_value = packet
        with self.assertRaises(ValueError):
            self.construct()._get_raw_data(0)

    def test_invalid_generated_hs_is_not_swallowed_as_a_skipped_row(self):
        for failure in ("token", "short", "shape"):
            with self.subTest(failure=failure):
                packet = self.packet(self.rows[0]["input_ids"][:3072])
                if failure == "token":
                    packet["token_ids"][0] = 9999
                elif failure == "short":
                    packet["hidden_states"] = packet["hidden_states"][:-1]
                else:
                    packet["hidden_states"] = packet["hidden_states"].reshape(3072, 12)
                self.generated_override = packet
                with self.assertRaises(ValueError):
                    self.construct()._get_raw_data(0)
                self.transfer.cache.assert_not_called()
                self.transfer.delete.assert_not_called()

    def test_fresh_response_must_be_exact_not_a_longer_matching_prefix(self):
        self.generated_override = self.packet(self.rows[0]["input_ids"])
        for policy in ("cache", "delete"):
            with (
                self.subTest(policy=policy),
                self.assertRaisesRegex(ValueError, "3072"),
            ):
                self.construct(on_generate=policy)._maybe_generate_hs(0)
        self.transfer.cache.assert_not_called()
        self.transfer.delete.assert_not_called()

    def test_cache_mismatch_error_names_original_split_row_and_required_length(self):
        packet = self.packet(self.rows[3]["input_ids"])
        packet["token_ids"][0] = 9999
        self.transfer.get_cached.return_value = packet
        with self.assertRaisesRegex(ValueError, r"row 3.*3072"):
            self.construct(split_ratio=-0.5)._get_raw_data(1)

    def test_generated_cache_and_delete_use_original_split_file_index(self):
        before = copy.deepcopy(self.rows)
        for split, local_index, original_index in ((0.5, 1, 1), (-0.5, 1, 3)):
            for policy in ("cache", "delete"):
                with self.subTest(split=split, policy=policy):
                    self.transfer.get_cached.reset_mock()
                    self.transfer.cache.reset_mock()
                    self.transfer.delete.reset_mock()
                    dataset = self.construct(split_ratio=split, on_generate=policy)
                    sample = dataset._get_raw_data(local_index)
                    self.assert_sample(sample, self.rows[original_index], 3072)
                    self.transfer.get_cached.assert_called_once_with(original_index)
                    self.assertEqual(
                        self.generator.call_args.args[2]["input_ids"],
                        self.rows[original_index]["input_ids"][:3072],
                    )
                    if policy == "cache":
                        self.transfer.cache.assert_called_once_with(
                            "generated-hs-handle", original_index
                        )
                        self.transfer.delete.assert_not_called()
                    else:
                        self.transfer.delete.assert_called_once_with(
                            "generated-hs-handle"
                        )
                        self.transfer.cache.assert_not_called()
        self.assertEqual(self.rows, before)

    def test_cached_validation_split_uses_original_file_index(self):
        self.transfer.get_cached.return_value = self.packet(self.rows[2]["input_ids"])
        sample = self.construct(split_ratio=-0.5)._get_raw_data(0)
        self.assert_sample(sample, self.rows[2], 3072)
        self.transfer.get_cached.assert_called_once_with(2)
        self.generator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
