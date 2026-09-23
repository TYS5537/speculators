"""Run real Arrow/loader entry points with stdlib-only dependency substitutes."""

# ruff: noqa: PT009, PT027 -- Keep the suite runnable without pytest/torch.

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


class TensorValue:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class DatasetView:
    """Model format-only views; underlying rows retain their source identity."""

    def __init__(self, rows, format_type=None, columns=None, output_all_columns=False):
        self.rows = rows
        self.format_type = format_type
        self.columns = columns
        self.output_all_columns = output_all_columns
        self.format_calls = []

    def with_format(self, format_type, columns=None, output_all_columns=False):
        self.format_calls.append((format_type, columns, output_all_columns))
        return DatasetView(self.rows, format_type, columns, output_all_columns)

    def select(self, indices):
        return DatasetView(
            [self.rows[index] for index in indices],
            self.format_type,
            self.columns,
            self.output_all_columns,
        )

    def __getitem__(self, index):
        if isinstance(index, str):
            return [row[index] for row in self.rows]
        row = self.rows[index]
        if self.format_type is None:
            return row
        columns = set(row) if self.columns is None else set(self.columns)
        return {
            key: TensorValue(value)
            if key in columns and key in {"input_ids", "loss_mask"}
            else value
            for key, value in row.items()
            if key in columns or self.output_all_columns
        }

    def __len__(self):
        return len(self.rows)


class DatasetBase:
    def __init__(self, max_len, transform=None, hidden_states_dtype=None):
        self.max_len = max_len
        self.transform = transform
        self.hidden_states_dtype = hidden_states_dtype
        self.approx_lengths = self._compute_approx_lengths()


def load_entry_points(source, names, namespace, *, class_methods=None):
    """Execute only named definitions from this checkout, without heavy imports."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    for node in definitions:
        if isinstance(node, ast.ClassDef) and class_methods is not None:
            node.body = [
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef) and method.name in class_methods
            ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 -- Only named local definitions, not external data/code.
        compile(ast.fix_missing_locations(module), str(source), "exec"), namespace
    )
    return namespace


class ArrowTextOnlyTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "input_ids": [0, 10 + index, 20, 30],
                "loss_mask": [0, 1, 0, 1],
                "seq_len": 4,
                # Even text-only content lists trigger the existing Chat path.
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hello"}]}
                ],
            }
            for index in range(4)
        ]
        self.source = DatasetView(self.rows)
        self.loader = Mock(return_value=self.source)
        self.namespace = {
            "BaseDataset": DatasetBase,
            "load_from_disk": self.loader,
            "FileTransfer": lambda path: SimpleNamespace(path=path),
            "Path": Path,
            "torch": SimpleNamespace(bfloat16="bf16"),
            "DEFAULT_REQUEST_TIMEOUT": 120,
            "DEFAULT_MAX_RETRIES": 2,
            "cast": cast,
        }
        load_entry_points(
            ROOT / "src/speculators/train/data.py",
            {"ArrowDataset", "build_client_item", "_has_multimodal_content"},
            self.namespace,
            class_methods={
                "__init__",
                "_select_split",
                "__len__",
                "_map_to_file_idx",
                "_compute_approx_lengths",
            },
        )
        self.dataset_class = self.namespace["ArrowDataset"]

    def construct(self, **kwargs):
        return self.dataset_class(max_len=16, datapath="external-arrow", **kwargs)

    def test_external_plain_lists_get_tensor_view_without_changing_source(self):
        before = copy.deepcopy(self.rows)
        dataset = self.construct(pretokenized_text_only=True)
        self.assertIsNot(dataset.data, self.source)
        self.assertIs(dataset.data.rows, self.source.rows)
        self.assertEqual(self.rows, before)
        self.assertIsNone(self.source.format_type)
        self.assertEqual(
            self.source.format_calls,
            [("torch", ["input_ids", "loss_mask"], False)],
        )
        self.assertEqual(dataset.approx_lengths, [4, 4, 4, 4])
        for index in range(4):
            row = dataset.data[index]
            self.assertEqual(set(row), {"input_ids", "loss_mask"})
            self.assertEqual(row["input_ids"].tolist(), before[index]["input_ids"])
            self.assertEqual(row["loss_mask"].tolist(), before[index]["loss_mask"])

    def test_external_messages_are_not_forwarded_to_retokenizing_chat_api(self):
        dataset = self.construct(pretokenized_text_only=True)
        payload = self.namespace["build_client_item"](dataset.data[0])
        self.assertEqual(payload, {"input_ids": self.rows[0]["input_ids"]})

    def test_split_keeps_original_row_indices_and_sequence_content(self):
        train = self.construct(pretokenized_text_only=True, split_ratio=0.5)
        validation = self.construct(pretokenized_text_only=True, split_ratio=-0.5)
        self.assertEqual([len(train), len(validation)], [2, 2])
        for dataset, expected_indices in ((train, [0, 1]), (validation, [2, 3])):
            self.assertEqual(dataset.approx_lengths, [4, 4])
            for index, original_index in enumerate(expected_indices):
                self.assertEqual(dataset._map_to_file_idx(index), original_index)
                row = dataset.data[index]
                self.assertEqual(
                    row["input_ids"].tolist(), self.rows[original_index]["input_ids"]
                )
                self.assertEqual(
                    row["loss_mask"].tolist(), self.rows[original_index]["loss_mask"]
                )

    def test_default_does_not_change_saved_format_or_forwarded_messages(self):
        dataset = self.construct()
        self.assertIs(dataset.data, self.source)
        self.assertIsInstance(dataset.data[0]["input_ids"], list)
        self.assertEqual(self.source.format_calls, [(None, None, False)])

        saved_torch = DatasetView(self.rows, format_type="torch")
        self.loader.return_value = saved_torch
        dataset = self.construct()
        self.assertIs(dataset.data, saved_torch)
        payload = self.namespace["build_client_item"](dataset.data[0])
        self.assertEqual(payload["messages"], self.rows[0]["messages"])

    def load_train_val_function(self):
        self.namespace.update(
            {
                "AddUniformNoise": lambda std: SimpleNamespace(std=std),
                "_setup_dataloader": Mock(side_effect=lambda dataset, *a, **k: dataset),
            }
        )
        load_entry_points(
            ROOT / "src/speculators/train/dataloader.py",
            {"create_train_val_loaders"},
            self.namespace,
        )
        return self.namespace["create_train_val_loaders"]

    def loader_args(self):
        return {
            "data_path": "external-arrow",
            "total_seq_len": 16,
            "hidden_states_dtype": "bf16",
            "noise_std": 0,
            "legacy_data": False,
            "vllm_endpoint": "http://localhost:8001/v1",
            "on_missing": "generate",
            "on_generate": "delete",
            "verifier_name_or_path": "target",
            "request_timeout": 120,
            "max_retries": 2,
            "hidden_size": 4096,
            "num_target_layers": 5,
            "num_workers": 0,
            "prefetch_factor": 1,
            "preprocess": None,
            "train_data_ratio": 0.5,
        }

    def test_loader_forwards_text_only_view_to_train_and_validation(self):
        create_loaders = self.load_train_val_function()
        train, validation = create_loaders(
            **self.loader_args(), pretokenized_text_only=True
        )
        self.assertEqual(train.data.format_type, "torch")
        self.assertEqual(validation.data.format_type, "torch")
        self.assertEqual(train.data.columns, ["input_ids", "loss_mask"])
        self.assertEqual(validation.data.columns, ["input_ids", "loss_mask"])
        self.assertEqual(validation._map_to_file_idx(0), 2)

    def test_loader_default_preserves_format_on_both_splits(self):
        create_loaders = self.load_train_val_function()
        train, validation = create_loaders(**self.loader_args())
        self.assertIsNone(train.data.format_type)
        self.assertIsNone(validation.data.format_type)

    def test_legacy_data_with_text_only_flag_is_rejected(self):
        create_loaders = self.load_train_val_function()
        args = {**self.loader_args(), "legacy_data": True}
        with self.assertRaisesRegex(ValueError, "requires Arrow"):
            create_loaders(**args, pretokenized_text_only=True)
        self.loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
