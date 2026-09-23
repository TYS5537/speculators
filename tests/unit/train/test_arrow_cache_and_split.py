"""Exercise actual Arrow read/split methods without optional data backends."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[3]


class TinyArrow:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, str):
            return [row[index] for row in self.rows]
        return self.rows[index]

    def with_format(self, *_args, **_kwargs):
        return self

    def select(self, indices):
        return TinyArrow([self.rows[i] for i in indices])


@pytest.fixture
def data_api():
    offline_path = ROOT / "src/speculators/data_generation/offline.py"
    spec = importlib.util.spec_from_file_location("arrow_cache_offline", offline_path)
    offline = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(offline)
    dataset = TinyArrow(
        [
            {
                "input_ids": torch.tensor([i, i + 1]),
                "loss_mask": torch.ones(2, dtype=torch.bool),
                "seq_len": 2,
                "row_id": i,
            }
            for i in range(10)
        ]
    )
    path = ROOT / "src/speculators/train/data.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    classes = [
        node
        for node in source.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("BaseDataset", "ArrowDataset", "_has_multimodal_content")
    ]
    namespace = {
        "torch": torch,
        "Dataset": torch.utils.data.Dataset,
        "SampleUnavailable": type("SampleUnavailable", (), {}),
        "Path": Path,
        "DEFAULT_REQUEST_TIMEOUT": 120,
        "DEFAULT_MAX_RETRIES": 0,
        "load_from_disk": lambda *_args: dataset,
        "check_hidden_states": offline.check_hidden_states,
        "align_hidden_states": offline.align_hidden_states,
    }
    module = ast.parse("from __future__ import annotations")
    module.body.extend(classes)
    exec(  # noqa: S102 -- Real dataset methods; Arrow storage is the only stub.
        compile(module, str(path), "exec"), namespace
    )
    return namespace, dataset


def _dataset(data_api, **kwargs):
    namespace, _ = data_api
    return namespace["ArrowDataset"](
        max_len=8, datapath="unused", transfer=SimpleNamespace(), **kwargs
    )


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.29, 0.3, 0.5, 0.9])
def test_named_splits_have_no_overlap_and_preserve_cache_row_ids(data_api, ratio):
    train = _dataset(data_api, split_ratio=ratio, split="train")
    valid = _dataset(data_api, split_ratio=ratio, split="validation")
    train_ids = train.data["row_id"]
    valid_ids = valid.data["row_id"]
    assert train_ids == list(range(int(10 * ratio)))
    assert not set(train_ids) & set(valid_ids)
    assert train_ids + valid_ids == list(range(10))
    assert [valid._map_to_file_idx(i) for i in range(len(valid))] == valid_ids


def test_legacy_signed_split_and_default_api_remain_compatible(data_api):
    assert len(_dataset(data_api)) == 10
    assert _dataset(data_api, split_ratio=0.9).data["row_id"] == list(range(9))
    assert _dataset(data_api, split_ratio=-0.1).data["row_id"] == [9]


@pytest.mark.parametrize("bad", ["nan", "inf", "shape", "length", "tokens"])
def test_native_cached_states_fail_closed(data_api, bad):
    model = _dataset(data_api)
    states = torch.ones(2, 3, 4)
    tokens = torch.tensor([0, 1])
    if bad in ("nan", "inf"):
        states[0, 0, 0] = float(bad)
    elif bad == "shape":
        states = torch.ones(2, 4)
    elif bad == "length":
        states = states[:1]
    elif bad == "tokens":
        tokens = torch.tensor([1, 2])
    model.transfer.get_cached = lambda _: {
        "token_ids": tokens,
        "hidden_states": states,
    }
    with pytest.raises(ValueError, match="Invalid hidden states for row 0"):
        model[0]


def test_valid_native_cache_keeps_mask_and_hidden_alignment(data_api):
    model = _dataset(data_api)
    states = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    model.transfer.get_cached = lambda _: {
        "token_ids": torch.tensor([0, 1]),
        "hidden_states": states,
    }
    row = model[0]
    torch.testing.assert_close(row["hidden_states"], states[:, :-1].flatten(1))
    torch.testing.assert_close(row["verifier_last_hidden_states"], states[:, -1])
    assert row["loss_mask"].tolist() == [True, True]
    assert row["lengths"].tolist() == [2]


def test_loader_factory_uses_identical_positive_split_ratio(data_api):
    namespace, _ = data_api
    path = ROOT / "src/speculators/train/dataloader.py"
    method = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "create_train_val_loaders"
    )
    namespace.update(
        AddUniformNoise=lambda **_kwargs: None,
        _setup_dataloader=lambda dataset, *_args, **_kwargs: dataset,
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    exec(  # noqa: S102 -- Verify the actual factory passes the shared split boundary.
        compile(module, str(path), "exec"), namespace
    )
    train, valid = namespace["create_train_val_loaders"](
        data_path="unused",
        total_seq_len=8,
        hidden_states_dtype=torch.bfloat16,
        noise_std=0,
        legacy_data=False,
        transfer=SimpleNamespace(),
        vllm_endpoint="unused",
        on_missing="raise",
        on_generate="cache",
        verifier_name_or_path="unused",
        request_timeout=120,
        max_retries=0,
        hidden_size=4,
        num_target_layers=2,
        num_workers=0,
        prefetch_factor=1,
        preprocess=None,
        train_data_ratio=0.1,
    )
    assert train.data["row_id"] == [0]
    assert valid.data["row_id"] == list(range(1, 10))
