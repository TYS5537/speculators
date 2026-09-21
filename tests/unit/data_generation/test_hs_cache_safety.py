"""Invalid hidden states must never become a reusable training cache."""

import ast
import asyncio
import importlib.util
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).parents[3]


@pytest.fixture
def offline():
    path = ROOT / "src/speculators/data_generation/offline.py"
    spec = importlib.util.spec_from_file_location("hs_cache_offline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _payload():
    return {
        "token_ids": torch.tensor([1, 2]),
        "hidden_states": torch.ones(2, 3, 4, dtype=torch.bfloat16),
    }


@pytest.mark.parametrize("bad_value", [torch.nan, torch.inf, -torch.inf])
def test_nonfinite_states_are_rejected(offline, bad_value):
    data = _payload()
    data["hidden_states"][0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="NaN/Inf"):
        offline.check_hidden_states(data, [1, 2])


@pytest.mark.parametrize("shape", [(2, 4), (2, 0, 4), (2, 3, 0), (1, 3, 4)])
def test_invalid_hidden_shapes_are_rejected(offline, shape):
    data = _payload()
    data["hidden_states"] = torch.ones(shape)
    with pytest.raises(ValueError, match="shape|Sequence length"):
        offline.check_hidden_states(data, [1, 2])


def test_teacher_only_payload_remains_valid(offline):
    data = _payload()
    data["hidden_states"] = data["hidden_states"][:, -1:]
    offline.check_hidden_states(data, [1, 2])


def test_existing_bad_cache_is_preserved_and_reenters_generation(offline, tmp_path):
    data = _payload()
    data["hidden_states"][0, 0, 0] = torch.nan
    bad = tmp_path / "hs_0.safetensors"
    save_file(data, bad)
    original_bytes = bad.read_bytes()
    good = tmp_path / "hs_1.safetensors"
    save_file(_payload(), good)
    good_bytes = good.read_bytes()
    dataset = [{"input_ids": torch.tensor([1, 2])}] * 2

    valid = offline.validate_existing_hidden_states(tmp_path, dataset, [0, 1])

    assert valid == [1]
    assert not bad.exists()
    quarantined = list(tmp_path.glob("hs_0.safetensors.invalid-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == original_bytes
    assert good.read_bytes() == good_bytes
    existing = offline.get_existing_hidden_state_indices(tmp_path)
    assert existing == [1]
    assert offline.get_indices_to_process(2, None, existing, 1, 0) == [0]


def test_only_requested_cache_rows_are_inspected(offline, tmp_path):
    # Another rank's file can be invalid without this rank touching it.
    untouched = tmp_path / "hs_1.safetensors"
    untouched.write_bytes(b"other rank's incomplete cache")
    save_file(_payload(), tmp_path / "hs_0.safetensors")
    valid = offline.validate_existing_hidden_states(
        tmp_path, [{"input_ids": torch.tensor([1, 2])}], [0]
    )
    assert valid == [0]
    assert untouched.read_bytes() == b"other rank's incomplete cache"
    assert not list(tmp_path.glob("*.invalid-*"))


def test_corrupt_safetensors_are_quarantined_not_deleted(offline, tmp_path):
    bad = tmp_path / "hs_0.safetensors"
    bad.write_bytes(b"incomplete safetensors")
    assert (
        offline.validate_existing_hidden_states(
            tmp_path, [{"input_ids": torch.tensor([1, 2])}], [0]
        )
        == []
    )
    quarantined = next(tmp_path.glob("*.invalid-*"))
    assert quarantined.read_bytes() == b"incomplete safetensors"


def _worker(offline, source, moves, *, interrupt_copy=False):
    path = ROOT / "scripts/data_generation_offline.py"
    methods = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name in ("worker", "_publish_hidden_states")
    ]

    async def generate(*_args, **_kwargs):
        return str(source)

    def move(src, dst):
        assert dst.name.startswith(".")
        assert dst.suffix == ".tmp"
        assert not (dst.parent / "hs_0.safetensors").exists()
        moves.append((src, dst))
        if interrupt_copy:
            dst.write_bytes(b"partial interrupted copy")
            raise OSError("Copy interrupted")
        return shutil.move(src, dst)

    namespace = {
        "asyncio": asyncio,
        "uuid4": uuid4,
        "Path": Path,
        "shutil": SimpleNamespace(move=move),
        "logger": logging.getLogger("test-cache-worker"),
        "load_file": load_file,
        "check_hidden_states": offline.check_hidden_states,
        "generate_hidden_states_async": generate,
    }
    module = ast.parse("from __future__ import annotations")
    module.body.extend(methods)
    exec(  # noqa: S102 -- Run the actual worker without network/client imports.
        compile(module, str(path), "exec"), namespace
    )
    return namespace["worker"]


@pytest.mark.parametrize(
    ("valid", "interrupt_copy"), [(False, False), (True, False), (True, True)]
)
def test_worker_validates_before_publishing(offline, tmp_path, valid, interrupt_copy):
    source = tmp_path / "response.safetensors"
    data = _payload()
    if not valid:
        data["hidden_states"][0, 0, 0] = torch.nan
    save_file(data, source)
    destination = tmp_path / "cache"
    destination.mkdir()
    moves, skipped = [], []
    worker = _worker(offline, source, moves, interrupt_copy=interrupt_copy)

    async def run():
        queue = asyncio.Queue()
        queue.put_nowait({"idx": 0, "input_ids": [1, 2]})
        queue.put_nowait(None)
        await worker(
            None,
            "target",
            queue,
            SimpleNamespace(update=lambda *_args: None),
            asyncio.Semaphore(1),
            asyncio.Semaphore(1),
            destination,
            True,
            None,
            0,
            False,
            skipped,
            asyncio.Event(),
            None,
        )
        await queue.join()

    asyncio.run(run())
    published = valid and not interrupt_copy
    assert skipped == ([] if published else [0])
    assert bool(moves) == valid
    assert (destination / "hs_0.safetensors").exists() == published
    assert source.exists() != published
    assert offline.get_existing_hidden_state_indices(destination) == (
        [0] if published else []
    )
    if interrupt_copy:
        assert (
            next(destination.glob("*.tmp")).read_bytes() == b"partial interrupted copy"
        )
