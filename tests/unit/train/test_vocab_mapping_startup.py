"""Vocabulary startup must not expose partial files or strand distributed peers."""

import argparse
import ast
import copy
import logging
import multiprocessing
import tempfile
import threading
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch


def _load_startup():
    root = Path(__file__).parents[3]
    module = ModuleType("vocab_startup_cpu")
    module.__dict__.update(
        argparse=argparse,
        logging=logging,
        logger=logging.getLogger("vocab-startup-test"),
        tempfile=tempfile,
        Path=Path,
        np=np,
        torch=torch,
        is_distributed=lambda: False,
        get_rank=lambda: 0,
        get_target_vocab_size=lambda *_: 8,
        get_verifier_config=lambda *_: SimpleNamespace(vocab_size=8),
    )
    for relative, names in (
        (
            "scripts/train.py",
            {
                "_load_mappings",
                "_save_vocab_mapping_atomically",
                "_parse_vocab_mappings_local",
                "parse_vocab_mappings",
            },
        ),
        (
            "src/speculators/train/vocab_mapping.py",
            {"build_vocab_mappings_from_distribution"},
        ),
    ):
        path = root / relative
        definitions = [
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        exec(  # noqa: S102 -- Run actual startup without importing model backends.
            compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
            module.__dict__,
        )
    return module


@pytest.fixture
def startup():
    return _load_startup()


def _args(tmp_path, **overrides):
    values = {
        "data_path": str(tmp_path),
        "d2t_path": None,
        "t2d_path": None,
        "token_freq_path": None,
        "draft_vocab_size": 3,
        "verifier_name_or_path": "mock-verifier",
    }
    return argparse.Namespace(**(values | overrides))


def test_first_local_startup_publishes_complete_maps_and_reuses_them(startup, tmp_path):
    torch.save({0: 4, 2: 3, 4: 2}, tmp_path / "token_freq.pt")
    first = startup.parse_vocab_mappings(_args(tmp_path))
    assert first[2] == 3
    assert first[0].tolist() == [0, 1, 2]
    assert first[1].nonzero().flatten().tolist() == [0, 2, 4]
    np.testing.assert_array_equal(np.load(tmp_path / "d2t.npy"), first[0].numpy())
    np.testing.assert_array_equal(np.load(tmp_path / "t2d.npy"), first[1].numpy())
    startup.build_vocab_mappings_from_distribution = Mock(
        side_effect=AssertionError("existing maps must not be regenerated")
    )
    second = startup.parse_vocab_mappings(_args(tmp_path))
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "d2t.npy",
        "t2d.npy",
        "token_freq.pt",
    ]


@pytest.mark.parametrize("existing", [False, True])
def test_atomic_publish_hides_partial_numpy_write(
    startup, tmp_path, monkeypatch, existing
):
    path = tmp_path / "d2t.npy"
    old = np.array([11, 12], dtype=np.int64)
    new = np.array([21, 22], dtype=np.int64)
    if existing:
        np.save(path, old)
    opened, release = threading.Event(), threading.Event()
    original_save = np.save
    errors = []

    def paused_save(stream, values):
        # The staging file is open, but its header and payload are not written.
        opened.set()
        if not release.wait(10):
            raise TimeoutError("test reader did not release writer")
        return original_save(stream, values)

    def writer():
        try:
            startup._save_vocab_mapping_atomically(path, new)
        except Exception as exc:  # noqa: BLE001 -- Propagate thread failures below.
            errors.append(exc)

    monkeypatch.setattr(np, "save", paused_save)
    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert opened.wait(10)
        if existing:
            np.testing.assert_array_equal(np.load(path), old)
        else:
            assert not path.exists()
    finally:
        release.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert errors == []
    np.testing.assert_array_equal(np.load(path), new)
    assert [p.name for p in tmp_path.iterdir()] == ["d2t.npy"]


def test_failed_numpy_write_preserves_old_cache_and_removes_staging_file(
    startup, tmp_path, monkeypatch
):
    path = tmp_path / "d2t.npy"
    original = np.array([4, 5], dtype=np.int64)
    np.save(path, original)
    monkeypatch.setattr(np, "save", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        startup._save_vocab_mapping_atomically(path, np.array([9, 10]))
    np.testing.assert_array_equal(np.load(path), original)
    assert [p.name for p in tmp_path.iterdir()] == ["d2t.npy"]


@pytest.mark.parametrize("source", ["generated", "existing", "explicit", "full"])
def test_rank_zero_broadcasts_result_without_peer_filesystem_reads(
    startup, tmp_path, monkeypatch, source
):
    args = _args(tmp_path)
    if source == "generated":
        torch.save({0: 4, 2: 3, 4: 2}, tmp_path / "token_freq.pt")
    elif source in {"existing", "explicit"}:
        d2t_path, t2d_path = tmp_path / "d2t.npy", tmp_path / "t2d.npy"
        np.save(d2t_path, np.array([0, 1, 2], dtype=np.int64))
        np.save(t2d_path, np.array([1, 0, 1, 0, 1, 0, 0, 0], dtype=np.bool_))
        if source == "explicit":
            args.d2t_path, args.t2d_path = str(d2t_path), str(t2d_path)

    startup.is_distributed = lambda: True
    payloads = []

    def broadcast(items, src):
        assert src == 0
        if startup.get_rank() == 0:
            payloads.append(copy.deepcopy(items))
        else:
            items[:] = copy.deepcopy(payloads[0])

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    first = startup.parse_vocab_mappings(args)
    startup.get_rank = lambda: 1
    startup._parse_vocab_mappings_local = Mock(
        side_effect=AssertionError("nonzero rank must not read or write mapping files")
    )
    other = startup.parse_vocab_mappings(args)
    assert len(payloads) == 1
    assert first[2] == other[2]
    for left, right in zip(first[:2], other[:2], strict=True):
        torch.testing.assert_close(left, right)
    startup._parse_vocab_mappings_local.assert_not_called()


def test_rank_zero_failure_is_reported_to_every_rank(startup, tmp_path, monkeypatch):
    startup.is_distributed = lambda: True
    startup._parse_vocab_mappings_local = Mock(side_effect=OSError("disk full"))
    stored = []

    def broadcast(items, src):
        assert src == 0
        if startup.get_rank() == 0:
            stored[:] = copy.deepcopy(items)
        else:
            items[:] = copy.deepcopy(stored)

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    for rank in (0, 1, 2):
        startup.get_rank = lambda rank=rank: rank
        with pytest.raises(ValueError, match="rank 0: OSError: disk full"):
            startup.parse_vocab_mappings(_args(tmp_path))
    assert startup._parse_vocab_mappings_local.call_count == 1


def test_single_rank_keeps_original_validation_error(startup, tmp_path):
    with pytest.raises(ValueError, match="Both t2d and d2t"):
        startup.parse_vocab_mappings(_args(tmp_path, d2t_path="only-one.npy"))


def _distributed_vocab_worker(rank, directory, failure, results):
    root = Path(directory)
    torch.distributed.init_process_group(
        "gloo",
        init_method=(root / "gloo-store").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        module = _load_startup()
        module.is_distributed = lambda: True
        module.get_rank = lambda: rank
        if rank != 0:
            module._parse_vocab_mappings_local = Mock(
                side_effect=AssertionError("peer performed mapping I/O")
            )
        args = _args(root if rank == 0 else root / "not-visible-to-this-rank")
        if failure:
            args.d2t_path = "only-one.npy"
        try:
            d2t, t2d, size = module.parse_vocab_mappings(args)
            results.put((rank, "ok", d2t.tolist(), t2d.tolist(), size))
        except ValueError as exc:
            results.put((rank, "error", str(exc)))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="CPU Gloo is unavailable"
)
@pytest.mark.parametrize("failure", [False, True])
def test_real_two_rank_mapping_startup_finishes_on_success_and_failure(
    tmp_path, failure
):
    torch.save({0: 4, 2: 3, 4: 2}, tmp_path / "token_freq.pt")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_distributed_vocab_worker,
            args=(rank, str(tmp_path), failure, results),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        received = sorted(results.get(timeout=40) for _ in processes)
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
        assert received[0][1:] == received[1][1:]
        if failure:
            assert received[0][1] == "error"
            assert "Both t2d and d2t" in received[0][2]
        else:
            assert received[0][1:] == (
                "ok",
                [0, 1, 2],
                [True, False, True, False, True, False, False, False],
                3,
            )
            assert (tmp_path / "d2t.npy").exists()
            assert (tmp_path / "t2d.npy").exists()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        results.close()
