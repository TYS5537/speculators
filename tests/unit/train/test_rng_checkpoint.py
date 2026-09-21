"""Real CPU/Gloo RNG replay, with simulated current-device CUDA/NPU API tests."""

import copy
import importlib.util
import json
import os
import random
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

# Intentionally exercise the legacy process-global RNG used by the trainer.
# ruff: noqa: NPY002


def _load(relative):
    path = Path(__file__).parents[3] / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rng():
    py_state, np_state, cpu_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    module = _load("src/speculators/train/rng.py")
    yield module
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(cpu_state)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _draw():
    return (
        [random.random(), random.gauss(0, 1)],
        [np.random.random(), np.random.normal()],
        torch.randperm(41).tolist(),
        torch.multinomial(torch.ones(17), 5, replacement=True).tolist(),
    )


def test_json_roundtrip_restores_all_cpu_generators_and_gaussian_cache(rng):
    _seed(47)
    random.gauss(0, 1)
    np.random.normal()
    payload = json.loads(json.dumps(rng.capture_rng_states("cpu")))
    expected = [_draw() for _ in range(3)]
    _seed(999)
    assert rng.restore_rng_states(payload, "cpu")
    assert [_draw() for _ in range(3)] == expected


def test_capture_does_not_consume_random_stream(rng):
    _seed(7)
    first = rng.capture_rng_states("cpu")
    assert first == rng.capture_rng_states("cpu")
    assert rng.restore_rng_states(first, "cpu")
    assert first == rng.capture_rng_states("cpu")


def test_decoded_rng_tensors_stay_on_cpu_when_default_device_changes(rng):
    payload = rng.capture_rng_states("cpu")
    with torch.device("meta"):
        prepared = rng._prepare_local(payload["states"][0], "cpu")
    assert prepared[2].device.type == "cpu"


@pytest.mark.parametrize("incompatibility", ["legacy", "world_size", "device_type"])
def test_compatibility_fallback_warns_without_changing_state(
    rng, caplog, incompatibility
):
    payload = rng.capture_rng_states("cpu")
    before = copy.deepcopy(payload)
    if incompatibility == "legacy":
        payload = None
    elif incompatibility == "world_size":
        payload["world_size"] = 2
    else:
        payload["device_type"] = "npu"
    assert not rng.restore_rng_states(payload, "cpu")
    assert "keeping startup RNG states" in caplog.text
    assert rng.capture_rng_states("cpu") == before


@pytest.mark.parametrize(
    "corruption", ["version", "count", "python", "numpy", "torch", "device"]
)
def test_corrupt_state_does_not_partially_restore_cpu_generators(rng, corruption):
    payload = rng.capture_rng_states("cpu")
    if corruption == "version":
        payload["version"] = 9
    elif corruption == "count":
        payload["states"] = []
    elif corruption == "python":
        payload["states"][0]["python"] = [3, [], None]
    elif corruption == "numpy":
        payload["states"][0]["numpy"][0] = "invalid"
    elif corruption == "torch":
        payload["states"][0]["torch_cpu"] = "AQ=="
    else:
        payload["states"][0]["accelerator"] = "AQ=="
    _seed(511)
    before = rng.capture_rng_states("cpu")
    with pytest.raises(RuntimeError, match="Cannot restore checkpoint RNG"):
        rng.restore_rng_states(payload, "cpu")
    assert rng.capture_rng_states("cpu") == before


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_current_device_rng_api_roundtrip(rng, monkeypatch, device_type):
    current_state = torch.tensor([11, 12, 13], dtype=torch.uint8)
    backend = SimpleNamespace(
        get_rng_state=Mock(side_effect=current_state.clone),
        set_rng_state=Mock(),
    )

    def set_state(state):
        current_state.copy_(state)

    backend.set_rng_state.side_effect = set_state
    monkeypatch.setattr(rng.torch, device_type, backend, raising=False)
    payload = rng.capture_rng_states(device_type)
    current_state.fill_(22)
    assert rng.restore_rng_states(payload, device_type)
    assert current_state.tolist() == [11, 12, 13]
    assert all(call.args == () for call in backend.get_rng_state.call_args_list)
    assert backend.set_rng_state.call_args.args[0].dtype == torch.uint8
    assert backend.set_rng_state.call_args.args[0].device.type == "cpu"


def test_backend_capture_failure_is_explicit(rng, monkeypatch):
    monkeypatch.setattr(rng.torch, "npu", SimpleNamespace(), raising=False)
    with pytest.raises(RuntimeError, match="rank 0.*get_rng_state"):
        rng.capture_rng_states("npu")


def test_accelerator_apply_failure_rolls_back_cpu_states(rng, monkeypatch):
    backend = SimpleNamespace(
        get_rng_state=lambda: torch.tensor([1], dtype=torch.uint8),
        set_rng_state=Mock(side_effect=[RuntimeError("bad device RNG"), None]),
    )
    monkeypatch.setattr(rng.torch, "npu", backend, raising=False)
    payload = rng.capture_rng_states("npu")
    _seed(355)
    before = rng.capture_rng_states("cpu")
    with pytest.raises(RuntimeError, match="bad device RNG"):
        rng.restore_rng_states(payload, "npu")
    assert rng.capture_rng_states("cpu") == before
    assert backend.set_rng_state.call_count == 2


def test_real_dflash_anchor_selection_replays_after_model_initialization(rng):
    utils = _load("src/speculators/models/dflash/utils.py")
    _seed(1009)
    mask = torch.ones(1, 128)
    for _ in range(3):
        utils.select_anchors(mask, 12, 7)
    payload = rng.capture_rng_states("cpu")
    expected, expected_valid = utils.select_anchors(mask, 12, 7)
    torch.nn.Linear(32, 32)  # Model setup on resume advances RNG.
    assert rng.restore_rng_states(payload, "cpu")
    actual, actual_valid = utils.select_anchors(mask, 12, 7)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_valid, expected_valid)


def test_mid_epoch_iterator_must_be_created_before_restoring(rng):
    _seed(41)
    loader = torch.utils.data.DataLoader(torch.arange(4), batch_size=1, num_workers=0)
    iterator = iter(loader)
    next(iterator)
    payload = rng.capture_rng_states("cpu")
    expected = torch.rand(5)
    rng.restore_rng_states(payload, "cpu")
    iterator = iter(loader)
    assert not torch.equal(torch.rand(5), expected)
    rng.restore_rng_states(payload, "cpu")
    assert torch.equal(torch.rand(5), expected)


def test_two_rank_rng_roundtrip_and_asymmetric_failures(tmp_path):
    script = textwrap.dedent("""
        import sys, json, datetime, copy
        sys.path[:] = json.loads(sys.argv[1])
        import torch, torch.distributed as dist
        from tests.unit.train.test_rng_checkpoint import _load, _seed, _draw
        rank, port = int(sys.argv[2]), sys.argv[3]
        dist.init_process_group(
            'gloo', init_method='tcp://127.0.0.1:'+port, rank=rank, world_size=2,
            timeout=datetime.timedelta(seconds=20),
        )
        try:
            rng = _load('src/speculators/train/rng.py')
            _seed(71 + rank)
            payload = json.loads(json.dumps(rng.capture_rng_states('cpu')))
            assert payload['states'][0] != payload['states'][1]
            expected = _draw()
            _seed(999)
            assert rng.restore_rng_states(payload, 'cpu')
            assert _draw() == expected
            original_capture = rng._capture_local
            def fail_capture(device):
                if rank == 1:
                    raise RuntimeError('rank-one capture failure')
                return original_capture(device)
            rng._capture_local = fail_capture
            try:
                rng.capture_rng_states('cpu')
            except RuntimeError as error:
                assert 'rank-one capture failure' in str(error)
            else:
                raise AssertionError('both ranks must observe capture failure')
            rng._capture_local = original_capture
            original_apply = rng._apply_local
            count = 0
            def fail_apply(*args):
                global count
                count += 1
                if rank == 1 and count == 1:
                    raise RuntimeError('rank-one restore failure')
                return original_apply(*args)
            before = original_capture('cpu')
            rng._apply_local = fail_apply
            try:
                rng.restore_rng_states(payload, 'cpu')
            except RuntimeError as error:
                assert 'rank-one restore failure' in str(error)
            else:
                raise AssertionError('both ranks must observe apply failure')
            assert original_capture('cpu') == before
            assert count == 2
            rng._apply_local = original_apply
            corrupted = copy.deepcopy(payload)
            if rank == 1:
                corrupted['states'][1]['torch_cpu'] = 'AQ=='
            try:
                rng.restore_rng_states(corrupted, 'cpu')
            except RuntimeError as error:
                assert 'rank 1' in str(error)
            else:
                raise AssertionError('both ranks must observe validation failure')
            assert original_capture('cpu') == before
            print('passed')
        finally:
            dist.destroy_process_group()
    """)
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    processes = [
        subprocess.Popen(  # noqa: S603 -- Fixed local test script.
            [
                sys.executable,
                "-B",
                "-c",
                script,
                json.dumps(sys.path),
                str(rank),
                str(port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            output, _ = process.communicate(timeout=35)
            assert process.returncode == 0, output
            assert "passed" in output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
