"""CPU-only writer lifecycle tests with fake vLLM interfaces and copy events."""

import importlib.util
import logging
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from hs_connectors.mooncake_buffers import CopySynchronizationError, HostBufferPool


@pytest.fixture
def module(monkeypatch):
    class Base:
        pass

    class HMA:
        pass

    fixtures = {
        "vllm.config": {"VllmConfig": Base, "get_layers_from_vllm_config": Mock()},
        "vllm.distributed.kv_transfer.kv_connector.v1.base": {
            "KVConnectorBase_V1": Base,
            "KVConnectorMetadata": Base,
            "KVConnectorRole": Base,
            "SupportsHMA": HMA,
        },
        (
            "vllm.distributed.kv_transfer.kv_connector.v1."
            "example_hidden_states_connector"
        ): {
            "ExampleHiddenStatesConnector": Base,
            "extract_from_kv_cache": lambda cache, slots, length: cache[:length],
        },
        "vllm.distributed.parallel_state": {
            "get_tensor_model_parallel_rank": lambda: 0
        },
        "vllm.logger": {"init_logger": logging.getLogger},
        "vllm.v1.attention.backend": {"AttentionMetadata": Base},
        "vllm.v1.core.sched.output": {"SchedulerOutput": Base},
    }
    for name, attributes in fixtures.items():
        parts = name.split(".")
        for end in range(1, len(parts) + 1):
            prefix = ".".join(parts[:end])
            if prefix not in sys.modules:
                monkeypatch.setitem(sys.modules, prefix, ModuleType(prefix))
        for name_, value in attributes.items():
            monkeypatch.setattr(sys.modules[name], name_, value, raising=False)
    path = (
        Path(__file__).resolve().parents[3]
        / "hs_connectors/src/hs_connectors/mooncake_hidden_states_connector.py"
    )
    spec = importlib.util.spec_from_file_location("fixture_mooncake_connector", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def writer(module, monkeypatch):
    connector = module.MooncakeHiddenStatesConnector.__new__(
        module.MooncakeHiddenStatesConnector
    )
    connector._kv_cache = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    connector._block_size = 2
    connector._copy_stream = Mock()
    connector._device_module = SimpleNamespace(
        stream=lambda _: nullcontext(), Event=Mock()
    )
    connector._buffer_pool = HostBufferPool(24)
    monkeypatch.setattr(connector._buffer_pool, "_allocate", torch.empty_like)
    connector._use_copy_events = True
    connector._sample_metadata = None
    connector._store = Mock()
    connector._is_tp_rank_zero = False
    connector._req_futures = {}
    connector._accumulated_finished_req_ids = set()
    pending = module.PendingSave("request", "key", torch.arange(2), [0])
    return connector, pending


@pytest.mark.parametrize("events", [False, True])
def test_store_write_follows_copy_completion(writer, events):
    connector, pending = writer
    connector._use_copy_events = events
    order = []
    copy_event = Mock()
    connector._device_module.Event.return_value = copy_event
    copy_event.synchronize.side_effect = lambda: order.append("copied")
    connector._copy_stream.synchronize.side_effect = lambda: order.append("copied")

    def publish(_key, tensors):
        assert order == ["copied"]
        assert torch.equal(tensors["hidden_states"], connector._kv_cache)
        assert connector._buffer_pool._available == []
        order.append("published")

    connector._store.put_sample.side_effect = publish
    connector._write_sample(pending, Mock())
    assert order == ["copied", "published"]
    assert len(connector._buffer_pool._available) == 1
    if events:
        copy_event.record.assert_called_once_with(connector._copy_stream)
        connector._copy_stream.synchronize.assert_not_called()
    else:
        connector._device_module.Event.assert_not_called()


def test_store_completion_gates_buffer_reuse_and_request_release(writer):
    connector, pending = writer
    started, finish = Event(), Event()

    def publish(*_args):
        started.set()
        assert finish.wait(timeout=5)

    connector._store.put_sample.side_effect = publish
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(connector._write_sample, pending, Mock())
        connector._req_futures[pending.req_id] = future
        try:
            assert started.wait(timeout=5)
            assert connector._buffer_pool._available == []
            assert connector.get_finished({pending.req_id}) == (None, None)
        finally:
            finish.set()
        future.result(timeout=5)
    assert connector.get_finished(set()) == ({pending.req_id}, None)
    assert len(connector._buffer_pool._available) == 1


def test_failed_store_write_discards_the_completed_buffer(writer):
    connector, pending = writer
    connector._store.put_sample.side_effect = RuntimeError("store failed")
    with pytest.raises(RuntimeError, match="store failed"):
        connector._write_sample(pending, Mock())
    connector._store.put_error.assert_called_once()
    assert connector._buffer_pool._available == []
    assert connector._buffer_pool.reserved_bytes == 0


def test_failed_event_wait_drains_stream_before_discard(writer):
    connector, pending = writer
    connector._device_module.Event.return_value.synchronize.side_effect = RuntimeError(
        "event failed"
    )
    with pytest.raises(RuntimeError, match="event failed"):
        connector._write_sample(pending, Mock())
    connector._copy_stream.synchronize.assert_called_once()
    connector._store.put_sample.assert_not_called()
    assert connector._buffer_pool.reserved_bytes == 0


def test_unknown_copy_completion_quarantines_buffer_and_refuses_kv_release(writer):
    connector, pending = writer
    connector._device_module.Event.return_value.synchronize.side_effect = RuntimeError(
        "event failed"
    )
    connector._copy_stream.synchronize.side_effect = RuntimeError("device unavailable")
    with pytest.raises(CopySynchronizationError) as failed:
        connector._write_sample(pending, Mock())
    assert len(connector._buffer_pool._quarantined) == 1
    assert connector._buffer_pool.reserved_bytes == 12
    assert connector._buffer_pool._available == []
    future = Future()
    future.set_exception(failed.value)
    connector._req_futures[pending.req_id] = future
    with pytest.raises(CopySynchronizationError):
        connector.get_finished({pending.req_id})
    assert pending.req_id in connector._req_futures
