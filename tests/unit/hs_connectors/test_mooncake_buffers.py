from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch
from hs_connectors.mooncake_buffers import HostBufferPool


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(HostBufferPool, "_allocate", staticmethod(torch.empty_like))
    return HostBufferPool(48)


def test_pool_reuses_only_returned_matching_buffers(pool):
    reference = torch.zeros(2, 3)
    first = pool.acquire(reference)
    second = pool.acquire(reference)
    assert first.tensor.data_ptr() != second.tensor.data_ptr()
    pool.release(first, reuse=True)
    reused = pool.acquire(reference)
    assert reused is first
    pool.release(second, reuse=False)
    pool.release(reused, reuse=True)
    assert pool.reserved_bytes == 24


def test_concurrent_pool_budget_covers_live_and_cached_buffers(pool):
    barrier = Barrier(4)

    def borrow(_):
        lease = pool.acquire(torch.zeros(2, 3))
        barrier.wait(timeout=5)
        assert pool.reserved_bytes <= pool.max_bytes
        pooled = lease.pooled_bytes
        barrier.wait(timeout=5)
        pool.release(lease, reuse=True)
        return pooled

    with ThreadPoolExecutor(max_workers=4) as workers:
        result = list(workers.map(borrow, range(4)))
    assert sorted(result) == [0, 0, 24, 24]
    assert pool.reserved_bytes == 48


def test_idle_other_shape_is_evicted_without_reusing_wrong_layout(pool):
    first = pool.acquire(torch.zeros(2, 3))
    pool.release(first, reuse=True)
    other = pool.acquire(torch.zeros(10))
    assert other.tensor.shape == (10,)
    assert other.pooled_bytes == 40
    assert pool.reserved_bytes == 40


def test_failed_allocation_does_not_consume_budget(pool, monkeypatch):
    def fail(_):
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(pool, "_allocate", fail)
    with pytest.raises(RuntimeError, match="allocation failed"):
        pool.acquire(torch.zeros(2, 3))
    assert pool.reserved_bytes == 0


def test_quarantined_buffer_is_retained_and_never_reused(pool):
    first = pool.acquire(torch.zeros(2, 3))
    pool.quarantine(first)
    second = pool.acquire(torch.zeros(2, 3))
    assert second.tensor.data_ptr() != first.tensor.data_ptr()
    assert pool.reserved_bytes == 48
    assert pool._quarantined == [first]


def test_default_pool_does_not_retain_buffers():
    pool = HostBufferPool()
    lease = pool.acquire(torch.zeros(2, 3))
    pool.release(lease, reuse=True)
    assert lease.pooled_bytes == 0
    assert pool.reserved_bytes == 0
    assert pool._available == []


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_pool_rejects_invalid_limits(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        HostBufferPool(value)
