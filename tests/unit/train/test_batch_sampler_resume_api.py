"""Public sampler resume offsets preserve the immutable full epoch schedule."""

import numpy as np
import pytest

from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)


def _sampler(rank=0, max_batches=None):
    return MultipackDistributedBatchSamplerV2(
        batch_max_length=64,
        lengths=np.arange(1, 301) % 31 + 1,
        num_replicas=3,
        rank=rank,
        seed=19,
        max_batches=max_batches,
    )


def _rows(sampler):
    return [batch.tolist() for batch in sampler]


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("max_batches", [None, 5])
def test_resume_is_absolute_and_retains_the_full_schedule(rank, max_batches):
    sampler = _sampler(rank, max_batches)
    sampler.set_epoch(2)
    complete = _rows(sampler)
    full_cache = sampler._cached_generated_batches[1]
    sampler.set_start_batch(2, 2)
    assert _rows(sampler) == complete[2:]
    assert len(sampler) == len(complete) - 2
    assert sampler._cached_generated_batches[1] is full_cache
    sampler.set_start_batch(2, 1)
    assert _rows(sampler) == complete[1:]
    sampler.set_start_batch(2, len(complete))
    assert len(sampler) == 0
    assert _rows(sampler) == []
    sampler.set_start_batch(2, 0)
    assert _rows(sampler) == complete
    assert len(full_cache) == len(complete)


@pytest.mark.parametrize("next_epoch", [2, 3])
def test_set_epoch_resets_resume_even_for_the_same_epoch(next_epoch):
    sampler = _sampler()
    sampler.set_start_batch(2, 2)
    sampler.set_epoch(next_epoch)
    reference = _sampler()
    reference.set_epoch(next_epoch)
    assert len(sampler) == len(reference)
    assert _rows(sampler) == _rows(reference)


@pytest.mark.parametrize("index", [-1, 100000, 1.5, "2", True])
def test_invalid_offset_does_not_change_visible_schedule(index):
    sampler = _sampler()
    sampler.set_start_batch(2, 1)
    before = _rows(sampler)
    with pytest.raises((TypeError, ValueError)):
        sampler.set_start_batch(3, index)
    assert sampler.epoch == 2
    assert _rows(sampler) == before


def test_resume_does_not_consume_global_numpy_rng():
    before = np.random.get_state()  # noqa: NPY002 -- Test legacy global RNG isolation.
    sampler = _sampler()
    sampler.set_start_batch(2, np.int64(1))
    _rows(sampler)
    after = np.random.get_state()  # noqa: NPY002
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
