"""Real CPU page storage tests, including padding/scales/compressor rollback."""

# ruff: noqa: INP001

from types import SimpleNamespace

import pytest
import torch

from speculators_dsv4.kv_snapshots import KVSnapshotStore


def _spec(name, **values):
    result = type(name, (), {})()
    vars(result).update(values)
    return result


def _fixture(*, budget=4096, entries=64):
    blocks, width = 16, 32
    # First/last bytes are sentinels: cache views need not begin at storage zero.
    raw = torch.arange(8 + blocks * width + 8, dtype=torch.int64).to(torch.uint8)
    data = raw.as_strided((blocks, 8), (width, 1), storage_offset=8)
    scales = raw.view(torch.float32).as_strided(
        (blocks, 1), (width // 4, 1), storage_offset=4
    )
    state = torch.arange(blocks * 8, dtype=torch.float32).reshape(blocks, 8)
    hidden = torch.ones(blocks, 16)
    groups = [
        SimpleNamespace(
            layer_names=["kv"],
            kv_cache_spec=_spec(
                "UniformTypeKVCacheSpecs",
                kv_cache_specs={
                    "kv": _spec("AscendMLAAttentionSpec", page_size_bytes=width)
                },
            ),
        ),
        SimpleNamespace(
            layer_names=["compressor"],
            kv_cache_spec=_spec("AscendSlidingWindowMLASpec", page_size_bytes=32),
        ),
        SimpleNamespace(
            layer_names=["hs"], kv_cache_spec=_spec("HiddenStateCacheSpec")
        ),
    ]
    caches = {"kv": [data, scales], "compressor": [state], "hs": hidden}
    config = SimpleNamespace(num_blocks=blocks, kv_cache_groups=groups)
    store = KVSnapshotStore(caches, config, max_bytes=budget, max_entries=entries)
    return SimpleNamespace(
        raw=raw, state=state, hidden=hidden, store=store, caches=caches, config=config
    )


def test_remaps_full_pages_with_padding_scales_and_nonzero_storage_offset():
    f = _fixture()
    before = f.raw.clone()
    state = f.state.clone()
    assert f.store.save("base", [1, 2, 3], ((1, 3), (None, 5), (9,)))
    # Overwrite original pages, as vLLM may do after finishing the HTTP request.
    f.raw[8:] = 77
    f.state.zero_()
    f.hidden.fill_(99)
    f.store.restore("base", [1, 2, 3, 4], 3, ((7, 8, 10), (None, 11, 12), (13,)))
    for source, destination in ((1, 7), (3, 8)):
        assert torch.equal(
            f.raw[8 + destination * 32 : 8 + (destination + 1) * 32],
            before[8 + source * 32 : 8 + (source + 1) * 32],
        )
    assert torch.equal(f.state[11], state[5])
    assert not f.state[0].any()  # The null block is never written.
    assert not f.state[12].any()  # New, not-yet-computed page is untouched.
    assert f.hidden.eq(99).all()  # HS cache is write-only, not target state.
    assert torch.equal(f.raw[:8], before[:8])
    assert f.store.size == 96


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("accepted", [0, 1, 3, 7])
def test_rejected_trial_does_not_mutate_confirmed_compressor_snapshot(ratio, accepted):
    f = _fixture()

    # Stateful recurrence simulates overlapping/pending compressed groups. Just
    # cropping K/V would leave these accumulators corrupted after a rejection.
    def forward(tokens, position, page):
        values = []
        for token in tokens:
            f.state[page, 0] += token
            f.state[page, 1] = f.state[page, 1] * 0.5 + token
            if (position + 1) % ratio == 0:
                f.state[page, 2] += f.state[page, 0]
                f.state[page, 0] = 0
            values.append(f.state[page].clone())
            position += 1
        return values

    f.state.zero_()
    prefix = [1] * (ratio - 1)
    proposal = [2, 3, 4, 1, 2, 3, 4, 2]
    forward(prefix, 0, 1)
    f.store.save("base", prefix, ((1,), (1,), (1,)))
    forward(proposal, len(prefix), 1)
    f.store.save("trial", prefix + proposal, ((1,), (1,), (1,)))
    committed = prefix + proposal[: accepted + 1]
    f.store.restore("base", committed, len(prefix), ((2,), (2,), (2,)))
    forward(proposal[: accepted + 1], len(prefix), 2)
    f.store.save("commit", committed, ((2,), (2,), (2,)))
    f.store.restore("commit", committed + [4], len(committed), ((3,), (3,), (3,)))
    actual = forward([4], len(committed), 3)[-1]
    f.state[4].zero_()
    expected = forward(committed + [4], 0, 4)[-1]
    assert torch.equal(actual, expected)


def test_budget_eviction_oversize_release_and_immutability():
    f = _fixture(budget=128, entries=2)
    tables = ((1,), (2,), (3,))
    for key in ("a", "b"):
        assert f.store.save(key, [1], tables)
    f.store.restore("a", [1, 2], 1, tables)
    assert f.store.save("c", [1], tables)
    assert list(f.store.entries) == ["a", "c"]
    assert not f.store.save("oversized", [1], ((1, 2, 3, 4, 5), (2,), (3,)))
    assert list(f.store.entries) == ["a", "c"]
    with pytest.raises(ValueError, match="immutable"):
        f.store.save("c", [1], tables)
    f.store.release(["a", "c", "absent"])
    assert f.store.size == 0
    assert not f.store.entries


@pytest.mark.parametrize(
    "tables",
    [
        ((1, 1), (2,), (3,)),
        ((-1,), (2,), (3,)),
        ((16,), (2,), (3,)),
        ((True,), (2,), (3,)),
        ((1,),),
    ],
)
def test_rejects_bad_physical_tables(tables):
    with pytest.raises(ValueError, match="block IDs|group count"):
        _fixture().store.save("x", [1], tables)


def test_missing_prefix_and_pruned_page_fail_closed():
    f = _fixture()
    f.store.save("a", [1], ((None, 2), (3,), (4,)))
    with pytest.raises(ValueError, match="different prefix"):
        f.store.restore("a", [2, 3], 1, ((None, 2), (3,), (4,)))
    with pytest.raises(ValueError, match="pruned"):
        f.store.restore("a", [1, 2], 1, ((1, 2), (3,), (4,)))
    with pytest.raises(ValueError, match="absent"):
        f.store.restore("missing", [1, 2], 1, ((1, 2), (3,), (4,)))


def test_unknown_or_nonpage_layout_rejected_at_startup():
    f = _fixture()
    f.caches["kv"] = [torch.zeros(16, 9)]
    with pytest.raises(ValueError, match="strides"):
        KVSnapshotStore(f.caches, f.config, max_bytes=4096)
    f = _fixture()
    f.config.kv_cache_groups[0].kv_cache_spec.kv_cache_specs["kv"] = _spec(
        "UnknownSpec", page_size_bytes=32
    )
    with pytest.raises(ValueError, match="native paged"):
        KVSnapshotStore(f.caches, f.config, max_bytes=4096)
