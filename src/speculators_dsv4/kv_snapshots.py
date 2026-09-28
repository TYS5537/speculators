"""Bounded, immutable host snapshots of native Ascend V4 cache pages.

Copy the complete physical page, including compressor state, indexer scales and
padding. Never interpret/crop a compressed state or alias a speculative write.
The cache-only HS group is write-only and is deliberately not snapshotted.
"""

from collections import OrderedDict
from dataclasses import dataclass

import torch

from speculators_dsv4.block_protocol import KV_MAX_ENTRIES

_PAGE_NDIM = 2


def _page_view(components, page_bytes, num_blocks):
    """Validate the pinned runner's page-strided views before accessing bytes."""
    if isinstance(components, torch.Tensor):
        components = [components]
    if not isinstance(components, (list, tuple)) or not components:
        raise ValueError("Unsupported DSV4 KV cache components")
    first = components[0]
    if not isinstance(first, torch.Tensor) or first.ndim < _PAGE_NDIM:
        raise ValueError("Unsupported DSV4 KV page layout")
    storage = first.untyped_storage()
    base = first.storage_offset() * first.element_size()
    for tensor in components:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim < _PAGE_NDIM
            or tensor.shape[0] != num_blocks
            or tensor.stride(0) * tensor.element_size() != page_bytes
            or tensor.untyped_storage().data_ptr() != storage.data_ptr()
            or any(stride < 0 for stride in tensor.stride())
        ):
            raise ValueError("Unsupported DSV4 KV page/component strides")
        offset = tensor.storage_offset() * tensor.element_size() - base
        extent = (
            1
            + sum(
                (size - 1) * stride
                for size, stride in zip(
                    tensor.shape[1:], tensor.stride()[1:], strict=True
                )
            )
        ) * tensor.element_size()
        if offset < 0 or offset + extent > page_bytes:
            raise ValueError("DSV4 KV components extend beyond their physical page")
    if (
        page_bytes % first.element_size()
        or base + num_blocks * page_bytes > storage.nbytes()
    ):
        raise ValueError("DSV4 KV page exceeds its backing storage")
    return first.as_strided(
        (num_blocks, page_bytes // first.element_size()),
        (first.stride(0), 1),
    ).view(torch.uint8)


@dataclass
class Snapshot:
    tokens: tuple[int, ...]
    # Logical page index -> row in the saved tensor. None is the allocator's
    # pruned/null block, not a physical page to read or overwrite.
    slots: tuple[dict[int, int], ...]
    tables: tuple[tuple[int | None, ...], ...]
    pages: dict[str, torch.Tensor]
    size: int


class KVSnapshotStore:
    def __init__(self, caches, config, *, max_bytes, max_entries=KV_MAX_ENTRIES):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("KV snapshot host-memory budget must be positive")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.num_blocks = config.num_blocks
        self.groups = []
        seen = set()
        for group in config.kv_cache_groups:
            spec = group.kv_cache_spec
            if type(spec).__name__ == "HiddenStateCacheSpec":
                self.groups.append({})
                seen.update(group.layer_names)
                continue
            specs = getattr(spec, "kv_cache_specs", None)
            views = {}
            for name in group.layer_names:
                layer_spec = specs[name] if specs is not None else spec
                if type(layer_spec).__name__ not in (
                    "AscendMLAAttentionSpec",
                    "AscendSlidingWindowMLASpec",
                ) or getattr(layer_spec, "store_on_host", False):
                    raise ValueError(
                        "KV reuse requires native paged V4 attention/compressor caches"
                    )
                views[name] = _page_view(
                    caches[name], layer_spec.page_size_bytes, self.num_blocks
                )
            seen.update(views)
            self.groups.append(views)
        if seen != set(caches) or not any(self.groups):
            raise ValueError("KV reuse must account for every native V4 cache layer")
        self.entries = OrderedDict()
        self.size = 0

    def release(self, keys):
        for key in keys:
            snapshot = self.entries.pop(key, None)
            if snapshot is not None:
                self.size -= snapshot.size

    def _validate_tables(self, tables):
        if len(tables) != len(self.groups):
            raise ValueError("KV snapshot cache-group count mismatch")
        for table in tables:
            ids = [value for value in table if value is not None]
            if any(
                type(value) is not int or not 0 <= value < self.num_blocks
                for value in ids
            ) or len(ids) != len(set(ids)):
                raise ValueError("KV snapshot physical block IDs are invalid")

    def save(self, key, tokens, tables):
        self._validate_tables(tables)
        if key in self.entries:
            raise ValueError("KV snapshots are immutable; duplicate write handle")
        size = sum(
            sum(value is not None for value in table)
            * sum(view.shape[1] for view in group.values())
            for table, group in zip(tables, self.groups, strict=True)
        )
        if size > self.max_bytes:
            return False  # No partial snapshot; next request safely recomputes.
        while self.entries and (
            self.size + size > self.max_bytes or len(self.entries) >= self.max_entries
        ):
            self.release([next(iter(self.entries))])
        pages, slots = {}, []
        for table, group in zip(tables, self.groups, strict=True):
            logical = [i for i, value in enumerate(table) if value is not None]
            slots.append({index: row for row, index in enumerate(logical)})
            for name, view in group.items():
                ids = torch.tensor(
                    [table[i] for i in logical], dtype=torch.long, device=view.device
                )
                # One layer-sized temporary, not an additional model-sized HBM
                # cache. Blocking D2H makes publication and reuse synchronous.
                pages[name] = view.index_select(0, ids).to(device="cpu", copy=True)
        self.entries[key] = Snapshot(
            tuple(tokens), tuple(slots), tuple(tuple(t) for t in tables), pages, size
        )
        self.size += size
        return True

    def restore(self, key, tokens, computed, tables):
        self._validate_tables(tables)
        snapshot = self.entries.get(key)
        if (
            snapshot is None
            or snapshot.tokens != tuple(tokens[:computed])
            or len(snapshot.tokens) != computed
        ):
            raise ValueError(
                "Scheduled KV snapshot is absent or has a different prefix"
            )
        for old_table, table, slots, group in zip(
            snapshot.tables, tables, snapshot.slots, self.groups, strict=True
        ):
            logical = [
                i
                for i, value in enumerate(table[: len(old_table)])
                if value is not None
            ]
            if group and any(i not in slots for i in logical):
                raise ValueError(
                    "A previously pruned KV page is required by the new request"
                )
            for name, view in group.items():
                rows = torch.tensor([slots[i] for i in logical], dtype=torch.long)
                ids = torch.tensor(
                    [table[i] for i in logical], dtype=torch.long, device=view.device
                )
                view.index_copy_(
                    0, ids, snapshot.pages[name].index_select(0, rows).to(view.device)
                )
        self.entries.move_to_end(key)
