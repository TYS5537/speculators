"""Opt-in V4 block verification with immutable, bounded host KV snapshots.

vLLM still owns allocation and all attention/compression kernels. The connector
loads whole native pages via its external-computed-token API, then the runner
forwards only the suffix. DP engines keep independent caches; clients use the
standard X-data-parallel-rank header after their first response.
"""

# ruff: noqa: ARG002 -- Preserve vLLM hook signatures.

from dataclasses import dataclass

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorWorkerMetadata
from vllm.distributed.parallel_state import get_dp_group, get_tp_group

from speculators_dsv4.block_connector import BlockRequest, DSV4BlockVerifyConnector
from speculators_dsv4.block_protocol import (
    BLOCK_KV_VERSION,
    BLOCK_REQUEST_KEY,
    validate_block_request,
)
from speculators_dsv4.kv_snapshots import KVSnapshotStore


@dataclass
class CachedBlockRequest(BlockRequest):
    cache_read: str | None = None
    cache_write: str | None = None
    cache_release: tuple[str, ...] = ()
    block_tables: tuple[tuple[int | None, ...], ...] = ()
    cache_saved: bool = False


@dataclass
class SnapshotAvailability(KVConnectorWorkerMetadata):
    keys: set[str]

    def aggregate(self, other):
        if not isinstance(other, SnapshotAvailability):
            raise ValueError("Unexpected KV snapshot worker metadata")
        # A hit is valid only if EVERY TP worker still owns the snapshot.
        return SnapshotAvailability(self.keys & other.keys)


class DSV4CachedBlockVerifyConnector(DSV4BlockVerifyConnector):
    supports_snapshot_reuse = True

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        mb = self._kv_transfer_config.get_from_extra_config("kv_cache_mb", 1024)
        if type(mb) is not int or mb <= 0:
            raise ValueError("kv_cache_mb must be a positive integer per target rank")
        self._max_bytes = mb * 1024 * 1024
        self._cache_config = kv_cache_config
        self._snapshots = {}  # Scheduler-side, intersected with worker feedback.
        self._allocations = {}
        self._store = None
        self._load_error = None

    def register_kv_caches(self, kv_caches):
        self._store = KVSnapshotStore(
            kv_caches, self._cache_config, max_bytes=self._max_bytes
        )

    def _cache_options(self, request):
        params = (request.sampling_params.extra_args or {}).get(
            "kv_transfer_params"
        ) or {}
        options = params.get(BLOCK_REQUEST_KEY)
        validate_block_request(options, len(request.prompt_token_ids))
        return options.get("cache") if options["version"] == BLOCK_KV_VERSION else None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        options = self._cache_options(request)
        if num_computed_tokens:
            raise ValueError("DSV4 KV reuse requires native prefix caching disabled")
        if options is None:
            return 0, False
        tokens = self._snapshots.get(options["read"])
        if tokens is None:
            return 0, False
        if list(tokens) != request.prompt_token_ids[: len(tokens)]:
            raise ValueError("KV snapshot does not match the requested prefix")
        block = request.sampling_params.extra_args["kv_transfer_params"][
            BLOCK_REQUEST_KEY
        ]
        if len(tokens) >= len(request.prompt_token_ids) or min(
            block["logits_start"], block["hidden_start"]
        ) < len(tokens):
            raise ValueError("KV snapshot must precede all requested output rows")
        return len(tokens), False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self._allocations[request.request_id] = (
            num_external_tokens,
            tuple(
                tuple(None if block.is_null else block.block_id for block in group)
                for group in blocks.blocks
            ),
        )

    def _computed_tokens(self, request):
        allocation = self._allocations.get(request.req_id)
        if allocation is None:
            raise ValueError("Missing KV snapshot allocation metadata")
        return allocation[0]

    def _build_request(self, request, scheduled):
        block = super()._build_request(request, scheduled)
        options = self._cache_options(request)
        if options is None:
            return block  # Stateless v1/v2 clients can share this eval server.
        if options["write"] in self._snapshots:
            raise ValueError("Cannot replace an immutable KV snapshot")
        return CachedBlockRequest(
            **vars(block),
            cache_read=options["read"],
            cache_write=options["write"],
            cache_release=tuple(options["release"]),
            block_tables=self._allocations[request.req_id][1],
        )

    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        writes = [
            request.cache_write
            for request in metadata.requests
            if isinstance(request, CachedBlockRequest)
        ]
        if len(writes) != len(set(writes)):
            raise ValueError("Duplicate KV snapshot write in a batch")
        for request in metadata.requests:
            self._allocations.pop(request.request_id, None)
            if isinstance(request, CachedBlockRequest):
                self._snapshots[request.cache_write] = tuple(request.token_ids)
        return metadata

    def request_finished(self, request, block_ids):
        self._allocations.pop(request.request_id, None)
        return super().request_finished(request, block_ids)

    def start_load_kv(self, *args, **kwargs):
        self._load_error = None
        try:
            if not self.has_connector_metadata():
                return
            requests = self._get_connector_metadata().requests
            if (
                any(isinstance(request, CachedBlockRequest) for request in requests)
                and self._store is None
            ):
                raise ValueError("Native KV buffers have not been registered")
            # Restore the ENTIRE batch before applying releases or LRU evictions.
            for request in requests:
                if isinstance(request, CachedBlockRequest) and request.computed_tokens:
                    self._store.restore(
                        request.cache_read,
                        request.token_ids,
                        request.computed_tokens,
                        request.block_tables,
                    )
            for request in requests:
                if isinstance(request, CachedBlockRequest):
                    self._store.release(request.cache_release)
        except Exception as exc:  # noqa: BLE001 -- Rendezvous before model collectives.
            self._load_error = exc

    def check_ready(self, device):
        # Called by the model on active AND idle DP ranks, before native MoE/TP
        # forward collectives. Never run with a partially restored cache.
        failed = torch.tensor(
            [int(self._load_error is not None)], dtype=torch.int32, device=device
        )
        failed = get_tp_group().all_reduce(failed)
        if self._dp_size > 1:
            failed = get_dp_group().all_reduce(failed)
        if failed.item():
            raise RuntimeError(
                "DSV4 KV snapshot restore failed on a TP/DP rank"
            ) from self._load_error

    def _capture_requests(self, input_ids, positions, output):
        batch = super()._capture_requests(input_ids, positions, output)
        for request, _ in batch:
            if isinstance(request, CachedBlockRequest):
                request.cache_saved = self._store.save(
                    request.cache_write, request.token_ids, request.block_tables
                )
        return batch

    def _packet_extras(self, request):
        if not isinstance(request, CachedBlockRequest):
            return {}
        return {
            "kv_reuse_metadata": torch.tensor(
                [request.computed_tokens, int(request.cache_saved), self._dp_rank],
                dtype=torch.int64,
            )
        }

    def build_connector_worker_meta(self):
        return SnapshotAvailability(set(self._store.entries) if self._store else set())

    def update_connector_output(self, connector_output):
        metadata = connector_output.kv_connector_worker_meta
        if isinstance(metadata, SnapshotAvailability):
            self._snapshots = {
                key: tokens
                for key, tokens in self._snapshots.items()
                if key in metadata.keys
            }

    def shutdown(self):
        if self._store is not None:
            self._store.release(list(self._store.entries))
