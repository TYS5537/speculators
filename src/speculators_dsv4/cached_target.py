"""Accepted-prefix replay for the opt-in V4 KV snapshot connector.

Rejection never crops compressor memory. Restore an immutable confirmed ancestor
and replay only the accepted tail before verifying another block. Full acceptance
promotes the trial snapshot without an extra target request.
"""

from dataclasses import dataclass
from uuid import uuid4

import torch

from speculators_dsv4.block_protocol import BLOCK_KV_VERSION
from speculators_dsv4.offline import DSV4OfflineTarget, TokenHistoryCache


@dataclass(frozen=True)
class RemoteSnapshot:
    key: str
    tokens: tuple[int, ...]


class SnapshotTokenHistory(TokenHistoryCache):
    def __init__(self):
        super().__init__()
        self.base = None
        self.trial = None
        self.dp_rank = None


class DSV4CachedTarget(DSV4OfflineTarget):
    def _initialize_contract(self, *args, **kwargs):
        if kwargs.get("verification_mode") != "block":
            raise ValueError("DSV4 KV reuse requires block verification")
        super()._initialize_contract(*args, **kwargs)

    @property
    def block_protocol_version(self):
        return BLOCK_KV_VERSION

    @staticmethod
    def new_cache():
        return SnapshotTokenHistory()

    def _cached_request(self, prefix, cache, *, logits_start, hidden_start, release=()):
        key = uuid4().hex
        result, hidden, metadata = self._request_block(
            prefix,
            logits_start=logits_start,
            hidden_start=hidden_start,
            cache_options={
                "read": cache.base.key if cache.base else None,
                "write": key,
                "release": list(dict.fromkeys(release)),
            },
            dp_rank=cache.dp_rank,
        )
        if metadata.dtype != torch.int64 or tuple(metadata.shape) != (3,):
            raise ValueError("Invalid target KV reuse metadata")
        computed, saved, rank = metadata.tolist()
        allowed = (0, len(cache.base.tokens)) if cache.base else (0,)
        if (
            computed not in allowed
            or saved not in (0, 1)
            or rank not in (0, 1)
            or (cache.dp_rank is not None and rank != cache.dp_rank)
        ):
            raise ValueError("Target KV reuse prefix/DP identity mismatch")
        cache.dp_rank = rank
        self.profiler.count("kv_reused_tokens", computed)
        self.profiler.count("kv_computed_tokens", len(prefix) - computed)
        self.profiler.count(
            "kv_snapshot_misses", int(cache.base is not None and computed == 0)
        )
        self.profiler.count("kv_snapshot_unsaved", int(not saved))
        snapshot = RemoteSnapshot(key, tuple(prefix)) if saved else None
        return result, hidden, snapshot

    def _confirmed_base(self, cache):
        releases = []
        committed = tuple(cache.tokens)
        if cache.trial is not None and cache.trial.tokens == committed:
            if cache.base is not None:
                releases.append(cache.base.key)
            cache.base, cache.trial = cache.trial, None
        if cache.base is not None and cache.base.tokens != committed:
            if committed[: len(cache.base.tokens)] != cache.base.tokens:
                raise ValueError(
                    "Cannot rewind V4 KV state before its confirmed ancestor"
                )
            if cache.trial is not None:
                releases.append(cache.trial.key)
            releases.append(cache.base.key)
            self.profiler.count("kv_commit_requests", 1)
            # No rejected token enters the new confirmed snapshot. No HS are
            # requested or reused in this accepted-tail-only replay.
            _, _, snapshot = self._cached_request(
                list(committed),
                cache,
                logits_start=len(committed) - 1,
                hidden_start=len(committed),
                release=releases,
            )
            cache.base, cache.trial = snapshot, None
            releases = []
        if cache.trial is not None:
            releases.append(cache.trial.key)
        return releases

    def _forward_block(self, prefix, old_length, cache, output_hidden_states):
        if not isinstance(cache, SnapshotTokenHistory):
            raise ValueError("KV reuse requires its own target cache")
        releases = self._confirmed_base(cache)
        result, hidden, snapshot = self._cached_request(
            prefix,
            cache,
            logits_start=old_length if old_length else len(prefix) - 1,
            hidden_start=old_length if output_hidden_states else len(prefix),
            release=releases,
        )
        output = self._finish_block(result, hidden, prefix, cache, output_hidden_states)
        if old_length == 0:
            cache.base = snapshot
        else:
            cache.trial = snapshot
        return output
