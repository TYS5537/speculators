"""Opt-in, request-local inference buffers; never used by autograd."""

import torch


class StaticKVBuffer:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("Static KV capacity must be positive")
        self.capacity = capacity
        self.length = 0
        self.key = self.value = None

    def append(self, key, value):
        if torch.is_grad_enabled():
            raise ValueError("Static KV buffers are inference-only")
        end = self.length + key.shape[-2]
        if key.shape != value.shape or end > self.capacity:
            raise ValueError("Static KV shape mismatch or capacity exceeded")
        if self.key is None:
            shape = (*key.shape[:-2], self.capacity, key.shape[-1])
            self.key = key.new_empty(shape)
            self.value = value.new_empty(shape)
        for source, storage in ((key, self.key), (value, self.value)):
            if (
                source.shape[:-2] != storage.shape[:-2]
                or source.shape[-1] != storage.shape[-1]
                or source.dtype != storage.dtype
                or source.device != storage.device
            ):
                raise ValueError("Static KV cannot change shape, device or dtype")
            storage[..., self.length : end, :].copy_(source)
        self.length = end
        return self.key[..., :end, :], self.value[..., :end, :]


class StaticDraftKV(StaticKVBuffer):
    """One token trajectory at one logical depth, discarded after each round."""

    def append_draft(self, context_key, context_value, key, value):
        if self.length == 0:
            self.append(context_key, context_value)
        return self.append(key, value)


class ReusableDraftKV(StaticDraftKV):
    """Request-local storage; replace every synthetic suffix before a new round."""

    def __init__(self, capacity):
        super().__init__(capacity)
        self.verified_length = 0

    def begin_round(self, context_key, context_value):
        if torch.is_grad_enabled():
            raise ValueError("Reusable draft KV is inference-only")
        length = context_key.shape[-2]
        if length < self.verified_length or context_key.shape != context_value.shape:
            raise ValueError("Reusable draft KV requires a growing verified prefix")
        # Drop accepted AND rejected synthetic states. Both must be replaced by
        # the target-conditioned projections, never promoted to verified context.
        self.length = self.verified_length
        self.append(
            context_key[..., self.verified_length : length, :],
            context_value[..., self.verified_length : length, :],
        )
        self.verified_length = length
