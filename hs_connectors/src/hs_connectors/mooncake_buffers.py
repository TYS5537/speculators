"""Bounded reuse of host staging buffers; no accelerator initialization."""

from dataclasses import dataclass
from threading import Lock

import torch


class CopySynchronizationError(RuntimeError):
    """A device copy may still own its source/destination; do not release KV."""


@dataclass
class HostBufferLease:
    tensor: torch.Tensor
    pooled_bytes: int


class HostBufferPool:
    """Limit retained pooled bytes, including buffers currently in use.

    A miss beyond the pool budget uses an unpooled allocation, as before.
    The writer concurrency still bounds these in-flight fallback allocations.
    Failed, unsynchronized copies are quarantined until worker termination.
    """

    def __init__(self, max_bytes=0):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("Pinned pool bytes must be a nonnegative integer")
        self.max_bytes = max_bytes
        self.reserved_bytes = 0
        self._available = []
        self._quarantined = []
        self._lock = Lock()

    @staticmethod
    def _allocate(reference):
        try:
            return torch.empty_like(reference, device="cpu", pin_memory=True)
        except Exception:  # noqa: BLE001 -- Some accelerator builds cannot pin memory.
            return torch.empty_like(reference, device="cpu")

    def acquire(self, reference):
        size = reference.numel() * reference.element_size()
        pooled_bytes = 0
        with self._lock:
            for index, lease in enumerate(self._available):
                if (
                    lease.tensor.shape == reference.shape
                    and lease.tensor.dtype == reference.dtype
                ):
                    return self._available.pop(index)
            if 0 < size <= self.max_bytes:
                while self._available and self.reserved_bytes + size > self.max_bytes:
                    self.reserved_bytes -= self._available.pop(0).pooled_bytes
                if self.reserved_bytes + size <= self.max_bytes:
                    pooled_bytes = size
                    self.reserved_bytes += size
        try:
            return HostBufferLease(self._allocate(reference), pooled_bytes)
        except BaseException:
            with self._lock:
                self.reserved_bytes -= pooled_bytes
            raise

    def release(self, lease, *, reuse):
        """Call only after the copy and the synchronous store operation finish."""
        with self._lock:
            if reuse and lease.pooled_bytes:
                self._available.append(lease)
            else:
                self.reserved_bytes -= lease.pooled_bytes

    def quarantine(self, lease):
        with self._lock:
            self._quarantined.append(lease)
