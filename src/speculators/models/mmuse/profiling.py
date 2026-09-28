"""Opt-in synchronized forward timings, separate from model/checkpoint config."""

import logging
import os
import time

import torch

logger = logging.getLogger(__name__)


class _NoopProfile:
    def mark(self, _name: str) -> None:
        pass

    def finish(self) -> None:
        pass


_NO_PROFILE = _NoopProfile()


class _ForwardProfile:
    def __init__(self, device: torch.device, call_index: int) -> None:
        self.device = device
        self.call_index = call_index
        self.timings: dict[str, float] = {}
        self.last: float | None = self._now()

    def _now(self) -> float | None:
        try:
            if self.device.type != "cpu":
                getattr(torch, self.device.type).synchronize(self.device)
            return time.perf_counter()
        except Exception:  # noqa: BLE001
            # A diagnostic backend failure must not abort or misreport training.
            logger.warning("MMuse forward profiling device sync failed", exc_info=True)
            return None

    @torch.compiler.disable
    def mark(self, name: str) -> None:
        if self.last is None:
            return
        now = self._now()
        if now is not None:
            self.timings[name] = (now - self.last) * 1000
        self.last = now

    @torch.compiler.disable
    def finish(self) -> None:
        if self.last is None:
            return
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        stages = " ".join(f"{name}_ms={ms:.2f}" for name, ms in self.timings.items())
        logger.info(
            "MMuse forward profile: call=%d %s total_ms=%.2f (synchronized sample)",
            self.call_index,
            stages,
            sum(self.timings.values()),
        )


def start_forward_profile(
    model: torch.nn.Module, device: torch.device
) -> _ForwardProfile | _NoopProfile:
    """Sample every N training forwards with ``MMUSE_PROFILE_FORWARD=N``.

    Zero/unset is a true no-op: no device synchronization or model state changes.
    Validation is excluded. The counter is process-local and not checkpointed;
    it is not the Trainer's global step. Only rank zero prints, but all ranks
    take the same samples. Profiling intentionally serializes sampled forwards.
    """
    interval = os.environ.get("MMUSE_PROFILE_FORWARD", "0")
    if not model.training or interval == "0":
        return _NO_PROFILE
    return _start_sample(model, device, interval) or _NO_PROFILE


@torch.compiler.disable
def _start_sample(
    model: torch.nn.Module, device: torch.device, interval: str
) -> _ForwardProfile | None:
    every = int(interval)
    if every <= 0:
        raise ValueError("MMUSE_PROFILE_FORWARD must be a positive interval or 0")
    call_index = getattr(model, "_mmuse_profile_forward_calls", 0)
    model._mmuse_profile_forward_calls = call_index + 1  # noqa: SLF001
    if call_index % every:
        return None
    return _ForwardProfile(device, call_index)
