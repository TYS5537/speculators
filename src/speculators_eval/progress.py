"""Model-free, per-dataset progress shared by sample-parallel eval workers."""

from __future__ import annotations

import logging
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional display dependency
    tqdm = None

logger = logging.getLogger("dspark_offline_eval")
_LOG_INTERVAL = 10.0


class WorkerProgress:
    """Publish completed samples atomically; reporting failures never abort eval."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def update(self, completed: int) -> None:
        if self.path is None:
            return
        try:
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(str(completed), encoding="ascii")
            temporary.replace(self.path)
        except OSError:
            logger.warning("Cannot update worker progress at %s", self.path)
            self.path = None


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ParallelProgress:
    """One terminal bar, or periodic log snapshots for nohup/redirected output."""

    def __init__(self, dataset: str, total: int, directory: Path, workers: int) -> None:
        self.dataset = dataset
        self.total = total
        self.paths = [directory / f"worker-{index}.count" for index in range(workers)]
        self.limits = [len(range(index, total, workers)) for index in range(workers)]
        self.counts = [0] * workers
        self.completed = 0
        self.started = time.monotonic()
        self.next_log = self.started
        self.last_reported = -1
        self.bar = (
            tqdm(
                total=total,
                desc=dataset,
                unit="sample",
                dynamic_ncols=True,
                mininterval=1.0,
                file=sys.stderr,
            )
            if tqdm is not None and sys.stderr.isatty()
            else None
        )
        self._render()

    def update(self, statuses: list[tuple[int, int | None]]) -> None:
        for index, status in statuses:
            if status == 0:
                # A successful worker also covers empty shards or lost telemetry.
                self.counts[index] = self.limits[index]
                continue
            try:
                value = int(self.paths[index].read_text(encoding="ascii"))
            except (OSError, ValueError):
                continue
            if self.counts[index] <= value <= self.limits[index]:
                self.counts[index] = value
        completed = sum(self.counts)
        if self.bar is not None:
            self.bar.update(completed - self.completed)
        self.completed = completed
        self._render()

    def _render(self, *, force: bool = False) -> None:
        now = time.monotonic()
        finished = self.completed == self.total and self.last_reported != self.completed
        if not force and not finished and now < self.next_log:
            return
        self.next_log = now + _LOG_INTERVAL
        self.last_reported = self.completed
        if self.bar is not None:
            self.bar.refresh()
            return
        elapsed = now - self.started
        remaining = (
            _duration(elapsed * (self.total - self.completed) / self.completed)
            if self.completed
            else ("00:00:00" if not self.total else "--:--:--")
        )
        percent = 100.0 * self.completed / self.total if self.total else 100.0
        logger.info(
            "[%s] progress %d/%d samples (%.1f%%) | elapsed=%s | ETA=%s",
            self.dataset,
            self.completed,
            self.total,
            percent,
            _duration(elapsed),
            remaining,
        )

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
        elif self.last_reported != self.completed:
            self._render(force=True)
