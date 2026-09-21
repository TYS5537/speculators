"""Graceful shutdown handler for saving checkpoints on interrupt.

When Ctrl+C is pressed during training:
- torchrun intercepts SIGINT and sends SIGINT/SIGTERM to all worker processes
- In single-GPU mode, the process receives SIGINT directly

This module handles both cases by registering handlers for SIGINT and SIGTERM.
The first signal requests a coordinated stop. The trainer finishes its current
forward/backward and ALL optimizer/scheduler updates before raising at a safe
boundary. A stuck collective/kernel cannot produce a new consistent checkpoint:
the timeout forces exit and leaves the previously published checkpoint intact.

Key design decisions:
- The handler only runs in the process that called install() (tracked via
  PID). Forked dataloader workers inherit the handler but ignore the signal,
  preventing worker crashes.
- After the first signal, subsequent rapid re-sends (torchrun sends SIGINT
  to the process group AND then again directly to each child) are silently
  ignored for one second. A later repeat forces exit, as does another Ctrl+C
  during the save, when the original signal handlers have been restored.
"""

import logging
import os
import signal
import threading
import time
from functools import wraps
from typing import Any

logger = logging.getLogger("speculators")

# Default timeout for coordinated shutdown save (seconds)
DEFAULT_SHUTDOWN_TIMEOUT = 120


class TrainingInterruptedError(Exception):
    """Raised at a complete training-step boundary to save consistent state."""


class GracefulShutdownHandler:
    """Manages graceful shutdown with checkpoint saving on interrupt.

    First interrupt: request a stop after the current complete step.
    Subsequent rapid signals (from torchrun re-sends): ignored for one second.
    After restore(): default handlers are active, so another Ctrl+C kills.
    """

    def __init__(self, timeout: int = DEFAULT_SHUTDOWN_TIMEOUT):
        self._interrupted = False
        self._requested_at = 0.0
        self._timer: threading.Timer | None = None
        self._original_sigint: Any = None
        self._original_sigterm: Any = None
        self._timeout = timeout
        self._owner_pid: int | None = None

    def install(self):
        """Register signal handlers for SIGINT and SIGTERM."""
        self._owner_pid = os.getpid()
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def restore(self):
        """Restore original signal handlers.

        Called before the checkpoint save attempt so that a deliberate
        second Ctrl+C during the save causes immediate exit.
        """
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handler(self, signum, frame):  # noqa: ARG002
        # Only handle in the process that installed the handler.
        # Forked dataloader workers inherit signal handlers but should
        # not raise TrainingInterruptedError (it would crash the worker).
        if os.getpid() != self._owner_pid:
            return

        if self._interrupted:
            if time.monotonic() - self._requested_at >= 1.0:
                os._exit(128 + signum)
            return
        self._interrupted = True
        self._requested_at = time.monotonic()
        self.start_watchdog()
        logger.warning(
            "Received %s — stopping after the current complete training step. "
            "Repeat after one second to force exit without a new checkpoint; "
            "shutdown timeout is %ss.",
            signal.Signals(signum).name,
            self.timeout,
        )

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def start_watchdog(self):
        if self._timer is not None:
            return

        def expire():
            logger.error(
                "Safe shutdown timed out after %ss — forcing exit", self.timeout
            )
            os._exit(1)

        self._timer = threading.Timer(self.timeout, expire)
        self._timer.daemon = True
        self._timer.start()

    def cancel_watchdog(self):
        if self._timer is not None:
            self._timer.cancel()

    @property
    def timeout(self) -> int:
        return self._timeout


def with_graceful_shutdown(
    save_label: str = "interrupted",
    timeout: int = DEFAULT_SHUTDOWN_TIMEOUT,
):
    """Decorator that wraps a Trainer method with graceful shutdown handling.

    The decorated method's `self` must have `maybe_save_checkpoint(label)` and
    `checkpointer.path`.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            handler = GracefulShutdownHandler(timeout=timeout)
            previous_handler = getattr(self, "_shutdown_handler", None)
            self._shutdown_handler = handler
            handler.install()

            try:
                return fn(self, *args, **kwargs)
            except TrainingInterruptedError:
                handler.restore()

                logger.warning(
                    "Training interrupted — attempting to save checkpoint "
                    f"(timeout={handler.timeout}s, send Ctrl+C again to force exit)..."
                )

                handler.start_watchdog()

                try:
                    self.maybe_save_checkpoint(save_label)
                    logger.info(
                        "Interrupt checkpoint saved to "
                        f"'{self.checkpointer.path / save_label}'"
                    )
                except Exception:
                    logger.exception("Failed to save interrupt checkpoint")
            finally:
                handler.cancel_watchdog()
                handler.restore()
                self._shutdown_handler = previous_handler

        return wrapper

    return decorator
