"""Own and clean up the processes created by a local evaluation launcher.

On POSIX, every child gets a new session and its own process group. Cleanup
signals only that recorded group, including workers left after its leader exits.
The Windows fallback manages only the immediate child and exists for lightweight
development tests; it is not Windows process-tree or NPU runtime support.

Descendants that deliberately detach into a different session are outside this
process-group contract. No PID discovery or global process-name killing is used.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from weakref import WeakKeyDictionary

_POLL_INTERVAL = 0.05
_KILL_WAIT = 5.0
# SIGKILL is absent on Windows; this value is used only for POSIX groups.
_SIGKILL = getattr(signal, "SIGKILL", 9)


@dataclass
class _OwnedProcess:
    pid: int
    posix_group: bool
    stopped: bool = False


_OWNED: WeakKeyDictionary[subprocess.Popen, _OwnedProcess] = WeakKeyDictionary()


def start_process(command, *, cwd=None, env=None, stdout=None, control_stdin=False):
    """Start a shell-free child and retain ownership of its process group.

    ``env`` follows Popen semantics: None inherits the parent environment, while
    a supplied mapping replaces it. ``stdout=None`` inherits stdout; stderr is
    always merged into that same destination. A requested control stdin is a
    UTF-8 text pipe; otherwise stdin is DEVNULL and cannot steal terminal input.
    """
    if (
        not isinstance(command, Sequence)
        or isinstance(command, (str, bytes))
        or not command
        or any(not isinstance(arg, str) or "\0" in arg for arg in command)
        or not command[0]
    ):
        raise ValueError("command must be a nonempty sequence of NUL-free strings")
    if type(control_stdin) is not bool:
        raise ValueError("control_stdin must be a boolean")
    posix_group = os.name == "posix"
    process = subprocess.Popen(  # noqa: S603 -- Caller supplies explicit argv, no shell.
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        stdout=stdout,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE if control_stdin else subprocess.DEVNULL,
        shell=False,
        start_new_session=posix_group,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _OWNED[process] = _OwnedProcess(process.pid, posix_group)
    return process


def _group_alive(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_exit(process, owner, timeout):
    """Reap the leader but also wait for its POSIX workers to leave the group."""
    if not owner.posix_group:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True
    deadline = time.monotonic() + timeout
    while True:
        process.poll()  # Reap an exited leader; its workers may still be alive.
        if not _group_alive(owner.pid):
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_POLL_INTERVAL, remaining))


def _signal_owned(process, owner, *, force=False):
    if owner.pid <= 1:
        raise ValueError("Refusing to signal an invalid owned process/group ID")
    if owner.posix_group:
        with suppress(ProcessLookupError):
            os.killpg(owner.pid, _SIGKILL if force else signal.SIGTERM)
    elif process.poll() is None:
        with suppress(ProcessLookupError):
            if force:
                process.kill()
            else:
                process.terminate()


def stop_process(process, *, timeout=20.0, close_stdin=False):
    """Terminate only a child/group created here and reap its leader.

    Normally SIGTERM gets ``timeout`` seconds before SIGKILL. With close_stdin,
    closing a control pipe first grants one additional ``timeout`` EOF-grace
    period. The final leader-reap wait is bounded to five seconds. Return the
    leader's exit code, or raise if it cannot be reaped after forced termination.
    Calls after successful cleanup are harmless. POSIX orphaned zombies are not
    ours to reap; their init/subreaper does so after remaining workers are killed.
    """
    owner = _OWNED.get(process)
    if owner is None:
        raise ValueError("Can only stop a process created by start_process")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (float, int))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive finite number")
    if type(close_stdin) is not bool:
        raise ValueError("close_stdin must be a boolean")
    if owner.stopped:
        return process.returncode
    if _wait_for_exit(process, owner, 0.0):
        owner.stopped = True
        return process.returncode
    if close_stdin and process.stdin is not None:
        with suppress(BrokenPipeError):
            process.stdin.close()
        if _wait_for_exit(process, owner, timeout):
            owner.stopped = True
            return process.returncode
    _signal_owned(process, owner)
    if not _wait_for_exit(process, owner, timeout):
        _signal_owned(process, owner, force=True)
        try:
            process.wait(timeout=_KILL_WAIT)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "Owned child did not exit after forced cleanup"
            ) from error
    owner.stopped = True
    return process.returncode
