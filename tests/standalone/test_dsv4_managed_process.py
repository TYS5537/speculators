"""Stdlib ownership, signal escalation and lightweight child lifecycle tests."""

# ruff: noqa: PT009, PT027 -- Keep this suite runnable with stdlib unittest.

import os
import signal
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, call, patch

from speculators_dsv4 import managed_process as managed


class StartProcessTests(unittest.TestCase):
    def test_posix_argv_session_and_noninteractive_io(self):
        process = MagicMock(pid=12345)
        command = ["python", "model path.py", "a;not-a-shell-command"]
        environment = {"ASCEND_RT_VISIBLE_DEVICES": "0,1"}
        with (
            patch.object(managed.os, "name", "posix"),
            patch.object(managed.subprocess, "Popen", return_value=process) as popen,
        ):
            result = managed.start_process(command, cwd="repo", env=environment)
        self.assertIs(result, process)
        args, kwargs = popen.call_args
        self.assertEqual(args, (command,))
        self.assertEqual(kwargs["cwd"], "repo")
        self.assertEqual(kwargs["env"], environment)
        self.assertIsNot(kwargs["env"], environment)
        self.assertFalse(kwargs["shell"])
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertIsNone(kwargs["stdout"])

    def test_control_stdin_is_a_text_pipe(self):
        process = MagicMock(pid=12345)
        with patch.object(managed.subprocess, "Popen", return_value=process) as popen:
            managed.start_process(
                ["python"], control_stdin=True, stdout=subprocess.PIPE
            )
        self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.PIPE)
        self.assertEqual(popen.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertTrue(popen.call_args.kwargs["text"])
        self.assertEqual(popen.call_args.kwargs["encoding"], "utf-8")

    def test_windows_does_not_claim_posix_group_support(self):
        process = MagicMock(pid=12345)
        with (
            patch.object(managed.os, "name", "nt"),
            patch.object(managed.subprocess, "Popen", return_value=process) as popen,
        ):
            managed.start_process(["python"])
        self.assertFalse(popen.call_args.kwargs["start_new_session"])
        self.assertFalse(managed._OWNED[process].posix_group)

    def test_rejects_string_commands_or_invalid_arguments(self):
        with patch.object(managed.subprocess, "Popen") as popen:
            for command in ("python script.py", b"python", [], [""], [2], ["a\0b"]):
                with self.subTest(command=command), self.assertRaises(ValueError):
                    managed.start_process(command)
        popen.assert_not_called()


class StopProcessTests(unittest.TestCase):
    def make_process(self, *, posix=True, returncode=None):
        process = MagicMock(pid=12345, returncode=returncode)
        process.poll.return_value = returncode
        process.stdin = None
        managed._OWNED[process] = managed._OwnedProcess(process.pid, posix)
        return process

    def test_rejects_unowned_process_without_signalling(self):
        process = MagicMock(pid=12345)
        with (
            patch.object(managed, "_signal_owned") as send,
            self.assertRaisesRegex(ValueError, "created by start_process"),
        ):
            managed.stop_process(process)
        send.assert_not_called()

    def test_already_exited_group_requires_no_signal(self):
        process = self.make_process(returncode=0)
        with (
            patch.object(managed, "_wait_for_exit", return_value=True),
            patch.object(managed, "_signal_owned") as send,
        ):
            self.assertEqual(managed.stop_process(process), 0)
            self.assertEqual(managed.stop_process(process), 0)
        send.assert_not_called()

    def test_exited_leader_still_terminates_remaining_workers(self):
        process = self.make_process(returncode=0)
        with (
            patch.object(managed, "_wait_for_exit", side_effect=[False, True]),
            patch.object(managed.os, "killpg", create=True) as killpg,
        ):
            self.assertEqual(managed.stop_process(process, timeout=0.1), 0)
        killpg.assert_called_once_with(process.pid, signal.SIGTERM)
        process.terminate.assert_not_called()

    def test_term_timeout_escalates_only_the_owned_group(self):
        process = self.make_process(returncode=0)
        with (
            patch.object(managed, "_wait_for_exit", side_effect=[False, False]),
            patch.object(managed.os, "killpg", create=True) as killpg,
        ):
            managed.stop_process(process, timeout=0.1)
        self.assertEqual(
            killpg.call_args_list,
            [call(process.pid, signal.SIGTERM), call(process.pid, managed._SIGKILL)],
        )
        process.wait.assert_called_once_with(timeout=managed._KILL_WAIT)

    def test_unreapable_child_reports_failure(self):
        process = self.make_process()
        process.wait.side_effect = subprocess.TimeoutExpired("owned child", 5)
        with (
            patch.object(managed, "_wait_for_exit", side_effect=[False, False]),
            patch.object(managed, "_signal_owned"),
            self.assertRaisesRegex(RuntimeError, "forced cleanup"),
        ):
            managed.stop_process(process, timeout=0.1)
        self.assertFalse(managed._OWNED[process].stopped)

    def test_eof_grace_can_avoid_signals(self):
        process = self.make_process(returncode=0)
        process.stdin = MagicMock()
        with (
            patch.object(managed, "_wait_for_exit", side_effect=[False, True]),
            patch.object(managed, "_signal_owned") as send,
        ):
            managed.stop_process(process, close_stdin=True, timeout=0.1)
        process.stdin.close.assert_called_once_with()
        send.assert_not_called()

    def test_windows_fallback_terminates_only_the_child(self):
        process = self.make_process(posix=False)
        with patch.object(managed, "_wait_for_exit", side_effect=[False, True]):
            managed.stop_process(process, timeout=0.1)
        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()

    def test_invalid_timeout_does_not_signal(self):
        process = self.make_process()
        with patch.object(managed, "_signal_owned") as send:
            for timeout in (0, -1, float("nan"), float("inf"), True, "20"):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    managed.stop_process(process, timeout=timeout)
        send.assert_not_called()

    def test_group_wait_does_not_finish_when_only_leader_has_exited(self):
        process = self.make_process(returncode=0)
        owner = managed._OWNED[process]
        with patch.object(managed, "_group_alive", return_value=True):
            self.assertFalse(managed._wait_for_exit(process, owner, 0))
        process.wait.assert_not_called()

    def test_group_wait_reaps_leader_after_group_is_gone(self):
        process = self.make_process(returncode=0)
        owner = managed._OWNED[process]
        with patch.object(managed, "_group_alive", return_value=False):
            self.assertTrue(managed._wait_for_exit(process, owner, 0))
        process.wait.assert_called_once()


class LightweightProcessTests(unittest.TestCase):
    def test_default_stdin_is_eof_and_child_is_reaped(self):
        process = managed.start_process(
            [sys.executable, "-u", "-c", "import sys; print(repr(sys.stdin.read()))"],
            stdout=subprocess.PIPE,
        )
        try:
            output, _ = process.communicate(timeout=5)
            self.assertEqual(output.strip(), "''")
            self.assertEqual(managed.stop_process(process, timeout=0.2), 0)
        finally:
            managed.stop_process(process, timeout=0.2)
            if process.stdout is not None:
                process.stdout.close()

    def test_running_child_is_stopped_and_waited(self):
        process = managed.start_process(
            [sys.executable, "-u", "-c", "import time; time.sleep(60)"],
        )
        result = managed.stop_process(process, timeout=0.2)
        self.assertIsNotNone(result)
        self.assertIsNotNone(process.poll())

    def test_control_pipe_eof_allows_clean_exit(self):
        process = managed.start_process(
            [sys.executable, "-u", "-c", "import sys; sys.stdin.read()"],
            control_stdin=True,
        )
        self.assertEqual(managed.stop_process(process, timeout=5, close_stdin=True), 0)
        self.assertTrue(process.stdin.closed)

    @unittest.skipUnless(os.name == "posix", "Real process groups require POSIX")
    def test_posix_child_has_its_own_session_and_process_group(self):
        process = managed.start_process(
            [
                sys.executable,
                "-u",
                "-c",
                "import os; print(os.getpid(), os.getpgrp(), os.getsid(0))",
            ],
            stdout=subprocess.PIPE,
        )
        try:
            output, _ = process.communicate(timeout=5)
            self.assertEqual(
                [int(value) for value in output.split()], [process.pid] * 3
            )
        finally:
            managed.stop_process(process, timeout=0.2)
            if process.stdout is not None:
                process.stdout.close()


if __name__ == "__main__":
    unittest.main()
