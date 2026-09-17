"""Run Bash wiring with fake commands: no NPU, HTTP, or real process-group kills."""

# ruff: noqa: PT009 -- Also runnable without pytest/torch.

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if os.name == "nt":
    BASH = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/bash.exe"
    BASH = str(BASH) if BASH.is_file() else None
else:
    BASH = shutil.which("bash")

SERVER_STUBS = r"""
python() { return 0; }
setsid() { printf '%s\n' "$@" > "$CAPTURE"; }
curl_calls=0
curl() {
  curl_calls=$((curl_calls + 1))
  case "$MODE" in
    occupied) return 0 ;;
    timeout|dead) return 1 ;;
    interrupt)
      if (( curl_calls > 1 )); then builtin kill -INT "$$"; fi
      return 1 ;;
    *) (( curl_calls > 1 )) ;;
  esac
}
kill() {
  if [[ "$1" == -0 ]]; then
    [[ "$2" != -- && "$MODE" != dead ]]
  else
    printf '%s\n' "$*" >> "$SIGNALS"
  fi
}
wait() { builtin wait "$@" || true; return "${WAIT_STATUS:-0}"; }
sleep() { SECONDS=$((SECONDS + 5)); }
"""

TRAINER_STUBS = r"""
# The fixture already owns this directory; avoid MSYS mkdir path translation.
mkdir() { [[ "$1" == -p && -d "$2" ]]; }
nohup() {
  printf '%s\n' "$@" > "$CAPTURE"
  echo training-stdout
  echo training-stderr >&2
}
exec() {
  printf '%s\n' "$@" > "$CAPTURE"
  exit 17
}
"""


@unittest.skipUnless(BASH, "Bash is required for shell wiring tests")
class LaunchScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.capture = self.root / "arguments"
        self.signals = self.root / "signals"
        self.output = self.root / "output with spaces"

    def run_script(self, kind, **overrides):
        environment = {
            **os.environ,
            "CAPTURE": self.capture.as_posix(),
            "SIGNALS": self.signals.as_posix(),
            "OUTPUT_DIR": self.output.as_posix(),
            "MODEL": "/fixture/model",
            "HS_PATH": "/fixture/hs",
            "DATA_PATH": "/fixture/data",
            "VLLM_HOST": "127.0.0.1",
            "VLLM_PORT": "9123",
            "VLLM_ENDPOINT": "http://127.0.0.1:9123/v1",
            "VLLM_STARTUP_TIMEOUT": "1",
            "TP_SIZE": "8",
            "DP_SIZE": "2",
            "VLLM_NPUS": ",".join(map(str, range(16))),
            "TRAIN_NPUS": ",".join(map(str, range(16))),
            "NUM_TRAIN_NPUS": "16",
            "TARGET_QUANTIZATION": "",
            "DSV4_EVAL": "0",
            "TRAINING_SMOKE": "0",
            "MODE": "ready",
            "WAIT_STATUS": "0",
            **overrides,
        }
        source = f"source examples/train/dspark_dsv4_flash_bf16_{kind}.sh\n"
        stubs = SERVER_STUBS if kind == "server" else TRAINER_STUBS
        if kind == "trainer":
            source += 'wait "$TRAIN_PID"\n'
        return subprocess.run(  # noqa: S603 -- Fixed repository scripts, fake commands.
            [BASH, "--noprofile", "--norc"],
            input=stubs + source,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_bash_syntax(self):
        for kind in ("server", "trainer"):
            with self.subTest(kind=kind):
                result = subprocess.run(  # noqa: S603 -- Syntax check only.
                    [BASH, "-n", f"examples/train/dspark_dsv4_flash_bf16_{kind}.sh"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_server_captures_pid_waits_and_cleans_only_owned_group(self):
        result = self.run_script("server")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("server ready", result.stdout)
        args = self.capture.read_text().splitlines()
        self.assertEqual(args[args.index("--data-parallel-size") + 1], "2")
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        self.assertEqual(args[args.index("--port") + 1], "9123")
        self.assertRegex(self.signals.read_text(), r"^-TERM -- -[1-9][0-9]*\n$")

    def test_server_early_exit_reports_failure_without_waiting_forever(self):
        result = self.run_script("server", MODE="dead", WAIT_STATUS="7")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("before becoming ready", result.stderr)
        self.assertNotIn("server ready", result.stdout)

    def test_server_readiness_timeout_cleans_up(self):
        result = self.run_script("server", MODE="timeout")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("timed out", result.stderr)
        self.assertTrue(self.signals.exists())

    def test_server_failure_after_readiness_keeps_exit_code(self):
        result = self.run_script("server", WAIT_STATUS="9")
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertIn("server ready", result.stdout)
        self.assertTrue(self.signals.exists())

    def test_existing_endpoint_is_not_mistaken_for_new_server(self):
        result = self.run_script("server", MODE="occupied")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("already responds", result.stderr)
        self.assertFalse(self.capture.exists())
        self.assertFalse(self.signals.exists())

    def test_server_interrupt_runs_cleanup(self):
        result = self.run_script("server", MODE="interrupt")
        self.assertEqual(result.returncode, 130, result.stderr)
        self.assertTrue(self.signals.exists())

    def test_background_training_redirects_both_streams_and_records_pid(self):
        (self.output / "logs").mkdir(parents=True)
        result = self.run_script("trainer")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertIn("scripts/train.py", args)
        self.assertEqual(args[args.index("--epochs") + 1], "10")
        self.assertEqual(
            args[args.index("--vllm-endpoint") + 1], "http://127.0.0.1:9123/v1"
        )
        self.assertEqual(
            args[args.index("--save-path") + 1], f"{self.output.as_posix()}/checkpoints"
        )
        logs = list((self.output / "logs").glob("train_*.log"))
        self.assertEqual(len(logs), 1)
        self.assertIn("training-stdout", logs[0].read_text())
        self.assertIn("training-stderr", logs[0].read_text())
        self.assertNotIn("training-stdout", result.stdout)
        pid = (self.output / "logs/train.pid").read_text().strip()
        self.assertTrue(pid.isdigit())
        self.assertIn(f"kill -TERM {pid}", result.stdout)

    def test_smoke_stays_foreground_and_propagates_failure(self):
        result = self.run_script(
            "trainer",
            TRAINING_SMOKE="1",
            SMOKE_PHASE="fresh",
            SMOKE_REPORT_DIR=(self.root / "reports").as_posix(),
        )
        self.assertEqual(result.returncode, 17, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertIn("scripts/check_dsv4_training.py", args)
        self.assertNotIn("scripts/train.py", args)
        self.assertFalse((self.output / "logs/train.pid").exists())


if __name__ == "__main__":
    unittest.main()
