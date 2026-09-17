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
# Set case-sensitive fixture variables inside Bash, including on Windows hosts.
export NO_PROXY="$FIXTURE_NO_PROXY_UPPER" no_proxy="$FIXTURE_NO_PROXY_LOWER"
if [[ -z "$FIXTURE_NO_PROXY_UPPER" && -z "$FIXTURE_NO_PROXY_LOWER" ]]; then
  unset NO_PROXY no_proxy
fi
export HTTP_PROXY=http://http-proxy.fixture:3128 http_proxy=http://http-lower.fixture:3128
export HTTPS_PROXY=http://https-proxy.fixture:3128 https_proxy=http://https-lower.fixture:3128
export ALL_PROXY=socks5://all-proxy.fixture:1080 all_proxy=socks5://all-lower.fixture:1080
capture_proxy_environment() {
  # env is a child process: unexported shell variables must not satisfy this test.
  env | while IFS='=' read -r name value; do
    case "$name" in
      NO_PROXY|no_proxy|HTTP_PROXY|http_proxy|HTTPS_PROXY|https_proxy|ALL_PROXY|all_proxy)
        printf '%s=%s\n' "$name" "$value" ;;
    esac
  done > "$ENV_CAPTURE"
}
# The fixture already owns this directory; avoid MSYS mkdir path translation.
mkdir() { [[ "$1" == -p && -d "$2" ]]; }
nohup() {
  capture_proxy_environment
  printf '%s\n' "$@" > "$CAPTURE"
  echo training-stdout
  echo training-stderr >&2
}
exec() {
  capture_proxy_environment
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
        self.environment_capture = self.root / "proxy-environment"
        self.signals = self.root / "signals"
        self.output = self.root / "output with spaces"

    def run_script(self, kind, **overrides):
        environment = {
            **os.environ,
            "CAPTURE": self.capture.as_posix(),
            "ENV_CAPTURE": self.environment_capture.as_posix(),
            "FIXTURE_NO_PROXY_UPPER": "",
            "FIXTURE_NO_PROXY_LOWER": "",
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
            "DSV4_EXTERNAL_ARROW": "0",
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
        self.assertNotIn("--dsv4-external-arrow", args)
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
        self.assertNotIn("--dsv4-external-arrow", args)
        self.assertFalse((self.output / "logs/train.pid").exists())

    def proxy_environment(self):
        return dict(
            line.split("=", 1)
            for line in self.environment_capture.read_text().splitlines()
        )

    def test_no_proxy_is_exported_to_background_and_smoke_with_existing_entries(self):
        (self.output / "logs").mkdir(parents=True)
        for smoke in ("0", "1"):
            with self.subTest(smoke=smoke):
                result = self.run_script(
                    "trainer",
                    VLLM_ENDPOINT="http://teacher.fixture:8001/v1",
                    FIXTURE_NO_PROXY_UPPER="upper.fixture,192.0.2.0/24",
                    FIXTURE_NO_PROXY_LOWER="lower.fixture,.svc",
                    TRAINING_SMOKE=smoke,
                    SMOKE_PHASE="fresh",
                    SMOKE_REPORT_DIR=(self.root / "reports").as_posix(),
                )
                self.assertEqual(
                    result.returncode, 17 if smoke == "1" else 0, result.stderr
                )
                environment = self.proxy_environment()
                self.assertEqual(environment["NO_PROXY"], environment["no_proxy"])
                self.assertEqual(
                    set(environment["NO_PROXY"].split(",")),
                    {
                        "upper.fixture",
                        "192.0.2.0/24",
                        "lower.fixture",
                        ".svc",
                        "localhost",
                        "127.0.0.1",
                        "teacher.fixture",
                    },
                )
                self.assertEqual(
                    {
                        key: value
                        for key, value in environment.items()
                        if key.upper() != "NO_PROXY"
                    },
                    {
                        "HTTP_PROXY": "http://http-proxy.fixture:3128",
                        "http_proxy": "http://http-lower.fixture:3128",
                        "HTTPS_PROXY": "http://https-proxy.fixture:3128",
                        "https_proxy": "http://https-lower.fixture:3128",
                        "ALL_PROXY": "socks5://all-proxy.fixture:1080",
                        "all_proxy": "socks5://all-lower.fixture:1080",
                    },
                )

    def test_no_proxy_extracts_hostname_ipv4_and_bracketed_ipv6_without_port_or_path(
        self,
    ):
        for endpoint, host in (
            ("https://teacher.fixture/v1", "teacher.fixture"),
            ("http://10.12.0.15:8001/v1", "10.12.0.15"),
            ("http://[2001:db8::3]:8001/v1", "2001:db8::3"),
            ("https://[::1]/v1/", "::1"),
        ):
            with self.subTest(endpoint=endpoint):
                result = self.run_script(
                    "trainer",
                    VLLM_ENDPOINT=endpoint,
                    TRAINING_SMOKE="1",
                    SMOKE_PHASE="fresh",
                    SMOKE_REPORT_DIR=(self.root / "reports").as_posix(),
                )
                self.assertEqual(result.returncode, 17, result.stderr)
                environment = self.proxy_environment()
                for key in ("NO_PROXY", "no_proxy"):
                    self.assertEqual(
                        set(environment[key].split(",")),
                        {"localhost", "127.0.0.1", host},
                    )

    def test_external_arrow_opt_in_reaches_normal_and_smoke_training(self):
        (self.output / "logs").mkdir(parents=True)
        for smoke in ("0", "1"):
            with self.subTest(smoke=smoke):
                result = self.run_script(
                    "trainer",
                    DSV4_EXTERNAL_ARROW="1",
                    TRAINING_SMOKE=smoke,
                    SMOKE_PHASE="fresh",
                    SMOKE_REPORT_DIR=(self.root / "reports").as_posix(),
                )
                self.assertEqual(
                    result.returncode, 17 if smoke == "1" else 0, result.stderr
                )
                args = self.capture.read_text().splitlines()
                self.assertEqual(args.count("--dsv4-external-arrow"), 1)
                if smoke == "1":
                    self.assertGreater(
                        args.index("--dsv4-external-arrow"), args.index("--")
                    )

    def test_invalid_external_arrow_setting_rejected_before_launch(self):
        for value in ("true", "2", "-1"):
            for smoke in ("0", "1"):
                with self.subTest(value=value, smoke=smoke):
                    result = self.run_script(
                        "trainer",
                        DSV4_EXTERNAL_ARROW=value,
                        TRAINING_SMOKE=smoke,
                        SMOKE_PHASE="fresh",
                        SMOKE_REPORT_DIR=(self.root / "reports").as_posix(),
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("DSV4_EXTERNAL_ARROW must be 0 or 1", result.stderr)
                    self.assertFalse(self.capture.exists())
                    self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
