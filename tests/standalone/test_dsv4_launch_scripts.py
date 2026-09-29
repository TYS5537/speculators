"""Run Bash wiring with fake commands: no NPU, HTTP, or real process-group kills."""

# ruff: noqa: PT009 -- Also runnable without pytest/torch.

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if os.name == "nt":
    BASH = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/bash.exe"
    BASH = str(BASH) if BASH.is_file() else None
else:
    BASH = shutil.which("bash")

SERVER_STUBS = r"""
python() { printf '%s\n' "$@" >> "$CHECKPOINT_CAPTURE"; }
setsid() {
  printf '%s\n' "$@" > "$CAPTURE"
  env | while IFS='=' read -r name value; do
    case "$name" in
      HCCL_IF_IP|GLOO_SOCKET_IFNAME|TP_SOCKET_IFNAME|HCCL_SOCKET_IFNAME)
        printf '%s=%s\n' "$name" "$value" ;;
    esac
  done > "$ENV_CAPTURE"
}
curl_calls=0
curl() {
  printf '%s\n' "$*" >> "$CURL_CAPTURE"
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
fixture_initial_pwd="$PWD"
# Set case-sensitive fixture variables inside Bash, including on Windows hosts.
export NO_PROXY="$FIXTURE_NO_PROXY_UPPER" no_proxy="$FIXTURE_NO_PROXY_LOWER"
if [[ -z "$FIXTURE_NO_PROXY_UPPER" && -z "$FIXTURE_NO_PROXY_LOWER" ]]; then
  unset NO_PROXY no_proxy
fi
export HTTP_PROXY=http://http-proxy.fixture:3128 http_proxy=http://http-lower.fixture:3128
export HTTPS_PROXY=http://https-proxy.fixture:3128 https_proxy=http://https-lower.fixture:3128
export ALL_PROXY=socks5://all-proxy.fixture:1080 all_proxy=socks5://all-lower.fixture:1080
capture_proxy_environment() {
  [[ "$PWD" == "$fixture_initial_pwd" ]] || exit 98
  # env is a child process: unexported shell variables must not satisfy this test.
  env | while IFS='=' read -r name value; do
    case "$name" in
      NO_PROXY|no_proxy|HTTP_PROXY|http_proxy|HTTPS_PROXY|https_proxy|ALL_PROXY|all_proxy)
        printf '%s=%s\n' "$name" "$value" ;;
    esac
  done > "$ENV_CAPTURE"
  env | while IFS='=' read -r name value; do
    case "$name" in
      OMP_PROC_BIND|OMP_NUM_THREADS|MKL_NUM_THREADS|VE_OMP_NUM_THREADS|\
      PYTORCH_NPU_ALLOC_CONF|TASK_QUEUE_ENABLE|ACLNN_CACHE_LIMIT|\
      NPU_ASD_ENABLE|ASCEND_LAUNCH_BLOCKING)
        printf '%s=%s\n' "$name" "$value" ;;
    esac
  done > "$RUNTIME_ENV_CAPTURE"
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


def _shell_environment(overrides=None):
    environment = {**os.environ, **(overrides or {})}
    environment.pop("BASH_ENV", None)
    environment.pop("ENV", None)
    return {key: value for key, value in environment.items() if value is not None}


@unittest.skipUnless(BASH, "Bash is required for shell wiring tests")
class LaunchScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.capture = self.root / "arguments"
        self.checkpoint_capture = self.root / "checkpoint-arguments"
        self.environment_capture = self.root / "proxy-environment"
        self.runtime_environment_capture = self.root / "runtime-environment"
        self.curl_capture = self.root / "curl-arguments"
        self.signals = self.root / "signals"
        self.output = self.root / "output with spaces"

    def run_script(self, kind, *, cwd=None, **overrides):
        environment = {
            "CAPTURE": self.capture.as_posix(),
            "CHECKPOINT_CAPTURE": self.checkpoint_capture.as_posix(),
            "ENV_CAPTURE": self.environment_capture.as_posix(),
            "RUNTIME_ENV_CAPTURE": self.runtime_environment_capture.as_posix(),
            "CURL_CAPTURE": self.curl_capture.as_posix(),
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
            "DP_SIZE_LOCAL": "",
            "DP_START_RANK": "",
            "DP_ADDRESS": "",
            "DP_RPC_PORT": "",
            "TARGET_MASTER_IP": "",
            "TARGET_WORKER_IP": "",
            "TARGET_LOCAL_IP": "10.0.0.10",
            "TARGET_IFNAME": "eth-fixture",
            "HCCL_IF_IP": "",
            "GLOO_SOCKET_IFNAME": "",
            "TP_SOCKET_IFNAME": "",
            "HCCL_SOCKET_IFNAME": "",
            "DSV4_MANIFEST_TIMEOUT": "",
            "VLLM_NPUS": ",".join(map(str, range(16))),
            "TRAIN_NPUS": ",".join(map(str, range(16))),
            "NUM_TRAIN_NPUS": "16",
            "TARGET_QUANTIZATION": "",
            "DSV4_EVAL": "0",
            "DSV4_BLOCK_VERIFY": "0",
            "DSV4_KV_REUSE": "0",
            "DSV4_GREEDY_REPLAY": "0",
            "MAX_NUM_SEQS": "",
            "MAX_NUM_BATCHED_TOKENS": "",
            "DSV4_EXECUTION_MODE": "",
            "DSV4_ASYNC_SCHEDULING": "",
            "DSV4_EXTERNAL_ARROW": "0",
            "RECOMPUTE": "",
            "TRAINING_SMOKE": "0",
            "MODE": "ready",
            "WAIT_STATUS": "0",
            **overrides,
        }
        environment = _shell_environment(environment)
        script = ROOT / f"examples/train/dspark_dsv4_flash_bf16_{kind}.sh"
        source = f"source {shlex.quote(script.as_posix())}\n"
        stubs = SERVER_STUBS if kind == "server" else TRAINER_STUBS
        if kind == "trainer":
            source += 'wait "$TRAIN_PID"\n'
        return subprocess.run(  # noqa: S603 -- Fixed repository scripts, fake commands.
            [BASH, "--noprofile", "--norc"],
            input=stubs + source,
            cwd=ROOT if cwd is None else cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_bash_syntax(self):
        for relative in (
            "dspark_dsv4_flash_bf16_server.sh",
            "dspark_dsv4_flash_bf16_trainer.sh",
            "dspark_qwen3_8b_trainer.sh",
            "common/ascend_training_env.sh",
            "../evaluate/dspark_dsv4_offline_eval.sh",
            "../evaluate/dspark_dsv4_single_eval.sh",
            "../evaluate/dspark_offline_jsonl.sh",
            "dspark_dsv4_hs_http_server.sh",
        ):
            script = ROOT / "examples/train" / relative
            with self.subTest(script=relative):
                self.assertNotIn(b"\r\n", script.read_bytes())
                result = subprocess.run(  # noqa: S603 -- Syntax check only.
                    [BASH, "-n", script.as_posix()],
                    cwd=ROOT,
                    env=_shell_environment(),
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_server_snapshot_switch_is_opt_in_and_requires_block(self):
        result = self.run_script(
            "server", DSV4_BLOCK_VERIFY="1", DSV4_KV_REUSE="1", DSV4_KV_CACHE_MB="256"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertLess(args.index("--dsv4-kv-reuse"), args.index("--"))
        self.assertEqual(args[args.index("--dsv4-kv-cache-mb") + 1], "256")
        for overrides in (
            {"DSV4_KV_REUSE": "1"},
            {"DSV4_BLOCK_VERIFY": "1", "DSV4_KV_REUSE": "bad"},
            {"DSV4_BLOCK_VERIFY": "1", "DSV4_KV_REUSE": "1", "DSV4_KV_CACHE_MB": "0"},
        ):
            with self.subTest(overrides=overrides):
                result = self.run_script("server", **overrides)
                self.assertNotEqual(result.returncode, 0)

    def test_server_replay_is_opt_in_and_cannot_use_host_snapshots(self):
        result = self.run_script("server", DSV4_BLOCK_VERIFY="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--dsv4-greedy-replay", self.capture.read_text().splitlines())
        result = self.run_script(
            "server", DSV4_BLOCK_VERIFY="1", DSV4_GREEDY_REPLAY="1"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertIn("--dsv4-block-verify", args)
        self.assertLess(args.index("--dsv4-greedy-replay"), args.index("--"))
        self.assertNotIn("--dsv4-kv-reuse", args)
        for overrides in (
            {"DSV4_GREEDY_REPLAY": "1"},
            {"DSV4_BLOCK_VERIFY": "1", "DSV4_GREEDY_REPLAY": "bad"},
            {"DSV4_BLOCK_VERIFY": "1", "DSV4_GREEDY_REPLAY": "1", "DSV4_KV_REUSE": "1"},
        ):
            with self.subTest(overrides=overrides):
                result = self.run_script("server", **overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("DSV4_GREEDY_REPLAY", result.stderr)

    def test_eval_dataset_and_device_wiring(self):
        for kind, devices in (
            ("offline", "2"),
            ("single", "2"),
            ("offline", "8,9,10,11,12,13,14,15"),
            ("single", "8,9,10,11,12,13,14,15"),
        ):
            script = ROOT / f"examples/evaluate/dspark_dsv4_{kind}_eval.sh"
            default_datasets = (
                "gsm8k,math500,humaneval,mbpp,mt-bench"
                if kind == "offline"
                else "gsm8k,math500"
            )
            for datasets, expected in (
                (None, default_datasets),
                ("aime24,humaneval", "aime24,humaneval"),
                ("", ""),
            ):
                with self.subTest(kind=kind, datasets=datasets, devices=devices):
                    environment = _shell_environment(
                        {
                            "VERIFIER_MODEL": "/fixture/target",
                            "DRAFT_MODEL": "/fixture/draft",
                            "DATASETS_ROOT": "/fixture/eval data",
                            "DATASETS": datasets,
                            "MAX_SAMPLES": None,
                            "HS_PATH": "/fixture/hs",
                            "VLLM_ENDPOINT": "http://target.fixture:8001/v1",
                            "VLLM_NPUS": "0,1",
                            "EVAL_NPU": devices,
                            "KEEP_TARGET_HS": "0",
                            "ALLOW_SHARED_DEVICE": "0",
                            "SKIP_ARTIFACTS": "0",
                            "DRY_RUN": "0",
                            "HS_HTTP_ENDPOINT": "",
                            "MAX_NUM_SEQS": "2" if "," in devices else "",
                            "MAX_NUM_BATCHED_TOKENS": "8192" if "," in devices else "",
                            "DSV4_MAX_MODEL_LEN": "4096",
                        }
                    )
                    # Capture the final argv; never invoke Python or a target service.
                    source = r"""exec() { printf '%s\0' "$@"; }"""
                    source += f"\nsource {shlex.quote(script.as_posix())}\n"
                    result = subprocess.run(  # noqa: S603 -- Fake exec only.
                        [BASH, "--noprofile", "--norc"],
                        input=source,
                        cwd=self.root,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(result.stdout.endswith("\0"))
                    args = result.stdout.split("\0")[:-1]
                    self.assertNotIn("--max-samples", args)
                    self.assertEqual(args.count("--datasets"), 1 if expected else 0)
                    if expected:
                        self.assertEqual(args[args.index("--datasets") + 1], expected)
                    self.assertEqual(
                        args[args.index("--datasets-root") + 1], "/fixture/eval data"
                    )
                    if kind == "single":
                        self.assertEqual(args[args.index("--eval-device") + 1], devices)
                        self.assertEqual(
                            args[args.index("--target-max-num-seqs") + 1],
                            "2" if "," in devices else "16",
                        )
                        self.assertEqual(
                            args[args.index("--target-max-num-batched-tokens") + 1],
                            "8192" if "," in devices else "4096",
                        )
                    else:
                        self.assertNotIn("--target-max-num-seqs", args)
                        self.assertNotIn("--target-max-num-batched-tokens", args)
                        self.assertIn(f"ASCEND_RT_VISIBLE_DEVICES={devices}", args)
                        if "," in devices:
                            self.assertEqual(
                                args[args.index("--ascend-devices") + 1], devices
                            )
                        else:
                            self.assertNotIn("--ascend-devices", args)

    def test_draft_cache_switch_in_all_eval_scripts(self):
        for filename in (
            "dspark_dsv4_offline_eval.sh",
            "dspark_dsv4_single_eval.sh",
            "dspark_offline_jsonl.sh",
        ):
            for value in ("0", "1", "bad"):
                with self.subTest(filename=filename, value=value):
                    environment = _shell_environment(
                        {
                            "VERIFIER_MODEL": "/fixture/target",
                            "DRAFT_MODEL": "/fixture/draft",
                            "DATASETS_ROOT": "/fixture/data",
                            "HS_PATH": "/fixture/hs",
                            "HS_HTTP_ENDPOINT": "",
                            "VLLM_NPUS": "0,1",
                            "EVAL_NPU": "2,3",
                            "DSV4_PROFILE": "0",
                            "DSV4_KV_REUSE": "0",
                            "DRAFT_KV_REUSE": value,
                        }
                    )
                    script = ROOT / "examples/evaluate" / filename
                    source = r"""exec() { printf '%s\0' "$@"; }"""
                    source += f"\nsource {shlex.quote(script.as_posix())}\n"
                    result = subprocess.run(  # noqa: S603 -- Only capture script argv.
                        [BASH, "--noprofile", "--norc"],
                        input=source,
                        cwd=self.root,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    if value == "bad":
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("DRAFT_KV_REUSE", result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        args = result.stdout.split("\0")[:-1]
                        self.assertEqual("--draft-kv-reuse" in args, value == "1")
                        self.assertNotIn("--dsv4-kv-reuse", args)

    def test_offline_eval_saved_paths_and_runtime_defaults_allow_overrides(self):
        settings = {
            "VERIFIER_MODEL": (
                "--verifier-model",
                "/mnt/nfs/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16",
                "/fixture/other target",
            ),
            "DRAFT_MODEL": (
                "--draft-model",
                "output/dspark_dsv4_flash_bestArch/checkpoints/9/",
                "/fixture/other draft",
            ),
            "DATASETS_ROOT": (
                "--datasets-root",
                "../DeepSpec/eval_datasets",
                "/fixture/other data",
            ),
            "HS_PATH": (
                "--hidden-states-path",
                "/mnt/nfs/dataset/tmp_hs",
                "/fixture/hs",
            ),
            "VLLM_ENDPOINT": (
                "--vllm-endpoint",
                "http://80.48.17.187:8001/v1",
                "http://target.fixture:9001/v1",
            ),
            "EVAL_NPU": ("--ascend-devices", ",".join(map(str, range(16))), "8,9"),
            "MAX_SAMPLES": ("--max-samples", None, "12"),
            "MAX_NEW_TOKENS": ("--max-new-tokens", "2048", "128"),
            "DSV4_MAX_MODEL_LEN": ("--dsv4-max-model-len", "8192", "4096"),
            "TARGET_REQUEST_TIMEOUT": ("--target-request-timeout", "1200", "45"),
            "VERIFICATION_MODE": ("--dsv4-verification-mode", "block", "reference"),
            "DSV4_BLOCK_OUTPUT": ("--dsv4-block-output", "full", "auto"),
            "DSV4_REPLAY_CACHE": (
                "--dsv4-replay-cache",
                "./dsv4_greedy_traces",
                "/fixture/cache traces",
            ),
            "DSV4_REPLAY_CACHE_TAG": (
                "--dsv4-replay-cache-tag",
                "stack-v1",
                "runtime-a",
            ),
            "DSV4_REPLAY_AUDIT_SAMPLES": ("--dsv4-replay-audit-samples", "0", "2"),
        }
        script = ROOT / "examples/evaluate/dspark_dsv4_offline_eval.sh"
        for use_overrides in (False, True):
            with self.subTest(use_overrides=use_overrides):
                environment = _shell_environment(
                    {
                        **{
                            name: override if use_overrides else None
                            for name, (_, _, override) in settings.items()
                        },
                        "HS_HTTP_ENDPOINT": "",
                        "KEEP_TARGET_HS": "0",
                        "SERVED_MODEL_NAME": "",
                        "DSV4_PROFILE": "1" if use_overrides else "0",
                        "DSV4_KV_REUSE": "1" if use_overrides else "0",
                        "DRAFT_KV_REUSE": "0" if use_overrides else None,
                    }
                )
                source = r"""exec() { printf '%s\0' "$@"; }"""
                source += f"\nsource {shlex.quote(script.as_posix())}\n"
                result = subprocess.run(  # noqa: S603 -- Capture argv, no real eval.
                    [BASH, "--noprofile", "--norc"],
                    input=source,
                    cwd=self.root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.endswith("\0"))
                args = result.stdout.split("\0")[:-1]
                self.assertEqual("--dsv4-profile" in args, use_overrides)
                self.assertEqual("--dsv4-kv-reuse" in args, use_overrides)
                self.assertEqual("--draft-kv-reuse" in args, not use_overrides)
                for flag, default, override in settings.values():
                    if not use_overrides and default is None:
                        self.assertNotIn(flag, args)
                        continue
                    self.assertEqual(args.count(flag), 1)
                    self.assertEqual(
                        args[args.index(flag) + 1],
                        override if use_overrides else default,
                    )
                devices = settings["EVAL_NPU"][2 if use_overrides else 1]
                self.assertIn(f"ASCEND_RT_VISIBLE_DEVICES={devices}", args)

    def test_eval_scripts_export_both_source_packages(self):
        for name in (
            "dspark_dsv4_offline_eval.sh",
            "dspark_dsv4_single_eval.sh",
            "dspark_offline_jsonl.sh",
        ):
            for inherited in ("", "/fixture/user code:/fixture/other"):
                with self.subTest(script=name, inherited=inherited):
                    environment = _shell_environment(
                        {
                            "PYTHONPATH": None,
                            "FIXTURE_PYTHONPATH": inherited,
                            "VERIFIER_MODEL": "/fixture/target",
                            "DRAFT_MODEL": "/fixture/draft",
                            "DATASETS_ROOT": "/fixture/eval",
                            "HS_PATH": "/fixture/hs",
                            "VLLM_NPUS": "0,1",
                            "EVAL_NPU": "2,3",
                            "HS_HTTP_ENDPOINT": "",
                            "KEEP_TARGET_HS": "0",
                            "ALLOW_SHARED_DEVICE": "0",
                            "SKIP_ARTIFACTS": "0",
                            "DRY_RUN": "0",
                        }
                    )
                    source = r"""
export PYTHONPATH="$FIXTURE_PYTHONPATH"
exec() {
  printf '%s\0' "$REPO_ROOT"
  env | while IFS='=' read -r name value; do
    if [[ "$name" == PYTHONPATH ]]; then printf '%s\0' "$value"; fi
  done
}
"""
                    script = ROOT / "examples/evaluate" / name
                    source += f"\nsource {shlex.quote(script.as_posix())}\n"
                    result = subprocess.run(  # noqa: S603 -- No actual eval process.
                        [BASH, "--noprofile", "--norc"],
                        input=source,
                        cwd=self.root,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    root, pythonpath, _ = result.stdout.split("\0")
                    self.assertEqual(
                        pythonpath,
                        f"{root}/src:{root}/hs_connectors/src:{root}:{inherited}",
                    )

    def test_http_eval_and_sidecar_wiring_without_shared_storage(self):
        for sidecar, use_defaults in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            relative = (
                "train/dspark_dsv4_hs_http_server.sh"
                if sidecar
                else "evaluate/dspark_dsv4_offline_eval.sh"
            )
            script = ROOT / "examples" / relative
            environment = _shell_environment(
                {
                    "VERIFIER_MODEL": "/fixture/target",
                    "DRAFT_MODEL": "/fixture/draft",
                    "DATASETS_ROOT": "/fixture/eval",
                    "HS_PATH": None
                    if use_defaults
                    else "/target/hs"
                    if sidecar
                    else "",
                    "VLLM_ENDPOINT": "http://target.fixture:8001/v1",
                    "EVAL_NPU": "8,9",
                    "HS_HTTP_ENDPOINT": None
                    if use_defaults
                    else "http://target.fixture:8002",
                    "HS_HTTP_HOST": None if use_defaults else "10.0.0.10",
                    "HS_HTTP_PORT": None if use_defaults else "8002",
                    "HS_HTTP_MAX_FILE_BYTES": None,
                    "DSV4_HS_HTTP_TOKEN": None
                    if use_defaults
                    else "fixture-secret-01234567890123456789",
                    "OUTPUT_DIR": "/eval/output",
                    "KEEP_TARGET_HS": "0",
                }
            )
            source = r"""exec() { printf '%s\0' "$@"; }"""
            source += f"\nsource {shlex.quote(script.as_posix())}\n"
            result = subprocess.run(  # noqa: S603 -- Fake exec, no service startup.
                [BASH, "--noprofile", "--norc"],
                input=source,
                cwd=self.root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            args = result.stdout.split("\0")[:-1]
            if not use_defaults:
                self.assertNotIn(environment["DSV4_HS_HTTP_TOKEN"], args)
            self.assertFalse(any(re.fullmatch(r"[0-9a-f]{64}", arg) for arg in args))
            if sidecar:
                self.assertIn("speculators_dsv4.hs_http_server", args)
                self.assertEqual(
                    args[args.index("--host") + 1],
                    "80.48.17.187" if use_defaults else "10.0.0.10",
                )
                self.assertEqual(args[args.index("--port") + 1], "8002")
                self.assertEqual(args[args.index("--max-file-bytes") + 1], "536870912")
                self.assertEqual(
                    args[args.index("--hidden-states-path") + 1],
                    "/home/s00969542/DSV4F/tmp_hs" if use_defaults else "/target/hs",
                )
                self.assertNotIn("scripts/launch_vllm.py", args)
            else:
                self.assertEqual(
                    args[args.index("--hs-http-endpoint") + 1],
                    "http://80.48.17.187:8002"
                    if use_defaults
                    else "http://target.fixture:8002",
                )
                self.assertEqual(
                    args[args.index("--hidden-states-path") + 1],
                    "/eval/output/target-hs-downloads",
                )
                self.assertEqual(args[args.index("--ascend-devices") + 1], "8,9")

    def test_saved_http_tokens_match_without_putting_credentials_in_argv(self):
        tokens = []
        for relative in (
            "train/dspark_dsv4_hs_http_server.sh",
            "evaluate/dspark_dsv4_offline_eval.sh",
        ):
            source = (ROOT / "examples" / relative).read_text(encoding="utf-8")
            match = re.search(r"DSV4_HS_HTTP_TOKEN:=([0-9a-f]{64})\}", source)
            self.assertIsNotNone(match)
            tokens.append(match.group(1))
            self.assertIn("export DSV4_HS_HTTP_TOKEN", source)
        # Avoid printing credentials in assertion failures.
        self.assertTrue(tokens[0] == tokens[1], "Saved client/server tokens differ")

    def test_inherited_shell_startup_is_not_executed(self):
        startup, marker = self.root / "startup.sh", self.root / "startup-ran"
        startup.write_bytes(b'printf "startup\\n" >> "$STARTUP_MARKER"\n')
        (self.output / "logs").mkdir(parents=True)
        with patch.dict(
            os.environ,
            {
                "BASH_ENV": startup.as_posix(),
                "ENV": startup.as_posix(),
                "STARTUP_MARKER": marker.as_posix(),
            },
        ):
            control = subprocess.run(  # noqa: S603 -- Only a temporary printf hook.
                [BASH, "--noprofile", "--norc"],
                input=":\n",
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(control.returncode, 0, control.stderr)
            self.assertEqual(marker.read_text(), "startup\n")
            marker.unlink()
            for kind, smoke in (("server", "0"), ("trainer", "0"), ("trainer", "1")):
                with self.subTest(kind=kind, smoke=smoke):
                    result = self.run_script(
                        kind,
                        TRAINING_SMOKE=smoke,
                        SMOKE_PHASE="fresh",
                        SMOKE_REPORT_DIR=(self.root / "reports").as_posix(),
                    )
                    self.assertEqual(
                        result.returncode, 17 if smoke == "1" else 0, result.stderr
                    )
                    self.assertTrue(self.capture.exists())
                    self.assertFalse(marker.exists())

    def test_trainer_exports_shared_ascend_settings_to_background_and_smoke(self):
        (self.output / "logs").mkdir(parents=True)
        for smoke in ("0", "1"):
            with self.subTest(smoke=smoke):
                result = self.run_script(
                    "trainer",
                    TRAINING_SMOKE=smoke,
                    SMOKE_PHASE="fresh",
                    SMOKE_REPORT_DIR=(self.root / "reports").as_posix(),
                )
                self.assertEqual(
                    result.returncode, 17 if smoke == "1" else 0, result.stderr
                )
                captured = self.runtime_environment_capture.read_text().splitlines()
                self.assertEqual(
                    dict(line.split("=", 1) for line in captured),
                    {
                        "OMP_PROC_BIND": "false",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "VE_OMP_NUM_THREADS": "1",
                        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                        "TASK_QUEUE_ENABLE": "2",
                        "ACLNN_CACHE_LIMIT": "100000",
                        "NPU_ASD_ENABLE": "0",
                        "ASCEND_LAUNCH_BLOCKING": "0",
                    },
                )

    def test_trainer_sources_helper_from_foreign_cwd_without_rebasing_output(self):
        caller = self.root / "caller with spaces"
        (caller / "relative output/logs").mkdir(parents=True)
        for smoke in ("0", "1"):
            with self.subTest(smoke=smoke):
                result = self.run_script(
                    "trainer",
                    cwd=caller,
                    OUTPUT_DIR="relative output",
                    TRAINING_SMOKE=smoke,
                    SMOKE_PHASE="fresh",
                    SMOKE_REPORT_DIR=(self.root / "reports").as_posix(),
                )
                self.assertEqual(
                    result.returncode, 17 if smoke == "1" else 0, result.stderr
                )
                args = self.capture.read_text().splitlines()
                self.assertEqual(
                    args[args.index("--save-path") + 1], "relative output/checkpoints"
                )
                self.assertIn("torchrun", args)

    def test_server_captures_pid_waits_and_cleans_only_owned_group(self):
        result = self.run_script("server")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("server ready", result.stdout)
        args = self.capture.read_text().splitlines()
        self.assertEqual(args[args.index("--data-parallel-size") + 1], "2")
        self.assertEqual(args[args.index("--data-parallel-size-local") + 1], "2")
        self.assertNotIn("--headless", args)
        self.assertNotIn("--data-parallel-address", args)
        self.assertNotIn("--data-parallel-start-rank", args)
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        self.assertEqual(args[args.index("--port") + 1], "9123")
        self.assertEqual(args[args.index("--dsv4-execution-mode") + 1], "eager")
        self.assertLess(args.index("--dsv4-execution-mode"), args.index("--"))
        self.assertGreater(args.index("--no-async-scheduling"), args.index("--"))
        self.assertNotIn("--async-scheduling", args)
        self.assertNotIn("--dsv4-block-verify", args)
        self.assertEqual(args[args.index("--max-num-seqs") + 1], "64")
        self.assertIn("execution mode: eager; async scheduling: 0", result.stdout)
        self.assertRegex(self.signals.read_text(), r"^-TERM -- -[1-9][0-9]*\n$")

    def test_server_preserves_user_paths_and_block_defaults(self):
        result = self.run_script(
            "server",
            MODEL=None,
            HS_PATH=None,
            VLLM_HOST="",
            VLLM_PORT=None,
            TARGET_LOCAL_IP=None,
            TARGET_IFNAME=None,
            DSV4_EVAL=None,
            DSV4_BLOCK_VERIFY=None,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertIn("/mnt/nfs/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16", args)
        for flag, expected in (
            ("--hidden-states-path", "/home/s00969542/DSV4F/tmp_hs"),
            ("--host", "80.48.17.187"),
            ("--port", "8001"),
            ("--dsv4-manifest-timeout", "600"),
            ("--max-num-seqs", "16"),
            ("--max-logprobs", "0"),
            ("--max-model-len", "8192"),
            ("--max-num-batched-tokens", "8192"),
        ):
            self.assertEqual(args[args.index(flag) + 1], expected)
        self.assertIn("--dsv4-block-verify", args)
        self.assertNotIn("--logprobs-mode", args)
        self.assertEqual(self.proxy_environment()["HCCL_IF_IP"], "80.48.17.187")
        self.assertEqual(self.proxy_environment()["HCCL_SOCKET_IFNAME"], "enp48s3u1u1")

    def test_server_block_opt_in_preserves_dp_and_disables_http_logprobs(self):
        for dp in ("1", "2"):
            with self.subTest(dp=dp):
                result = self.run_script(
                    "server", DP_SIZE=dp, DSV4_BLOCK_VERIFY="1", DSV4_EVAL="1"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                args = self.capture.read_text().splitlines()
                self.assertLess(args.index("--dsv4-block-verify"), args.index("--"))
                for flag, expected in (
                    ("--tensor-parallel-size", "8"),
                    ("--data-parallel-size", dp),
                    ("--data-parallel-size-local", dp),
                    ("--max-num-seqs", "16"),
                    ("--max-num-batched-tokens", "8192"),
                    ("--max-logprobs", "0"),
                    ("--generation-config", "vllm"),
                ):
                    self.assertEqual(args[args.index(flag) + 1], expected)
                self.assertNotIn("--logprobs-mode", args)
                self.assertIn("--no-async-scheduling", args)
                self.assertIn("max sequences per DP engine: 16", result.stdout)

    def test_server_block_concurrency_can_restore_low_memory_limit(self):
        result = self.run_script("server", DSV4_BLOCK_VERIFY="1", MAX_NUM_SEQS="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertEqual(args[args.index("--max-num-seqs") + 1], "1")
        self.assertEqual(args[args.index("--max-num-batched-tokens") + 1], "8192")

    def test_server_rejects_unsafe_block_options_before_launch(self):
        for overrides in (
            {"DSV4_BLOCK_VERIFY": "2"},
            {"DP_SIZE": "4"},
            {"MAX_NUM_SEQS": "0"},
            {"MAX_NUM_SEQS": "-1"},
            {"MAX_NUM_SEQS": "1.5"},
            {"DSV4_ASYNC_SCHEDULING": "1"},
            {"DSV4_EXECUTION_MODE": "full-decode-only"},
        ):
            with self.subTest(overrides=overrides):
                result = self.run_script(
                    "server", **{"DSV4_BLOCK_VERIFY": "1", **overrides}
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.checkpoint_capture.exists())
                self.assertFalse(self.capture.exists())

    def test_server_allows_explicit_block_batch_and_token_budget(self):
        result = self.run_script(
            "server",
            DSV4_BLOCK_VERIFY="1",
            MAX_NUM_SEQS="2",
            MAX_NUM_BATCHED_TOKENS="16384",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertEqual(args[args.index("--max-num-seqs") + 1], "2")
        self.assertEqual(args[args.index("--max-num-batched-tokens") + 1], "16384")

    def test_two_host_head_starts_local_engines_and_keeps_http_readiness(self):
        result = self.run_script("server", DP_SIZE="4", DP_ADDRESS="10.0.0.10")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("server ready", result.stdout)
        args = self.capture.read_text().splitlines()
        for flag, expected in (
            ("--tensor-parallel-size", "8"),
            ("--data-parallel-size", "4"),
            ("--data-parallel-size-local", "2"),
            ("--data-parallel-start-rank", "0"),
            ("--data-parallel-address", "10.0.0.10"),
            ("--data-parallel-rpc-port", "13345"),
            ("--data-parallel-backend", "mp"),
            ("--dsv4-manifest-timeout", "600"),
        ):
            self.assertEqual(args[args.index(flag) + 1], expected)
        self.assertLess(args.index("--dsv4-manifest-timeout"), args.index("--"))
        self.assertNotIn("--headless", args)
        self.assertIn("--enable-tokenizer-info-endpoint", args)
        self.assertIn("--host", args)
        self.assertIn("--port", args)
        self.assertEqual(len(self.curl_capture.read_text().splitlines()), 2)
        for name in (
            "VLLM_DP_SIZE",
            "VLLM_DP_RANK",
            "VLLM_DP_RANK_LOCAL",
            "VLLM_DP_MASTER_IP",
            "VLLM_DP_MASTER_PORT",
        ):
            self.assertEqual(args[args.index(name) - 1], "-u")

    def test_two_host_communication_environment_is_not_overridden(self):
        communication = {
            "HCCL_IF_IP": "10.0.0.20",
            "GLOO_SOCKET_IFNAME": "gloo-fixture",
            "TP_SOCKET_IFNAME": "tp-fixture",
            "HCCL_SOCKET_IFNAME": "hccl-fixture",
        }
        result = self.run_script(
            "server",
            DP_SIZE="4",
            DP_START_RANK="2",
            DP_ADDRESS="10.0.0.10",
            TARGET_LOCAL_IP="",
            TARGET_IFNAME="",
            **communication,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.proxy_environment(), communication)

    def test_explicit_dp4_master_sets_api_and_rendezvous(self):
        result = self.run_script(
            "server",
            DP_SIZE="4",
            VLLM_HOST="",
            TARGET_LOCAL_IP="80.48.17.186",
            TARGET_MASTER_IP="80.48.17.186",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        for flag in ("--host", "--data-parallel-address"):
            self.assertEqual(args[args.index(flag) + 1], "80.48.17.186")
        self.assertEqual(
            self.proxy_environment(),
            {
                "HCCL_IF_IP": "80.48.17.186",
                "GLOO_SOCKET_IFNAME": "eth-fixture",
                "TP_SOCKET_IFNAME": "eth-fixture",
                "HCCL_SOCKET_IFNAME": "eth-fixture",
            },
        )

    def test_user_network_defaults_select_local_ip_by_start_rank(self):
        for rank, local_ip in (("0", "80.48.17.186"), ("2", "80.48.17.187")):
            with self.subTest(rank=rank):
                result = self.run_script(
                    "server",
                    DP_SIZE="4",
                    DP_START_RANK=rank,
                    TARGET_MASTER_IP="80.48.17.186",
                    VLLM_HOST="",
                    TARGET_LOCAL_IP=None,
                    TARGET_IFNAME=None,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                args = self.capture.read_text().splitlines()
                self.assertEqual(
                    args[args.index("--data-parallel-address") + 1], "80.48.17.186"
                )
                self.assertEqual(
                    self.proxy_environment(),
                    {
                        "HCCL_IF_IP": local_ip,
                        "GLOO_SOCKET_IFNAME": "enp48s3u1u1",
                        "TP_SOCKET_IFNAME": "enp48s3u1u1",
                        "HCCL_SOCKET_IFNAME": "enp48s3u1u1",
                    },
                )

    def test_custom_worker_ip_is_used_only_for_worker_local_interface(self):
        result = self.run_script(
            "server",
            DP_SIZE="4",
            DP_START_RANK="2",
            TARGET_WORKER_IP="10.0.0.88",
            TARGET_MASTER_IP="80.48.17.186",
            TARGET_LOCAL_IP=None,
            TARGET_IFNAME=None,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertEqual(
            args[args.index("--data-parallel-address") + 1], "80.48.17.186"
        )
        self.assertEqual(self.proxy_environment()["HCCL_IF_IP"], "10.0.0.88")

    def test_worker_uses_own_local_ip_and_common_nic_not_master_ip(self):
        result = self.run_script(
            "server",
            DP_SIZE="4",
            DP_START_RANK="2",
            TARGET_LOCAL_IP="80.48.17.187",
            TARGET_MASTER_IP="80.48.17.186",
            TARGET_IFNAME="worker-nic",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertEqual(
            args[args.index("--data-parallel-address") + 1], "80.48.17.186"
        )
        self.assertEqual(
            self.proxy_environment(),
            {
                "HCCL_IF_IP": "80.48.17.187",
                "GLOO_SOCKET_IFNAME": "worker-nic",
                "TP_SOCKET_IFNAME": "worker-nic",
                "HCCL_SOCKET_IFNAME": "worker-nic",
            },
        )
        self.assertIn(
            "master=80.48.17.186:13345, local HCCL IP=80.48.17.187", result.stdout
        )

    def test_native_network_overrides_take_precedence_over_common_defaults(self):
        result = self.run_script(
            "server",
            DP_SIZE="4",
            TARGET_LOCAL_IP="10.0.0.20",
            TARGET_IFNAME="fallback-nic",
            HCCL_IF_IP="10.0.0.21",
            GLOO_SOCKET_IFNAME="gloo-override",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.proxy_environment(),
            {
                "HCCL_IF_IP": "10.0.0.21",
                "GLOO_SOCKET_IFNAME": "gloo-override",
                "TP_SOCKET_IFNAME": "fallback-nic",
                "HCCL_SOCKET_IFNAME": "fallback-nic",
            },
        )

    def test_custom_master_ip_updates_api_and_dp_address(self):
        result = self.run_script(
            "server", DP_SIZE="4", TARGET_MASTER_IP="10.0.0.99", VLLM_HOST=""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        for flag in ("--host", "--data-parallel-address"):
            self.assertEqual(args[args.index(flag) + 1], "10.0.0.99")
        self.assertEqual(self.proxy_environment()["HCCL_IF_IP"], "10.0.0.10")

    def test_single_host_network_can_remain_unconfigured(self):
        result = self.run_script("server", TARGET_LOCAL_IP="", TARGET_IFNAME="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(not value for value in self.proxy_environment().values()))

    def test_trainer_default_endpoint_tracks_master_and_no_proxy(self):
        (self.output / "logs").mkdir(parents=True)
        for master, expected in (("", "80.48.17.186"), ("10.0.0.99", "10.0.0.99")):
            with self.subTest(master=master):
                result = self.run_script(
                    "trainer", VLLM_ENDPOINT="", TARGET_MASTER_IP=master
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                args = self.capture.read_text().splitlines()
                self.assertEqual(
                    args[args.index("--vllm-endpoint") + 1],
                    f"http://{expected}:8001/v1",
                )
                self.assertIn(expected, self.proxy_environment()["NO_PROXY"].split(","))

    def test_single_host_dp1_local_size_still_defaults_to_one(self):
        result = self.run_script("server", DP_SIZE="1", VLLM_NPUS="0,1,2,3,4,5,6,7")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.capture.read_text().splitlines()
        self.assertEqual(args[args.index("--data-parallel-size") + 1], "1")
        self.assertEqual(args[args.index("--data-parallel-size-local") + 1], "1")
        self.assertNotIn("--data-parallel-address", args)
        self.assertNotIn("--data-parallel-start-rank", args)
        self.assertNotIn("--headless", args)

    def test_headless_host_never_checks_http_or_claims_cluster_readiness(self):
        result = self.run_script(
            "server",
            DP_SIZE="4",
            DP_SIZE_LOCAL="2",
            DP_START_RANK="2",
            DP_ADDRESS="10.0.0.10",
            DP_RPC_PORT="24455",
            DSV4_MANIFEST_TIMEOUT="600",
            MODE="occupied",  # Would reject an API launch; headless never calls curl.
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NOT cluster readiness", result.stdout)
        self.assertNotIn("server ready", result.stdout)
        self.assertFalse(self.curl_capture.exists())
        args = self.capture.read_text().splitlines()
        self.assertIn("--headless", args)
        self.assertEqual(args[args.index("--data-parallel-start-rank") + 1], "2")
        self.assertEqual(args[args.index("--data-parallel-rpc-port") + 1], "24455")
        self.assertEqual(args[args.index("--dsv4-manifest-timeout") + 1], "600")
        for flag in ("--host", "--port", "--enable-tokenizer-info-endpoint"):
            self.assertNotIn(flag, args)
        self.assertRegex(self.signals.read_text(), r"^-TERM -- -[1-9][0-9]*\n$")

    def test_headless_host_propagates_worker_failure_and_cleans_owned_group(self):
        result = self.run_script(
            "server",
            DP_SIZE="4",
            DP_START_RANK="2",
            DP_ADDRESS="10.0.0.10",
            WAIT_STATUS="7",
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertFalse(self.curl_capture.exists())
        self.assertTrue(self.signals.exists())

    def test_invalid_parallel_settings_fail_before_checkpoint_or_process_launch(self):
        for settings in (
            {"DP_SIZE": "3"},
            {"DP_SIZE": "8"},
            {"DP_SIZE": "4", "TARGET_LOCAL_IP": ""},
            {"DP_SIZE": "4", "TARGET_IFNAME": ""},
            {"DP_SIZE": "4", "TARGET_IFNAME": "", "GLOO_SOCKET_IFNAME": "gloo-only"},
            {"DP_SIZE": "4", "DP_ADDRESS": "10.0.0.10", "DP_SIZE_LOCAL": "4"},
            {"DP_SIZE": "4", "DP_ADDRESS": "10.0.0.10", "DP_START_RANK": "1"},
            {"DP_SIZE": "4", "DP_ADDRESS": "10.0.0.10", "DP_START_RANK": "4"},
            {"DP_SIZE": "4", "DP_ADDRESS": "10.0.0.10", "DP_RPC_PORT": "65536"},
            {"DP_SIZE": "4", "DP_ADDRESS": "10.0.0.10", "DP_RPC_PORT": "abc"},
            {"DP_SIZE": "4", "DP_ADDRESS": "10.0.0.10", "HS_PATH": "relative/hs"},
            {"DP_SIZE": "2", "DP_SIZE_LOCAL": "1"},
            {"DP_SIZE": "2", "DP_START_RANK": "2"},
            {"DP_SIZE": "1", "DP_ADDRESS": "10.0.0.10"},
            {"DSV4_MANIFEST_TIMEOUT": "0"},
            {"DSV4_MANIFEST_TIMEOUT": "nan"},
        ):
            with self.subTest(settings=settings):
                result = self.run_script("server", **settings)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(self.checkpoint_capture.exists())
                self.assertFalse(self.capture.exists())
                self.assertFalse(self.curl_capture.exists())
                self.assertFalse(self.signals.exists())

    def test_server_execution_mode_and_async_scheduling_are_independent(self):
        for execution_mode in ("eager", "full-decode-only"):
            for async_scheduling in ("0", "1"):
                with self.subTest(
                    mode=execution_mode, async_scheduling=async_scheduling
                ):
                    result = self.run_script(
                        "server",
                        DSV4_EXECUTION_MODE=execution_mode,
                        DSV4_ASYNC_SCHEDULING=async_scheduling,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    args = self.capture.read_text().splitlines()
                    self.assertEqual(args.count("--dsv4-execution-mode"), 1)
                    self.assertEqual(
                        args[args.index("--dsv4-execution-mode") + 1], execution_mode
                    )
                    self.assertLess(
                        args.index("--dsv4-execution-mode"), args.index("--")
                    )
                    enabled_flag, disabled_flag = (
                        ("--async-scheduling", "--no-async-scheduling")
                        if async_scheduling == "1"
                        else ("--no-async-scheduling", "--async-scheduling")
                    )
                    self.assertEqual(args.count(enabled_flag), 1)
                    self.assertNotIn(disabled_flag, args)
                    self.assertGreater(args.index(enabled_flag), args.index("--"))
                    self.assertIn(
                        f"execution mode: {execution_mode}; "
                        f"async scheduling: {async_scheduling}",
                        result.stdout,
                    )
                    for flag, value in (
                        ("--tensor-parallel-size", "8"),
                        ("--data-parallel-size", "2"),
                        ("--data-parallel-size-local", "2"),
                        ("--max-model-len", "8192"),
                        ("--max-num-batched-tokens", "8192"),
                        ("--max-num-seqs", "64"),
                    ):
                        self.assertEqual(args[args.index(flag) + 1], value)

    def test_invalid_server_performance_settings_fail_before_checkpoint_check(self):
        for variable, values, expected in (
            (
                "DSV4_EXECUTION_MODE",
                ("full", "FULL_DECODE_ONLY", "0"),
                "DSV4_EXECUTION_MODE must be eager or full-decode-only",
            ),
            (
                "DSV4_ASYNC_SCHEDULING",
                ("true", "2", "-1"),
                "DSV4_ASYNC_SCHEDULING must be 0 or 1",
            ),
        ):
            for value in values:
                with self.subTest(variable=variable, value=value):
                    result = self.run_script("server", **{variable: value})
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(expected, result.stderr)
                    self.assertFalse(self.checkpoint_capture.exists())
                    self.assertFalse(self.capture.exists())
                    self.assertFalse(self.signals.exists())

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

    def test_recompute_default_and_overrides_reach_normal_and_smoke_training(self):
        (self.output / "logs").mkdir(parents=True)
        for recompute in ("", "0", "1"):
            for smoke in ("0", "1"):
                with self.subTest(recompute=recompute, smoke=smoke):
                    result = self.run_script(
                        "trainer",
                        RECOMPUTE=recompute,
                        TRAINING_SMOKE=smoke,
                        SMOKE_PHASE="fresh",
                        SMOKE_REPORT_DIR=(self.root / "reports").as_posix(),
                    )
                    self.assertEqual(
                        result.returncode, 17 if smoke == "1" else 0, result.stderr
                    )
                    args = self.capture.read_text().splitlines()
                    self.assertEqual(
                        args.count("--activation-checkpointing"),
                        0 if recompute == "0" else 1,
                    )
                    self.assertNotIn("--fsdp-shard", args)
                    if smoke == "1" and recompute != "0":
                        self.assertGreater(
                            args.index("--activation-checkpointing"), args.index("--")
                        )

    def test_invalid_recompute_setting_rejected_before_launch(self):
        for value in ("true", "2", "-1"):
            for smoke in ("0", "1"):
                with self.subTest(value=value, smoke=smoke):
                    result = self.run_script(
                        "trainer", RECOMPUTE=value, TRAINING_SMOKE=smoke
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("RECOMPUTE must be 0 or 1", result.stderr)
                    self.assertFalse(self.capture.exists())
                    self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
