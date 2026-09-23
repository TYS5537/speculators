"""Single-host lifecycle wrapper around the DSV4 offline evaluator.

The controller deliberately imports neither torch nor vLLM. Target and draft
initialize NPU runtimes in separate, explicitly configured child processes.
The block and reference target transports share speculative acceptance statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import secrets
import signal
import socket
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4

from speculators_dsv4.contract import (
    ensure_manifest,
    inspect_checkpoint,
    make_manifest,
)
from speculators_dsv4.eval_contract import validate_draft_target
from speculators_dsv4.managed_process import start_process, stop_process

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
RANK_ENVIRONMENT = (
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)
_MAX_PORT = 65535
_HTTP_TIMEOUT = 3.0
_PROGRESS_INTERVAL = 30.0
_POLL_INTERVAL = 0.25
_MAX_RESPONSE_BYTES = 1024 * 1024


@dataclass
class LaunchPlan:
    run_id: str
    output_dir: Path
    hidden_states_path: Path
    model_name: str
    port: int
    endpoint: str
    target_command: list[str]
    eval_command: list[str]
    target_env: dict[str, str] = field(repr=False)
    eval_env: dict[str, str] = field(repr=False)
    api_key: str = field(repr=False)
    report: dict
    layer_ids: list[int]
    target_devices: list[int]
    eval_devices: list[int]
    shared_device: bool
    target_memory_utilization: float
    target_quantization: str | None
    verification_mode: str
    startup_timeout: float
    shutdown_timeout: float
    interrupted_signal: int | None = field(default=None, repr=False)

    @property
    def eval_device(self):
        """Preserve the single-device metadata field without hiding extra workers."""
        return self.eval_devices[0] if len(self.eval_devices) == 1 else None

    def public_metadata(self):
        """Never serialize API tokens or inherited process environments."""
        return {
            "run_id": self.run_id,
            "mode": f"single-host-managed-{self.verification_mode}",
            "verification_mode": self.verification_mode,
            "online_speedup_benchmark": False,
            "output_dir": str(self.output_dir),
            "hidden_states_path": str(self.hidden_states_path),
            "served_model_name": self.model_name,
            "endpoint": self.endpoint,
            "target_devices": self.target_devices,
            "eval_device": self.eval_device,
            "eval_devices": self.eval_devices,
            "eval_num_workers": len(self.eval_devices),
            "shared_device": self.shared_device,
            "target_memory_utilization": self.target_memory_utilization,
            "runtime_quantization": {"method": self.target_quantization},
            "checkpoint_signature": self.report["checkpoint_signature"],
            "auxiliary_hs_ids": self.layer_ids,
            "startup_timeout": self.startup_timeout,
            "shutdown_timeout": self.shutdown_timeout,
            "target_command": self.target_command,
            "eval_command": self.eval_command,
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument(
        "--hidden-states-path",
        type=Path,
        required=True,
        help="Parent directory for a fresh per-run HS directory",
    )
    parser.add_argument(
        "--target-devices", required=True, help="Physical NPU IDs, e.g. 0,1,2,3"
    )
    parser.add_argument(
        "--eval-device",
        "--eval-devices",
        required=True,
        help="Comma-separated physical evaluation NPU IDs; one draft worker per NPU",
    )
    parser.add_argument("--target-tp-size", type=int, default=None)
    parser.add_argument("--target-python", default=sys.executable)
    parser.add_argument("--eval-python", default=sys.executable)
    parser.add_argument("--target-quantization", default=None)
    parser.add_argument(
        "--verification-mode",
        choices=["block", "reference"],
        default="block",
        help="Block: one full-prefix forward; reference: one request per position",
    )
    parser.add_argument(
        "--target-memory-utilization",
        type=float,
        default=None,
        help="Explicit target budget required when sharing an NPU",
    )
    parser.add_argument("--allow-shared-device", action="store_true")
    parser.add_argument(
        "--port", type=int, default=0, help="Loopback port; 0 selects a free port"
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--shutdown-timeout", type=float, default=30.0)
    parser.add_argument("--target-request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dspark_dsv4_single_eval")
    )
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument("--datasets", default=None)
    parser.add_argument(
        "--enable-thinking", choices=["false", "true", "default"], default="false"
    )
    parser.add_argument(
        "--raw-prompt-mode", choices=["auto", "chat_template", "raw"], default="auto"
    )
    parser.add_argument("--keep-target-hs", action="store_true")
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print commands without launching or writing files",
    )
    return parser.parse_args(argv)


def parse_devices(value):
    parts = value.split(",")
    if any(re.fullmatch(r"0|[1-9]\d*", part.strip()) is None for part in parts):
        raise ValueError("NPU devices must be comma-separated nonnegative integer IDs")
    devices = [int(part.strip()) for part in parts]
    if len(set(devices)) != len(devices):
        raise ValueError("NPU device IDs must not repeat")
    return devices


def choose_port(requested):
    if type(requested) is not int or not 0 <= requested <= _MAX_PORT:
        raise ValueError("--port must be between 0 and 65535")
    # Do not reuse or terminate an existing server. A later bind race is guarded
    # by the fresh authentication token AND served-model name during readiness.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", requested))
        return listener.getsockname()[1]


def _validate_options(args):
    if args.verification_mode not in {"block", "reference"}:
        raise ValueError("--verification-mode must be block or reference")
    for name in ("startup_timeout", "shutdown_timeout", "target_request_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("--temperature must be finite and nonnegative")
    if args.max_samples <= 0 or args.max_new_tokens <= 0 or args.max_model_len <= 1:
        raise ValueError("Invalid sample count, generation length or context limit")
    if not args.datasets_root.exists():
        raise ValueError(f"Dataset path does not exist: {args.datasets_root}")
    for executable in (args.target_python, args.eval_python):
        if not executable or executable.startswith("-") or "\x00" in executable:
            raise ValueError("Specify a Python executable, not shell command arguments")


def _device_config(args):
    target_devices = parse_devices(args.target_devices)
    eval_devices = parse_devices(args.eval_device)
    shared = bool(set(eval_devices) & set(target_devices))
    if shared and not args.allow_shared_device:
        raise ValueError(
            "Target/eval devices overlap; explicitly set --allow-shared-device"
        )
    if shared and args.target_memory_utilization is None:
        raise ValueError(
            "Shared NPU requires an explicit --target-memory-utilization budget"
        )
    memory = args.target_memory_utilization
    if memory is None:
        memory = 0.9
    if not math.isfinite(memory) or not 0 < memory < 1:
        raise ValueError("--target-memory-utilization must be strictly between 0 and 1")
    tp_size = (
        args.target_tp_size if args.target_tp_size is not None else len(target_devices)
    )
    if tp_size <= 0 or tp_size != len(target_devices):
        raise ValueError(
            "With DP=PP=1, target TP size must match selected target devices"
        )
    return target_devices, eval_devices, shared, memory, tp_size


def _draft_layers(args, report):
    config_path = args.draft_model.resolve() / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return validate_draft_target(config, report)


def _child_env(devices):
    environment = dict(os.environ)
    for name in RANK_ENVIRONMENT:
        environment.pop(name, None)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), str(REPO_ROOT), environment.get("PYTHONPATH", "")]
    )
    # OpenAI/httpx inherits proxy settings. Keep this run's prompts and API token
    # on loopback, just as the readiness client does, without disabling proxies
    # needed for unrelated package/model downloads.
    exclusions = [
        value.strip()
        for name in ("NO_PROXY", "no_proxy")
        for value in environment.get(name, "").split(",")
        if value.strip()
    ]
    exclusions.extend(["127.0.0.1", "localhost"])
    environment["NO_PROXY"] = environment["no_proxy"] = ",".join(
        dict.fromkeys(exclusions)
    )
    return environment


def _target_command(
    args, *, report, layers, hs_path, model_name, port, memory, tp_size
):
    command = [
        args.target_python,
        "-u",
        str(REPO_ROOT / "scripts/launch_vllm.py"),
        report["model_path"],
        "--dsv4",
        "--hidden-states-path",
        str(hs_path),
        "--target-layer-ids",
        *map(str, layers),
        *(["--dsv4-block-verify"] if args.verification_mode == "block" else []),
        "--",
        "--tensor-parallel-size",
        str(tp_size),
        "--data-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--enable-expert-parallel",
        "--tokenizer-mode",
        "deepseek_v4",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_model_len),
        "--max-num-seqs",
        # Block export supports one scheduled request. Multiple draft workers
        # share this target through its request queue, not batched verification.
        "1",
        "--block-size",
        "128",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        model_name,
        "--gpu-memory-utilization",
        str(memory),
        "--max-logprobs",
        # Block probabilities use the shared artifact, not full-vocabulary JSON.
        "0"
        if args.verification_mode == "block"
        else str(report["config"]["vocab_size"]),
        "--logprobs-mode",
        "raw_logprobs",
        "--generation-config",
        "vllm",
        "--additional-config",
        json.dumps({"enable_flashcomm1": False, "enable_dsa_cp": False}),
    ]
    if args.target_quantization:
        command.extend(["--quantization", args.target_quantization])
    return command


def _eval_command(
    args, *, report, hs_path, output_dir, endpoint, model_name, eval_devices
):
    command = [
        args.eval_python,
        "-u",
        str(REPO_ROOT / "scripts/evaluate/dspark_offline_eval.py"),
        "--target-backend",
        "dsv4-vllm",
        "--dsv4-verification-mode",
        args.verification_mode,
        "--verifier-model",
        report["model_path"],
        "--draft-model",
        str(args.draft_model.resolve()),
        "--datasets-root",
        str(args.datasets_root.resolve()),
        "--hidden-states-path",
        str(hs_path),
        "--output-dir",
        str(output_dir),
        "--vllm-endpoint",
        endpoint,
        "--served-model-name",
        model_name,
        "--dsv4-max-model-len",
        str(args.max_model_len),
        "--target-request-timeout",
        str(args.target_request_timeout),
        "--max-samples",
        str(args.max_samples),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--enable-thinking",
        args.enable_thinking,
        "--raw-prompt-mode",
        args.raw_prompt_mode,
        "--device",
        "npu:0",
        "--dtype",
        "bfloat16",
        "--draft-attn-impl",
        "sdpa",
        "--no-progress",
    ]
    if len(eval_devices) > 1:
        command.extend(["--ascend-devices", ",".join(map(str, eval_devices))])
    if args.datasets:
        command.extend(["--datasets", args.datasets])
    if args.keep_target_hs:
        command.append("--keep-target-hs")
    if args.skip_artifacts:
        command.append("--skip-artifacts")
    return command


def build_plan(args, *, run_id=None, port=None):
    """Read-only planning: no directories, child processes or NPU imports."""
    _validate_options(args)
    devices, eval_devices, shared, memory, tp_size = _device_config(args)
    report = inspect_checkpoint(args.verifier_model.resolve())
    layers = _draft_layers(args, report)
    run_id = (
        run_id
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:12]
    )
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("Invalid run identifier")
    output_dir = args.output_dir.resolve() / f"run-{run_id}"
    hs_path = args.hidden_states_path.resolve() / f"hs-{run_id}"
    if output_dir.exists() or hs_path.exists():
        raise ValueError("Refusing to reuse an existing run directory")
    port = choose_port(args.port) if port is None else port
    if type(port) is not int or not 0 < port <= _MAX_PORT:
        raise ValueError("Invalid target port")
    model_name = f"dspark-eval-{run_id}"
    endpoint = f"http://127.0.0.1:{port}/v1"
    api_key = secrets.token_urlsafe(32)
    target_env = _child_env(devices)
    target_env.update(
        {
            "VLLM_API_KEY": api_key,
            "OMP_PROC_BIND": "false",
            "OMP_NUM_THREADS": "1",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        }
    )
    eval_env = _child_env(eval_devices)
    eval_env["OPENAI_API_KEY"] = api_key
    return LaunchPlan(
        run_id=run_id,
        output_dir=output_dir,
        hidden_states_path=hs_path,
        model_name=model_name,
        port=port,
        endpoint=endpoint,
        target_command=_target_command(
            args,
            report=report,
            layers=layers,
            hs_path=hs_path,
            model_name=model_name,
            port=port,
            memory=memory,
            tp_size=tp_size,
        ),
        eval_command=_eval_command(
            args,
            report=report,
            hs_path=hs_path,
            output_dir=output_dir,
            endpoint=endpoint,
            model_name=model_name,
            eval_devices=eval_devices,
        ),
        target_env=target_env,
        eval_env=eval_env,
        api_key=api_key,
        report=report,
        layer_ids=layers,
        target_devices=devices,
        eval_devices=eval_devices,
        shared_device=shared,
        target_memory_utilization=memory,
        target_quantization=args.target_quantization,
        verification_mode=args.verification_mode,
        startup_timeout=args.startup_timeout,
        shutdown_timeout=args.shutdown_timeout,
    )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002 -- urllib override.
        raise RuntimeError("Refusing readiness redirect away from the managed target")


def _read_response(opener, url, api_key, timeout):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Readiness requests must stay on the local HTTP endpoint")
    request = Request(url, headers={"Authorization": f"Bearer {api_key}"})  # noqa: S310 -- Validated loopback HTTP only.
    with opener.open(request, timeout=timeout) as response:
        payload = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(payload) > _MAX_RESPONSE_BYTES:
            raise RuntimeError("Oversized target readiness response")
        return payload


def _probe_ready(opener, plan, timeout):
    _read_response(
        opener,
        plan.endpoint.removesuffix("/v1") + "/health",
        plan.api_key,
        timeout,
    )
    payload = _read_response(opener, plan.endpoint + "/models", plan.api_key, timeout)
    models = json.loads(payload)
    entries = models.get("data") if isinstance(models, dict) else None
    if not isinstance(entries, list) or not any(
        isinstance(item, dict) and item.get("id") == plan.model_name for item in entries
    ):
        raise RuntimeError(
            "Ready endpoint is not this run's target; refusing to use it"
        )


def wait_until_ready(process, plan):
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    deadline = time.monotonic() + plan.startup_timeout
    next_progress = time.monotonic()
    while True:
        _check_interrupted(plan)
        code = process.poll()
        if code is not None:
            raise RuntimeError(
                f"Target exited during startup (code {code}); inspect target.log"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Target startup timed out; inspect target.log")
        try:
            timeout = min(_HTTP_TIMEOUT, remaining)
            _probe_ready(opener, plan, timeout)
            _check_interrupted(plan)
            if process.poll() is not None:
                raise RuntimeError("Target exited just after readiness")
            return
        except HTTPError as error:
            if error.code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
                raise RuntimeError(
                    "Target authentication failed; possible port collision"
                ) from error
            if error.code != HTTPStatus.SERVICE_UNAVAILABLE:
                raise RuntimeError(
                    f"Unexpected target readiness HTTP status {error.code}"
                ) from error
        except (URLError, TimeoutError, ConnectionError):
            pass  # The child may still be loading its checkpoint.
        if time.monotonic() >= next_progress:
            logger.info("Waiting for target; log: %s", plan.output_dir / "target.log")
            next_progress = time.monotonic() + _PROGRESS_INTERVAL
        time.sleep(min(_POLL_INTERVAL, max(0, deadline - time.monotonic())))


class _LaunchInterruptedError(Exception):
    def __init__(self, signum):
        super().__init__(f"Interrupted by signal {signum}")
        self.signum = signum


def _check_interrupted(plan):
    if plan.interrupted_signal is not None:
        raise _LaunchInterruptedError(plan.interrupted_signal)


@contextmanager
def _handle_signals(plan):
    previous = {}

    def interrupt(signum, _frame):
        # Never raise asynchronously: a signal between Popen and callback
        # registration (or during cleanup itself) could orphan an owned child.
        # The wait loops raise at safe points after ownership is registered.
        if plan.interrupted_signal is None:
            plan.interrupted_signal = signum

    try:
        for name in (signal.SIGINT, signal.SIGTERM):
            previous[name] = signal.signal(name, interrupt)
        yield
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


def _record(plan, status, **extra):
    payload = plan.public_metadata()
    payload.update(
        status=status, updated_at=datetime.now(timezone.utc).isoformat(), **extra
    )
    (plan.output_dir / "launcher.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _wait_for_eval(target, evaluator, plan):
    next_progress = time.monotonic() + _PROGRESS_INTERVAL
    while True:
        _check_interrupted(plan)
        target_code = target.poll()
        if target_code is not None:
            raise RuntimeError(
                f"Target exited during evaluation (code {target_code}); "
                "inspect target.log"
            )
        eval_code = evaluator.poll()
        if eval_code is not None:
            return eval_code if eval_code >= 0 else 128 - eval_code
        if time.monotonic() >= next_progress:
            logger.info("Evaluation running; log: %s", plan.output_dir / "eval.log")
            next_progress = time.monotonic() + _PROGRESS_INTERVAL
        time.sleep(_POLL_INTERVAL)


def _require_posix():
    if os.name != "posix":
        raise RuntimeError(
            "Managed Ascend evaluation requires Linux; "
            "use --dry-run to inspect elsewhere"
        )


def run_plan(plan):
    _require_posix()
    plan.output_dir.mkdir(parents=True, exist_ok=False)
    # Reserve only this run's HS directory. The launcher creates its manifest;
    # no pre-existing HS files or services are adopted, moved or removed.
    plan.hidden_states_path.mkdir(parents=True, exist_ok=False)
    logger.info("Run directory: %s", plan.output_dir)
    logger.info(
        "Evaluation NPUs: %s (%d draft worker(s)); target requests are queued "
        "with max-num-seqs=1",
        plan.eval_devices,
        len(plan.eval_devices),
    )
    if plan.shared_device:
        logger.warning(
            "Shared NPU explicitly enabled; memory budget is not an OOM guarantee"
        )
    status, code, error_message = "failed", 1, None
    with _handle_signals(plan):
        try:
            with ExitStack() as resources:
                target_log = resources.enter_context(
                    (plan.output_dir / "target.log").open("w", encoding="utf-8")
                )
                eval_log = resources.enter_context(
                    (plan.output_dir / "eval.log").open("w", encoding="utf-8")
                )
                _record(plan, "starting_target")
                target = start_process(
                    plan.target_command,
                    cwd=REPO_ROOT,
                    env=plan.target_env,
                    stdout=target_log,
                )
                resources.callback(stop_process, target, timeout=plan.shutdown_timeout)
                _check_interrupted(plan)
                wait_until_ready(target, plan)
                ensure_manifest(
                    plan.hidden_states_path,
                    make_manifest(plan.report, plan.layer_ids),
                    runtime_quantization={"method": plan.target_quantization},
                )
                _record(plan, "evaluating")
                logger.info(
                    "Target ready; starting evaluation. Log: %s",
                    plan.output_dir / "eval.log",
                )
                evaluator = start_process(
                    plan.eval_command, cwd=REPO_ROOT, env=plan.eval_env, stdout=eval_log
                )
                resources.callback(
                    stop_process, evaluator, timeout=plan.shutdown_timeout
                )
                _check_interrupted(plan)
                code = _wait_for_eval(target, evaluator, plan)
                status = "completed" if code == 0 else "failed"
                if code:
                    error_message = (
                        f"Evaluator exited with code {code}; inspect eval.log"
                    )
            _check_interrupted(plan)
        except _LaunchInterruptedError as error:
            status, code, error_message = "interrupted", 128 + error.signum, str(error)
        except Exception as error:  # noqa: BLE001 -- Persist failure after ExitStack cleanup.
            status, code, error_message = "failed", 1, str(error)
        _record(plan, status, exit_code=code, error=error_message)
    if error_message:
        logger.error("%s", error_message)
    logger.info("%s; outputs and logs retained at %s", status, plan.output_dir)
    return code


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            sys.stdout.write(json.dumps(plan.public_metadata(), indent=2) + "\n")
            return 0
        return run_plan(plan)
    except (ValueError, OSError, RuntimeError) as error:
        logger.error("%s", error)
        return 1
