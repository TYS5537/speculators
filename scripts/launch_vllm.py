import argparse
import datetime
import hashlib
import json
import math
import os
import shlex
import sys
import warnings
from pathlib import Path

try:
    import _provenance
except ModuleNotFoundError as exc:
    if exc.name != "_provenance":
        raise
    from scripts import _provenance

atomic_write = _provenance.atomic_write
find_package_repo = _provenance.find_package_repo
git_diff = _provenance.git_diff
pkg_version = _provenance.pkg_version
_git_sha = _provenance.git_sha

try:
    from hs_connectors import HiddenStatesBackend

    _backend_registry: dict[str, type[HiddenStatesBackend]] = dict(
        HiddenStatesBackend.registry  # type: ignore[misc]
    )
except ImportError:
    _backend_registry = {}  # type: ignore[assignment]


if "file" not in _backend_registry:
    # Vendored File backend in case hs_connectors is not available
    class _InlineFileBackend:
        @staticmethod
        def add_launch_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument(
                "--hidden-states-path",
                type=str,
                default="/tmp/hidden_states",  # noqa: S108
                help=(
                    "The directory to save hidden states to. "
                    "Default '/tmp/hidden_states'"
                ),
            )

        @staticmethod
        def build_kv_transfer_config(args: argparse.Namespace) -> dict:
            return {
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": args.hidden_states_path,
                },
            }

    _backend_registry["file"] = _InlineFileBackend  # type: ignore[assignment]


# Keep the preprocessing workers and vLLM front end within one CPU budget.
# Four preprocessing workers are paired with one API server. The former is
# estimated at 3 CPUs and the latter at 4 CPUs, so each preprocessing worker
# represents 4 CPUs of combined capacity. Leave 25% for native runtime
# threads and other application work.
CPU_BUDGET_FRACTION = 0.75
DEFAULT_RENDERER_NUM_WORKERS = 2
MAX_API_SERVER_COUNT = 32
WORKERS_PER_API_SERVER = 4
CPUS_PER_API_SERVER = 4
MAX_PREPROCESSING_WORKERS = 128
EFFECTIVE_CPUS_PER_PREPROCESSING_WORKER = 4
VLLM_SCALE_OUT_VERSION_BOUNDARY = (0, 29, 1)

# The vLLM API-server processes each create their own native thread pools.
# Keep those pools bounded by default; explicit environment settings still
# take precedence for users who have measured a different configuration.
DEFAULT_RENDER_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "2",
}


def _usable_cpu_count() -> int:
    """Return the CPUs available to this process, respecting affinity."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):  # Linux
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _preprocessing_workers(cpus: int) -> int:
    """Mirror prepare_data.py's shared render CPU budget."""
    return max(
        1,
        min(
            MAX_PREPROCESSING_WORKERS,
            int(cpus * CPU_BUDGET_FRACTION) // EFFECTIVE_CPUS_PER_PREPROCESSING_WORKER,
        ),
    )


def render_throughput_defaults(cpus: int | None = None) -> tuple[int, int]:
    """Return affinity-aware API-server and renderer-worker defaults."""
    if cpus is None:
        cpus = _usable_cpu_count()
    api_servers = max(
        1,
        min(
            MAX_API_SERVER_COUNT,
            _preprocessing_workers(cpus) // WORKERS_PER_API_SERVER,
            cpus // CPUS_PER_API_SERVER,
        ),
    )
    return api_servers, DEFAULT_RENDERER_NUM_WORKERS


def _vllm_supports_scale_out_flag() -> bool:
    """Return whether the installed vLLM accepts ``--enable-scale-out``."""
    try:
        from vllm import __version_tuple__  # noqa: PLC0415

        release = tuple(int(part) for part in __version_tuple__[:3])
    except (ImportError, AttributeError, TypeError, ValueError):
        return False

    return release > VLLM_SCALE_OUT_VERSION_BOUNDARY


def _with_render_defaults(vllm_args: list[str]) -> list[str]:
    """Enable and tune render endpoints, unless no API server is wanted."""
    if "--headless" in vllm_args:
        return vllm_args
    api_servers, renderer_workers = render_throughput_defaults()
    scale_out_args = ["--enable-scale-out"] if _vllm_supports_scale_out_flag() else []
    return [
        *scale_out_args,
        "--api-server-count",
        str(api_servers),
        "--renderer-num-workers",
        str(renderer_workers),
        *vllm_args,
    ]


def _set_render_thread_defaults() -> None:
    """Bound native pools inherited by vLLM's API-server processes."""
    for name, value in DEFAULT_RENDER_THREAD_ENV.items():
        os.environ.setdefault(name, value)


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provenance-dir",
        type=str,
        default=None,
        help=(
            "Directory to write vllm_command.txt, vllm.patch, and "
            "checkpoint_sha256.txt (plus drafter_checkpoint_sha256.txt in "
            "eval mode). Provenance is only captured when this is set; omit "
            "it to skip logging (e.g. for ad-hoc test/debug runs)."
        ),
    )
    parser.add_argument(
        "--no-hash-checkpoints",
        action="store_true",
        default=False,
        help=(
            "Skip SHA256 hashing of .safetensors files. Useful "
            "for large checkpoints where hashing adds significant "
            "launch latency. When set, checkpoint_sha256.txt "
            "records file sizes and modification times instead."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch vLLM for training or evaluation",
    )
    sub = parser.add_subparsers(dest="subcommand")

    # --- train subcommand (default when no subcommand given) ---
    train_parser = sub.add_parser(
        "train",
        help="Hidden-states extraction for training data generation",
    )
    train_parser.add_argument(
        "model",
        type=str,
        help="Model name or path to extract hidden states from",
    )
    train_parser.add_argument(
        "--hidden-states-backend",
        choices=list(_backend_registry.keys()),
        default="file",
        help=(
            "Hidden states transfer backend. Each backend may "
            "add its own CLI arguments (see below). "
            "Default: 'file'."
        ),
    )
    for backend_cls in _backend_registry.values():
        backend_cls.add_launch_args(train_parser)
    train_parser.add_argument(
        "--dsv4",
        "--dsv4-bf16",
        dest="dsv4",
        action="store_true",
        help=(
            "Experimental DSV4-Flash BF16 HS bridge for vLLM Ascend 0.26.0rc1. "
            "Target weight quantization is handled by the backend; "
            "--dsv4-bf16 is a compatibility alias."
        ),
    )
    train_parser.add_argument(
        "--dsv4-execution-mode",
        choices=("eager", "full-decode-only"),
        default=None,
        help=(
            "DSV4 HS execution mode (default: eager). FULL_DECODE_ONLY uses "
            "ACL graphs only for decode; prefill remains eager. Native "
            "--async-scheduling is an independent opt-in."
        ),
    )
    train_parser.add_argument(
        "--dsv4-block-verify",
        action="store_true",
        help=(
            "Dedicated DSV4 offline block-verification service; requires --dsv4. "
            "Only one sequence and max_tokens=1 requests are supported."
        ),
    )
    train_parser.add_argument(
        "--dsv4-manifest-timeout",
        type=float,
        default=None,
        help=(
            "Seconds for a secondary DSV4 target to wait for the shared HS "
            "manifest (default: 300)."
        ),
    )

    train_parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        help=(
            "(Optional) Space-separated list of integer layer "
            "ids. Defaults to "
            "[2, num_hidden_layers // 2, num_hidden_layers - 3]."
            " Note: if set, you must also pass the same value "
            "into the training process"
        ),
    )
    train_parser.add_argument(
        "--include-last-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append the last layer (num_hidden_layers) to "
            "target_layer_ids for verifier hidden states "
            "extraction. Default: True"
        ),
    )
    train_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allow custom model configuration code while resolving "
            "hidden-state layer ids. Pass the same flag after '--' for "
            "vLLM itself."
        ),
    )
    _add_shared_args(train_parser)

    # --- eval subcommand ---
    eval_parser = sub.add_parser(
        "eval",
        help="Speculative decoding serving for evaluation",
    )
    eval_parser.add_argument(
        "model",
        type=str,
        help="Target model name or path",
    )
    eval_parser.add_argument(
        "--spec-model",
        type=str,
        required=True,
        help="Drafter model name or path",
    )
    eval_parser.add_argument(
        "--spec-tokens",
        type=int,
        default=None,
        help="Number of speculative tokens",
    )
    eval_parser.add_argument(
        "--spec-method",
        type=str,
        default=None,
        help="Speculative decoding method (optional)",
    )
    _add_shared_args(eval_parser)

    subcommands = set(sub.choices)
    argv = sys.argv[1:]
    if not argv or (argv[0] not in subcommands and not argv[0].startswith("-")):
        argv = ["train", *argv]
    args, vllm_args = parser.parse_known_args(argv)
    if args.subcommand is None:
        args, vllm_args = parser.parse_known_args(["train"] + sys.argv[1:])
    return args, vllm_args


def _warn(msg: str) -> None:
    print(f"Warning: {msg}", file=sys.stderr)


def _find_vllm_repo() -> str | None:
    """Find the vllm git checkout by walking up from the installed package."""
    repo = find_package_repo("vllm")
    if repo and (repo / "vllm" / "__init__.py").is_file():
        return str(repo)
    return None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Individual provenance writers — each is self-contained and best-effort.
# ---------------------------------------------------------------------------


def _save_vllm_command(
    prov_dir: Path,
    cmd: list[str],
    sha: str,
    diff: str,
    vllm_ver: str,
) -> None:
    sha_label = f"{sha} (dirty)" if diff else sha
    ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    header = "\n".join(
        [
            f"# Timestamp: {ts}",
            f"# Python: {sys.executable}",
            f"# Git SHA: {sha_label}",
            f"# vllm: {vllm_ver}",
        ]
    )
    atomic_write(
        prov_dir / "vllm_command.txt",
        f"{header}\n{shlex.join(cmd)}\n",
    )


def _save_vllm_patch(
    prov_dir: Path,
    vllm_repo: str | None,
    sha: str,
    diff: str,
    vllm_ver: str,
) -> None:
    if vllm_repo:
        content = f"# repo: {vllm_repo} ({sha})\n{diff}\n"
    else:
        content = f"# vllm {vllm_ver} (wheel install, no git repo found)\n"
    atomic_write(prov_dir / "vllm.patch", content)


def _save_checkpoint_sha256(
    prov_dir: Path,
    model: str,
    *,
    skip_hash: bool = False,
    dest_name: str = "checkpoint_sha256.txt",
) -> None:
    dest = prov_dir / dest_name
    model_path = os.path.expanduser(model)
    if not os.path.isdir(model_path):
        atomic_write(dest, f"# model: {model} (not a local path)\n")
        return
    safetensors = sorted(
        f for f in os.listdir(model_path) if f.endswith(".safetensors")
    )
    if not safetensors:
        atomic_write(dest, f"# no .safetensors files in {model_path}\n")
        return
    if skip_hash:
        lines = []
        for name in safetensors:
            fp = os.path.join(model_path, name)
            st = os.stat(fp)
            lines.append(f"size={st.st_size}  mtime={st.st_mtime}  {name}")
        header = "# hashing skipped (--no-hash-checkpoints)\n"
        atomic_write(dest, header + "\n".join(lines) + "\n")
    else:
        lines = [
            f"{_sha256_file(os.path.join(model_path, name))}  {name}"
            for name in safetensors
        ]
        atomic_write(dest, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def _save_vllm_provenance(
    cmd: list[str],
    provenance_dir: str,
    model: str,
    *,
    skip_hash: bool = False,
    spec_model: str | None = None,
) -> None:
    """Write vllm_command.txt, vllm.patch, and checkpoint_sha256.txt.

    In eval mode (``spec_model`` given) also writes
    drafter_checkpoint_sha256.txt. Best-effort — failures warn but never
    block the vLLM launch.
    """
    prov_dir = Path(provenance_dir)
    try:
        prov_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _warn(f"could not create provenance dir: {exc}")
        return

    vllm_repo = _find_vllm_repo()
    vllm_root = Path(vllm_repo) if vllm_repo else None
    sha = _git_sha(vllm_root)
    diff = git_diff(vllm_root)
    vllm_ver = pkg_version("vllm")

    writers = [
        (
            "vllm_command.txt",
            lambda: _save_vllm_command(prov_dir, cmd, sha, diff, vllm_ver),
        ),
        (
            "vllm.patch",
            lambda: _save_vllm_patch(prov_dir, vllm_repo, sha, diff, vllm_ver),
        ),
        (
            "checkpoint_sha256.txt",
            lambda: _save_checkpoint_sha256(prov_dir, model, skip_hash=skip_hash),
        ),
    ]
    if spec_model is not None:
        drafter_dest = "drafter_checkpoint_sha256.txt"
        writers.append(
            (
                drafter_dest,
                lambda: _save_checkpoint_sha256(
                    prov_dir,
                    spec_model,
                    skip_hash=skip_hash,
                    dest_name=drafter_dest,
                ),
            )
        )
    for artifact, write in writers:
        try:
            write()
        except Exception as exc:  # noqa: BLE001
            _warn(f"could not save {artifact}: {exc}")


def _build_dsv4_train_cmd(args, vllm_args):  # noqa: C901
    """Build the Ascend bridge without applying CUDA vLLM render defaults."""
    dsv4_manifest = None
    dsv4_parallel = None
    dsv4_runtime_quantization = None
    if args.dsv4:
        from importlib.metadata import entry_points  # noqa: PLC0415

        from speculators_dsv4 import ARCHITECTURE, KV_CACHE_COMPAT_ENV  # noqa: PLC0415
        from speculators_dsv4.contract import (  # noqa: PLC0415
            DEFAULT_LAYERS,
            ensure_manifest,
            inspect_checkpoint,
            make_manifest,
            validate_layers,
            wait_for_manifest,
        )
        from speculators_dsv4.execution import configure_execution_args  # noqa: PLC0415
        from speculators_dsv4.parallel import (  # noqa: PLC0415
            DP4_SIZE,
            configure_parallel_args,
        )

        configure_execution_args(
            vllm_args,
            args.dsv4_execution_mode or "eager",
            block_verify=args.dsv4_block_verify,
        )
        dsv4_parallel = configure_parallel_args(
            vllm_args, os.environ, block_verify=args.dsv4_block_verify
        )
        if (
            dsv4_parallel.data_parallel_size == DP4_SIZE
            and not Path(args.hidden_states_path).is_absolute()
        ):
            raise ValueError(
                "DSV4 multi-host HS export requires an absolute shared HS path."
            )
        report = inspect_checkpoint(args.model)
        num_hidden_layers = report["config"]["num_hidden_layers"]
        if args.hidden_states_backend != "file" or not args.include_last_layer:
            raise ValueError("DSV4 requires the file backend and --include-last-layer.")
        args.target_layer_ids = args.target_layer_ids or list(DEFAULT_LAYERS)
        validate_layers(args.target_layer_ids)
        dsv4_manifest = make_manifest(report, args.target_layer_ids)
        quantization_parser = argparse.ArgumentParser(
            add_help=False, allow_abbrev=False
        )
        quantization_parser.add_argument("--quantization", "-q")
        quantization_args, _ = quantization_parser.parse_known_args(vllm_args)
        dsv4_runtime_quantization = {"method": quantization_args.quantization}
        if args.dsv4_block_verify:
            sequence_parser = argparse.ArgumentParser(
                add_help=False, allow_abbrev=False
            )
            sequence_parser.add_argument(
                "--max-num-seqs", "--max_num_seqs", type=int, action="append"
            )
            sequence_args, _ = sequence_parser.parse_known_args(vllm_args)
            if any(value != 1 for value in sequence_args.max_num_seqs or []):
                raise ValueError("DSV4 block verification requires --max-num-seqs 1.")
            if sequence_args.max_num_seqs is None:
                vllm_args.extend(["--max-num-seqs", "1"])
        for arg in vllm_args:
            if arg.split("=", 1)[0] in {
                "--hf-overrides",
                "--dtype",
                "--speculative-config",
                "--speculative_config",
                "--kv-transfer-config",
                "--kv_transfer_config",
            }:
                raise ValueError(
                    "DSV4 HS bridge owns HF overrides and runtime dtype, "
                    "speculative config and KV transfer config."
                )
        plugins = entry_points(group="vllm.general_plugins")
        if not args.dry_run and not any(p.name == "speculators_dsv4" for p in plugins):
            raise RuntimeError(
                "Install this checkout in the vLLM environment: "
                "pip install -e . --no-deps"
            )
        allowlist = os.environ.get("VLLM_PLUGINS")
        if allowlist is not None and "speculators_dsv4" not in allowlist.split(","):
            raise ValueError(
                "Add speculators_dsv4 to VLLM_PLUGINS (preserve your Ascend plugins)."
            )
        vllm_args.extend(
            [
                "--hf-overrides",
                json.dumps({"architectures": [ARCHITECTURE]}),
                "--dtype",
                "bfloat16",
                "--no-enable-prefix-caching",
            ]
        )
    else:
        from transformers import AutoConfig  # noqa: PLC0415

        config = AutoConfig.from_pretrained(args.model)
        if hasattr(config, "text_config"):
            config = config.text_config
        num_hidden_layers = config.num_hidden_layers
        if getattr(config, "model_type", None) == "deepseek_v4":
            raise ValueError(
                "Use --dsv4: native mean-only final HS is not a valid teacher."
            )

    if args.target_layer_ids:
        target_layer_ids = args.target_layer_ids
        if args.include_last_layer and num_hidden_layers not in target_layer_ids:
            target_layer_ids.append(num_hidden_layers)
        if dsv4_manifest is not None:
            warnings.warn(
                f"DSV4 export slots: {target_layer_ids}. Pass only auxiliary IDs "
                f"{dsv4_manifest['auxiliary_hs_ids']} to training; "
                "slot 43 is the separate teacher.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Using custom target layer ids {target_layer_ids}. These "
                "must also be explicitly passed into the training script.",
                stacklevel=2,
            )
    else:
        target_layer_ids = [
            2,
            num_hidden_layers // 2,
            num_hidden_layers - 3,
            num_hidden_layers,
        ]

    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": target_layer_ids}
        },
    }
    backend_cls = _backend_registry[args.hidden_states_backend]
    kv_transfer_config = backend_cls.build_kv_transfer_config(args)
    if args.dsv4_block_verify:
        from speculators_dsv4.block_protocol import (  # noqa: PLC0415
            BLOCK_CONNECTOR,
            BLOCK_CONNECTOR_MODULE,
        )

        kv_transfer_config = {
            "kv_connector": BLOCK_CONNECTOR,
            "kv_connector_module_path": BLOCK_CONNECTOR_MODULE,
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": args.hidden_states_path,
            },
        }

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--speculative_config",
        json.dumps(speculative_config),
        "--kv_transfer_config",
        json.dumps(kv_transfer_config),
        *vllm_args,
    ]

    disable_cp_arg = "--no-enable-chunked-prefill"
    if disable_cp_arg not in cmd:
        cmd.append(disable_cp_arg)

    def before_launch():
        if dsv4_manifest is not None:
            if dsv4_parallel.data_parallel_size == DP4_SIZE and dsv4_parallel.headless:
                print(
                    "Waiting for the target head node's shared DSV4 HS manifest...",
                    flush=True,
                )
                wait_for_manifest(
                    args.hidden_states_path,
                    dsv4_manifest,
                    runtime_quantization=dsv4_runtime_quantization,
                    timeout=args.dsv4_manifest_timeout or 300,
                )
            else:
                ensure_manifest(
                    args.hidden_states_path,
                    dsv4_manifest,
                    create=True,
                    runtime_quantization=dsv4_runtime_quantization,
                )
            # Inherited by EngineCore/worker children. The general plugin must
            # patch KV planning before initialization, not at model construction.
            os.environ[KV_CACHE_COMPAT_ENV] = "1"

    return cmd, before_launch


def _build_train_cmd(args, vllm_args):
    from transformers import AutoConfig  # noqa: PLC0415

    config = AutoConfig.from_pretrained(
        args.model,
        **({"trust_remote_code": True} if args.trust_remote_code else {}),
    )
    if hasattr(config, "text_config"):
        config = config.text_config
    num_hidden_layers = config.num_hidden_layers

    if args.target_layer_ids:
        target_layer_ids = args.target_layer_ids
        if args.include_last_layer and num_hidden_layers not in target_layer_ids:
            target_layer_ids.append(num_hidden_layers)
        warnings.warn(
            f"Using custom target layer ids {target_layer_ids}. These "
            "must also be explicitly passed into the training script.",
            stacklevel=2,
        )
    else:
        target_layer_ids = [
            2,
            num_hidden_layers // 2,
            num_hidden_layers - 3,
            num_hidden_layers,
        ]
    # Layer id ``num_hidden_layers`` (the final hidden state) is valid: the
    # default above and --include-last-layer both emit it.
    if (
        min(target_layer_ids) < 0
        or max(target_layer_ids) > num_hidden_layers
        or len(set(target_layer_ids)) != len(target_layer_ids)
    ):
        raise ValueError(
            f"Invalid target layer ids {target_layer_ids}; ids must be "
            f"distinct and within [0, {num_hidden_layers}]."
        )

    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": target_layer_ids}
        },
    }
    backend_cls = _backend_registry[args.hidden_states_backend]
    kv_transfer_config = backend_cls.build_kv_transfer_config(args)

    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--speculative_config",
        json.dumps(speculative_config),
        "--kv_transfer_config",
        json.dumps(kv_transfer_config),
        *_with_render_defaults(vllm_args),
    ]


def _build_eval_cmd(args, vllm_args):
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--spec-model",
        args.spec_model,
    ]
    if args.spec_tokens is not None:
        cmd.extend(["--spec-tokens", str(args.spec_tokens)])
    if args.spec_method is not None:
        cmd.extend(["--spec-method", args.spec_method])
    cmd.extend(vllm_args)
    return cmd


def _validate_dsv4_flags(args):
    if args.dsv4_block_verify and not args.dsv4:
        raise ValueError("--dsv4-block-verify requires --dsv4.")
    if args.dsv4_execution_mode is not None and not args.dsv4:
        raise ValueError("--dsv4-execution-mode requires --dsv4.")
    if args.dsv4_manifest_timeout is not None:
        if not args.dsv4:
            raise ValueError("--dsv4-manifest-timeout requires --dsv4.")
        if (
            not math.isfinite(args.dsv4_manifest_timeout)
            or args.dsv4_manifest_timeout <= 0
        ):
            raise ValueError("DSV4 manifest timeout must be finite and positive.")


def main():
    args, vllm_args = parse_args()
    if "--" in vllm_args:
        vllm_args.remove("--")

    before_launch = None
    if args.subcommand == "train":
        _validate_dsv4_flags(args)
        if args.dsv4:
            cmd, before_launch = _build_dsv4_train_cmd(args, vllm_args)
        else:
            cmd = _build_train_cmd(args, vllm_args)
    elif args.subcommand == "eval":
        cmd = _build_eval_cmd(args, vllm_args)
    else:
        raise ValueError(f"Unknown subcommand: {args.subcommand}")

    print("Running command:")
    if before_launch is not None:
        from speculators_dsv4 import KV_CACHE_COMPAT_ENV  # noqa: PLC0415

        print(f"{KV_CACHE_COMPAT_ENV}=1", end=" ")
    print(" ".join(cmd))

    if args.provenance_dir:
        _save_vllm_provenance(
            cmd,
            args.provenance_dir,
            args.model,
            skip_hash=args.no_hash_checkpoints,
            spec_model=getattr(args, "spec_model", None),
        )

    if not args.dry_run:
        # Render tuning applies to the train pipeline only; eval serving skips it.
        if before_launch is not None:
            before_launch()
        if (
            args.subcommand == "train"
            and not args.dsv4
            and "--headless" not in vllm_args
        ):
            _set_render_thread_defaults()
        os.execvp(cmd[0], cmd)  # noqa: S606


if __name__ == "__main__":
    main()
