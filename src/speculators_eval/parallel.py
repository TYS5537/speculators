"""Model-free sample-parallel worker launch, collection and cleanup.

The caller supplies the evaluator entrypoint. This module must not import the
training/model packages or initialize Torch before child device isolation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from speculators_eval.data import (
    dataset_id,
    discover_datasets,
    load_jsonl,
    select_eval_records,
    split_csv,
)
from speculators_eval.profiling import collect_worker_timings
from speculators_eval.progress import ParallelProgress
from speculators_eval.reporting import (
    aggregate_rows,
    dataset_output_path,
    read_worker_artifacts,
    read_worker_row,
    write_outputs,
)

if TYPE_CHECKING:
    import argparse


def target_worker_args(args: argparse.Namespace) -> list[str]:
    if getattr(args, "target_backend", "hf") != "dsv4-vllm":
        return []
    result = [
        "--target-backend",
        "dsv4-vllm",
        "--vllm-endpoint",
        args.vllm_endpoint,
        "--hidden-states-path",
        str(args.hidden_states_path),
        "--dsv4-max-model-len",
        str(args.dsv4_max_model_len),
        "--dsv4-verification-mode",
        getattr(args, "dsv4_verification_mode", "reference"),
        "--dsv4-block-output",
        getattr(args, "dsv4_block_output", "auto"),
        "--target-request-timeout",
        str(args.target_request_timeout),
    ]
    if getattr(args, "dsv4_profile", False):
        result.append("--dsv4-profile")
    if getattr(args, "dsv4_kv_reuse", False):
        result.append("--dsv4-kv-reuse")
    if args.served_model_name:
        result.extend(["--served-model-name", args.served_model_name])
    if args.keep_target_hs:
        result.append("--keep-target-hs")
    if getattr(args, "hs_http_endpoint", None):
        result.extend(["--hs-http-endpoint", args.hs_http_endpoint])
    return result


def worker_command(
    args: argparse.Namespace,
    *,
    entrypoint: Path,
    dataset_path: Path,
    shard_index: int,
    num_shards: int,
    output_dir: Path,
    progress_path: Path | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(entrypoint.resolve()),
        "--verifier-model",
        args.verifier_model,
        "--draft-model",
        args.draft_model,
        "--datasets-root",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
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
        args.device,
        "--dtype",
        args.dtype,
        "--draft-attn-impl",
        args.draft_attn_impl,
        "--log-every",
        str(args.log_every),
        "--worker-shard-index",
        str(shard_index),
        "--worker-num-shards",
        str(num_shards),
        "--no-progress",
    ]
    if progress_path is not None:
        cmd.extend(["--worker-progress-path", str(progress_path)])
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.d2t_path is not None:
        cmd.extend(["--d2t-path", str(args.d2t_path)])
    if args.t2d_path is not None:
        cmd.extend(["--t2d-path", str(args.t2d_path)])
    if args.skip_artifacts:
        cmd.append("--skip-artifacts")
    if getattr(args, "draft_kv_reuse", False):
        cmd.append("--draft-kv-reuse")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.sample_from_anchor is not None:
        cmd.extend(["--sample-from-anchor", str(args.sample_from_anchor).lower()])
    cmd.extend(target_worker_args(args))
    if args.measure_base_speedup:
        cmd.extend(
            [
                "--measure-base-speedup",
                "--throughput-warmup-samples",
                str(args.throughput_warmup_samples),
            ]
        )
    return cmd


def stop_eval_worker(process) -> None:
    """Reap only our child; managed launches also own its surrounding group."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def wait_eval_workers(
    processes, dataset: str, *, progress: ParallelProgress | None = None
) -> None:
    while True:
        statuses = [(index, process.poll()) for index, _, process in processes]
        if progress is not None:
            progress.update(statuses)
        failed = [(index, code) for index, code in statuses if code not in (None, 0)]
        if failed:
            raise RuntimeError(f"{dataset} worker failures: {failed}")
        if all(code is not None for _, code in statuses):
            for _, _, process in processes:
                process.wait()
            return
        time.sleep(0.25)


def run_ascend_data_parallel(
    args: argparse.Namespace,
    *,
    entrypoint: Path,
) -> None:
    devices = split_csv(args.ascend_devices)
    if not devices:
        raise ValueError("--ascend-devices must contain at least one device id")
    if getattr(args, "target_backend", "hf") == "dsv4-vllm":
        from speculators_dsv4.eval_launcher import parse_devices  # noqa: PLC0415

        devices = [str(device) for device in parse_devices(args.ascend_devices)]
        if args.device != "npu:0":
            raise ValueError("DSV4 data-parallel workers require --device npu:0")
    dataset_paths = discover_datasets(
        args.datasets_root,
        split_csv(args.datasets) or None,
    )
    rows: list[dict[str, Any]] = []
    artifacts_by_dataset: dict[str, list[dict[str, Any]]] = {}
    profiles = {}
    for dataset_path in dataset_paths:
        dataset_start = time.perf_counter()
        dataset = dataset_id(dataset_path, args.datasets_root)
        shard_root = dataset_output_path(args.output_dir / "_shards", dataset)
        processes = []
        with ExitStack() as workers:
            progress = None
            if not getattr(args, "no_progress", False):
                total = len(
                    select_eval_records(
                        load_jsonl(dataset_path),
                        dataset_name=dataset_path.stem,
                        max_samples=args.max_samples,
                        seed=args.seed,
                    )
                )
                # Unique to this dataset/run: no stale counts on repeated evals.
                directory = workers.enter_context(
                    TemporaryDirectory(prefix="speculators-eval-progress-")
                )
                progress = ParallelProgress(
                    dataset, total, Path(directory), len(devices)
                )
                workers.callback(progress.close)
            for shard_index, visible_device in enumerate(devices):
                shard_output_dir = shard_root / f"shard_{shard_index}"
                cmd = worker_command(
                    args,
                    entrypoint=entrypoint,
                    dataset_path=dataset_path,
                    shard_index=shard_index,
                    num_shards=len(devices),
                    output_dir=shard_output_dir,
                    progress_path=(
                        progress.paths[shard_index] if progress is not None else None
                    ),
                )
                env = os.environ.copy()
                env["ASCEND_RT_VISIBLE_DEVICES"] = visible_device
                # A caller-owned Python entrypoint and argv, never shell input.
                process = subprocess.Popen(cmd, env=env)  # noqa: S603
                workers.callback(stop_eval_worker, process)
                processes.append((shard_index, shard_output_dir, process))
            wait_eval_workers(processes, dataset, progress=progress)
        shard_rows = [
            read_worker_row(shard_output_dir) for _, shard_output_dir, _ in processes
        ]
        row = aggregate_rows(dataset, shard_rows)
        if not args.measure_base_speedup:
            row["elapsed_s"] = time.perf_counter() - dataset_start
            row["requests_per_second"] = (
                row["num_requests"] / row["elapsed_s"] if row["elapsed_s"] else 0
            )
            row["output_tokens_per_second"] = (
                row["total_output_tokens"] / row["elapsed_s"] if row["elapsed_s"] else 0
            )
        rows.append(row)
        collect_worker_timings(
            args.output_dir,
            processes,
            dataset,
            profiles,
            enabled=getattr(args, "dsv4_profile", False),
        )
        if not args.skip_artifacts:
            artifacts = []
            for _, shard_output_dir, _ in processes:
                artifacts.extend(
                    # Each worker receives a single-file root, so its local
                    # artifact keeps the flat stem; the parent restores the ID.
                    read_worker_artifacts(shard_output_dir, dataset_path.stem)
                )
            artifacts.sort(key=lambda item: int(item.get("source_index", 0)))
            artifacts_by_dataset[dataset] = artifacts
        write_outputs(args.output_dir, rows, artifacts_by_dataset)
