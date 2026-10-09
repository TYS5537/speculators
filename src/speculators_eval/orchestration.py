"""Offline evaluation setup and dataset orchestration with explicit callbacks."""

from __future__ import annotations

import argparse  # noqa: TC003 -- Runtime type hints.
from typing import Any

from speculators_eval.loading import LoadedEvaluation, TargetSetup


def validate_run_options(args: argparse.Namespace) -> None:
    if getattr(args, "dsv4_verification_mode", "reference") == "replay":
        if (
            getattr(args, "target_backend", "hf") != "dsv4-vllm"
            or args.temperature != 0.0
            or getattr(args, "dsv4_kv_reuse", False)
            or getattr(args, "dsv4_block_output", "auto") != "auto"
            or getattr(args, "measure_base_speedup", False)
        ):
            raise ValueError(
                "Replay requires dsv4-vllm, temperature=0, auto output, "
                "no target snapshots/base-speedup"
            )
        if args.dsv4_replay_audit_samples < 0:
            raise ValueError("Replay audit sample count must be nonnegative")
    if getattr(args, "dsv4_kv_reuse", False) and (
        getattr(args, "target_backend", "hf") != "dsv4-vllm"
        or getattr(args, "dsv4_verification_mode", "reference") != "block"
    ):
        raise ValueError("--dsv4-kv-reuse requires dsv4-vllm and block verification")


def prepare_target_backend(
    args: argparse.Namespace,
    *,
    prepare_hs_http,
    validate_target_cache_support,
    write_backend_metadata,
) -> TargetSetup:
    """Validate transport/cache contracts and publish metadata before dispatch."""
    target_backend = getattr(args, "target_backend", "hf")
    if getattr(args, "dsv4_profile", False) and target_backend != "dsv4-vllm":
        raise ValueError("--dsv4-profile requires --target-backend dsv4-vllm")
    hs_http_endpoint = prepare_hs_http(args, target_backend)
    target_config = validate_target_cache_support(args.verifier_model, target_backend)
    report = None
    if target_backend == "dsv4-vllm":
        from speculators_dsv4.contract import inspect_checkpoint  # noqa: PLC0415

        if not args.vllm_endpoint or not args.hidden_states_path:
            raise ValueError("DSV4 needs --vllm-endpoint and --hidden-states-path")
        if args.measure_base_speedup:
            raise ValueError(
                "DSV4 recompute evaluates acceptance, not online base speedup"
            )
        if args.dtype != "bfloat16":
            raise ValueError("DSV4 evaluation requires --dtype bfloat16")
        report = inspect_checkpoint(args.verifier_model)
        write_backend_metadata(args, report)
    return TargetSetup(target_backend, target_config, report, hs_http_endpoint)


def evaluate_loaded_models(
    args: argparse.Namespace,
    models: LoadedEvaluation,
    *,
    make_runner,
    make_base_runner,
    resolve_stop_token_ids,
    discover_datasets,
    split_csv,
    evaluate_dataset,
    write_outputs,
    write_timings,
    logger,
) -> None:
    """Construct runners and execute the unchanged dataset/reporting pipeline."""
    target_model, draft_model, tokenizer = (
        models.target_model,
        models.draft_model,
        models.tokenizer,
    )
    runner = make_runner(target_model, draft_model, tokenizer, args)
    base_runner = (
        make_base_runner(target_model, tokenizer, args)
        if args.measure_base_speedup
        else None
    )
    if base_runner is not None and float(args.temperature) > 0:
        logger.warning(
            "Base speedup at temperature > 0 compares sampled output throughput; "
            "use --temperature 0 for identical greedy output paths."
        )
    stop_token_ids = resolve_stop_token_ids(target_model, tokenizer)
    dataset_paths = discover_datasets(
        args.datasets_root,
        split_csv(args.datasets) or None,
    )
    rows: list[dict[str, Any]] = []
    artifacts_by_dataset: dict[str, list[dict[str, Any]]] = {}
    profiler = getattr(target_model, "profiler", None)
    profiles = {}
    for path in dataset_paths:
        if profiler is not None:
            profiler.reset()
        row, artifacts = evaluate_dataset(
            path=path,
            runner=runner,
            base_runner=base_runner,
            args=args,
            stop_token_ids=stop_token_ids,
        )
        rows.append(row)
        if profiler is not None and profiler.enabled:
            profiles[row["dataset"]] = profiler.snapshot()
            write_timings(args.output_dir, profiles)
        if not args.skip_artifacts:
            artifacts_by_dataset[row["dataset"]] = artifacts
    write_outputs(args.output_dir, rows, artifacts_by_dataset)
    logger.info("Wrote results to %s", args.output_dir)
