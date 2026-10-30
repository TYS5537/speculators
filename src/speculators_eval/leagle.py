"""LEAGLE acceptance evaluation. Parent worker dispatch stays model-free."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

from speculators_eval import data, parallel, provenance, reporting
from speculators_eval.diagnostics import capture_sample_failure
from speculators_eval.progress import WorkerProgress

logger = logging.getLogger("speculators_eval")


def build_parser():
    parser = argparse.ArgumentParser(
        description="EAGLE3 / LEAGLE single-chain speculation with an HF target."
    )
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--datasets", help="Comma-separated JSONL dataset names")
    parser.add_argument("--output-dir", type=Path, default=Path("output/leagle_eval"))
    parser.add_argument("--spec-tokens", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 uses greedy decoding; >0 samples draft and target at this temperature.",
    )
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--enable-thinking", choices=["false", "true", "default"], default="false"
    )
    parser.add_argument(
        "--raw-prompt-mode", choices=["auto", "chat_template", "raw"], default="auto"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32", "auto"],
        default="bfloat16",
    )
    parser.add_argument("--draft-attn-impl", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--target-attn-impl", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument(
        "--no-draft-kv-reuse",
        action="store_true",
        help="Use the uncached reference proposals for comparison (slower)",
    )
    parser.add_argument(
        "--candidate-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="LEAGLE candidate output head; by default follow the trained checkpoint",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        help="Candidate vocabulary count; defaults to checkpoint (not draft length)",
    )
    parser.add_argument(
        "--leagle-sparse-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental compact candidate sampling and exact sparse-q "
        "rejection correction",
    )
    parser.add_argument(
        "--leagle-static-kv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental request-local preallocated LEAGLE prefix/rollout KV buffers",
    )
    parser.add_argument(
        "--leagle-reuse-rollout-buffers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental reuse of rollout buffer storage across rounds; "
        "requires --leagle-static-kv and verified-prefix KV reuse",
    )
    parser.add_argument(
        "--check-greedy",
        action="store_true",
        help="Audit every output against target-only HF greedy generation (extra work)",
    )
    for flag, help_text in (
        ("leagle-static-target-kv", "Preallocated target KV with speculative rollback"),
        (
            "leagle-selective-target-hidden-states",
            "Capture only the required Qwen3 target states",
        ),
        (
            "leagle-share-output-head",
            "Share identical frozen draft/teacher output weights",
        ),
    ):
        parser.add_argument(
            "--" + flag,
            action=argparse.BooleanOptionalAction,
            default=False,
            help=help_text,
        )
    parser.add_argument(
        "--ascend-devices",
        help="Sample-parallel NPU IDs, e.g. 0,1,2,3; each device holds both models",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep each NPU worker's models loaded across datasets; "
        "reset RNG per dataset",
    )
    parser.add_argument(
        "--worker-job-stream", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--d2t-path", type=Path)
    parser.add_argument("--t2d-path", type=Path)
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--worker-shard-index", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-num-shards", type=int, default=1, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-progress-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-manifest", type=Path, help=argparse.SUPPRESS)
    # Shared worker dispatcher fields; not LEAGLE model options.
    parser.set_defaults(
        target_backend="hf", sample_from_anchor=None, measure_base_speedup=False
    )
    return parser


def _validate_persistent_workers(args):
    if args.persistent_workers and (
        not args.ascend_devices or args.worker_shard_index is not None
    ):
        raise ValueError(
            "--persistent-workers requires a parent --ascend-devices launch"
        )
    if args.worker_job_stream and args.worker_shard_index is None:
        raise ValueError("--worker-job-stream is reserved for sharded workers")


def validate_args(args):
    _validate_persistent_workers(args)
    if args.spec_tokens < 1 or args.max_new_tokens < 0:
        raise ValueError("--spec-tokens must be >= 1 and --max-new-tokens >= 0")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("--temperature must be finite and >= 0")
    if args.check_greedy and args.temperature != 0:
        raise ValueError("--check-greedy requires temperature=0")
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.log_every < 1 or args.worker_num_shards < 1:
        raise ValueError("--log-every and --worker-num-shards must be positive")
    optimization_settings(args)
    if (args.d2t_path is None) != (args.t2d_path is None):
        raise ValueError("--d2t-path and --t2d-path must be provided together")
    if args.candidate_top_k is not None and (
        args.candidate_head is False or args.candidate_top_k < 1
    ):
        raise ValueError(
            "--candidate-top-k must be positive and requires --candidate-head"
        )
    if args.ascend_devices:
        devices = data.split_csv(args.ascend_devices)
        if (
            not devices
            or any(not device.isdecimal() for device in devices)
            or len(set(devices)) != len(devices)
            or args.device != "npu:0"
        ):
            raise ValueError(
                "--ascend-devices requires distinct numeric IDs and --device npu:0"
            )
    data.discover_datasets(args.datasets_root, data.split_csv(args.datasets) or None)


def candidate_head_settings(args, config=None):
    """Validate the switch without initializing Torch in the NPU parent process."""
    config = config or {}
    enabled = (
        config.get("candidate_head", False)
        if args.candidate_head is None
        else args.candidate_head
    )
    rank = (
        config.get("candidate_head_rank")
        if config.get("candidate_head", False)
        else None
    )
    top_k = args.candidate_top_k or config.get("candidate_head_top_k")
    if enabled and config:
        if config.get("speculators_model_type") != "leagle":
            raise ValueError(
                "--candidate-head is supported only for LEAGLE, not EAGLE3"
            )
        if rank is None:
            raise ValueError(
                "Checkpoint has no candidate head. Train LEAGLE with --candidate-head."
            )
        if top_k is None or not 1 <= top_k <= config["draft_vocab_size"]:
            raise ValueError("Candidate top_k must be in [1, draft_vocab_size]")
    if not enabled and args.candidate_top_k is not None:
        raise ValueError("--candidate-top-k requires an enabled candidate head")
    return {
        "enabled": enabled,
        "rank": rank,
        "top_k": top_k if enabled else None,
        "scoring": "low-rank-retrieval-exact-rescore" if enabled else "dense",
        "proposal_support": "retrieved-candidates" if enabled else "draft-vocabulary",
    }


def optimization_settings(args, config=None):
    settings = {
        "sparse_candidates": getattr(args, "leagle_sparse_candidates", False),
        "static_kv": getattr(args, "leagle_static_kv", False),
        "reuse_rollout_buffers": getattr(args, "leagle_reuse_rollout_buffers", False),
        "static_target_kv": getattr(args, "leagle_static_target_kv", False),
        "selective_target_hidden_states": getattr(
            args, "leagle_selective_target_hidden_states", False
        ),
        "share_output_head": getattr(args, "leagle_share_output_head", False),
    }
    if settings["reuse_rollout_buffers"] and (
        not settings["static_kv"] or getattr(args, "no_draft_kv_reuse", False)
    ):
        raise ValueError(
            "Rollout buffer reuse requires --leagle-static-kv and "
            "cannot use --no-draft-kv-reuse"
        )
    if config and any(settings.values()):
        if config.get("speculators_model_type") != "leagle":
            raise ValueError(
                "LEAGLE inference optimizations cannot be enabled for EAGLE3"
            )
        if (
            settings["sparse_candidates"]
            and not candidate_head_settings(args, config)["enabled"]
        ):
            raise ValueError(
                "Sparse candidates require an enabled trained candidate head"
            )
    return settings


def load_models(args):
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from speculators import SpeculatorModel, SpeculatorModelConfig  # noqa: PLC0415

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401, PLC0415 -- Register the Ascend backend.

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    config = SpeculatorModelConfig.from_pretrained(args.draft_model)
    if config.speculators_model_type not in ("leagle", "eagle3"):
        raise ValueError("This entrypoint requires an EAGLE3 or LEAGLE checkpoint")
    head_settings = candidate_head_settings(args, config.to_dict())
    optimizations = optimization_settings(args, config.to_dict())
    config.speculators_config.verifier.name_or_path = args.verifier_model
    config.transformer_layer_config._attn_implementation = args.draft_attn_impl  # noqa: SLF001
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.verifier_model,
            torch_dtype=dtype,
            attn_implementation=args.target_attn_impl,
            trust_remote_code=args.trust_remote_code,
        )
        .to(device)
        .eval()
    )
    if target.config.model_type != "qwen3" or any(
        kind != "full_attention" for kind in target.config.layer_types
    ):
        raise ValueError(
            "LEAGLE evaluation currently requires a full-attention Qwen3 target"
        )
    layer_ids = (
        config.eagle_aux_hidden_state_layer_ids
        if config.speculators_model_type == "eagle3"
        else config.aux_hidden_state_layer_ids
    )
    if (
        target.config.hidden_size != config.transformer_layer_config.hidden_size
        or target.config.vocab_size != config.transformer_layer_config.vocab_size
        or min(layer_ids) < 0
        or max(layer_ids) > target.config.num_hidden_layers
    ):
        raise ValueError(
            "Target dimensions or auxiliary layer IDs do not match the draft"
        )
    mappings = {}
    if args.d2t_path is not None:
        mappings = {
            "d2t": torch.load(args.d2t_path, map_location="cpu", weights_only=True),
            "t2d": torch.load(args.t2d_path, map_location="cpu", weights_only=True),
        }
    draft = (
        SpeculatorModel.from_pretrained(
            args.draft_model, config=config, torch_dtype=target.dtype, **mappings
        )
        .to(device=device, dtype=target.dtype)
        .eval()
    )
    if draft.use_draft_vocab:
        # Validate checkpoint-resident offsets too: zero placeholders would
        # otherwise silently propose the wrong target-vocabulary token IDs.
        draft.load_vocab_mappings(draft.t2d, draft.d2t)
    if config.speculators_model_type == "leagle":
        draft.set_candidate_head(head_settings["enabled"], top_k=head_settings["top_k"])
        draft.set_inference_optimizations(**optimizations)
    if head_settings["enabled"]:
        logger.info(
            "LEAGLE candidate head enabled: rank=%d candidates=%d (exact rescore)",
            head_settings["rank"],
            head_settings["top_k"],
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model, trust_remote_code=args.trust_remote_code
    )
    logger.info(
        "%s D=%d R=%d effective_depth=%d | K=%d | target=%s draft=%s | %s",
        config.speculators_model_type,
        config.transformer_layer_config.num_hidden_layers,
        getattr(config, "num_loops", 1),
        config.transformer_layer_config.num_hidden_layers
        * getattr(config, "num_loops", 1),
        args.spec_tokens,
        args.target_attn_impl,
        args.draft_attn_impl,
        target.dtype,
    )
    return target, draft, tokenizer


def save_provenance(args, draft_config=None, *, argv=None, persistent=False):
    config = draft_config.to_dict() if draft_config is not None else None
    config_path = Path(args.draft_model) / "config.json"
    if config is None and config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = {
        "proposal": (
            "single-chain-sampling" if args.temperature > 0 else "single-chain-greedy"
        ),
        "verification": (
            "HF rejection sampling with residual correction and KV rollback"
            if args.temperature > 0
            else "HF full-attention target with KV rollback"
        ),
        "target_kv_reuse": True,
        "candidate_head": candidate_head_settings(args, config),
        "experimental_optimizations": optimization_settings(args, config),
        "draft_kv_reuse_between_rounds": not args.no_draft_kv_reuse,
        "draft_cache_policy": (
            "rebuild-each-round" if args.no_draft_kv_reuse else "verified-prefix-only"
        ),
        "online_speedup_benchmark": False,
        "probability_diagnostics_available": args.temperature > 0,
        "persistent_workers": args.persistent_workers
        or persistent
        or args.worker_job_stream,
        "worker_rng_policy": "restore-post-load-per-dataset"
        if (args.persistent_workers or persistent or args.worker_job_stream)
        else "fresh-worker-per-dataset"
        if args.worker_shard_index is not None or args.ascend_devices
        else "continuous-local-run",
        "timing_scope": (
            "persistent worker wall time; first dataset includes model loads"
            if args.persistent_workers
            else "parallel worker wall time including model loads"
            if args.ascend_devices and args.worker_shard_index is None
            else "local generation including target-only greedy audit"
            if args.check_greedy
            else "local generation without serving optimizations"
        ),
    }
    if draft_config is not None:
        metadata["draft_config"] = draft_config.to_dict()
        metadata["algorithm"] = draft_config.speculators_model_type
    if argv is None:
        provenance.write_eval_provenance(args, metadata)
    else:
        provenance.write_eval_provenance(args, metadata, argv=argv)


def evaluate_dataset(path, runner, args, stop_token_ids):
    import torch  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    name = data.dataset_id(path, args.datasets_root)
    selected = data.select_eval_records(
        data.load_jsonl(path),
        dataset_name=path.stem,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    indexed = data.shard_records(
        selected, shard_index=args.worker_shard_index, num_shards=args.worker_num_shards
    )
    stats = reporting.EvalStats(probability_diagnostics_available=args.temperature > 0)
    artifacts = []
    progress = WorkerProgress(args.worker_progress_path)
    progress.update(0)
    synchronize = getattr(getattr(torch, runner.device.type, None), "synchronize", None)
    for processed, (index, record) in enumerate(
        tqdm(indexed, desc=name, unit="sample", disable=args.no_progress), start=1
    ):
        with capture_sample_failure(
            args=args, path=path, dataset=name, selected_index=index, record=record
        ):
            prompt = data.prompt_from_record(
                record, runner.tokenizer, source=f"{path}:{index}", args=args
            )
            if synchronize is not None:
                synchronize(runner.device)
            started = time.perf_counter()
            response = runner.generate_one(prompt, stop_token_ids)
            if synchronize is not None:
                synchronize(runner.device)
            stats.elapsed_s += time.perf_counter() - started
        stats.add_response(response)
        if not args.skip_artifacts:
            artifacts.append(
                {
                    "prompt": prompt,
                    "output_token_ids": response.output_ids[0].tolist(),
                    "num_input_tokens": response.num_input_tokens,
                    "source_index": index,
                    "proposal_lengths": response.proposal_lengths,
                    "accepted_draft_lengths": response.accepted_draft_lengths,
                    "accept_prob_lists": response.accept_prob_lists,
                    "support_accept_rate_lists": response.support_accept_rate_lists,
                }
            )
        progress.update(processed)
        if not args.worker_progress_path and (
            processed == 1
            or processed % args.log_every == 0
            or processed == len(indexed)
        ):
            logger.info(
                "[%s] %d/%d | acceptance_length=%.4f",
                name,
                processed,
                len(indexed),
                stats.acceptance_length,
            )
    return reporting.summary_row(name, len(indexed), stats), artifacts


def run(args, *, entrypoint):
    validate_args(args)
    provenance.prepare_checkpoints(args)
    save_provenance(args)
    if args.ascend_devices and args.worker_shard_index is None:
        try:
            parallel.run_ascend_data_parallel(args, entrypoint=entrypoint)
        finally:
            provenance.validate_checkpoint_files(args)
        return
    target, draft, tokenizer = load_models(args)
    provenance.validate_checkpoint_files(args)
    save_provenance(args, draft.config)
    from speculators_eval.leagle_decoding import (  # noqa: PLC0415
        LEagleOfflineRunner,
        resolve_stop_token_ids,
    )

    runner = LEagleOfflineRunner(target, draft, tokenizer, args)
    stop_ids = resolve_stop_token_ids(target, tokenizer)
    if args.worker_job_stream:
        from speculators_eval.persistent import (  # noqa: PLC0415
            serve_jobs,
            validate_worker_models,
        )

        validate_worker_models(target, draft)
        serve_jobs(args, runner, stop_ids)
        return
    rows, artifacts = [], {}
    for path in data.discover_datasets(
        args.datasets_root, data.split_csv(args.datasets) or None
    ):
        row, samples = evaluate_dataset(path, runner, args, stop_ids)
        rows.append(row)
        if not args.skip_artifacts:
            artifacts[row["dataset"]] = samples
        provenance.validate_checkpoint_files(args)
        reporting.write_outputs(args.output_dir, rows, artifacts)
    logger.info(
        "Wrote %s acceptance results to %s",
        draft.config.speculators_model_type,
        args.output_dir,
    )


def main(*, entrypoint=None):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(build_parser().parse_args(), entrypoint=entrypoint or Path(__file__))


if __name__ == "__main__":
    main()
