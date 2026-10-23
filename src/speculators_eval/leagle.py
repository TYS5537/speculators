"""LEAGLE acceptance evaluation. Parent worker dispatch stays model-free."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from speculators_eval import data, parallel, provenance, reporting
from speculators_eval.diagnostics import capture_sample_failure
from speculators_eval.progress import WorkerProgress

logger = logging.getLogger("speculators_eval")


def build_parser():
    parser = argparse.ArgumentParser(
        description="EAGLE3 / LEAGLE single-chain greedy acceptance with an HF target."
    )
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--datasets", help="Comma-separated JSONL dataset names")
    parser.add_argument("--output-dir", type=Path, default=Path("output/leagle_eval"))
    parser.add_argument("--spec-tokens", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, choices=[0.0], default=0.0)
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
        "--check-greedy",
        action="store_true",
        help="Audit every output against target-only HF greedy generation (extra work)",
    )
    parser.add_argument(
        "--ascend-devices",
        help="Sample-parallel NPU IDs, e.g. 0,1,2,3; each device holds both models",
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


def validate_args(args):
    if args.spec_tokens < 1 or args.max_new_tokens < 0:
        raise ValueError("--spec-tokens must be >= 1 and --max-new-tokens >= 0")
    if args.temperature != 0:
        raise ValueError("LEAGLE evaluation currently supports temperature=0 only")
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.log_every < 1 or args.worker_num_shards < 1:
        raise ValueError("--log-every and --worker-num-shards must be positive")
    if (args.d2t_path is None) != (args.t2d_path is None):
        raise ValueError("--d2t-path and --t2d-path must be provided together")
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


def save_provenance(args, draft_config=None):
    metadata = {
        "proposal": "single-chain-greedy",
        "verification": "HF full-attention target with KV rollback",
        "target_kv_reuse": True,
        "draft_kv_reuse_between_rounds": False,
        "online_speedup_benchmark": False,
        "probability_diagnostics_available": False,
        "timing_scope": (
            "parallel worker wall time including model loads"
            if args.ascend_devices and args.worker_shard_index is None
            else "local generation including target-only greedy audit"
            if args.check_greedy
            else "local generation without serving optimizations"
        ),
    }
    if draft_config is not None:
        metadata["draft_config"] = draft_config.to_dict()
        metadata["algorithm"] = draft_config.speculators_model_type
    provenance.write_eval_provenance(args, metadata)


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
    stats = reporting.EvalStats(probability_diagnostics_available=False)
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
