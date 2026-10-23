"""Argument parsing for offline evaluation, without model imports."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline DSpark/MMuse/DFlash2 evaluation on JSONL data.",
    )
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-backend", choices=["hf", "dsv4-vllm"], default="hf")
    parser.add_argument("--vllm-endpoint", default=None)
    parser.add_argument("--hidden-states-path", type=Path, default=None)
    parser.add_argument(
        "--hs-http-endpoint",
        default=None,
        help="Optional authenticated HS sidecar URL (not the vLLM /v1 URL). "
        "Use DSV4_HS_HTTP_TOKEN; --hidden-states-path becomes "
        "a local download directory.",
    )
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--dsv4-max-model-len", type=int, default=4096)
    parser.add_argument(
        "--dsv4-verification-mode",
        choices=["reference", "block", "replay"],
        default="reference",
    )
    parser.add_argument(
        "--dsv4-replay-cache", type=Path, default=Path("dsv4_greedy_traces")
    )
    parser.add_argument("--dsv4-replay-cache-tag", default="")
    parser.add_argument(
        "--dsv4-replay-audit-samples",
        type=int,
        default=0,
        help="Compare the first N samples per worker against live block decoding",
    )
    parser.add_argument(
        "--dsv4-kv-reuse",
        action="store_true",
        help="Reuse native V4 target KV via a matching --dsv4-kv-reuse block server",
    )
    parser.add_argument(
        "--dsv4-block-output",
        choices=["auto", "full"],
        default="auto",
        help="Auto uses compact greedy IDs for temperature=0 only when Correction "
        "does not need target logits. Full retains the version-1 probability path.",
    )
    parser.add_argument(
        "--dsv4-profile",
        action="store_true",
        help="Write timing.json with synchronized diagnostic timings; block mode "
        "also requests server timings and requires the updated server.",
    )
    parser.add_argument("--target-request-timeout", type=float, default=120.0)
    parser.add_argument("--keep-target-hs", action="store_true")
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("dspark_offline_eval"))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Override the DeepSpec per-dataset sample cap. When unset, known "
            "DeepSpec datasets use the limits from DeepSpec-Ascend/eval.py."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--spec-tokens",
        type=int,
        default=None,
        help="Cap draft tokens per round; must fit the checkpoint's slots.",
    )
    parser.add_argument("--target-attn-impl", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument(
        "--check-greedy",
        action="store_true",
        help="Audit HF temperature=0 outputs against target-only generation.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--enable-thinking",
        choices=["false", "true", "default"],
        default="false",
    )
    parser.add_argument(
        "--raw-prompt-mode",
        choices=["auto", "chat_template", "raw"],
        default="auto",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--draft-kv-reuse",
        action="store_true",
        help="Cache confirmed draft context projections/KV across rounds (HF or "
        "DSV4 target); opt-in, with no checkpoint changes",
    )
    parser.add_argument(
        "--draft-attn-impl",
        choices=["auto", "simple_flex_attention", "sdpa", "eager"],
        default="auto",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--measure-base-speedup",
        action="store_true",
        help=(
            "Also run verifier-only autoregressive decoding and report measured "
            "speculative output-throughput speedup."
        ),
    )
    parser.add_argument(
        "--throughput-warmup-samples",
        type=int,
        default=1,
        help="Warmup prompts per dataset before paired throughput timing.",
    )
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--d2t-path", type=Path, default=None)
    parser.add_argument("--t2d-path", type=Path, default=None)
    parser.add_argument(
        "--sample-from-anchor",
        choices=["true", "false"],
        default=None,
        help="Override checkpoint config. Leave unset to use checkpoint value.",
    )
    parser.add_argument("--ascend-devices", default=None)
    parser.add_argument("--worker-shard-index", type=int, default=None)
    parser.add_argument("--worker-num-shards", type=int, default=1)
    parser.add_argument("--worker-progress-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-manifest", type=Path, help=argparse.SUPPRESS)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
