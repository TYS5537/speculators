"""Audit DSV4 training teacher probabilities against native target logprobs.

reference: existing HS training server with DSV4_EVAL=1 / raw full-vocab logprobs.
block: dedicated DSV4 block-evaluation service (not an ordinary training server).
The service, checkpoint, and request-owned HS directory must refer to the same
target. Nothing is started automatically; no automatic protocol fallback occurs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speculators_dsv4.contract import DEFAULT_LAYERS, inspect_checkpoint
from speculators_dsv4.offline import DSV4OfflineTarget
from speculators_dsv4.parity import (
    ParityThresholds,
    check_teacher_parity,
    load_teacher_head,
    select_positions,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hidden-states-path", required=True)
    parser.add_argument("--vllm-endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument(
        "--served-model-name", help="Exact server model alias; defaults to --model"
    )
    parser.add_argument(
        "--verification-mode", choices=("reference", "block"), default="reference"
    )
    parser.add_argument(
        "--target-layer-ids", nargs="+", type=int, default=DEFAULT_LAYERS
    )
    parser.add_argument("--input-ids", nargs="+", type=int, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--positions", nargs="+", type=int, help="Ascending zero-based HS rows"
    )
    selection.add_argument("--tail-positions", type=int, default=4)
    parser.add_argument("--position-chunk-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--device", default="cpu", help="cpu or the trainer device, e.g. npu:0"
    )
    parser.add_argument(
        "--head-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Projection compute dtype; BF16 matches the trainer autocast",
    )
    parser.add_argument(
        "--norm-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="FP32 matches fresh/config-only training parameters; select BF16 only "
        "if the training model itself was loaded with BF16 norm weights",
    )
    parser.add_argument("--max-tv", type=float, default=0.02)
    parser.add_argument("--max-logprob-error", type=float, default=0.5)
    parser.add_argument("--min-argmax-agreement", type=float, default=1.0)
    parser.add_argument("--keep-target-hs", action="store_true")
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def run(args):
    import httpx  # noqa: PLC0415
    import openai  # noqa: PLC0415

    if args.device.startswith("npu"):
        import torch_npu  # noqa: PLC0415, F401

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if args.position_chunk_size <= 0:
        raise ValueError("position-chunk-size must be positive")
    endpoint = urlsplit(args.vllm_endpoint)
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.hostname
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
        or endpoint.path.rstrip("/") != "/v1"
    ):
        raise ValueError("vllm-endpoint must be an explicit HTTP(S) /v1 API URL")
    thresholds = ParityThresholds(
        args.max_tv, args.max_logprob_error, args.min_argmax_agreement
    )
    thresholds.validate()
    select_positions(
        len(args.input_ids),
        positions=args.positions,
        tail_positions=args.tail_positions,
    )
    report = inspect_checkpoint(args.model)
    model_name = args.served_model_name or args.model
    with openai.OpenAI(
        base_url=args.vllm_endpoint,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        max_retries=0,
        timeout=args.timeout,
        http_client=httpx.Client(
            trust_env=False, follow_redirects=False, timeout=args.timeout
        ),
    ) as client:
        models = client.models.list()
        if model_name not in [model.id for model in models.data]:
            raise ValueError(
                "Requested model alias is not advertised by this target service"
            )
        target = DSV4OfflineTarget.from_contract(
            report,
            args.target_layer_ids,
            hidden_states_path=args.hidden_states_path,
            client=client,
            model_name=model_name,
            max_model_len=args.max_model_len,
            timeout=args.timeout,
            keep_hidden_states=args.keep_target_hs,
            verification_mode=args.verification_mode,
        )
        teacher = load_teacher_head(
            report,
            device=args.device,
            dtype=args.head_dtype,
            norm_dtype=args.norm_dtype,
        )
        result = check_teacher_parity(
            target,
            teacher,
            args.input_ids,
            positions=args.positions,
            tail_positions=args.tail_positions,
            position_chunk_size=args.position_chunk_size,
            thresholds=thresholds,
        )
    result.update(
        model_path=report["model_path"],
        checkpoint_signature=report["checkpoint_signature"],
        served_model_name=model_name,
        device=args.device,
        target_request_count=target.num_target_requests,
        auxiliary_hs_ids=list(args.target_layer_ids),
        default_training_dtypes_checked=(
            args.head_dtype == "bfloat16" and args.norm_dtype == "float32"
        ),
    )
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as error:  # noqa: BLE001 -- Diagnostic CLI must fail closed.
        print(
            f"DSV4 teacher check failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
