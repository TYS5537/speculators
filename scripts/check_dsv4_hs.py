"""Request HS exports and check token alignment, shape, dtype and finiteness.

Run in the Linux trainer environment with the checkpoint and HS directory shared
with the server. This checks transport/layout, NOT target quantization accuracy
or teacher logits. The generated HS file is retained for further inspection.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from speculators_dsv4.contract import (
    DEFAULT_LAYERS,
    ensure_manifest,
    inspect_checkpoint,
    make_manifest,
    validate_layers,
)


def check_requests(input_ids, requests, concurrency, check_one):
    """Exercise varying prefix lengths and reject reused response filenames."""
    if requests < 1 or concurrency < 1 or not input_ids:
        raise ValueError("requests, concurrency and input length must be positive")
    prefixes = [
        input_ids[: len(input_ids) - i % len(input_ids)] for i in range(requests)
    ]
    with ThreadPoolExecutor(max_workers=min(requests, concurrency)) as executor:
        results = list(executor.map(check_one, prefixes))
    handles = [result["hidden_states_file"] for result in results]
    if len(set(handles)) != len(handles):
        raise ValueError("Different HS requests returned the same output file")
    return results


def validate_payload(payload, input_ids, layer_count, torch):
    if payload is None:
        raise ValueError(
            "HS file is missing; verify shared mounts and file permissions."
        )
    if payload["token_ids"].tolist() != input_ids:
        raise ValueError("Exported token IDs do not match the request.")
    hidden = payload["hidden_states"]
    expected = (len(input_ids), layer_count, 4096)
    if tuple(hidden.shape) != expected:
        raise ValueError(f"Expected HS shape {expected}, got {tuple(hidden.shape)}.")
    if hidden.dtype != torch.bfloat16 or not torch.isfinite(hidden).all().item():
        raise ValueError("Hidden states must be finite BF16 tensors.")
    return hidden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Shared DSV4 checkpoint path")
    parser.add_argument("--hidden-states-path", required=True)
    parser.add_argument("--vllm-endpoint", default="http://localhost:8001/v1")
    parser.add_argument("--input-ids", nargs="+", type=int, required=True)
    parser.add_argument(
        "--target-layer-ids", nargs="+", type=int, default=DEFAULT_LAYERS
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--requests",
        type=int,
        default=1,
        help="Number of probes, cycling through decreasing input prefix lengths",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent HS requests (use 2 or more to exercise DP2 load balancing)",
    )
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")

    import openai  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from hs_connectors import FileTransfer  # noqa: PLC0415
    from speculators.data_generation.vllm_client import (  # noqa: PLC0415
        generate_hidden_states,
    )

    validate_layers(args.target_layer_ids)
    report = inspect_checkpoint(args.model)
    directory = Path(args.hidden_states_path).resolve()
    ensure_manifest(directory, make_manifest(report, args.target_layer_ids))
    if any(
        token < 0 or token >= report["config"]["vocab_size"] for token in args.input_ids
    ):
        raise ValueError("Input token ID outside target vocabulary.")

    def check_one(input_ids):
        # Each thread owns its client; the producer uses a fresh request ID.
        with openai.OpenAI(
            base_url=args.vllm_endpoint,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            max_retries=0,
        ) as client:
            handle = generate_hidden_states(
                client,
                args.model,
                {"input_ids": input_ids},
                timeout=args.timeout,
                max_retries=0,
            )
        if not handle or not Path(handle).resolve().is_relative_to(directory):
            raise ValueError(
                "Server returned an HS path outside the shared HS directory."
            )
        payload = FileTransfer(directory).get_generated(handle)
        hidden = validate_payload(
            payload, input_ids, len(args.target_layer_ids) + 1, torch
        )
        return {
            "hidden_states_file": str(Path(handle).resolve()),
            "shape": list(hidden.shape),
            "dtype": str(hidden.dtype),
            "auxiliary_hs_ids": args.target_layer_ids,
            "teacher_hs_id": 43,
            "per_slot_rms": hidden.float().square().mean(dim=(0, 2)).sqrt().tolist(),
        }

    results = check_requests(args.input_ids, args.requests, args.concurrency, check_one)
    print(
        json.dumps(
            results[0]
            if len(results) == 1
            else {"requests": results, "concurrency": args.concurrency},
            indent=2,
        )
    )
    if args.requests > 1:
        print(
            "Concurrent probes do NOT prove both DP engines were used; "
            "check server per-engine request metrics/logs. All HS files are retained."
        )
    print(
        "HS transport/layout passed. "
        "Teacher-logit and target numerical parity remain to be checked with "
        "scripts/check_dsv4_teacher.py --verification-mode reference (requires "
        "the training server's DSV4_EVAL=1 full-logprob options), or block "
        "against a dedicated block-evaluation service."
    )


if __name__ == "__main__":
    main()
