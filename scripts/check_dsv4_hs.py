"""Request one HS export and check token alignment, shape, dtype and finiteness.

Run in the Linux trainer environment with the checkpoint and HS directory shared
with the server. This checks transport/layout, NOT target quantization accuracy
or teacher logits. The generated HS file is retained for further inspection.
"""

import argparse
import json
import os
from pathlib import Path

from speculators_dsv4.contract import (
    DEFAULT_LAYERS,
    ensure_manifest,
    inspect_checkpoint,
    make_manifest,
    validate_layers,
)


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
    args = parser.parse_args()

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
    with openai.OpenAI(
        base_url=args.vllm_endpoint,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        max_retries=0,
    ) as client:
        handle = generate_hidden_states(
            client,
            args.model,
            {"input_ids": args.input_ids},
            timeout=args.timeout,
            max_retries=0,
        )
    if not handle or not Path(handle).resolve().is_relative_to(directory):
        raise ValueError("Server returned an HS path outside the shared HS directory.")
    payload = FileTransfer(directory).get_generated(handle)
    if payload is None:
        raise ValueError(
            "HS file is missing; verify shared mounts and file permissions."
        )
    if payload["token_ids"].tolist() != args.input_ids:
        raise ValueError("Exported token IDs do not match the request.")
    hidden = payload["hidden_states"]
    expected = (len(args.input_ids), len(args.target_layer_ids) + 1, 4096)
    if tuple(hidden.shape) != expected:
        raise ValueError(f"Expected HS shape {expected}, got {tuple(hidden.shape)}.")
    if hidden.dtype != torch.bfloat16 or not torch.isfinite(hidden).all().item():
        raise ValueError("Hidden states must be finite BF16 tensors.")
    print(
        json.dumps(
            {
                "hidden_states_file": handle,
                "shape": list(hidden.shape),
                "dtype": str(hidden.dtype),
                "auxiliary_hs_ids": args.target_layer_ids,
                "teacher_hs_id": 43,
                "per_slot_rms": hidden.float()
                .square()
                .mean(dim=(0, 2))
                .sqrt()
                .tolist(),
            },
            indent=2,
        )
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
