"""Execution policy for the opt-in DSV4 HS bridge (no torch/vLLM imports)."""

import argparse
import json

EXECUTION_MODES = ("eager", "full-decode-only")


def configure_execution_args(vllm_args, mode, *, block_verify=False):
    """Own graph configuration; forward native async scheduling explicitly.

    Keep torch.compile disabled: a compiled inner model may bypass the Python
    norm hook while the outer bridge still expects a fresh teacher capture.
    FULL_DECODE_ONLY instead captures the whole ForCausalLM in ACLGraphWrapper;
    prefill stays eager. Do not silently fall back from a requested graph mode.
    """
    if mode not in EXECUTION_MODES:
        raise ValueError(f"Unknown DSV4 execution mode: {mode}")
    normalized = [
        arg.split("=", 1)[0].replace("_", "-")
        + ("=" + arg.split("=", 1)[1] if "=" in arg else "")
        if arg.startswith("--")
        else arg
        for arg in vllm_args
    ]
    for arg in normalized:
        key = arg.split("=", 1)[0]
        if key.startswith(
            ("--compilation-config", "-cc", "--optimization-level", "-O")
        ):
            raise ValueError(
                "DSV4 HS bridge owns compilation config; use "
                "--dsv4-execution-mode eager or full-decode-only."
            )
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--async-scheduling", action=argparse.BooleanOptionalAction, default=None
    )
    parsed, _ = parser.parse_known_args(normalized)
    eager = mode == "eager"
    if parsed.enforce_eager is not None and parsed.enforce_eager != eager:
        raise ValueError("--enforce-eager conflicts with --dsv4-execution-mode.")
    if block_verify and (not eager or parsed.async_scheduling):
        raise ValueError(
            "DSV4 block verification requires eager, synchronous execution."
        )
    if parsed.enforce_eager is None:
        vllm_args.append("--enforce-eager" if eager else "--no-enforce-eager")
    if parsed.async_scheduling is None:
        # Avoid version-dependent automatic scheduling choices for this bridge.
        vllm_args.append("--no-async-scheduling")
    vllm_args.extend(
        [
            "--compilation-config",
            json.dumps(
                {"mode": 0, "cudagraph_mode": "NONE" if eager else "FULL_DECODE_ONLY"}
            ),
        ]
    )


def validate_execution_config(vllm_config, *, block_verify=False):
    """Check resolved worker config, including backend rewrites of CLI options."""
    eager = vllm_config.model_config.enforce_eager
    compilation = vllm_config.compilation_config
    mode = getattr(compilation, "cudagraph_mode", "NONE")
    mode = getattr(mode, "name", mode)
    compile_mode = getattr(compilation, "mode", 0)
    asynchronous = bool(
        getattr(vllm_config.scheduler_config, "async_scheduling", False)
    )
    if block_verify and (not eager or asynchronous):
        raise ValueError(
            "DSV4 block verification requires eager, synchronous execution."
        )
    if compile_mode != 0:
        raise ValueError("DSV4 HS export requires compilation mode NONE (0).")
    if eager:
        if mode != "NONE":
            raise ValueError("Eager DSV4 HS export requires graph mode NONE.")
        return "eager", asynchronous
    if mode != "FULL_DECODE_ONLY":
        raise ValueError("DSV4 HS graph execution requires FULL_DECODE_ONLY.")
    return "full-decode-only", asynchronous
