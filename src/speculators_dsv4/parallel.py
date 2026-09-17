"""Single-host topology checks for the experimental DSV4 HS producer."""

import argparse
import re

DP2_SIZE = 2


def validate_parallel_config(parallel, *, block_verify=False):
    for name in (
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"DSV4 HS bridge requires {name}=1; TP/EP are allowed.")
    dp = getattr(parallel, "data_parallel_size", 1)
    if dp not in (1, DP2_SIZE):
        raise ValueError("DSV4 HS bridge supports data_parallel_size=1 or 2 only.")
    if block_verify and dp != 1:
        raise ValueError("DSV4 block verification still requires data_parallel_size=1.")
    if getattr(parallel, "tensor_parallel_size", 1) < 1:
        raise ValueError("DSV4 tensor_parallel_size must be positive.")
    if dp == 1:
        return
    if (
        getattr(parallel, "data_parallel_size_local", 1) != dp
        or getattr(parallel, "nnodes", 1) != 1
        or getattr(parallel, "node_rank", 0) != 0
        or getattr(parallel, "data_parallel_external_lb", False)
        or getattr(parallel, "data_parallel_hybrid_lb", False)
        or getattr(parallel, "data_parallel_multi_port_external_lb", False)
        or getattr(parallel, "data_parallel_backend", "mp") != "mp"
        or getattr(parallel, "distributed_executor_backend", None) not in (None, "mp")
    ):
        raise ValueError(
            "DSV4 DP2 requires two local DP engines on one host, internal load "
            "balancing and the mp backend (data_parallel_size_local=2)."
        )
    if not getattr(parallel, "enable_expert_parallel", False):
        raise ValueError("DSV4 DP2 requires --enable-expert-parallel.")


def configure_parallel_args(vllm_args, environment, *, block_verify=False):  # noqa: C901
    """Validate before creating the HS manifest or starting any engine.

    Leave DP1 launch arguments unchanged. DP2 opts into two *local* engines;
    the resolved runtime is checked again by the model plugin on every rank.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name, short, default in (
        ("tensor-parallel-size", "-tp", 1),
        ("data-parallel-size", "-dp", 1),
        ("data-parallel-size-local", "-dpl", None),
        ("pipeline-parallel-size", "-pp", 1),
        ("prefill-context-parallel-size", "-pcp", 1),
        ("decode-context-parallel-size", "-dcp", 1),
    ):
        parser.add_argument(f"--{name}", short, type=int, default=default)
    parser.add_argument("--nnodes", "-n", type=int, default=1)
    parser.add_argument("--node-rank", "-r", type=int, default=0)
    parser.add_argument("--data-parallel-backend", "-dpb", default="mp")
    parser.add_argument("--distributed-executor-backend")
    for name, short in (
        ("enable-expert-parallel", "-ep"),
        ("data-parallel-external-lb", "-dpe"),
        ("data-parallel-hybrid-lb", "-dph"),
        ("data-parallel-multi-port-external-lb", "-dpm"),
    ):
        parser.add_argument(
            f"--{name}", short, action=argparse.BooleanOptionalAction, default=False
        )
    parser.add_argument("--data-parallel-rank", "-dpn", type=int)
    parser.add_argument("--data-parallel-start-rank", "-dpr", type=int)
    parser.add_argument("--headless", action="store_true")
    # vLLM's flexible parser accepts underscores as well as hyphens.
    normalized = [
        arg.split("=", 1)[0].replace("_", "-")
        + ("=" + arg.split("=", 1)[1] if "=" in arg else "")
        if arg.startswith("--")
        else arg
        for arg in vllm_args
    ]
    parallel, _ = parser.parse_known_args(normalized)
    dp = parallel.data_parallel_size
    if dp == DP2_SIZE:
        if (
            parallel.headless
            or parallel.data_parallel_rank is not None
            or parallel.data_parallel_start_rank is not None
        ):
            raise ValueError(
                "DSV4 DP2 uses a single API with internal load balancing; "
                "do not set DP ranks or --headless."
            )
        if parallel.data_parallel_size_local is None:
            parallel.data_parallel_size_local = DP2_SIZE
            vllm_args.extend(["--data-parallel-size-local", "2"])
    validate_parallel_config(parallel, block_verify=block_verify)
    visible = environment.get("ASCEND_RT_VISIBLE_DEVICES")
    if dp == DP2_SIZE and not visible:
        raise ValueError("Set ASCEND_RT_VISIBLE_DEVICES explicitly for DSV4 DP2.")
    if visible is not None:
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", visible):
            raise ValueError(
                "ASCEND_RT_VISIBLE_DEVICES must list numeric device IDs "
                "separated by commas."
            )
        devices = visible.split(",")
        if len(devices) != len(set(devices)):
            raise ValueError("ASCEND_RT_VISIBLE_DEVICES contains duplicate device IDs.")
        if len(devices) != parallel.tensor_parallel_size * dp:
            raise ValueError("DSV4 visible device count must equal TP * DP (PP=PCP=1).")
