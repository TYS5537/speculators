"""Bounded single-host and two-host topologies for the DSV4 HS producer."""

import argparse
import ipaddress
import re

DP2_SIZE = 2
DP4_SIZE = 4
_MAX_PORT = 65535


def _validate_shared_endpoint(address, port):
    message = (
        "DSV4 DP4 requires an explicit, non-loopback --data-parallel-address "
        "and --data-parallel-rpc-port (1-65535), identical on both target hosts."
    )
    if not address or port is None or not 1 <= port <= _MAX_PORT:
        raise ValueError(message)
    try:
        host = ipaddress.ip_address(address)
    except ValueError:
        if (
            not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.?", address)
            or address.rstrip(".").lower() == "localhost"
        ):
            raise ValueError(message) from None
    else:
        if host.is_loopback or host.is_unspecified or host.is_multicast:
            raise ValueError(message)


def validate_parallel_config(parallel, *, block_verify=False):
    for name in (
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"DSV4 HS bridge requires {name}=1; TP/EP are allowed.")
    dp = getattr(parallel, "data_parallel_size", 1)
    if dp not in (1, DP2_SIZE, DP4_SIZE):
        raise ValueError("DSV4 HS bridge supports data_parallel_size=1, 2 or 4 only.")
    if block_verify and dp != 1:
        raise ValueError("DSV4 block verification still requires data_parallel_size=1.")
    if getattr(parallel, "tensor_parallel_size", 1) < 1:
        raise ValueError("DSV4 tensor_parallel_size must be positive.")
    if dp == 1:
        return
    _validate_dp_topology(parallel, dp)
    if dp == DP4_SIZE:
        _validate_shared_endpoint(
            getattr(
                parallel,
                "data_parallel_master_ip",
                getattr(parallel, "data_parallel_address", None),
            ),
            getattr(parallel, "data_parallel_rpc_port", None),
        )


def _validate_dp_topology(parallel, dp):
    if (
        getattr(parallel, "data_parallel_size_local", 1) != DP2_SIZE
        or getattr(parallel, "nnodes", 1) != 1
        or getattr(parallel, "node_rank", 0) != 0
        or getattr(parallel, "data_parallel_external_lb", False)
        or getattr(parallel, "data_parallel_hybrid_lb", False)
        or getattr(parallel, "data_parallel_multi_port_external_lb", False)
        or getattr(parallel, "data_parallel_backend", "mp") != "mp"
        or getattr(parallel, "distributed_executor_backend", None) not in (None, "mp")
    ):
        raise ValueError(
            f"DSV4 DP{dp} requires two local DP engines per host, internal load "
            "balancing and the mp backend (data_parallel_size_local=2). "
            "Keep nnodes=1 and node_rank=0: TP stays within each target host."
        )
    if not getattr(parallel, "enable_expert_parallel", False):
        raise ValueError(f"DSV4 DP{dp} requires --enable-expert-parallel.")
    # Native EngineArgs resolves the launch start rank to data_parallel_rank;
    # each EngineCore subsequently sets its own rank. All four ranks are valid.
    rank = getattr(parallel, "data_parallel_rank", None)
    if rank is not None and not 0 <= rank < dp:
        raise ValueError(f"DSV4 data_parallel_rank must be in [0, {dp}).")
    local_rank = getattr(parallel, "data_parallel_rank_local", None)
    if (
        dp == DP4_SIZE
        and local_rank is not None
        and (rank is None or local_rank != rank % DP2_SIZE)
    ):
        raise ValueError(
            "DSV4 DP4 data_parallel_rank_local must equal "
            "data_parallel_rank % 2 (local ranks 0 and 1 on each host)."
        )


def _configure_launch_ranks(parallel, vllm_args):
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
    elif dp == DP4_SIZE:
        if parallel.data_parallel_rank is not None:
            raise ValueError(
                "DSV4 DP4 uses internal load balancing; do not set "
                "--data-parallel-rank (it enables external load balancing)."
            )
        start = parallel.data_parallel_start_rank
        if (parallel.headless and start != DP2_SIZE) or (
            not parallel.headless and start not in (None, 0)
        ):
            raise ValueError(
                "DSV4 DP4 requires a head API at start rank 0 (or omitted), "
                "and a second host with --headless --data-parallel-start-rank 2."
            )


def configure_parallel_args(vllm_args, environment, *, block_verify=False):
    """Validate before creating the HS manifest or starting any engine.

    Leave DP1 launch arguments unchanged. DP2 has two local engines; DP4 has
    two local engines on each of two hosts, with TP confined to each host.
    Return the parsed topology, including the launch-only ``headless`` flag.
    The model plugin rechecks the resolved runtime independently on every rank.
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
    parser.add_argument("--data-parallel-address", "-dpa")
    parser.add_argument("--data-parallel-rpc-port", "-dpp", type=int)
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
    _configure_launch_ranks(parallel, vllm_args)
    validate_parallel_config(parallel, block_verify=block_verify)
    visible = environment.get("ASCEND_RT_VISIBLE_DEVICES")
    if dp in (DP2_SIZE, DP4_SIZE) and not visible:
        raise ValueError(f"Set ASCEND_RT_VISIBLE_DEVICES explicitly for DSV4 DP{dp}.")
    if parallel.data_parallel_size_local is None:
        parallel.data_parallel_size_local = dp
    if visible is not None:
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", visible):
            raise ValueError(
                "ASCEND_RT_VISIBLE_DEVICES must list numeric device IDs "
                "separated by commas."
            )
        devices = visible.split(",")
        if len(devices) != len(set(devices)):
            raise ValueError("ASCEND_RT_VISIBLE_DEVICES contains duplicate device IDs.")
        # Only the new DP4 topology spans hosts. Preserve the existing DP1/DP2
        # visible-device check even when a caller supplied another local size.
        device_dp_size = parallel.data_parallel_size_local if dp == DP4_SIZE else dp
        if len(devices) != parallel.tensor_parallel_size * device_dp_size:
            raise ValueError(
                "DSV4 visible device count must equal TP * local DP (PP=PCP=DCP=1)."
            )
    return parallel
