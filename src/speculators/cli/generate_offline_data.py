"""
Offline Hidden States Generation Pipeline

This module generates hidden states and saves them to disk for offline training.

Usage::

    speculators generate-offline-data \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --preprocessed-data sharegpt \
        --output ./training_data \
        --max-samples 5000
"""

import asyncio
import logging
import sys
from functools import partial
from types import SimpleNamespace
from typing import Annotated

import typer
from datasets import load_from_disk

from speculators.data_generation import offline_runner as _offline_runner
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
)
from speculators.train.logger import setup_root_logger

logger = logging.getLogger(__name__)

# Preserve the historical helper imports while sharing their implementation.
_FailureTracker = _offline_runner.FailureTracker
_validate_and_publish = _offline_runner.validate_and_publish
_worker = partial(_offline_runner.worker, log=logger)
_feed_queue = _offline_runner.feed_queue
_shutdown_workers = partial(_offline_runner.shutdown_workers, log=logger)


async def _generate_and_save_hidden_states(
    model: str | None,
    endpoint: str,
    preprocessed_data: str,
    output: str | None,
    max_samples: int | None,
    concurrency: int,
    validate_outputs: bool,
    request_timeout: float,
    max_retries: int,
    fail_on_error: bool,
    max_consecutive_errors: int | None,
    world_size: int,
    rank: int,
):
    dataset = load_from_disk(preprocessed_data)
    args = SimpleNamespace(
        model=model,
        endpoint=endpoint,
        preprocessed_data=preprocessed_data,
        output=output,
        max_samples=max_samples,
        concurrency=concurrency,
        validate_outputs=validate_outputs,
        request_timeout=request_timeout,
        max_retries=max_retries,
        fail_on_error=fail_on_error,
        max_consecutive_errors=max_consecutive_errors,
        world_size=world_size,
        rank=rank,
    )
    return await _offline_runner.run_generation(args, dataset, log=logger)


def generate_offline_data(
    model: Annotated[
        str | None,
        typer.Option(
            help=(
                "HuggingFace model ID or local path for target model "
                "(default auto select). For verification purposes only."
            ),
        ),
    ] = None,
    endpoint: Annotated[
        str,
        typer.Option(
            help=(
                "The address of the vLLM instance to use for hidden states "
                "generation. The instance must be configured for hidden states "
                "extraction."
            ),
        ),
    ] = "http://localhost:8000/v1",
    preprocessed_data: Annotated[
        str,
        typer.Option(
            help="Path to preprocessed dataset (produced by prepare-data)",
        ),
    ] = "./output",
    max_samples: Annotated[
        int | None,
        typer.Option(help="Maximum number of samples to process"),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            help=(
                "Directory to save generated hidden states files "
                "(default: {preprocessed-data}/hidden_states)"
            ),
        ),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(
            help=(
                "Number of active vLLM requests at a time. "
                "Note: number of async workers set to 2*concurrency"
            ),
        ),
    ] = 32,
    validate_outputs: Annotated[
        bool,
        typer.Option(
            "--validate-outputs",
            help=(
                "Load generated safetensor files and check output token ids "
                "match prompt tokens and hidden states seq_len matches num tokens"
            ),
        ),
    ] = False,
    request_timeout: Annotated[
        float,
        typer.Option(
            help="Timeout in seconds for each individual vLLM request",
        ),
    ] = DEFAULT_REQUEST_TIMEOUT,
    max_retries: Annotated[
        int,
        typer.Option(
            help="Maximum number of retry attempts per request on failure",
        ),
    ] = DEFAULT_MAX_RETRIES,
    fail_on_error: Annotated[
        bool,
        typer.Option(
            "--fail-on-error",
            help=(
                "Abort when a request fails after all retries. "
                "By default, failed samples are skipped."
            ),
        ),
    ] = False,
    max_consecutive_errors: Annotated[
        int | None,
        typer.Option(
            help=(
                "Abort after this many consecutive sample failures (each sample "
                "already retried --max-retries times). Prevents silently churning "
                "through the entire dataset when the server is down. "
                "Ignored when --fail-on-error is set. "
                "(default: value of --concurrency)"
            ),
        ),
    ] = None,
    world_size: Annotated[
        int,
        typer.Option(
            help=(
                "World size for multi-node data generation offline. "
                "This is the number of nodes (not GPUs)."
            ),
        ),
    ] = 1,
    rank: Annotated[
        int,
        typer.Option(
            help=(
                "Rank for multi-node data generation offline. "
                "This is the node index, not a GPU index. "
                "Must be in range [0, world_size)."
            ),
        ),
    ] = 0,
) -> None:
    """Generate hidden states offline from a vLLM server.

    Connects to a running vLLM instance, sends preprocessed samples, and saves
    the extracted hidden states to disk for offline training.
    """
    if concurrency < 1:
        raise typer.BadParameter("--concurrency must be >= 1")
    if rank < 0 or rank >= world_size:
        raise typer.BadParameter("--rank must be in range [0, world_size)")
    setup_root_logger()

    logger.info("EAGLE Offline Data Generation")

    try:
        asyncio.run(
            _generate_and_save_hidden_states(
                model=model,
                endpoint=endpoint,
                preprocessed_data=preprocessed_data,
                output=output,
                max_samples=max_samples,
                concurrency=concurrency,
                validate_outputs=validate_outputs,
                request_timeout=request_timeout,
                max_retries=max_retries,
                fail_on_error=fail_on_error,
                max_consecutive_errors=max_consecutive_errors,
                world_size=world_size,
                rank=rank,
            )
        )
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        logger.exception("Data generation failed")
        sys.exit(1)

    logger.info("Data generation complete!")
