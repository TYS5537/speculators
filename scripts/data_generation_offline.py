#!/usr/bin/env python3
"""
Offline Hidden States Generation Pipeline

This script generates hidden states and saves them to disk for offline training.

Usage:
    python data_generation_offline.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --preprocessed-data sharegpt \
        --output ./training_data \
        --max-samples 5000
"""

import argparse
import asyncio
import logging
import sys
from functools import partial

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
_publish_hidden_states = _offline_runner.publish_hidden_states
worker = partial(_offline_runner.worker, legacy=True, log=logger)
_feed_queue = partial(_offline_runner.feed_queue, legacy=True)
_shutdown_workers = partial(_offline_runner.shutdown_workers, log=logger)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate EAGLE training data offline")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "HuggingFace model ID or local path for target model "
            "(default auto select). For verification purposes only."
        ),
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help=(
            "The address of the vLLM instance to use for hidden states generation "
            "(default: 'http://localhost:8000/v1'). "
            "Note: the vLLM instance must be configured for hidden states extraction."
        ),
    )

    # Data arguments
    parser.add_argument(
        "--preprocessed-data",
        type=str,
        default="./output",
        help="Path to preprocessed dataset (dataset produced by prepare_data.py)"
        " (default: ./output)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: None, process all)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Directory to generated hidden states files "
            "(default args.preprocessed_data / 'hidden_states')"
        ),
    )

    # Hidden states generation arguments
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help=(
            "Number of active vLLM requests at a time. "
            "Note: number of async workers set to 2*concurrency"
        ),
    )
    parser.add_argument(
        "--validate-outputs",
        action="store_true",
        help=(
            "Load generated safetensor files and check output token ids match "
            "prompt tokens and hidden states seq_len matches num tokens"
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=(
            "Timeout in seconds for each individual vLLM request "
            f"(default: {DEFAULT_REQUEST_TIMEOUT})"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Maximum number of retry attempts per request on failure "
            f"(default: {DEFAULT_MAX_RETRIES})"
        ),
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help=(
            "Abort when a request fails after all retries. "
            "By default, failed samples are skipped."
        ),
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=None,
        help=(
            "Abort after this many consecutive sample failures (each sample "
            "already retried --max-retries times). Prevents silently churning "
            "through the entire dataset when the server is down. "
            "Ignored when --fail-on-error is set. "
            "(default: value of --concurrency)"
        ),
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help=(
            "World size for multi-node data generation offline. IMPORTANT: this "
            "is the number of nodes (not the number of gpus). Defaults to 1"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help=(
            "Rank for multi-node data generation offline. IMPORTANT: this is "
            "the node index, not an index for a gpu. Must be in range[0, world_size)."
            " Defaults to 0"
        ),
    )
    return parser.parse_args()


async def generate_and_save_hidden_states(args, dataset):
    return await _offline_runner.run_generation(args, dataset, legacy=True, log=logger)


def main():
    args = parse_args()
    if int(args.rank) < 0 or int(args.rank) >= int(args.world_size):
        raise ValueError("--rank must be in range [0, world_size)")
    setup_root_logger()

    logger.info("EAGLE Offline Data Generation")

    dataset = load_from_disk(args.preprocessed_data)

    try:
        asyncio.run(generate_and_save_hidden_states(args, dataset))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        logger.exception("Data generation failed")
        sys.exit(1)

    logger.info("Data generation complete!")


if __name__ == "__main__":
    main()
