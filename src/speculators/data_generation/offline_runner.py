"""Shared offline HS scheduling, with explicit legacy publication policy."""

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import openai
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from speculators.data_generation.offline import (
    align_hidden_states,
    get_existing_hidden_state_indices,
    get_indices_to_process,
    validate_existing_hidden_states,
)
from speculators.data_generation.vllm_client import (
    generate_hidden_states_async,
    wait_for_lock_async,
)
from speculators.train.data import build_client_item

logger = logging.getLogger(__name__)


class FailureTracker:
    """Count consecutive failures across workers on the same asyncio loop."""

    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consecutive = 0

    def record_success(self) -> None:
        self._consecutive = 0

    def record_failure(self) -> bool:
        self._consecutive += 1
        return self._consecutive >= self.threshold


def publish_hidden_states(
    source: str, target: Path, prefix: dict | None = None
) -> None:
    """Publish atomically, even when the source is on another filesystem."""
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    if prefix is None:
        shutil.move(source, temporary)
    else:
        save_file(prefix, temporary)
    temporary.replace(target)
    if prefix is not None:
        Path(source).unlink()


def _validated_payload(
    source: str, item: dict, validate: bool, *, legacy: bool
) -> dict | None:
    allow_prefix = item.get("messages") is not None if legacy else "messages" in item
    loaded = None
    if validate or allow_prefix:
        loaded = align_hidden_states(
            load_file(source), item["input_ids"], allow_prefix=allow_prefix
        )
        if legacy and not allow_prefix:
            loaded = None  # Legacy validated text is moved, not reserialized.
    return loaded


def validate_and_publish(
    source: str, target: Path, item: dict, validate: bool, *, legacy: bool = False
) -> None:
    """Validate before publication; keep legacy text artifacts byte-for-byte."""
    publish_hidden_states(
        source, target, _validated_payload(source, item, validate, legacy=legacy)
    )


async def worker(
    client,
    model: str,
    queue: asyncio.Queue[dict[str, Any] | None],
    pbar: tqdm,
    vllm_semaphore: asyncio.Semaphore,
    write_semaphore: asyncio.Semaphore,
    hidden_states_output_dir: Path,
    validate_outputs: bool,
    request_timeout: float | None,
    max_retries: int,
    fail_on_error: bool,
    skipped_indices: list[int],
    cancel_event: asyncio.Event,
    failure_tracker: FailureTracker | None,
    stats: dict[str, Any] | None = None,
    *,
    legacy: bool = False,
    log: logging.Logger = logger,
) -> None:
    """Generate and publish queued samples, retaining failed source artifacts."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        idx = item["idx"]
        if cancel_event.is_set():
            queue.task_done()
            continue
        target = hidden_states_output_dir / f"hs_{idx}.safetensors"
        try:
            async with vllm_semaphore:
                started = time.perf_counter() if stats is not None else 0
                source = await generate_hidden_states_async(
                    client,
                    model,
                    item,
                    timeout=request_timeout,
                    max_retries=max_retries,
                )
                vllm_s = time.perf_counter() - started if stats is not None else 0
            lock_path = source + ".lock"
            if Path(lock_path).exists():  # noqa: ASYNC240
                await wait_for_lock_async(lock_path)
            async with write_semaphore:
                started = time.perf_counter() if stats is not None else 0
                if legacy:
                    prefix = None
                    if validate_outputs or item.get("messages") is not None:
                        prefix = await asyncio.to_thread(
                            _validated_payload,
                            source,
                            item,
                            validate_outputs,
                            legacy=True,
                        )
                    # Keep the legacy cancellation boundary: a canceled validation
                    # must not proceed to publish or remove the server artifact.
                    await asyncio.to_thread(
                        publish_hidden_states, source, target, prefix
                    )
                else:
                    await asyncio.to_thread(
                        validate_and_publish, source, target, item, validate_outputs
                    )
                write_s = time.perf_counter() - started if stats is not None else 0
        except Exception as error:
            if fail_on_error:
                log.exception(
                    "Fatal: sample %d aborted with --fail-on-error: %s", idx, error
                )
                logging.shutdown()
                os._exit(1)
            log.warning("Skipping sample %d due to error: %s", idx, error)
            skipped_indices.append(idx)
            if stats is not None:
                stats["errors"] += 1
            if failure_tracker is not None and failure_tracker.record_failure():
                cancel_event.set()
                raise RuntimeError(
                    f"Aborting: {failure_tracker.threshold} consecutive samples "
                    "errored out. The vLLM server may be unreachable."
                ) from error
        else:
            if stats is not None:
                stats["ok"] += 1
                stats["total_vllm_s"] += vllm_s
                stats["total_write_s"] += write_s
                log.debug(
                    "Sample %d: vLLM %.0f ms, write %.0f ms",
                    idx,
                    vllm_s * 1000,
                    write_s * 1000,
                )
            if failure_tracker is not None:
                failure_tracker.record_success()
        finally:
            if stats is not None:
                _update_progress(pbar, stats)
            pbar.update(1)
            queue.task_done()


def _update_progress(pbar, stats):
    elapsed = time.perf_counter() - stats["start_time"]
    postfix = {"ok": stats["ok"], "err": stats["errors"]}
    if elapsed > 0 and stats["ok"] > 0:
        postfix["rps"] = f"{stats['ok'] / elapsed:.1f}"
        postfix["vllm"] = f"{stats['total_vllm_s'] / stats['ok'] * 1000:.0f}ms"
        postfix["write"] = f"{stats['total_write_s'] / stats['ok'] * 1000:.0f}ms"
    pbar.set_postfix(postfix, refresh=False)


async def feed_queue(to_process, dataset, queue, cancel_event, *, legacy=False):
    for index in to_process:
        if cancel_event.is_set():
            break
        item = build_client_item(dataset[index], legacy_final_message=legacy)
        item = item | {"idx": index}
        while not cancel_event.is_set():
            try:
                queue.put_nowait(item)
                break
            except asyncio.QueueFull:
                await asyncio.sleep(0.1)


async def shutdown_workers(workers, queue, cancel_event, *, log=logger):
    """Stop workers and propagate the first non-cancellation exception."""
    log.info("Waiting for remaining file saves to complete...")
    if cancel_event.is_set():
        for task in workers:
            if not task.done():
                task.cancel()
    else:
        for _ in workers:
            await queue.put(None)
    results = await asyncio.gather(*workers, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception) and not isinstance(
            result, asyncio.CancelledError
        ):
            raise result


async def run_generation(args, dataset, *, legacy=False, log=logger):
    """Run either existing entrypoint's recipe over its already-loaded dataset."""
    directory = (
        Path(args.preprocessed_data) / "hidden_states"
        if args.output is None
        else Path(args.output)
    )
    directory.mkdir(parents=True, exist_ok=True)
    existing = get_existing_hidden_state_indices(directory)
    num_samples = len(dataset)
    if args.validate_outputs:
        requested = set(
            get_indices_to_process(
                num_samples, args.max_samples, [], args.world_size, args.rank
            )
        )
        local_existing = [index for index in existing if index in requested]
        valid = await asyncio.to_thread(
            validate_existing_hidden_states, directory, dataset, local_existing
        )
        invalid = set(local_existing) - set(valid)
        existing = [index for index in existing if index not in invalid]
    to_process = get_indices_to_process(
        num_samples, args.max_samples, existing, args.world_size, args.rank
    )
    if not to_process:
        return
    log.info(f"Processing {len(to_process)} samples")
    queue = asyncio.Queue(maxsize=args.concurrency * 4)
    vllm_semaphore = asyncio.Semaphore(args.concurrency)
    write_semaphore = asyncio.Semaphore(args.concurrency)
    skipped_indices = []
    cancel_event = asyncio.Event()
    stats = (
        None
        if legacy
        else {
            "ok": 0,
            "errors": 0,
            "total_vllm_s": 0.0,
            "total_write_s": 0.0,
            "start_time": time.perf_counter(),
        }
    )
    threshold = args.max_consecutive_errors
    if threshold is None:
        threshold = args.concurrency
    tracker = None if args.fail_on_error else FailureTracker(threshold)
    async with openai.AsyncOpenAI(
        base_url=args.endpoint, api_key="EMPTY", max_retries=0
    ) as client:
        models = await client.models.list()
        if not legacy and not models.data:
            raise RuntimeError(
                "No models found on the vLLM server. "
                "Make sure the server is fully loaded."
            )
        model_id = models.data[0].id
        if args.model and args.model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({args.model}) which doesn't match"
                f" found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )
        with tqdm(total=len(to_process)) as pbar:
            workers = [
                asyncio.create_task(
                    worker(
                        client,
                        model_id,
                        queue,
                        pbar,
                        vllm_semaphore,
                        write_semaphore,
                        directory,
                        args.validate_outputs,
                        args.request_timeout,
                        args.max_retries,
                        args.fail_on_error,
                        skipped_indices,
                        cancel_event,
                        tracker,
                        stats,
                        legacy=legacy,
                        log=log,
                    )
                )
                for _ in range(args.concurrency * 2)
            ]
            await feed_queue(to_process, dataset, queue, cancel_event, legacy=legacy)
            await shutdown_workers(workers, queue, cancel_event, log=log)
    if stats is not None and stats["ok"] > 0:
        elapsed = time.perf_counter() - stats["start_time"]
        log.info(
            "Timing: %.1fs elapsed, %.1f samples/s, "
            "avg vLLM request %.0f ms, avg file write %.0f ms",
            elapsed,
            stats["ok"] / elapsed if elapsed > 0 else 0,
            stats["total_vllm_s"] / stats["ok"] * 1000,
            stats["total_write_s"] / stats["ok"] * 1000,
        )
    num_saved = len(to_process) - len(skipped_indices)
    log.info(
        f"Saved {num_saved} new data points to {args.output if legacy else directory}"
    )
    if skipped_indices:
        log.warning(
            f"Skipped {len(skipped_indices)} samples due to errors: {skipped_indices}"
        )
