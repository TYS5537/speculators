"""Opt-in, batched DSV4 verification export for vLLM 0.26.0.

The model hook runs on every TP rank and uses the native target LM head. Each
DP engine batches independent full prefixes; TP rank zero publishes a separate
packet per request before its completion response. Idle DP engines join error
synchronization without projecting logits or writing files. No target KV state
is exported or rolled back by this base connector. The optional cached subclass
supplies a computed-prefix offset. Neither constructs a full-prefix vocabulary
tensor.
"""

# ruff: noqa: ARG002 -- Preserve vLLM connector method keyword signatures.

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from functools import wraps
from importlib import import_module
from pathlib import Path

import torch
from safetensors.torch import save_file
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_tensor_model_parallel_rank,
    get_tp_group,
)

from speculators_dsv4.block_protocol import (
    BLOCK_KV_VERSION,
    BLOCK_PROFILE_STAGES,
    BLOCK_PROTOCOL_VERSION,
    BLOCK_REQUEST_KEY,
    validate_block_request,
)
from speculators_dsv4.parallel import validate_parallel_config
from speculators_eval.profiling import EvaluationProfiler

_HIDDEN_NDIM = 2


@dataclass
class BlockRequest:
    request_id: str
    filename: str
    token_ids: list[int]
    logits_start: int
    hidden_start: int
    version: int = BLOCK_PROTOCOL_VERSION
    output_mode: str = "logprobs"
    profile: bool = False
    computed_tokens: int = 0

    @property
    def query_length(self):
        return len(self.token_ids) - self.computed_tokens


@dataclass
class BlockMetadata(KVConnectorMetadata):
    requests: list[BlockRequest] = field(default_factory=list)
    data_parallel_rank: int = 0
    # Filled by the worker AFTER native input preparation/reordering. Scheduler
    # list order is not a reliable map into the packed forward tensors.
    forward_layout: list[tuple[str, int, int]] | None = None


def _save_packet(tensors, filename):
    """Publish only a complete packet; never replace an earlier request file."""
    path = Path(filename)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Block verification file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", dir=path.parent)
    os.close(fd)
    try:
        save_file(tensors, temporary)
        # An exclusive hard-link publish also prevents request-ID collisions
        # from silently replacing an artifact between the check and publish.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _pack_verification_rows(batch, normalized):
    if not batch:
        return None
    suffixes = [
        normalized[
            start + request.logits_start - request.computed_tokens : start
            + request.query_length
        ]
        for request, start in batch
    ]
    return suffixes[0] if len(suffixes) == 1 else torch.cat(suffixes)


class DSV4BlockVerifyConnector(KVConnectorBase_V1, SupportsHMA):
    """Transport request ranges to the model, and return its completed packet."""

    supports_snapshot_reuse = False

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        parallel = vllm_config.parallel_config
        validate_parallel_config(parallel, block_verify=True)
        self._dp_size = parallel.data_parallel_size
        self._dp_rank = parallel.data_parallel_rank
        self._max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        if type(self._max_num_seqs) is not int or self._max_num_seqs < 1:
            raise ValueError("DSV4 block verification requires positive max-num-seqs")
        self._storage_path = Path(
            self._kv_transfer_config.get_from_extra_config("shared_storage_path", "")
        ).resolve()
        if not self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", ""
        ):
            raise ValueError("DSV4 block verification requires a storage directory")
        speculation = vllm_config.speculative_config
        if speculation is None or speculation.method != "extract_hidden_states":
            raise ValueError("Block verification requires extract_hidden_states")
        self._layer_ids = list(
            speculation.draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids
        )
        self._vocab_size = vllm_config.model_config.hf_config.vocab_size
        self._hidden_size = vllm_config.model_config.hf_config.hidden_size
        self._max_verify_rows = self._kv_transfer_config.get_from_extra_config(
            "max_verify_rows", 128
        )
        if type(self._max_verify_rows) is not int or self._max_verify_rows <= 0:
            raise ValueError("max_verify_rows must be a positive integer")
        self._requests = {}

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            raise ValueError("Block verification cannot load external KV state")

    def build_connector_meta(self, scheduler_output):
        requests = scheduler_output.scheduled_new_reqs
        scheduled = scheduler_output.num_scheduled_tokens
        if not requests:
            if any(scheduled.values()):
                raise ValueError(
                    "Block verification requires a fresh full-prefix request"
                )
            return BlockMetadata(data_parallel_rank=self._dp_rank)
        request_ids = [request.req_id for request in requests]
        if (
            len(requests) > self._max_num_seqs
            or len(set(request_ids)) != len(request_ids)
            or set(scheduled) != set(request_ids)
        ):
            raise ValueError(
                "Block verification requires distinct full-prefix requests "
                "within max-num-seqs"
            )
        # Validate the whole batch before registering anything as in-flight.
        blocks = [self._build_request(request, scheduled) for request in requests]
        self._requests.update((block.request_id, block) for block in blocks)
        return BlockMetadata(blocks, data_parallel_rank=self._dp_rank)

    def _build_request(self, request, scheduled):
        tokens = request.prompt_token_ids
        computed = self._computed_tokens(request)
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(
                type(token) is not int or not 0 <= token < self._vocab_size
                for token in tokens
            )
            or request.num_computed_tokens != computed
            or scheduled[request.req_id] != len(tokens) - computed
            or request.prompt_embeds is not None
            or request.mm_features
            or request.lora_request is not None
        ):
            raise ValueError(
                "Block verification requires an uncached, unchunked token prefix"
            )
        sampling = request.sampling_params
        if sampling is None or sampling.max_tokens != 1 or sampling.n != 1:
            raise ValueError("Block verification requires max_tokens=1 and n=1")
        if sampling.logprobs is not None or sampling.prompt_logprobs is not None:
            raise ValueError(
                "Block verification returns probabilities in its packet, "
                "not HTTP logprobs"
            )
        params = (sampling.extra_args or {}).get("kv_transfer_params") or {}
        if not isinstance(params, dict) or set(params) != {BLOCK_REQUEST_KEY}:
            raise ValueError(
                "This target requires the DSV4 block verification protocol"
            )
        options = params[BLOCK_REQUEST_KEY]
        logits_start, hidden_start = validate_block_request(options, len(tokens))
        if options["version"] == BLOCK_KV_VERSION and not self.supports_snapshot_reuse:
            raise ValueError("Start the block server with --dsv4-kv-reuse")
        if min(logits_start, hidden_start) < computed:
            raise ValueError("Block output cannot request rows inside cached KV")
        if len(tokens) - logits_start > self._max_verify_rows:
            raise ValueError(
                f"Block verification exceeds {self._max_verify_rows} logit rows"
            )
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", request.req_id):
            raise ValueError("Unsafe block verification request ID")
        filename = str(self._storage_path / f"{request.req_id}.safetensors")
        if request.req_id in self._requests:
            raise ValueError("Duplicate in-flight block verification request ID")
        return BlockRequest(
            request.req_id,
            filename,
            list(tokens),
            logits_start,
            hidden_start,
            version=options["version"],
            output_mode=options.get("output_mode", "logprobs"),
            profile=options.get("profile", False),
            computed_tokens=computed,
        )

    def _computed_tokens(self, request):
        return 0

    def request_finished(self, request, block_ids):
        block_request = self._requests.pop(request.request_id, None)
        if block_request is None:
            # Aborted before being scheduled: no artifact was produced.
            return False, None
        return False, {
            "hidden_states_path": block_request.filename,
            "dsv4_block_verify_version": block_request.version,
        }

    def request_finished_all_groups(self, request, block_ids):
        return self.request_finished(request, block_ids)

    def start_load_kv(self, *args, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, *args, **kwargs):
        pass

    def wait_for_save(self):
        pass  # Publication is synchronous inside the model's forward call.

    def capture(self, model, input_ids, positions, output, *, profiler=None):
        normalized, _ = output
        batch, error = [], None
        try:
            batch = self._capture_requests(input_ids, positions, output)
            # Packing allocates a small suffix buffer. Report rank-local packing
            # failures before any peer enters the native head's TP collective.
            verification_hidden = _pack_verification_rows(batch, normalized)
        except Exception as exc:  # noqa: BLE001 -- Synchronize before raising.
            error = exc
        # A rank-local validation error must not leave its TP peers in the head's
        # gather. Idle/profile forwards also participate: with DP2 the native
        # Ascend runner coordinates their MoE forwards with the active engine.
        failed = torch.tensor(
            [int(error is not None)], dtype=torch.int32, device=normalized.device
        )
        failed = get_tp_group().all_reduce(failed)
        if not failed.item() and batch:
            if profiler is None:
                profiler = EvaluationProfiler(
                    enabled=any(request.profile for request, _ in batch),
                    device=normalized.device,
                )
            # The native head communicates ONLY within this TP group, so DP
            # engines may project different batch/row counts (or none when idle).
            # One projection/collective handles ALL verification rows in this
            # engine, not one collective per request interleaved with file IO.
            with profiler.measure("server_head"):
                logits = model.compute_logits(verification_hidden)
            if get_tensor_model_parallel_rank() == 0:
                try:
                    self._write_batch(batch, logits, output, profiler)
                except Exception as exc:  # noqa: BLE001 -- Report IO errors to peers.
                    error = exc
                    failed.fill_(1)
        failed = get_tp_group().all_reduce(failed)
        if self._dp_size > 1:
            # Do not return early for missing/empty metadata: the idle engine's
            # dummy forward must rendezvous here too, before its next MoE step.
            failed = get_dp_group().all_reduce(failed)
        if failed.item():
            raise RuntimeError(
                "DSV4 block packet export failed on a target TP/DP rank"
            ) from error

    def _capture_requests(self, input_ids, positions, output):
        if not self.has_connector_metadata():
            return []  # Profiling/warmup or a DP engine's idle dummy forward.
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, BlockMetadata):
            raise ValueError("Unexpected DSV4 block connector metadata")
        if metadata.data_parallel_rank != self._dp_rank:
            raise ValueError("Block request metadata belongs to another DP engine")
        if not metadata.requests:
            return []
        batch = self._batch_layout(metadata)
        length = sum(request.query_length for request, _ in batch)
        if (
            input_ids is None
            or input_ids.ndim != 1
            or input_ids.shape[0] < length
            or positions.ndim != 1
            or positions.shape[0] < length
        ):
            raise ValueError("Block forward tokens/positions have invalid shape")
        actual_tokens = input_ids[:length].tolist()
        actual_positions = positions[:length].tolist()
        for request, start in batch:
            end = start + request.query_length
            if actual_tokens[start:end] != request.token_ids[
                request.computed_tokens :
            ] or actual_positions[start:end] != list(
                range(request.computed_tokens, len(request.token_ids))
            ):
                raise ValueError(
                    "Block forward tokens/positions do not match the request layout"
                )
        normalized, auxiliary = output
        if (
            normalized.ndim != _HIDDEN_NDIM
            or normalized.shape[0] < length
            or normalized.shape[1] != self._hidden_size
            or len(auxiliary) != len(self._layer_ids)
            or any(
                value.ndim != _HIDDEN_NDIM
                or value.shape[0] < length
                or value.shape[1] != self._hidden_size
                or value.dtype != torch.bfloat16
                for value in auxiliary
            )
        ):
            raise ValueError("Block forward hidden-state shape/dtype mismatch")
        return batch

    def _batch_layout(self, metadata):
        requests = {request.request_id: request for request in metadata.requests}
        if (
            len(requests) != len(metadata.requests)
            or len(requests) > self._max_num_seqs
        ):
            raise ValueError("Invalid block request batch size/identities")
        layout = metadata.forward_layout
        if layout is None and len(requests) == 1:
            # A single full prefix has an unambiguous layout, preserving v1.
            request = metadata.requests[0]
            layout = [(request.request_id, 0, request.query_length)]
        if layout is None or len(layout) != len(requests):
            raise ValueError("Missing block worker request layout")
        offset, batch = 0, []
        for request_id, start, end in layout:
            request = requests.pop(request_id, None)
            if (
                request is None
                or type(start) is not int
                or type(end) is not int
                or start != offset
                or end - start != request.query_length
            ):
                raise ValueError("Invalid block worker request layout")
            batch.append((request, start))
            offset = end
        return batch

    def _write_batch(self, batch, logits, output, profiler):
        normalized, auxiliary = output
        rows = sum(
            len(request.token_ids) - request.logits_start for request, _ in batch
        )
        if logits is None or tuple(logits.shape) != (rows, self._vocab_size):
            raise ValueError(
                f"Expected block target logits shape {(rows, self._vocab_size)}"
            )
        row = 0
        for request, start in batch:
            count = len(request.token_ids) - request.logits_start
            request_profiler = EvaluationProfiler(
                enabled=request.profile, device=normalized.device
            )
            # Shared forward/head durations are equal per-request shares, not
            # independent per-request latency. Do not multiply batch work by N.
            for name in ("server_forward", "server_head"):
                seconds = profiler.stages.get(name, {}).get("seconds", 0.0)
                request_profiler.record(name, seconds / len(batch))
            end = start + request.query_length
            self._write_output(
                request,
                logits[row : row + count],
                [value[start:end] for value in auxiliary],
                profiler=request_profiler,
            )
            row += count

    def _write_output(self, request, logits, auxiliary, *, profiler):
        length = len(request.token_ids)
        expected = (length - request.logits_start, self._vocab_size)
        if logits is None or tuple(logits.shape) != expected:
            raise ValueError(f"Expected block target logits shape {expected}")
        with profiler.measure("server_packet_prepare"):
            # Keep the same FP32 normalization even in greedy mode: removing it
            # can change argmax ties caused by log-softmax rounding. Only the
            # compact IDs cross the device/host boundary in the greedy path.
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            if request.output_mode == "logprobs":
                logprobs = logprobs.detach().cpu().contiguous()
            # Also rejects all-masked rows while allowing zero-probability tokens.
            if not torch.isfinite(torch.logsumexp(logprobs, dim=-1)).all().item():
                raise ValueError(
                    "Block verification produced nonfinite probabilities/HS"
                )
            if request.output_mode == "greedy":
                result = {"greedy_token_ids": logprobs.argmax(-1).detach().cpu()}
            else:
                result = {"logprobs": logprobs}
            hidden = (
                torch.stack(
                    [
                        value[request.hidden_start - request.computed_tokens :]
                        for value in auxiliary
                    ],
                    dim=1,
                )
                .detach()
                .cpu()
                .contiguous()
            )
            if not torch.isfinite(hidden).all().item():
                raise ValueError(
                    "Block verification produced nonfinite probabilities/HS"
                )
        if request.profile:
            result["server_timings"] = torch.tensor(
                [
                    profiler.stages.get(name, {}).get("seconds", 0.0)
                    for name in BLOCK_PROFILE_STAGES
                ],
                dtype=torch.float64,
                device="cpu",
            )
        _save_packet(
            {
                "token_ids": torch.tensor(
                    request.token_ids, dtype=torch.int64, device="cpu"
                ),
                "verification_metadata": torch.tensor(
                    [
                        request.version,
                        length,
                        request.logits_start,
                        request.hidden_start,
                    ],
                    dtype=torch.int64,
                    device="cpu",
                ),
                "layer_ids": torch.tensor(
                    self._layer_ids, dtype=torch.int64, device="cpu"
                ),
                "hidden_states": hidden,
                **result,
                **self._packet_extras(request),
            },
            request.filename,
        )

    def _packet_extras(self, request):
        return {}


def block_forward_profiler(device):
    """Only explicit per-request profiling may synchronize the native forward."""
    enabled = False
    if has_kv_transfer_group():
        connector = get_kv_transfer_group()
        check_ready = getattr(connector, "check_ready", None)
        if isinstance(connector, DSV4BlockVerifyConnector) and check_ready is not None:
            check_ready(device)
        if (
            isinstance(connector, DSV4BlockVerifyConnector)
            and connector.has_connector_metadata()
        ):
            metadata = connector._get_connector_metadata()  # noqa: SLF001
            if isinstance(metadata, BlockMetadata):
                enabled = any(request.profile for request in metadata.requests)
    return EvaluationProfiler(enabled=enabled, device=device)


def install_worker_block_layout():
    """Bind request IDs to packed rows using the pinned Ascend worker's real order."""
    runner = import_module("vllm_ascend.worker.model_runner_v1")
    _install_block_layout(runner.NPUModelRunner)


def _install_block_layout(runner_class):
    original = runner_class._prepare_inputs  # noqa: SLF001 -- Pinned worker API.
    if getattr(original, "_speculators_dsv4_block_layout", False):
        return

    @wraps(original)
    def prepare_inputs(runner, scheduler_output, *args, **kwargs):
        result = original(runner, scheduler_output, *args, **kwargs)
        metadata = getattr(scheduler_output, "kv_connector_metadata", None)
        if isinstance(metadata, BlockMetadata):
            request_ids = list(runner.input_batch.req_ids)
            offsets = runner.query_start_loc.np[: len(request_ids) + 1].tolist()
            metadata.forward_layout = [
                (request_id, offsets[i], offsets[i + 1])
                for i, request_id in enumerate(request_ids)
            ]
        return result

    prepare_inputs._speculators_dsv4_block_layout = True  # noqa: SLF001 -- Idempotence.
    runner_class._prepare_inputs = prepare_inputs  # noqa: SLF001 -- Scoped wrapper.


def export_block(model, input_ids, positions, output, *, profiler=None):
    """Called only by the opt-in DSV4 model, never by Qwen/reference training."""
    if not has_kv_transfer_group():
        return  # Model initialization/profiling can precede connector setup.
    connector = get_kv_transfer_group()
    if not isinstance(connector, DSV4BlockVerifyConnector):
        raise ValueError("DSV4 block export requires its dedicated connector")
    connector.capture(model, input_ids, positions, output, profiler=profiler)
