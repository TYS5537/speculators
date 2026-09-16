"""Opt-in, single-request DSV4 verification export for vLLM 0.26.0.

The model hook runs on every TP rank and uses the native target LM head. Only
TP rank zero publishes the selected probabilities and HS, synchronously and
atomically before the completion response. No target KV state is exported or
rolled back, and no full-prefix vocabulary tensor is constructed.
"""

# ruff: noqa: ARG002 -- Preserve vLLM connector method keyword signatures.

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tp_group

from speculators_dsv4.block_protocol import (
    BLOCK_PROTOCOL_VERSION,
    BLOCK_REQUEST_KEY,
    validate_block_request,
)

_HIDDEN_NDIM = 2


@dataclass
class BlockRequest:
    request_id: str
    filename: str
    token_ids: list[int]
    logits_start: int
    hidden_start: int


@dataclass
class BlockMetadata(KVConnectorMetadata):
    request: BlockRequest | None = None


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


class DSV4BlockVerifyConnector(KVConnectorBase_V1, SupportsHMA):
    """Transport request ranges to the model, and return its completed packet."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        if vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError("DSV4 block verification requires --max-num-seqs 1")
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
        self._request_filenames = {}

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            raise ValueError("Block verification cannot load external KV state")

    def build_connector_meta(self, scheduler_output):  # noqa: C901
        requests = scheduler_output.scheduled_new_reqs
        scheduled = scheduler_output.num_scheduled_tokens
        if not requests:
            if any(scheduled.values()):
                raise ValueError(
                    "Block verification requires a fresh full-prefix request"
                )
            return BlockMetadata()
        if len(requests) != 1 or set(scheduled) != {requests[0].req_id}:
            raise ValueError("Block verification supports only one complete prefix")
        request = requests[0]
        tokens = request.prompt_token_ids
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(
                type(token) is not int or not 0 <= token < self._vocab_size
                for token in tokens
            )
            or request.num_computed_tokens != 0
            or scheduled[request.req_id] != len(tokens)
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
        logits_start, hidden_start = validate_block_request(
            params[BLOCK_REQUEST_KEY], len(tokens)
        )
        if len(tokens) - logits_start > self._max_verify_rows:
            raise ValueError(
                f"Block verification exceeds {self._max_verify_rows} logit rows"
            )
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", request.req_id):
            raise ValueError("Unsafe block verification request ID")
        filename = str(self._storage_path / f"{request.req_id}.safetensors")
        if request.req_id in self._request_filenames:
            raise ValueError("Duplicate in-flight block verification request ID")
        self._request_filenames[request.req_id] = filename
        return BlockMetadata(
            BlockRequest(
                request.req_id, filename, list(tokens), logits_start, hidden_start
            )
        )

    def request_finished(self, request, block_ids):
        filename = self._request_filenames.pop(request.request_id, None)
        if filename is None:
            # Aborted before being scheduled: no artifact was produced.
            return False, None
        return False, {
            "hidden_states_path": filename,
            "dsv4_block_verify_version": BLOCK_PROTOCOL_VERSION,
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

    def capture(self, model, input_ids, positions, output):
        if not self.has_connector_metadata():
            return  # vLLM profiling/warmup, not an evaluation request.
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, BlockMetadata):
            raise ValueError("Unexpected DSV4 block connector metadata")
        request = metadata.request
        if request is None:
            return
        length = len(request.token_ids)
        if (
            input_ids is None
            or input_ids.ndim != 1
            or input_ids.shape[0] < length
            or input_ids[:length].tolist() != request.token_ids
            or positions.ndim != 1
            or positions.shape[0] < length
            or positions[:length].tolist() != list(range(length))
        ):
            raise ValueError(
                "Block forward tokens/positions do not match its full prefix"
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
        # All TP ranks must enter the native head's collectives, even when
        # LogitsProcessor returns None on non-root ranks. Slice BEFORE the head.
        logits = model.compute_logits(normalized[request.logits_start : length])
        error = None
        if get_tensor_model_parallel_rank() == 0:
            try:
                self._write_output(request, logits, auxiliary)
            except Exception as exc:  # noqa: BLE001 -- Sync failures across TP ranks.
                error = exc
        failed = torch.tensor(
            [int(error is not None)], dtype=torch.int32, device=normalized.device
        )
        failed = get_tp_group().all_reduce(failed)
        if failed.item():
            raise RuntimeError(
                "DSV4 block packet export failed on the target TP rank"
            ) from error

    def _write_output(self, request, logits, auxiliary):
        length = len(request.token_ids)
        expected = (length - request.logits_start, self._vocab_size)
        if logits is None or tuple(logits.shape) != expected:
            raise ValueError(f"Expected block target logits shape {expected}")
        logprobs = torch.log_softmax(logits.float(), dim=-1).detach().cpu().contiguous()
        hidden = (
            torch.stack(
                [value[request.hidden_start : length] for value in auxiliary], dim=1
            )
            .detach()
            .cpu()
            .contiguous()
        )
        if not torch.isfinite(hidden).all().item() or (
            torch.isnan(logprobs).any().item()
            or torch.isposinf(logprobs).any().item()
            or not torch.isfinite(torch.logsumexp(logprobs, dim=-1)).all().item()
        ):
            raise ValueError("Block verification produced nonfinite probabilities/HS")
        _save_packet(
            {
                "token_ids": torch.tensor(request.token_ids, dtype=torch.int64),
                "verification_metadata": torch.tensor(
                    [
                        BLOCK_PROTOCOL_VERSION,
                        length,
                        request.logits_start,
                        request.hidden_start,
                    ],
                    dtype=torch.int64,
                ),
                "layer_ids": torch.tensor(self._layer_ids, dtype=torch.int64),
                "logprobs": logprobs,
                "hidden_states": hidden,
            },
            request.filename,
        )


def export_block(model, input_ids, positions, output):
    """Called only by the opt-in DSV4 model, never by Qwen/reference training."""
    if not has_kv_transfer_group():
        return  # Model initialization/profiling can precede connector setup.
    connector = get_kv_transfer_group()
    if not isinstance(connector, DSV4BlockVerifyConnector):
        raise ValueError("DSV4 block export requires its dedicated connector")
    connector.capture(model, input_ids, positions, output)
