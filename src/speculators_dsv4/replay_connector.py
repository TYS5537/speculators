"""Opt-in greedy generation alongside stateless block HS extraction.

Generation uses the native scheduler/KV cache and sampler, with no per-token
files or host snapshots. A separate block request extracts the finished trace.
Legacy block and cached connectors retain their strict one-step contracts.
"""

from dataclasses import dataclass, field

from speculators_dsv4.block_connector import (
    BlockMetadata,
    DSV4BlockVerifyConnector,
)
from speculators_dsv4.block_protocol import GREEDY_REQUEST_KEY, GREEDY_VERSION


@dataclass
class ReplayMetadata(BlockMetadata):
    generation_queries: dict[str, int] = field(default_factory=dict)


class DSV4ReplayConnector(DSV4BlockVerifyConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generating = set()

    @staticmethod
    def _is_generation(request):
        params = (request.sampling_params.extra_args or {}).get(
            "kv_transfer_params"
        ) or {}
        return GREEDY_REQUEST_KEY in params

    def _validate_generation(self, request, scheduled):
        sampling = request.sampling_params
        params = (sampling.extra_args or {}).get("kv_transfer_params")
        options = params.get(GREEDY_REQUEST_KEY) if isinstance(params, dict) else None
        if (
            params != {GREEDY_REQUEST_KEY: {"version": GREEDY_VERSION}}
            or type(options.get("version")) is not int
            or sampling.temperature != 0.0
            or sampling.n != 1
            or sampling.max_tokens < 1
            or sampling.logprobs is not None
            or sampling.prompt_logprobs is not None
            or request.num_computed_tokens != 0
            or scheduled != len(request.prompt_token_ids)
            or not request.prompt_token_ids
            or request.prompt_embeds is not None
            or request.mm_features
            or request.lora_request is not None
        ):
            raise ValueError(
                "Replay generation requires an unchunked greedy token prompt"
            )

    def build_connector_meta(self, scheduler_output):
        # Native decoding requests may span steps; block extraction may not.
        self._generating.difference_update(
            getattr(scheduler_output, "finished_req_ids", ())
        )
        scheduled = scheduler_output.num_scheduled_tokens
        fresh = scheduler_output.scheduled_new_reqs
        fresh_ids = [request.req_id for request in fresh]
        if len(set(fresh_ids)) != len(fresh_ids) or len(scheduled) > self._max_num_seqs:
            raise ValueError("Invalid replay batch identities/size")
        blocks, generation = [], {}
        for request in fresh:
            if request.req_id not in scheduled:
                raise ValueError("Missing replay request token budget")
            if self._is_generation(request):
                self._validate_generation(request, scheduled[request.req_id])
                generation[request.req_id] = scheduled[request.req_id]
            else:
                blocks.append(self._build_request(request, scheduled))
        for request_id, length in scheduled.items():
            if request_id not in fresh_ids:
                # extract_hidden_states uses one native lookahead slot. Decode
                # can therefore schedule the current token plus that slot.
                if request_id not in self._generating or length not in (1, 2):
                    raise ValueError(
                        "Only native replay generation may continue decoding"
                    )
                generation[request_id] = length
        if set(fresh_ids) & self._generating:
            raise ValueError(
                "Replay preemption/resume is unsupported; reduce concurrency"
            )
        self._generating.update(generation)
        self._requests.update((block.request_id, block) for block in blocks)
        return ReplayMetadata(
            requests=blocks,
            data_parallel_rank=self._dp_rank,
            generation_queries=generation,
        )

    def _batch_layout(self, metadata):
        if not isinstance(metadata, ReplayMetadata) or not metadata.generation_queries:
            return super()._batch_layout(metadata)
        blocks = {request.request_id: request for request in metadata.requests}
        queries = {key: request.query_length for key, request in blocks.items()}
        if set(queries) & set(metadata.generation_queries):
            raise ValueError("Replay block and generation identities overlap")
        queries.update(metadata.generation_queries)
        layout = metadata.forward_layout
        if layout is None or len(layout) != len(queries):
            raise ValueError("Missing mixed replay batch layout")
        offset, result = 0, []
        for request_id, start, end in layout:
            length = queries.pop(request_id, None)
            if length is None or start != offset or end - start != length:
                raise ValueError("Invalid mixed replay batch layout")
            if request_id in blocks:
                result.append((blocks[request_id], start))
            offset = end
        return result

    def request_finished(self, request, block_ids):
        if request.request_id in self._generating:
            self._generating.remove(request.request_id)
            return False, {GREEDY_REQUEST_KEY: {"version": GREEDY_VERSION}}
        return super().request_finished(request, block_ids)
