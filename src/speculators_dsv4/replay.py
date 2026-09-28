"""Greedy-only acceptance replay on checked native target trajectories.

Only prefix-acceptance counts are available: target probabilities after the first
rejected candidate are intentionally NOT invented from an unrelated trajectory.
Cache preparation and optional live audits are not serving throughput.
"""

import logging
from types import SimpleNamespace

import torch

from speculators_dsv4.block_protocol import GREEDY_REQUEST_KEY, GREEDY_VERSION
from speculators_dsv4.offline import DSV4OfflineTarget
from speculators_dsv4.replay_cache import GreedyTraceCache, trace_identity

logger = logging.getLogger(__name__)
_TOKEN_INPUT_NDIM = 2


class DSV4GreedyReplayTarget(DSV4OfflineTarget):
    is_greedy_replay = True
    probability_diagnostics_available = False

    def __init__(self, *args, replay_cache, replay_tag="", audit_samples=0, **kwargs):
        kwargs["verification_mode"] = "block"
        super().__init__(*args, **kwargs)
        self.trace_cache = GreedyTraceCache(replay_cache)
        self.replay_tag = replay_tag
        self.audit_remaining = audit_samples
        self.live_audit = False
        self.trace = None

    def configure_evaluation(
        self, *, temperature, requires_target_logits, block_output
    ):
        if temperature != 0.0 or requires_target_logits or block_output != "auto":
            raise ValueError(
                "Greedy replay requires temperature=0, block-output=auto, and a "
                "draft that does not consume full target logits; use block otherwise"
            )
        super().configure_evaluation(
            temperature=temperature,
            requires_target_logits=requires_target_logits,
            block_output=block_output,
        )

    def prepare_sample(self, input_ids, max_new_tokens, stop_token_ids):
        self.trace = None
        if (
            input_ids.ndim != _TOKEN_INPUT_NDIM
            or input_ids.shape[0] != 1
            or input_ids.shape[1] == 0
        ):
            raise ValueError("Replay requires one nonempty token prompt")
        self.validate_request_budget(input_ids.shape[1], max_new_tokens, 0)
        prompt = input_ids[0].tolist()
        identity = trace_identity(
            self, prompt, max_new_tokens, stop_token_ids, self.replay_tag
        )
        with self.profiler.measure("trace_cache_load", synchronize=False):
            trace = self.trace_cache.load(identity)
        if trace is None:
            logger.info("Replay cache miss: generating and checking target trajectory")
            with self.profiler.measure("trace_prepare", synchronize=False):
                trace = self._build_trace(prompt, max_new_tokens, stop_token_ids)
                trace = self.trace_cache.publish(identity, trace)
            self.profiler.count("trace_cache_misses", 1)
        else:
            self.profiler.count("trace_cache_hits", 1)
        self.trace = trace
        self.prompt_length = len(prompt)

    def _build_trace(self, prompt, max_new_tokens, stop_token_ids):
        request_id = self._new_request_id()
        self.num_target_requests += 1
        response = self._completion(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_new_tokens,
            n=1,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=[],
            timeout=self.timeout,
            extra_headers={"X-Request-Id": request_id},
            extra_body={
                "request_id": request_id,
                "top_k": 0,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "ignore_eos": True,
                "stop_token_ids": [],
                "skip_special_tokens": False,
                "add_special_tokens": False,
                "return_token_ids": True,
                "kv_transfer_params": {GREEDY_REQUEST_KEY: {"version": GREEDY_VERSION}},
            },
        )
        self._validate_response(response, request_id)
        if (getattr(response, "kv_transfer_params", None) or {}).get(
            GREEDY_REQUEST_KEY
        ) != {"version": GREEDY_VERSION}:
            raise ValueError("Start the dedicated target with --dsv4-greedy-replay")
        choice = response.choices[0]
        generated = getattr(choice, "token_ids", None)
        if (
            getattr(choice, "prompt_token_ids", None) != prompt
            or not isinstance(generated, list)
            or len(generated) != max_new_tokens
            or any(
                type(token) is not int or not 0 <= token < self.vocab_size
                for token in generated
            )
        ):
            raise ValueError(
                "Native greedy generation returned invalid/incomplete token IDs"
            )
        # Request the full budget to avoid backend-dependent EOS stripping of IDs.
        # Only the causal prefix through the first evaluator stop token is cached.
        for index, token in enumerate(generated):
            if token in (stop_token_ids or []):
                generated = generated[: index + 1]
                break
        tokens = prompt + generated
        pieces = []
        logits_start, hidden_start = len(prompt) - 1, 0
        while logits_start < len(tokens) - 1:
            end = min(logits_start + 128, len(tokens) - 1)
            ids, hidden = self._request_block(
                tokens[:end], logits_start=logits_start, hidden_start=hidden_start
            )
            if ids.tolist() != tokens[logits_start + 1 : end + 1]:
                raise ValueError(
                    "Native greedy decode and block extraction disagree; trace was "
                    "NOT cached. Use live block evaluation for this numeric path."
                )
            pieces.append(hidden[:, : len(self.layer_ids), :])
            logits_start = hidden_start = end
        return {
            "tokens": torch.tensor(tokens, dtype=torch.int64),
            "hidden_states": torch.cat(pieces).contiguous(),
        }

    def _states(self, start, end):
        hidden = self.trace["hidden_states"][start:end]
        return {
            layer: hidden[:, slot, :].unsqueeze(0).to(self.device)
            for slot, layer in enumerate(self.layer_ids)
        }

    def __call__(self, **kwargs):
        if self.live_audit:
            return super().__call__(**kwargs)
        cache = kwargs["past_key_values"]
        prompt = kwargs["input_ids"][0].tolist()
        if (
            self.trace is None
            or cache.get_seq_length() != 0
            or prompt != self.trace["tokens"][: self.prompt_length].tolist()
            or kwargs["position_ids"].tolist() != [list(range(len(prompt)))]
        ):
            raise ValueError("Replay prefill must match the prepared prompt")
        cache.tokens = list(prompt)
        return SimpleNamespace(
            logits=None,
            greedy_token_ids=self.trace["tokens"][len(prompt) : len(prompt) + 1]
            .unsqueeze(0)
            .to(self.device),
            vocab_size=self.vocab_size,
            hidden_states=self._states(0, len(prompt)),
        )

    def verify_proposal(self, *, proposal, cache, start, stop_token_ids):
        if self.live_audit:
            return None
        tokens = self.trace["tokens"].tolist()
        proposed = proposal.verify_input_ids[0].tolist()
        count = proposal.draft_token_count
        if (
            start != len(cache.tokens)
            or cache.tokens != tokens[:start]
            or len(proposed) != count + 1
            or proposed[0] != tokens[start]
        ):
            raise ValueError("Replay proposal does not follow the committed trajectory")
        accepted, stopped = 0, False
        for offset, token in enumerate(proposed[1:], start=1):
            if start + offset >= len(tokens) or token != tokens[start + offset]:
                break
            accepted += 1
            if token in (stop_token_ids or []):
                stopped = True
                break
        next_index = start + accepted + 1
        if not stopped and next_index >= len(tokens):
            raise ValueError("Replay exhausted before the requested continuation")
        next_token = torch.tensor(
            [tokens[next_index - 1] if stopped else tokens[next_index]],
            device=self.device,
            dtype=torch.long,
        )
        cache.tokens = tokens[:next_index]
        # No hidden state or probability on an off-trajectory suffix is exposed.
        output = SimpleNamespace(
            logits=None,
            hidden_states=None if stopped else self._states(start, next_index),
        )
        self.profiler.count("replay_proposals", 1)
        return {
            "target_output": output,
            "target_probs": None,
            "accept_prefix_mask": None,
            "accept_probs": None,
            "support_accept_rates": None,
            "accepted_draft_tokens": accepted,
            "next_token": next_token,
            "effective_proposal_length": accepted if stopped else count,
            "terminated_by_stop_token": stopped,
            "committed_tokens": torch.cat(
                [proposal.verify_input_ids[:, 1 : accepted + 1], next_token[:, None]],
                dim=1,
            ),
        }

    def audit(self, generate, replayed):
        if self.audit_remaining <= 0:
            return
        self.live_audit = True
        try:
            with self.profiler.measure("replay_live_audit", synchronize=False):
                live = generate()
        finally:
            self.live_audit = False
        if (
            not torch.equal(live.output_ids, replayed.output_ids)
            or live.proposal_lengths != replayed.proposal_lengths
            or live.accepted_draft_lengths != replayed.accepted_draft_lengths
        ):
            raise ValueError(
                "Replay/live block audit diverged; do not use these replay metrics"
            )
        self.audit_remaining -= 1
        self.profiler.count("replay_audits_passed", 1)
        logger.info(
            "Replay/live block audit passed (tokens and every acceptance boundary)"
        )
