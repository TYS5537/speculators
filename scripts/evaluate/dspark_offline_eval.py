#!/usr/bin/env python3
"""Offline DSpark/DFlash2 evaluation on JSONL datasets.

This evaluator intentionally mirrors the training-time DSpark alignment in this
repository.  In particular, DSpark defaults to ``sample_from_anchor=True``:
proposal slot ``k`` predicts the token after base position ``anchor + k``.  When
``sample_from_anchor=False``, slot 0 is the anchor slot and the first real draft
token is slot 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

logger = logging.getLogger("dspark_offline_eval")
torch = None
DynamicCache = None

PROMPT_FIELDS = (
    "prompt",
    "input",
    "question",
    "instruction",
    "text",
    "problem",
    "problem_statement",
    "question_content",
)
RESULT_COLUMNS = [
    "dataset",
    "num_requests",
    "elapsed_s",
    "requests_per_second",
    "output_tokens_per_second",
    "total_output_tokens",
    "base_elapsed_s",
    "base_output_tokens_per_second",
    "base_total_output_tokens",
    "speedup_vs_base",
    "num_proposals",
    "num_proposed_draft_tokens",
    "num_accepted_draft_tokens",
    "draft_length",
    "acceptance_length",
    "accepted_draft_length",
    "position_accept_rates",
    "position_accept_prob_means",
    "position_support_accept_rate_means",
    "position_accept_prob_sums",
    "position_support_accept_rate_sums",
    "position_accepted_counts",
    "position_proposed_counts",
]

# Keep the default offline-evaluation workload aligned with DeepSpec-Ascend's
# top-level eval.py. An explicit --max-samples overrides these per-dataset caps.
DEEPSPEC_EVAL_SAMPLE_LIMITS = {
    "gsm8k": 500,
    "math500": 500,
    "aime25": 30,
    "humaneval": 164,
    "mbpp": 256,
    "livecodebench": 500,
    "mt-bench": 80,
    "alpaca": 500,
    "arena-hard-v2": 500,
}


@dataclass
class DraftProposal:
    draft_token_count: int
    verify_input_ids: Any
    draft_probs: Any | None


@dataclass
class VerificationResult:
    target_output: Any
    target_probs: Any
    accept_prefix_mask: Any | None
    accept_probs: Any | None
    support_accept_rates: Any | None
    accepted_draft_tokens: int
    next_token: Any
    effective_proposal_length: int
    terminated_by_stop_token: bool = False
    committed_tokens: Any | None = None


@dataclass
class EvalStats:
    elapsed_s: float = 0.0
    total_output_tokens: int = 0
    num_proposals: int = 0
    num_proposed_draft_tokens: int = 0
    num_accepted_draft_tokens: int = 0
    position_proposed_counts: list[int] = field(default_factory=list)
    position_accepted_counts: list[int] = field(default_factory=list)
    position_accept_prob_sums: list[float] = field(default_factory=list)
    position_support_accept_rate_sums: list[float] = field(default_factory=list)

    @property
    def acceptance_length(self) -> float:
        if self.num_proposals == 0:
            return 1.0
        return 1.0 + self.num_accepted_draft_tokens / self.num_proposals

    @property
    def draft_length(self) -> float:
        if self.num_proposals == 0:
            return 0.0
        return self.num_proposed_draft_tokens / self.num_proposals

    @property
    def accepted_draft_length(self) -> float:
        if self.num_proposals == 0:
            return 0.0
        return self.num_accepted_draft_tokens / self.num_proposals

    @property
    def position_accept_rates(self) -> list[float]:
        return [
            accepted / proposed if proposed else 0.0
            for accepted, proposed in zip(
                self.position_accepted_counts,
                self.position_proposed_counts,
                strict=True,
            )
        ]

    @property
    def position_accept_prob_means(self) -> list[float]:
        return [
            value / proposed if proposed else 0.0
            for value, proposed in zip(
                self.position_accept_prob_sums,
                self.position_proposed_counts,
                strict=True,
            )
        ]

    @property
    def position_support_accept_rate_means(self) -> list[float]:
        return [
            value / proposed if proposed else 0.0
            for value, proposed in zip(
                self.position_support_accept_rate_sums,
                self.position_proposed_counts,
                strict=True,
            )
        ]

    def add_response(self, response: SimpleNamespace) -> None:
        self.total_output_tokens += int(response.num_output_tokens)
        proposal_lengths = getattr(response, "proposal_lengths", [])
        accepted_lengths = getattr(response, "accepted_draft_lengths", [])
        accept_prob_lists = getattr(response, "accept_prob_lists", [])
        support_accept_rate_lists = getattr(response, "support_accept_rate_lists", [])
        self.num_proposals += len(proposal_lengths)
        self.num_proposed_draft_tokens += sum(int(x) for x in proposal_lengths)
        self.num_accepted_draft_tokens += sum(int(x) for x in accepted_lengths)
        for proposal_len, accepted_len in zip(
            proposal_lengths,
            accepted_lengths,
            strict=True,
        ):
            self.add_proposal_positions(int(proposal_len), int(accepted_len))
        for proposal_len, accept_probs, support_accept_rates in zip(
            proposal_lengths,
            accept_prob_lists,
            support_accept_rate_lists,
            strict=True,
        ):
            self.add_proposal_probability_stats(
                int(proposal_len),
                accept_probs,
                support_accept_rates,
            )

    def add_proposal_positions(self, proposal_len: int, accepted_len: int) -> None:
        if accepted_len > proposal_len:
            raise ValueError(
                f"accepted_len must not exceed proposal_len: {accepted_len}"
            )
        missing = proposal_len - len(self.position_proposed_counts)
        if missing > 0:
            self.position_proposed_counts.extend([0] * missing)
            self.position_accepted_counts.extend([0] * missing)
        for pos in range(proposal_len):
            self.position_proposed_counts[pos] += 1
            if pos < accepted_len:
                self.position_accepted_counts[pos] += 1

    def add_proposal_probability_stats(
        self,
        proposal_len: int,
        accept_probs: list[float],
        support_accept_rates: list[float] | None,
    ) -> None:
        if len(accept_probs) != proposal_len:
            raise ValueError("accept_probs length does not match proposal_len")
        if (
            support_accept_rates is not None
            and len(support_accept_rates) != proposal_len
        ):
            raise ValueError("support_accept_rates length does not match proposal_len")

        missing = proposal_len - len(self.position_accept_prob_sums)
        if missing > 0:
            self.position_accept_prob_sums.extend([0.0] * missing)
            self.position_support_accept_rate_sums.extend([0.0] * missing)
        for pos in range(proposal_len):
            self.position_accept_prob_sums[pos] += float(accept_probs[pos])
            if support_accept_rates is not None:
                self.position_support_accept_rate_sums[pos] += float(
                    support_accept_rates[pos]
                )


def _parse_count_list(value: Any) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value) if value else []
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _parse_float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value) if value else []
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]


def logits_to_probs(logits, temperature: float):
    if temperature <= 0:
        return torch.nn.functional.one_hot(
            torch.argmax(logits, dim=-1),
            num_classes=logits.shape[-1],
        ).to(logits.dtype)
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample_from_probs(probs):
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1)
    return sampled.reshape(*probs.shape[:-1])


def sample_from_logits(logits, temperature: float):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    return sample_from_probs(logits_to_probs(logits, temperature))


def gather_token_probs(probs, token_ids):
    return torch.gather(probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def sample_residual(target_probs, draft_probs):
    residual = (target_probs - draft_probs).clamp_min(0)
    denom = residual.sum(dim=-1, keepdim=True)
    residual = torch.where(denom > 0, residual / denom.clamp_min(1e-8), target_probs)
    return sample_from_probs(residual)


def has_stop_token(token_ids, stop_token_ids: list[int] | None) -> bool:
    if stop_token_ids is None:
        return False
    stop_tensor = torch.tensor(stop_token_ids, device=token_ids.device)
    return bool(torch.isin(token_ids, stop_tensor).any().item())


def trim_output_ids(
    output_ids,
    num_input_tokens: int,
    stop_token_ids: list[int] | None,
):
    if stop_token_ids is None:
        return output_ids
    stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
    stop_indices = torch.isin(output_ids[0][num_input_tokens:], stop_tensor).nonzero(
        as_tuple=True,
    )[0]
    if stop_indices.numel() == 0:
        return output_ids
    return output_ids[:, : num_input_tokens + int(stop_indices[0].item()) + 1]


def resolve_stop_token_ids(target_model, tokenizer) -> list[int] | None:
    generation_config = getattr(target_model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]
    return list(dict.fromkeys(int(token_id) for token_id in eos_token_id))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(item)
    return records


def _canonical_dataset_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    aliases = {
        "aime-25": "aime25",
        "human-eval": "humaneval",
        "live-code-bench": "livecodebench",
        "math-500": "math500",
    }
    return aliases.get(normalized, normalized)


def _select_eval_records(
    records: list[dict[str, Any]],
    *,
    dataset_name: str,
    max_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    limit = (
        int(max_samples)
        if max_samples is not None
        else DEEPSPEC_EVAL_SAMPLE_LIMITS.get(_canonical_dataset_name(dataset_name))
    )
    if limit is None or len(records) <= limit:
        return records
    if limit < 0:
        raise ValueError("--max-samples must be >= 0")

    selected = list(records)
    random.Random(int(seed)).shuffle(selected)
    return selected[:limit]


def _string_turns(value: Any) -> list[str] | None:
    if isinstance(value, str) and value.strip():
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        turns = [item for item in value if item.strip()]
        return turns or None
    return None


def _messages_from_conversations(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    messages: list[dict[str, str]] = []
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }
    for item in value:
        if not isinstance(item, dict):
            return None
        raw_role = item.get("from", item.get("role"))
        raw_content = item.get("value", item.get("content"))
        if not isinstance(raw_role, str) or not isinstance(raw_content, str):
            return None
        role = role_map.get(raw_role)
        content = raw_content.strip()
        if role is None or not content:
            return None
        if role == "assistant":
            break
        messages.append({"role": role, "content": content})
    return messages or None


def _chat_template_kwargs(args: argparse.Namespace | None) -> dict[str, Any]:
    if args is None:
        return {}
    enable_thinking = getattr(args, "enable_thinking", "false")
    if enable_thinking == "default":
        return {}
    return {"enable_thinking": enable_thinking == "true"}


def _looks_like_chatml(text: str) -> bool:
    return "<|im_start|>" in text or "<|im_end|>" in text


def _format_raw_prompt(
    prompt: str,
    tokenizer,
    *,
    args: argparse.Namespace | None,
) -> str:
    mode = getattr(args, "raw_prompt_mode", "auto") if args is not None else "auto"
    if mode == "raw" or (mode == "auto" and _looks_like_chatml(prompt)):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **_chat_template_kwargs(args),
    )


def _prompt_from_record(
    record: dict[str, Any],
    tokenizer,
    *,
    source: str,
    args: argparse.Namespace | None = None,
) -> str:
    turns = _string_turns(record.get("turns"))
    if turns is not None:
        # Match DeepSpec's DSpark evaluation protocol: rows with `turns` contain
        # user turns, and acceptance eval uses only the first turn.
        return _format_raw_prompt(turns[0], tokenizer, args=args)

    messages = record.get("messages")
    if isinstance(messages, list):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **_chat_template_kwargs(args),
        )

    messages = _messages_from_conversations(record.get("conversations"))
    if messages is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **_chat_template_kwargs(args),
        )

    for field in PROMPT_FIELDS:
        turns = _string_turns(record.get(field))
        if turns is not None:
            return _format_raw_prompt("\n\n".join(turns), tokenizer, args=args)

    keys = ", ".join(sorted(record.keys()))
    supported = ", ".join(["turns", "messages", "conversations", *PROMPT_FIELDS])
    raise ValueError(
        f"{source}: record has no supported prompt field ({supported}); keys=[{keys}]"
    )


def _discover_datasets(root: Path, names: list[str] | None) -> list[Path]:
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    if names:
        wanted = set(names)
        paths = [
            path
            for path in paths
            if path.stem in wanted or path.name in wanted or str(path) in wanted
        ]
    if not paths:
        raise FileNotFoundError(f"No JSONL datasets found under {root}")
    return paths


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _shard_records(
    records: list[dict[str, Any]],
    *,
    shard_index: int | None,
    num_shards: int,
) -> list[tuple[int, dict[str, Any]]]:
    indexed_records = list(enumerate(records, start=1))
    if shard_index is None or num_shards <= 1:
        return indexed_records
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards})")
    return [
        item
        for zero_based_index, item in enumerate(indexed_records)
        if zero_based_index % num_shards == shard_index
    ]


def _draft_sample_from_anchor(draft) -> bool:
    return bool(getattr(getattr(draft, "config", None), "sample_from_anchor", False))


def _is_preprojection_correction(draft) -> bool:
    """Return whether ``draft`` exposes the native causal Correction interface."""
    correction = getattr(draft, "correction_head", None)
    return correction is not None and hasattr(correction, "position_embedding")


def _run_preprojection_correction_rollout(
    draft,
    *,
    hidden_states,
    anchor_token_ids,
    temperature: float,
    initial_previous_logits=None,
    base_logits=None,
):
    """Run native Correction rollout, including optional previous-logit feedback."""
    if not _is_preprojection_correction(draft):
        raise RuntimeError("This evaluator expects the native causal CorrectionHead")
    rollout_kwargs = {
        "anchor_token_ids": anchor_token_ids,
        "temperature": temperature,
    }
    if initial_previous_logits is not None:
        rollout_kwargs["initial_previous_logits"] = initial_previous_logits
    reuse_base_logits = bool(
        getattr(getattr(draft, "config", None), "correction_lm_head_fusion", False)
    )
    if (
        base_logits is not None
        or getattr(draft, "candidate_selector", None) is not None
        or reuse_base_logits
    ):
        rollout_kwargs["base_logits"] = (
            base_logits
            if base_logits is not None
            else draft.lm_head(hidden_states.to(draft.lm_head.weight.dtype))
        )
    return draft.rollout_correction(hidden_states, **rollout_kwargs)


def _prepare_dflash_target_context(draft, hidden_states):
    """Mirror the training/validation target-layer preparation exactly."""
    return draft._fuse_target_hidden(hidden_states)


def speculative_slots_for_draft(draft) -> int:
    block = int(draft.block_size)
    if _draft_sample_from_anchor(draft):
        return block
    if block <= 1:
        raise ValueError(
            "sample_from_anchor=False requires block_size >= 2 for offline "
            "speculative evaluation"
        )
    return block - 1


def first_draft_slot_for_draft(draft) -> int:
    return 0 if _draft_sample_from_anchor(draft) else 1


def target_position_for_slot(draft, anchor: int, slot: int) -> int:
    if _draft_sample_from_anchor(draft):
        return int(anchor) + int(slot) + 1
    return int(anchor) + int(slot)


def _draft_ids_to_target_ids(draft, draft_ids: list[int]) -> list[int]:
    if draft.use_draft_vocab and draft.d2t is not None:
        d2t = draft.d2t
        return [int(token_id + d2t[token_id].item()) for token_id in draft_ids]
    return [int(token_id) for token_id in draft_ids]


def _load_vocab_mapping_tensors(
    *,
    draft_model_path: str,
    d2t_path: Path | None,
    t2d_path: Path | None,
):
    if d2t_path is None and t2d_path is None:
        draft_path = Path(draft_model_path)
        d2t_path = draft_path / "d2t.npy"
        t2d_path = draft_path / "t2d.npy"
        if not d2t_path.exists() and not t2d_path.exists():
            return None, None
    elif d2t_path is None or t2d_path is None:
        raise ValueError("--d2t-path and --t2d-path must be provided together.")

    if d2t_path is None or t2d_path is None:
        return None, None
    if not d2t_path.exists():
        raise FileNotFoundError(f"d2t mapping file not found: {d2t_path}")
    if not t2d_path.exists():
        raise FileNotFoundError(f"t2d mapping file not found: {t2d_path}")

    import numpy as np  # noqa: PLC0415

    logger.info("Loading vocab mappings: d2t=%s t2d=%s", d2t_path, t2d_path)
    return torch.from_numpy(np.load(d2t_path)), torch.from_numpy(np.load(t2d_path))


def _ensure_loaded_vocab_mappings(draft_model, args: argparse.Namespace) -> None:
    if not draft_model.use_draft_vocab:
        return
    if draft_model.t2d is not None and int(
        draft_model.t2d.sum(dtype=torch.long).item()
    ) == int(draft_model.draft_vocab_size):
        return
    d2t, t2d = _load_vocab_mapping_tensors(
        draft_model_path=args.draft_model,
        d2t_path=args.d2t_path,
        t2d_path=args.t2d_path,
    )
    if d2t is None or t2d is None:
        raise ValueError(
            "DSpark draft uses a pruned draft vocab, but no real d2t/t2d mapping "
            "was loaded. Pass --d2t-path and --t2d-path, or place d2t.npy and "
            "t2d.npy under --draft-model."
        )
    draft_model.load_vocab_mappings(t2d, d2t)


def verify_draft_tokens(
    *,
    target_model,
    proposal: DraftProposal,
    position_ids,
    start: int,
    past_key_values_target,
    temperature: float,
    max_proposal_tokens: int,
    current_token_ids=None,
    stop_token_ids: list[int] | None = None,
) -> VerificationResult:
    if proposal.draft_token_count > max_proposal_tokens:
        raise ValueError("DraftProposal.draft_token_count exceeds max_proposal_tokens")
    if current_token_ids is not None and not torch.equal(
        proposal.verify_input_ids[:, :1],
        current_token_ids,
    ):
        raise ValueError(
            "DraftProposal.verify_input_ids must start with current token."
        )

    draft_token_count = int(proposal.draft_token_count)
    verify_length = draft_token_count + 1
    target_output = target_model(
        input_ids=proposal.verify_input_ids,
        position_ids=position_ids[:, start : start + verify_length],
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
    )
    target_probs = logits_to_probs(target_output.logits, float(temperature))

    accept_prefix_mask = None
    accept_probs = None
    support_accept_rates = None
    if draft_token_count > 0:
        if proposal.draft_probs is None:
            raise ValueError("draft_probs is required when draft_token_count > 0")
        proposed_tokens = proposal.verify_input_ids[:, 1:]
        selected_target_probs = gather_token_probs(
            target_probs[:, :-1, :],
            proposed_tokens,
        )
        selected_draft_probs = gather_token_probs(
            proposal.draft_probs,
            proposed_tokens,
        ).clamp_min(1e-8)
        accept_probs = torch.clamp(
            selected_target_probs / selected_draft_probs,
            max=1.0,
        )
        support_accept_rates = torch.minimum(
            proposal.draft_probs[:, :draft_token_count, :],
            target_probs[:, :draft_token_count, :],
        ).sum(dim=-1)
        accept_mask = (torch.rand_like(accept_probs) < accept_probs).to(torch.int64)
        accept_prefix_mask = accept_mask.cumprod(dim=1)
        accepted_draft_tokens = int(accept_prefix_mask.sum(dim=1)[0].item())
    else:
        accepted_draft_tokens = 0

    effective_proposal_length = draft_token_count
    terminated_by_stop_token = False
    if stop_token_ids and accepted_draft_tokens > 0:
        accepted_slice = proposal.verify_input_ids[0, 1 : accepted_draft_tokens + 1]
        stop_tensor = torch.tensor(
            stop_token_ids,
            device=accepted_slice.device,
            dtype=accepted_slice.dtype,
        )
        eos_hits = torch.isin(accepted_slice, stop_tensor).nonzero(as_tuple=True)[0]
        if eos_hits.numel() > 0:
            accepted_draft_tokens = int(eos_hits[0].item()) + 1
            effective_proposal_length = accepted_draft_tokens
            terminated_by_stop_token = True

    # A verifier call still scores the whole proposal block, but positions after
    # the first accepted stop token are not part of the effective proposal.  Keep
    # the probability diagnostics aligned with ``effective_proposal_length`` so
    # per-position aggregation uses the same EOS-truncated denominator as the
    # accepted/proposed token counters.
    if effective_proposal_length < draft_token_count:
        if accept_probs is not None:
            accept_probs = accept_probs[:, :effective_proposal_length]
        if support_accept_rates is not None:
            support_accept_rates = support_accept_rates[
                :, :effective_proposal_length
            ]

    if 0 < draft_token_count and accepted_draft_tokens < draft_token_count:
        next_token = sample_residual(
            target_probs[:, accepted_draft_tokens, :],
            proposal.draft_probs[:, accepted_draft_tokens, :],
        )
    else:
        next_token = sample_from_probs(target_probs[:, -1:, :]).squeeze(1)

    committed_tokens = torch.cat(
        [
            proposal.verify_input_ids[:, 1 : accepted_draft_tokens + 1],
            next_token.unsqueeze(1),
        ],
        dim=1,
    )
    return VerificationResult(
        target_output=target_output,
        target_probs=target_probs,
        accept_prefix_mask=accept_prefix_mask,
        accept_probs=accept_probs,
        support_accept_rates=support_accept_rates,
        accepted_draft_tokens=accepted_draft_tokens,
        next_token=next_token,
        effective_proposal_length=effective_proposal_length,
        terminated_by_stop_token=terminated_by_stop_token,
        committed_tokens=committed_tokens,
    )


def generate_decoding_sample(
    *,
    target_model,
    input_ids,
    max_new_tokens: int,
    max_proposal_tokens: int,
    temperature: float,
    stop_token_ids: list[int] | None,
    init_context: Callable[..., Any],
    propose: Callable[..., DraftProposal],
    update: Callable[[Any, VerificationResult], None],
) -> SimpleNamespace:
    if max_proposal_tokens < 1:
        raise ValueError("max_proposal_tokens must be >= 1")
    device = input_ids.device
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + int(max_new_tokens)
    output_ids = torch.empty(
        (1, max_length + max_proposal_tokens + 1),
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    past_key_values_target = DynamicCache()

    output = target_model(
        input_ids=input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample_from_probs(
        logits_to_probs(output.logits[:, -1:, :], float(temperature))
    )
    start = num_input_tokens
    proposal_lengths: list[int] = []
    accepted_draft_lengths: list[int] = []
    accept_prob_lists: list[list[float]] = []
    support_accept_rate_lists: list[list[float]] = []

    initial_token = output_ids[:, num_input_tokens : num_input_tokens + 1]
    if has_stop_token(initial_token, stop_token_ids):
        output_ids = trim_output_ids(
            output_ids[:, : num_input_tokens + 1],
            num_input_tokens,
            stop_token_ids,
        )
        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=output_ids.shape[1] - num_input_tokens,
            proposal_lengths=proposal_lengths,
            accepted_draft_lengths=accepted_draft_lengths,
            accept_prob_lists=accept_prob_lists,
            support_accept_rate_lists=support_accept_rate_lists,
        )

    context = init_context(initial_output=output, initial_token=initial_token)
    del output

    while start < max_length:
        proposal = propose(
            context=context,
            output_ids=output_ids,
            position_ids=position_ids,
            start=start,
            stop_token_ids=stop_token_ids,
        )
        verification = verify_draft_tokens(
            target_model=target_model,
            proposal=proposal,
            position_ids=position_ids,
            start=start,
            past_key_values_target=past_key_values_target,
            temperature=temperature,
            max_proposal_tokens=max_proposal_tokens,
            current_token_ids=output_ids[:, start : start + 1],
            stop_token_ids=stop_token_ids,
        )

        proposal_lengths.append(int(verification.effective_proposal_length))
        accepted = int(verification.accepted_draft_tokens)
        accepted_draft_lengths.append(accepted)
        accept_prob_lists.append(
            []
            if verification.accept_probs is None
            else verification.accept_probs.detach().float()[0].tolist()
        )
        support_accept_rate_lists.append(
            []
            if verification.support_accept_rates is None
            else verification.support_accept_rates.detach().float()[0].tolist()
        )
        output_ids[:, start : start + accepted + 1] = proposal.verify_input_ids[
            :, : accepted + 1
        ]
        if verification.terminated_by_stop_token:
            start += accepted
            past_key_values_target.crop(start)
            break

        output_ids[:, start + accepted + 1] = verification.next_token
        new_token_ids = output_ids[:, start + 1 : start + accepted + 2]
        start += accepted + 1
        past_key_values_target.crop(start)
        update(context, verification)
        if has_stop_token(new_token_ids, stop_token_ids):
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=output_ids.shape[1] - num_input_tokens,
        proposal_lengths=proposal_lengths,
        accepted_draft_lengths=accepted_draft_lengths,
        accept_prob_lists=accept_prob_lists,
        support_accept_rate_lists=support_accept_rate_lists,
    )


def generate_base_model_sample(
    *,
    target_model,
    input_ids,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: list[int] | None,
) -> SimpleNamespace:
    """Generate autoregressively with only the verifier and its KV cache."""
    num_input_tokens = input_ids.shape[1]
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens <= 0:
        return SimpleNamespace(
            output_ids=input_ids.clone(),
            num_input_tokens=num_input_tokens,
            num_output_tokens=0,
        )

    device = input_ids.device
    max_length = num_input_tokens + max_new_tokens
    output_ids = torch.empty(
        (1, max_length),
        dtype=torch.long,
        device=device,
    )
    output_ids[:, :num_input_tokens] = input_ids
    position_ids = torch.arange(max_length, device=device).unsqueeze(0)
    past_key_values = DynamicCache()

    output = target_model(
        input_ids=input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values,
        use_cache=True,
    )
    next_token = sample_from_logits(
        output.logits[:, -1:, :],
        float(temperature),
    )
    end = num_input_tokens

    while end < max_length:
        output_ids[:, end : end + 1] = next_token
        end += 1
        if has_stop_token(next_token, stop_token_ids) or end >= max_length:
            break
        output = target_model(
            input_ids=next_token,
            position_ids=position_ids[:, end - 1 : end],
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = sample_from_logits(
            output.logits[:, -1:, :],
            float(temperature),
        )

    output_ids = output_ids[:, :end]
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=end - num_input_tokens,
    )


class DSparkOfflineRunner:
    def __init__(self, target_model, draft_model, tokenizer, args) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.args = args
        self.device = next(target_model.parameters()).device
        self.sample_from_anchor = _draft_sample_from_anchor(draft_model)
        self.first_draft_slot = first_draft_slot_for_draft(draft_model)
        self.max_proposal_tokens = speculative_slots_for_draft(draft_model)
        correction_output_mode = getattr(
            getattr(draft_model, "correction_head", None),
            "output_mode",
            "hidden",
        )
        self.uses_initial_correction_logits = bool(
            getattr(draft_model, "correction_head", None) is not None
            and not self.sample_from_anchor
            and correction_output_mode == "logits"
        )
        self._draft_target_logit_indices = None
        if self.uses_initial_correction_logits and draft_model.use_draft_vocab:
            if draft_model.d2t is None:
                raise RuntimeError("Draft-to-target vocabulary mapping is not loaded")
            draft_ids = torch.arange(
                draft_model.draft_vocab_size,
                device=draft_model.d2t.device,
                dtype=draft_model.d2t.dtype,
            )
            self._draft_target_logit_indices = (draft_ids + draft_model.d2t).long()
    def _extract_context_feature(self, hidden_states):
        return torch.cat(
            [hidden_states[i] for i in self.draft_model.target_layer_ids],
            dim=-1,
        )

    def _target_logits_to_draft_vocab(self, target_logits):
        """Select the draft-vocabulary logits without another LM-head call."""
        draft = self.draft_model
        if target_logits.ndim != 2:
            raise ValueError("Target logits for the current anchor must be rank-2")
        if not draft.use_draft_vocab:
            if target_logits.shape[-1] != draft.draft_vocab_size:
                raise ValueError("Target and draft vocabulary sizes do not align")
            return target_logits
        if draft.d2t is None:
            raise RuntimeError("Draft-to-target vocabulary mapping is not loaded")
        target_ids = getattr(self, "_draft_target_logit_indices", None)
        if target_ids is None:
            draft_ids = torch.arange(
                draft.draft_vocab_size,
                device=target_logits.device,
                dtype=draft.d2t.dtype,
            )
            target_ids = (draft_ids + draft.d2t.to(target_logits.device)).long()
            self._draft_target_logit_indices = target_ids
        elif target_ids.device != target_logits.device:
            target_ids = target_ids.to(target_logits.device)
            self._draft_target_logit_indices = target_ids
        return target_logits.index_select(-1, target_ids)

    def _init_context(
        self,
        *,
        initial_output,
        **_kwargs,
    ) -> SimpleNamespace:
        correction_previous_logits = None
        if self.uses_initial_correction_logits:
            correction_previous_logits = self._target_logits_to_draft_vocab(
                initial_output.logits[:, -1, :]
            )
        return SimpleNamespace(
            target_hidden_states=self._extract_context_feature(
                initial_output.hidden_states,
            ),
            correction_previous_logits=correction_previous_logits,
        )

    def _single_anchor_backbone(
        self,
        hidden_states,
        input_ids,
        start: int,
    ):
        draft = self.draft_model
        block = int(draft.block_size)
        if hidden_states.shape[1] != start:
            raise ValueError(
                "Draft context states must contain exactly the prefix before the "
                f"current anchor; got length {hidden_states.shape[1]} and "
                f"start={start}."
            )
        hidden_states = torch.cat(
            [hidden_states, hidden_states.new_zeros(hidden_states[:, :1, :].shape)],
            dim=1,
        )
        total_seq_len = hidden_states.shape[1]
        current_ids = input_ids[:, :total_seq_len]
        anchor_positions = torch.tensor([start], dtype=torch.long, device=self.device)
        document_ids = torch.zeros_like(current_ids)

        full_attn_mask = None
        if draft.uses_full_attn:
            full_attn_mask = draft._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=self.device,
                sliding_window=None,
            )

        sliding_window_attn_mask = None
        if draft.uses_sliding_window_attn:
            sliding_window_attn_mask = draft._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=self.device,
                sliding_window=draft.sliding_window,
                sliding_window_non_causal=draft.sliding_window_non_causal,
            )

        mask_token_ids = torch.full(
            (1, block),
            draft.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        mask_token_ids[:, 0] = input_ids[:, start]
        noise_embedding = draft.embed_tokens(mask_token_ids)
        fc_output = _prepare_dflash_target_context(draft, hidden_states)
        noise_embedding = draft._condition_noise_embedding(
            noise_embedding,
            fc_output,
            anchor_positions,
            document_ids,
        )
        base_position_ids = torch.arange(
            total_seq_len,
            dtype=torch.long,
            device=self.device,
        )
        block_offsets = torch.arange(block, dtype=torch.long, device=self.device)
        position_ids = torch.cat(
            [base_position_ids, base_position_ids[start] + block_offsets],
            dim=0,
        ).unsqueeze(0)
        position_embeddings = draft.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(draft.layers):
            attention_mask = (
                sliding_window_attn_mask
                if layer_idx in draft.sliding_window_indices
                else full_attn_mask
            )
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
            )

        hidden = draft.norm(noise_embedding)
        # Selector-conditioned Correction needs pure DFlash logits before the
        # correction pass. LM-head fusion also needs one parallel base projection
        # so sequential Correction can add only low-rank vocabulary residuals.
        # With both features disabled, retain the historical per-position path.
        reuse_base_logits = bool(
            getattr(draft.config, "correction_lm_head_fusion", False)
        )
        correction_head = getattr(draft, "correction_head", None)
        candidate_selector = getattr(draft, "candidate_selector", None)
        base_logits = (
            None
            if (
                correction_head is not None
                and candidate_selector is None
                and not reuse_base_logits
            )
            else draft.lm_head(hidden)
        )
        return hidden, base_logits

    def _sample_correction_tokens(
        self,
        base_logits,
        hidden_states,
        first_prev_token_id,
        initial_previous_logits,
    ):
        """Sample with native causal Correction rollout."""
        draft = self.draft_model
        temperature = float(self.args.temperature)
        draft_ids, final_logits = _run_preprojection_correction_rollout(
            draft,
            hidden_states=hidden_states,
            anchor_token_ids=first_prev_token_id.reshape(-1).long(),
            temperature=temperature,
            initial_previous_logits=initial_previous_logits,
            base_logits=base_logits,
        )

        # The model returns all block slots.  With sample_from_anchor=False,
        # slot 0 is reserved and must not be proposed to the verifier.
        first_slot = self.first_draft_slot
        last_slot = first_slot + self.max_proposal_tokens
        draft_ids = draft_ids[:, first_slot:last_slot]
        final_logits = final_logits[:, first_slot:last_slot]

        if draft_ids.shape[1] != self.max_proposal_tokens:
            raise RuntimeError(
                "Correction rollout returned "
                f"{draft_ids.shape[1]} tokens, expected {self.max_proposal_tokens}"
            )

        proposed_target_ids = _draft_ids_to_target_ids(
            draft,
            [int(token_id) for token_id in draft_ids[0].tolist()],
        )
        return proposed_target_ids, logits_to_probs(final_logits, temperature)

    def _sample_dspark_tokens(
        self,
        base_logits,
        hidden_states,
        first_prev_token_id,
        initial_previous_logits=None,
    ):
        draft = self.draft_model
        correction_head = getattr(draft, "correction_head", None)
        candidate_selector = getattr(draft, "candidate_selector", None)
        markov_head = getattr(draft, "markov_head", None)
        if correction_head is not None:
            return self._sample_correction_tokens(
                base_logits,
                hidden_states,
                first_prev_token_id,
                initial_previous_logits,
            )

        if base_logits is None:
            raise RuntimeError(
                "Sequential/plain draft evaluation requires base logits"
            )
        selector_search_mode = getattr(
            getattr(draft, "config", None),
            "dflash2_selector_search_mode",
            "greedy",
        )
        if candidate_selector is not None and selector_search_mode == "global":
            if markov_head is not None:
                raise RuntimeError(
                    "Global DFlash2 path search is not compatible with a standalone "
                    "predecessor-dependent Markov head; use Correction collaboration"
                )
            _, _, selected_ids = draft.dflash2_select_path(
                base_logits,
                hidden_states,
                first_prev_token_id.reshape(-1).long(),
            )
            first_slot = self.first_draft_slot
            last_slot = first_slot + self.max_proposal_tokens
            selected_ids = selected_ids[:, first_slot:last_slot]
            if selected_ids.shape[1] != self.max_proposal_tokens:
                raise RuntimeError(
                    "Global DFlash2 selector returned the wrong proposal length"
                )
            # Viterbi is a deterministic block-level proposal. Returning its exact
            # one-hot q keeps speculative rejection sampling lossless even when the
            # evaluator's target temperature is non-zero.
            draft_probs = torch.zeros_like(
                base_logits[:, first_slot:last_slot], dtype=torch.float32
            )
            draft_probs.scatter_(-1, selected_ids.unsqueeze(-1), 1.0)
            proposed_target_ids = _draft_ids_to_target_ids(
                draft, [int(token_id) for token_id in selected_ids[0].tolist()]
            )
            return proposed_target_ids, draft_probs
        proposed_target_ids: list[int] = []
        draft_probs = []
        prev_token = first_prev_token_id.reshape(1, 1).long()

        for token_idx in range(self.max_proposal_tokens):
            slot = self.first_draft_slot + token_idx
            logits = base_logits[:, slot : slot + 1, :]
            if markov_head is not None:
                logits = logits + markov_head.block_bias(
                    prev_token_ids=prev_token,
                    hidden_states=hidden_states[:, slot : slot + 1, :],
                )
            if candidate_selector is not None:
                candidate_ids, candidate_logits = draft.dflash2_select_candidates(
                    logits,
                    hidden_states[:, slot : slot + 1, :],
                    prev_token,
                )
                candidate_probs = logits_to_probs(
                    candidate_logits, float(self.args.temperature)
                )
                selected = sample_from_probs(candidate_probs)
                draft_id = int(
                    candidate_ids.gather(-1, selected.unsqueeze(-1))[0, 0, 0].item()
                )
                probs = torch.zeros_like(logits, dtype=candidate_probs.dtype)
                probs.scatter_(-1, candidate_ids, candidate_probs)
            else:
                probs = logits_to_probs(logits, float(self.args.temperature))
                draft_id = int(sample_from_probs(probs)[0, 0].item())
            target_id = _draft_ids_to_target_ids(draft, [draft_id])[0]
            proposed_target_ids.append(target_id)
            draft_probs.append(probs)
            prev_token = torch.tensor(
                [[target_id]],
                dtype=torch.long,
                device=self.device,
            )

        return proposed_target_ids, torch.cat(draft_probs, dim=1)

    def _expand_draft_probs_to_target_vocab(self, draft_probs):
        draft = self.draft_model
        if not draft.use_draft_vocab or draft.d2t is None:
            return draft_probs
        if draft.t2d is not None:
            target_vocab_size = int(draft.t2d.shape[0])
        else:
            target_vocab_size = int(draft.verifier_vocab_size)
        expanded = draft_probs.new_zeros(*draft_probs.shape[:-1], target_vocab_size)
        draft_ids = torch.arange(
            draft_probs.shape[-1],
            device=draft_probs.device,
            dtype=draft.d2t.dtype,
        )
        target_ids = (draft_ids + draft.d2t.to(draft_probs.device)).long()
        expanded.index_copy_(-1, target_ids, draft_probs)
        return expanded

    def _propose(
        self,
        *,
        context: SimpleNamespace,
        output_ids,
        position_ids,
        start: int,
        stop_token_ids: list[int] | None = None,
    ) -> DraftProposal:
        del position_ids, stop_token_ids
        hidden, base_logits = self._single_anchor_backbone(
            context.target_hidden_states,
            output_ids,
            start,
        )
        proposed_target_ids, draft_probs = self._sample_dspark_tokens(
            base_logits,
            hidden,
            output_ids[:, start],
            context.correction_previous_logits,
        )
        verify_input_ids = torch.cat(
            [
                output_ids[:, start : start + 1],
                torch.tensor(
                    [proposed_target_ids],
                    dtype=torch.long,
                    device=self.device,
                ),
            ],
            dim=1,
        )
        return DraftProposal(
            draft_token_count=len(proposed_target_ids),
            verify_input_ids=verify_input_ids,
            draft_probs=self._expand_draft_probs_to_target_vocab(draft_probs),
        )

    def _update(
        self,
        context: SimpleNamespace,
        verification: VerificationResult,
    ) -> None:
        hidden = self._extract_context_feature(verification.target_output.hidden_states)
        committed_hidden = hidden[:, : verification.accepted_draft_tokens + 1, :]
        context.target_hidden_states = torch.cat(
            [context.target_hidden_states, committed_hidden],
            dim=1,
        )
        if self.uses_initial_correction_logits:
            context.correction_previous_logits = self._target_logits_to_draft_vocab(
                verification.target_output.logits[
                    :, verification.accepted_draft_tokens, :
                ]
            )

    def generate_one(self, prompt: str, stop_token_ids: list[int] | None):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        with torch.inference_mode():
            return generate_decoding_sample(
                target_model=self.target_model,
                input_ids=input_ids,
                max_new_tokens=int(self.args.max_new_tokens),
                max_proposal_tokens=self.max_proposal_tokens,
                temperature=float(self.args.temperature),
                stop_token_ids=stop_token_ids,
                init_context=self._init_context,
                propose=self._propose,
                update=self._update,
            )


class BaseModelOfflineRunner:
    def __init__(self, target_model, tokenizer, args) -> None:
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.args = args
        self.device = next(target_model.parameters()).device

    def generate_one(self, prompt: str, stop_token_ids: list[int] | None):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        with torch.inference_mode():
            return generate_base_model_sample(
                target_model=self.target_model,
                input_ids=input_ids,
                max_new_tokens=int(self.args.max_new_tokens),
                temperature=float(self.args.temperature),
                stop_token_ids=stop_token_ids,
            )


def _synchronize_device(device) -> None:
    backend = getattr(torch, device.type, None)
    synchronize = getattr(backend, "synchronize", None)
    if synchronize is not None:
        synchronize(device)


def _timed_generate(runner, prompt: str, stop_token_ids: list[int] | None):
    _synchronize_device(runner.device)
    start_time = time.perf_counter()
    response = runner.generate_one(prompt, stop_token_ids)
    _synchronize_device(runner.device)
    return response, time.perf_counter() - start_time


def _aggregate_rows(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    position_proposed_counts: list[int] = []
    position_accepted_counts: list[int] = []
    position_accept_prob_sums: list[float] = []
    position_support_accept_rate_sums: list[float] = []
    for row in rows:
        proposed = _parse_count_list(row.get("position_proposed_counts", []))
        accepted = _parse_count_list(row.get("position_accepted_counts", []))
        accept_prob_sums = _parse_float_list(row.get("position_accept_prob_sums", []))
        support_sums = _parse_float_list(
            row.get("position_support_accept_rate_sums", [])
        )
        size = max(len(position_proposed_counts), len(proposed))
        if len(position_proposed_counts) < size:
            position_proposed_counts.extend(
                [0] * (size - len(position_proposed_counts))
            )
            position_accepted_counts.extend(
                [0] * (size - len(position_accepted_counts))
            )
            position_accept_prob_sums.extend(
                [0.0] * (size - len(position_accept_prob_sums))
            )
            position_support_accept_rate_sums.extend(
                [0.0] * (size - len(position_support_accept_rate_sums))
            )
        for idx, count in enumerate(proposed):
            position_proposed_counts[idx] += count
        for idx, count in enumerate(accepted):
            position_accepted_counts[idx] += count
        for idx, value in enumerate(accept_prob_sums):
            position_accept_prob_sums[idx] += value
        for idx, value in enumerate(support_sums):
            position_support_accept_rate_sums[idx] += value

    stats = EvalStats(
        elapsed_s=max((float(row["elapsed_s"]) for row in rows), default=0.0),
        total_output_tokens=sum(int(row["total_output_tokens"]) for row in rows),
        num_proposals=sum(int(row["num_proposals"]) for row in rows),
        num_proposed_draft_tokens=sum(
            int(row["num_proposed_draft_tokens"]) for row in rows
        ),
        num_accepted_draft_tokens=sum(
            int(row["num_accepted_draft_tokens"]) for row in rows
        ),
        position_proposed_counts=position_proposed_counts,
        position_accepted_counts=position_accepted_counts,
        position_accept_prob_sums=position_accept_prob_sums,
        position_support_accept_rate_sums=position_support_accept_rate_sums,
    )
    num_requests = sum(int(row["num_requests"]) for row in rows)
    summary = _summary_row(dataset, num_requests, stats)
    base_elapsed_s = max(
        (float(row.get("base_elapsed_s", 0.0)) for row in rows),
        default=0.0,
    )
    if base_elapsed_s:
        base_total_output_tokens = sum(
            int(row.get("base_total_output_tokens", 0)) for row in rows
        )
        base_tps = base_total_output_tokens / base_elapsed_s
        summary.update(
            {
                "base_elapsed_s": base_elapsed_s,
                "base_output_tokens_per_second": base_tps,
                "base_total_output_tokens": base_total_output_tokens,
                "speedup_vs_base": (
                    summary["output_tokens_per_second"] / base_tps if base_tps else 0.0
                ),
            }
        )
    return summary


def _summary_row(dataset: str, num_requests: int, stats: EvalStats) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "num_requests": num_requests,
        "elapsed_s": stats.elapsed_s,
        "requests_per_second": num_requests / stats.elapsed_s if stats.elapsed_s else 0,
        "output_tokens_per_second": (
            stats.total_output_tokens / stats.elapsed_s if stats.elapsed_s else 0
        ),
        "total_output_tokens": stats.total_output_tokens,
        "base_elapsed_s": 0.0,
        "base_output_tokens_per_second": 0.0,
        "base_total_output_tokens": 0,
        "speedup_vs_base": 0.0,
        "num_proposals": stats.num_proposals,
        "num_proposed_draft_tokens": stats.num_proposed_draft_tokens,
        "num_accepted_draft_tokens": stats.num_accepted_draft_tokens,
        "draft_length": stats.draft_length,
        "acceptance_length": stats.acceptance_length,
        "accepted_draft_length": stats.accepted_draft_length,
        "position_accept_rates": json.dumps(stats.position_accept_rates),
        "position_accept_prob_means": json.dumps(stats.position_accept_prob_means),
        "position_support_accept_rate_means": json.dumps(
            stats.position_support_accept_rate_means
        ),
        "position_accept_prob_sums": json.dumps(stats.position_accept_prob_sums),
        "position_support_accept_rate_sums": json.dumps(
            stats.position_support_accept_rate_sums
        ),
        "position_accepted_counts": json.dumps(stats.position_accepted_counts),
        "position_proposed_counts": json.dumps(stats.position_proposed_counts),
    }


def _evaluate_dataset(
    *,
    path: Path,
    runner: DSparkOfflineRunner,
    base_runner: BaseModelOfflineRunner | None,
    args: argparse.Namespace,
    stop_token_ids: list[int] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = _load_jsonl(path)
    total_records = len(records)
    records = _select_eval_records(
        records,
        dataset_name=path.stem,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if len(records) != total_records:
        logger.info(
            "[%s] selected %d/%d samples with seed=%d",
            path.stem,
            len(records),
            total_records,
            int(args.seed),
        )
    indexed_records = _shard_records(
        records,
        shard_index=getattr(args, "worker_shard_index", None),
        num_shards=getattr(args, "worker_num_shards", 1),
    )

    stats = EvalStats()
    artifacts: list[dict[str, Any]] = []
    base_elapsed_s = 0.0
    base_total_output_tokens = 0

    if base_runner is not None:
        warmup_samples = int(args.throughput_warmup_samples)
        if warmup_samples < 0:
            raise ValueError("--throughput-warmup-samples must be >= 0")
        warmup_records = indexed_records[:warmup_samples]
        if warmup_records:
            logger.info(
                "[%s] warming up DSpark and base model with %d sample(s)",
                path.stem,
                len(warmup_records),
            )
        for idx, record in warmup_records:
            prompt = _prompt_from_record(
                record,
                runner.tokenizer,
                source=f"{path}:{idx}",
                args=args,
            )
            runner.generate_one(prompt, stop_token_ids)
            base_runner.generate_one(prompt, stop_token_ids)
        _synchronize_device(runner.device)
        torch.manual_seed(args.seed)

    start_time = time.perf_counter()
    iterator = indexed_records
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(
            iterator,
            total=len(indexed_records),
            desc=path.stem,
            unit="sample",
        )

    for processed, (idx, record) in enumerate(iterator, start=1):
        prompt = _prompt_from_record(
            record,
            runner.tokenizer,
            source=f"{path}:{idx}",
            args=args,
        )
        if base_runner is None:
            response = runner.generate_one(prompt, stop_token_ids)
        else:
            response, dspark_elapsed = _timed_generate(
                runner,
                prompt,
                stop_token_ids,
            )
            base_response, base_elapsed = _timed_generate(
                base_runner,
                prompt,
                stop_token_ids,
            )
            stats.elapsed_s += dspark_elapsed
            base_elapsed_s += base_elapsed
            base_total_output_tokens += int(base_response.num_output_tokens)
        stats.add_response(response)
        if not args.skip_artifacts:
            artifacts.append(
                {
                    "prompt": prompt,
                    "output_token_ids": response.output_ids[0].tolist(),
                    "num_input_tokens": int(response.num_input_tokens),
                    "source_index": idx,
                }
            )
        if (
            processed == 1
            or processed % args.log_every == 0
            or processed == len(indexed_records)
        ):
            elapsed = (
                stats.elapsed_s
                if base_runner is not None
                else time.perf_counter() - start_time
            )
            out_tps = stats.total_output_tokens / elapsed if elapsed else 0.0
            if base_runner is None:
                logger.info(
                    "[%s] %d/%d samples | out_tok=%d | tok/s=%.2f | acc_len=%.3f",
                    path.stem,
                    processed,
                    len(indexed_records),
                    stats.total_output_tokens,
                    out_tps,
                    stats.acceptance_length,
                )
            else:
                base_tps = (
                    base_total_output_tokens / base_elapsed_s if base_elapsed_s else 0.0
                )
                speedup = out_tps / base_tps if base_tps else 0.0
                logger.info(
                    "[%s] %d/%d samples | DSpark=%.2f tok/s | "
                    "base=%.2f tok/s | speedup=%.3fx | acc_len=%.3f",
                    path.stem,
                    processed,
                    len(indexed_records),
                    out_tps,
                    base_tps,
                    speedup,
                    stats.acceptance_length,
                )

    if base_runner is None:
        stats.elapsed_s = time.perf_counter() - start_time
    row = _summary_row(path.stem, len(indexed_records), stats)
    if base_runner is not None:
        base_tps = base_total_output_tokens / base_elapsed_s if base_elapsed_s else 0.0
        row.update(
            {
                "base_elapsed_s": base_elapsed_s,
                "base_output_tokens_per_second": base_tps,
                "base_total_output_tokens": base_total_output_tokens,
                "speedup_vs_base": (
                    row["output_tokens_per_second"] / base_tps if base_tps else 0.0
                ),
            }
        )
    return row, artifacts


def _write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    artifacts_by_dataset: dict[str, list[dict[str, Any]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    if not artifacts_by_dataset:
        return
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for dataset, artifacts in artifacts_by_dataset.items():
        with (artifacts_dir / f"{dataset}.jsonl").open("w", encoding="utf-8") as f:
            for artifact in artifacts:
                f.write(json.dumps(artifact) + "\n")


def _read_worker_row(output_dir: Path) -> dict[str, Any]:
    with (output_dir / "summary.json").open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"{output_dir}/summary.json must contain one result row")
    return rows[0]


def _read_worker_artifacts(output_dir: Path, dataset: str) -> list[dict[str, Any]]:
    path = output_dir / "artifacts" / f"{dataset}.jsonl"
    if not path.exists():
        return []
    return _load_jsonl(path)


def _worker_command(
    args: argparse.Namespace,
    *,
    dataset_path: Path,
    shard_index: int,
    num_shards: int,
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--verifier-model",
        args.verifier_model,
        "--draft-model",
        args.draft_model,
        "--datasets-root",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--enable-thinking",
        args.enable_thinking,
        "--raw-prompt-mode",
        args.raw_prompt_mode,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--draft-attn-impl",
        args.draft_attn_impl,
        "--log-every",
        str(args.log_every),
        "--worker-shard-index",
        str(shard_index),
        "--worker-num-shards",
        str(num_shards),
        "--no-progress",
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.d2t_path is not None:
        cmd.extend(["--d2t-path", str(args.d2t_path)])
    if args.t2d_path is not None:
        cmd.extend(["--t2d-path", str(args.t2d_path)])
    if args.skip_artifacts:
        cmd.append("--skip-artifacts")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.sample_from_anchor is not None:
        cmd.extend(["--sample-from-anchor", str(args.sample_from_anchor).lower()])
    if args.measure_base_speedup:
        cmd.extend(
            [
                "--measure-base-speedup",
                "--throughput-warmup-samples",
                str(args.throughput_warmup_samples),
            ]
        )
    return cmd


def run_ascend_data_parallel(args: argparse.Namespace) -> None:
    devices = _split_csv(args.ascend_devices)
    if not devices:
        raise ValueError("--ascend-devices must contain at least one device id")
    dataset_paths = _discover_datasets(
        args.datasets_root,
        _split_csv(args.datasets) or None,
    )
    rows: list[dict[str, Any]] = []
    artifacts_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset_path in dataset_paths:
        dataset_start = time.perf_counter()
        shard_root = args.output_dir / "_shards" / dataset_path.stem
        processes = []
        for shard_index, visible_device in enumerate(devices):
            shard_output_dir = shard_root / f"shard_{shard_index}"
            cmd = _worker_command(
                args,
                dataset_path=dataset_path,
                shard_index=shard_index,
                num_shards=len(devices),
                output_dir=shard_output_dir,
            )
            env = os.environ.copy()
            env["ASCEND_RT_VISIBLE_DEVICES"] = visible_device
            processes.append(
                (shard_index, shard_output_dir, subprocess.Popen(cmd, env=env))
            )
        failed = []
        for shard_index, _, process in processes:
            returncode = process.wait()
            if returncode != 0:
                failed.append((shard_index, returncode))
        if failed:
            raise RuntimeError(f"{dataset_path.stem} worker failures: {failed}")
        shard_rows = [
            _read_worker_row(shard_output_dir) for _, shard_output_dir, _ in processes
        ]
        row = _aggregate_rows(dataset_path.stem, shard_rows)
        if not args.measure_base_speedup:
            row["elapsed_s"] = time.perf_counter() - dataset_start
            row["requests_per_second"] = (
                row["num_requests"] / row["elapsed_s"] if row["elapsed_s"] else 0
            )
            row["output_tokens_per_second"] = (
                row["total_output_tokens"] / row["elapsed_s"] if row["elapsed_s"] else 0
            )
        rows.append(row)
        if not args.skip_artifacts:
            artifacts = []
            for _, shard_output_dir, _ in processes:
                artifacts.extend(
                    _read_worker_artifacts(shard_output_dir, dataset_path.stem)
                )
            artifacts.sort(key=lambda item: int(item.get("source_index", 0)))
            artifacts_by_dataset[dataset_path.stem] = artifacts
        _write_outputs(args.output_dir, rows, artifacts_by_dataset)


def _resolve_draft_attn_impl(device: str, draft_attn_impl: str) -> str | None:
    if draft_attn_impl != "auto":
        return draft_attn_impl
    if str(device).startswith("npu"):
        return "sdpa"
    return None


def _parse_bool_override(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected boolean value, got {value}")


def run(args: argparse.Namespace) -> None:
    global torch, DynamicCache
    if (
        getattr(args, "ascend_devices", None)
        and getattr(args, "worker_shard_index", None) is None
    ):
        run_ascend_data_parallel(args)
        return

    import torch as torch_module  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from transformers import DynamicCache as DynamicCacheClass  # noqa: PLC0415

    from speculators.models.dflash2.core import DFlash2DraftModel  # noqa: PLC0415
    from speculators.models.dspark.core import DSparkDraftModel  # noqa: PLC0415

    torch = torch_module
    DynamicCache = DynamicCacheClass
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model,
        trust_remote_code=args.trust_remote_code,
    )
    target_model = (
        AutoModelForCausalLM.from_pretrained(
            args.verifier_model,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        .to(device)
        .eval()
    )

    draft_config_dict, _ = DSparkDraftModel.config_class.get_config_dict(
        args.draft_model
    )
    draft_type = str(draft_config_dict.get("speculators_model_type", "dspark"))
    if draft_type == "dflash2":
        draft_model_class = DFlash2DraftModel
    elif draft_type == "dspark":
        draft_model_class = DSparkDraftModel
    else:
        raise ValueError(
            "This offline evaluator supports DSpark and standalone DFlash2 "
            f"checkpoints, got speculators_model_type={draft_type!r}."
        )

    draft_config = draft_model_class.config_class.from_pretrained(args.draft_model)
    sample_from_anchor = _parse_bool_override(args.sample_from_anchor)
    if sample_from_anchor is not None:
        draft_config.sample_from_anchor = sample_from_anchor
    draft_attn_impl = _resolve_draft_attn_impl(args.device, args.draft_attn_impl)
    if draft_attn_impl is not None:
        draft_config.transformer_layer_config._attn_implementation = draft_attn_impl
    d2t, t2d = _load_vocab_mapping_tensors(
        draft_model_path=args.draft_model,
        d2t_path=args.d2t_path,
        t2d_path=args.t2d_path,
    )
    draft_model = draft_model_class.from_pretrained(
        args.draft_model,
        config=draft_config,
        d2t=d2t,
        t2d=t2d,
    )
    draft_model = draft_model.to(device).eval()
    _ensure_loaded_vocab_mappings(draft_model, args)
    correction_head = getattr(draft_model, "correction_head", None)
    markov_head = getattr(draft_model, "markov_head", None)
    candidate_selector = getattr(draft_model, "candidate_selector", None)
    if correction_head is not None:
        if not _is_preprojection_correction(draft_model):
            raise RuntimeError(
                "Loaded checkpoint does not use the native causal CorrectionHead"
            )
        sequential_head = (
            f"correction:{draft_config.correction_output_mode}"
            f"+markov:{draft_config.markov_head_type}"
            if markov_head is not None
            else f"correction:{draft_config.correction_output_mode}"
        )
    elif markov_head is not None:
        sequential_head = f"markov:{draft_config.markov_head_type}"
    else:
        sequential_head = "none"
    logger.info(
        "Loaded %s | block_size=%d sample_from_anchor=%s "
        "max_proposal_tokens=%d sequential_head=%s lm_head_fusion=%s "
        "dflash2_conv=%s dflash2_selector=%s",
        draft_type,
        int(draft_model.block_size),
        bool(draft_config.sample_from_anchor),
        speculative_slots_for_draft(draft_model),
        sequential_head,
        bool(getattr(draft_config, "correction_lm_head_fusion", False)),
        bool(
            draft_type == "dflash2"
            or getattr(draft_config, "dflash2_dynamic_conv", False)
        ),
        bool(
            candidate_selector is not None
            or getattr(draft_config, "dflash2_candidate_selector", False)
        ),
    )
    logger.info(
        "%s implementation: %s",
        draft_type,
        sys.modules[draft_model_class.__module__].__file__,
    )

    runner = DSparkOfflineRunner(target_model, draft_model, tokenizer, args)
    base_runner = (
        BaseModelOfflineRunner(target_model, tokenizer, args)
        if args.measure_base_speedup
        else None
    )
    if base_runner is not None and float(args.temperature) > 0:
        logger.warning(
            "Base speedup at temperature > 0 compares sampled output throughput; "
            "use --temperature 0 for identical greedy output paths."
        )
    stop_token_ids = resolve_stop_token_ids(target_model, tokenizer)
    dataset_paths = _discover_datasets(
        args.datasets_root,
        _split_csv(args.datasets) or None,
    )
    rows: list[dict[str, Any]] = []
    artifacts_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for path in dataset_paths:
        row, artifacts = _evaluate_dataset(
            path=path,
            runner=runner,
            base_runner=base_runner,
            args=args,
            stop_token_ids=stop_token_ids,
        )
        rows.append(row)
        if not args.skip_artifacts:
            artifacts_by_dataset[path.stem] = artifacts
    _write_outputs(args.output_dir, rows, artifacts_by_dataset)
    logger.info("Wrote results to %s", args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline DSpark/speculators evaluation on JSONL data.",
    )
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("dspark_offline_eval"))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Override the DeepSpec per-dataset sample cap. When unset, known "
            "DeepSpec datasets use the limits from DeepSpec-Ascend/eval.py."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--enable-thinking",
        choices=["false", "true", "default"],
        default="false",
    )
    parser.add_argument(
        "--raw-prompt-mode",
        choices=["auto", "chat_template", "raw"],
        default="auto",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--draft-attn-impl",
        choices=["auto", "simple_flex_attention", "sdpa", "eager"],
        default="auto",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--measure-base-speedup",
        action="store_true",
        help=(
            "Also run verifier-only autoregressive decoding and report measured "
            "draft-model output-throughput speedup."
        ),
    )
    parser.add_argument(
        "--throughput-warmup-samples",
        type=int,
        default=1,
        help="Warmup prompts per dataset before paired throughput timing.",
    )
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--d2t-path", type=Path, default=None)
    parser.add_argument("--t2d-path", type=Path, default=None)
    parser.add_argument(
        "--sample-from-anchor",
        choices=["true", "false"],
        default=None,
        help="Override checkpoint config. Leave unset to use checkpoint value.",
    )
    parser.add_argument("--ascend-devices", default=None)
    parser.add_argument("--worker-shard-index", type=int, default=None)
    parser.add_argument("--worker-num-shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
