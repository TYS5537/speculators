#!/usr/bin/env python3
"""Offline DSpark evaluation on JSONL datasets.

This evaluator intentionally mirrors the training-time DSpark alignment in this
repository.  In particular, DSpark defaults to ``sample_from_anchor=True``:
proposal slot ``k`` predicts the token after base position ``anchor + k``.  When
``sample_from_anchor=False``, slot 0 is the anchor slot and the first real draft
token is slot 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable  # noqa: TC003 -- Runtime type hints.
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from speculators_eval import data as _eval_data
from speculators_eval import parallel as _eval_parallel
from speculators_eval import reporting as _eval_reporting

# Keep the script's existing helper API while the package owns implementation.
_load_jsonl = _eval_data.load_jsonl
_select_eval_records = _eval_data.select_eval_records
_prompt_from_record = _eval_data.prompt_from_record
_discover_datasets = _eval_data.discover_datasets
_dataset_id = _eval_data.dataset_id
_split_csv = _eval_data.split_csv
_shard_records = _eval_data.shard_records
_dataset_output_path = _eval_reporting.dataset_output_path
_aggregate_rows = _eval_reporting.aggregate_rows
_summary_row = _eval_reporting.summary_row
_write_outputs = _eval_reporting.write_outputs
_read_worker_row = _eval_reporting.read_worker_row
_read_worker_artifacts = _eval_reporting.read_worker_artifacts
_target_worker_args = _eval_parallel.target_worker_args
_stop_eval_worker = _eval_parallel.stop_eval_worker
_wait_eval_workers = _eval_parallel.wait_eval_workers
EvalStats = _eval_reporting.EvalStats
PROMPT_FIELDS = _eval_data.PROMPT_FIELDS
DEEPSPEC_EVAL_SAMPLE_LIMITS = _eval_data.DEEPSPEC_EVAL_SAMPLE_LIMITS
RESULT_COLUMNS = _eval_reporting.RESULT_COLUMNS

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

logger = logging.getLogger("dspark_offline_eval")
torch = None
DynamicCache = None
_TOKEN_LOGIT_RANK = 2


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


def _rejection_acceptance_probs(target_probs, draft_probs):
    if not torch.all(torch.isfinite(draft_probs) & (draft_probs > 0)):
        raise ValueError("Proposed tokens must have finite, positive draft probability")
    # Keep the actual q, including probabilities below 1e-8: flooring it
    # changes acceptance without changing the rejection residual p - q.
    return torch.clamp(target_probs / draft_probs, max=1.0)


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


def _compute_draft_acceptance(
    *,
    proposal: DraftProposal,
    target_probs,
    draft_token_count: int,
) -> tuple[Any | None, Any | None, Any | None, int]:
    """Return the full prefix mask, probability diagnostics and accepted count."""
    if draft_token_count <= 0:
        return None, None, None, 0
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
    )
    accept_probs = _rejection_acceptance_probs(
        selected_target_probs,
        selected_draft_probs,
    )
    support_accept_rates = torch.minimum(
        proposal.draft_probs[:, :draft_token_count, :],
        target_probs[:, :draft_token_count, :],
    ).sum(dim=-1)
    # Do not skip greedy or post-EOS positions: RNG advances for the full proposal.
    accept_mask = (torch.rand_like(accept_probs) < accept_probs).to(torch.int64)
    accept_prefix_mask = accept_mask.cumprod(dim=1)
    accepted_draft_tokens = int(accept_prefix_mask.sum(dim=1)[0].item())
    return (
        accept_prefix_mask,
        accept_probs,
        support_accept_rates,
        accepted_draft_tokens,
    )


def _accepted_stop_prefix_length(
    *,
    verify_input_ids,
    accepted_draft_tokens: int,
    stop_token_ids: list[int] | None,
) -> int | None:
    """Find the first stop within accepted draft tokens, excluding the anchor."""
    if not stop_token_ids or accepted_draft_tokens <= 0:
        return None
    accepted_slice = verify_input_ids[0, 1 : accepted_draft_tokens + 1]
    stop_tensor = torch.tensor(
        stop_token_ids,
        device=accepted_slice.device,
        dtype=accepted_slice.dtype,
    )
    eos_hits = torch.isin(accepted_slice, stop_tensor).nonzero(as_tuple=True)[0]
    if eos_hits.numel() > 0:
        return int(eos_hits[0].item()) + 1
    return None


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

    (
        accept_prefix_mask,
        accept_probs,
        support_accept_rates,
        accepted_draft_tokens,
    ) = _compute_draft_acceptance(
        proposal=proposal,
        target_probs=target_probs,
        draft_token_count=draft_token_count,
    )

    effective_proposal_length = draft_token_count
    terminated_by_stop_token = False
    stop_prefix_length = _accepted_stop_prefix_length(
        verify_input_ids=proposal.verify_input_ids,
        accepted_draft_tokens=accepted_draft_tokens,
        stop_token_ids=stop_token_ids,
    )
    if stop_prefix_length is not None:
        accepted_draft_tokens = stop_prefix_length
        effective_proposal_length = accepted_draft_tokens
        terminated_by_stop_token = True

    # Keep probability diagnostics aligned with the EOS-truncated proposal.
    if effective_proposal_length < draft_token_count:
        if accept_probs is not None:
            accept_probs = accept_probs[:, :effective_proposal_length]
        if support_accept_rates is not None:
            support_accept_rates = support_accept_rates[:, :effective_proposal_length]

    # Keep the continuation draw after an accepted stop too. The outer decoder
    # discards that token, but skipping the draw would change later RNG state.
    if draft_token_count > 0 and accepted_draft_tokens < draft_token_count:
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


def _new_target_cache(target_model):
    factory = getattr(target_model, "new_cache", None)
    return factory() if factory is not None else DynamicCache()


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
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens <= 0:
        return SimpleNamespace(
            output_ids=input_ids.clone(),
            num_input_tokens=num_input_tokens,
            num_output_tokens=0,
            proposal_lengths=[],
            accepted_draft_lengths=[],
            accept_prob_lists=[],
            support_accept_rate_lists=[],
        )
    max_length = num_input_tokens + max_new_tokens
    output_ids = torch.empty(
        (1, max_length + max_proposal_tokens + 1),
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    past_key_values_target = _new_target_cache(target_model)

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
    if max_new_tokens == 1 or has_stop_token(initial_token, stop_token_ids):
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

    # The token at start has already been generated. Reserve one remaining slot
    # for the target's replacement/bonus token before verifying any draft tokens.
    while start + 1 < max_length:
        remaining = max_length - start - 1
        if remaining == 1:
            # No draft token can fit alongside the target token in this round.
            proposal = DraftProposal(
                draft_token_count=0,
                verify_input_ids=output_ids[:, start : start + 1],
                draft_probs=None,
            )
        else:
            proposal = propose(
                context=context,
                output_ids=output_ids,
                position_ids=position_ids,
                start=start,
                stop_token_ids=stop_token_ids,
            )
            if proposal.draft_token_count > max_proposal_tokens:
                raise ValueError(
                    "DraftProposal.draft_token_count exceeds max_proposal_tokens"
                )
            if proposal.draft_token_count >= remaining:
                # The drafter may require a full block internally. Only its
                # in-budget prefix is sent to the verifier and counted in stats.
                draft_token_count = remaining - 1
                proposal = DraftProposal(
                    draft_token_count=draft_token_count,
                    verify_input_ids=proposal.verify_input_ids[
                        :, : draft_token_count + 1
                    ],
                    draft_probs=(
                        None
                        if proposal.draft_probs is None
                        else proposal.draft_probs[:, :draft_token_count, :]
                    ),
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
        if start + 1 >= max_length or has_stop_token(new_token_ids, stop_token_ids):
            break
        update(context, verification)

    output_ids = output_ids[:, : start + 1]
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
    past_key_values = _new_target_cache(target_model)

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


def _load_draft_config(model_path):
    """Resolve baseline DSpark and MMuse (including legacy enhanced DSpark)."""
    from speculators.config import SpeculatorModelConfig  # noqa: PLC0415

    config = SpeculatorModelConfig.from_pretrained(model_path)
    if config.speculators_model_type not in ("dspark", "mmuse"):
        raise ValueError(
            "This evaluator supports DSpark and MMuse checkpoints; "
            f"received {config.speculators_model_type!r}."
        )
    return config


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
            draft_model.correction_head is not None
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
        if target_logits.ndim != _TOKEN_LOGIT_RANK:
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
                "DSpark context states must contain exactly the prefix before the "
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
        base_logits = (
            None
            if (
                draft.correction_head is not None
                and getattr(draft, "candidate_selector", None) is None
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
        if draft.correction_head is not None:
            return self._sample_correction_tokens(
                base_logits,
                hidden_states,
                first_prev_token_id,
                initial_previous_logits,
            )

        if base_logits is None:
            raise RuntimeError("Markov/plain DSpark evaluation requires base logits")
        selector_search_mode = getattr(
            getattr(draft, "config", None),
            "dflash2_selector_search_mode",
            "greedy",
        )
        if (
            getattr(draft, "candidate_selector", None) is not None
            and selector_search_mode == "global"
        ):
            if draft.markov_head is not None:
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
        # RNN training also processes the reserved anchor slot when sampling
        # starts at slot 1. Replaying that prefix warms its recurrent state.
        markov_previous_ids = prev_token.expand(-1, self.first_draft_slot)

        for token_idx in range(self.max_proposal_tokens):
            slot = self.first_draft_slot + token_idx
            logits = base_logits[:, slot : slot + 1, :]
            if draft.markov_head is not None:
                if getattr(draft.markov_head, "head_type", None) == "rnn":
                    markov_previous_ids = torch.cat(
                        [markov_previous_ids, prev_token], dim=1
                    )
                    # Blocks are short: recompute the known prefix to retain RNN
                    # state without changing the training head or its weights.
                    markov_bias = draft.markov_head.block_bias(
                        prev_token_ids=markov_previous_ids,
                        hidden_states=hidden_states[:, : slot + 1, :],
                    )[:, -1:, :]
                else:
                    markov_bias = draft.markov_head.block_bias(
                        prev_token_ids=prev_token,
                        hidden_states=hidden_states[:, slot : slot + 1, :],
                    )
                logits = logits + markov_bias
            if getattr(draft, "candidate_selector", None) is not None:
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
        validate_budget = getattr(self.target_model, "validate_request_budget", None)
        if validate_budget is not None and int(self.args.max_new_tokens) > 0:
            validate_budget(
                input_ids.shape[1],
                int(self.args.max_new_tokens),
                self.max_proposal_tokens,
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


def _warmup_dataset(
    *,
    dataset: str,
    path: Path,
    indexed_records: list[tuple[int, dict[str, Any]]],
    runner: DSparkOfflineRunner,
    base_runner: BaseModelOfflineRunner | None,
    args: argparse.Namespace,
    stop_token_ids: list[int] | None,
) -> None:
    """Warm up paired throughput runs, then reset RNG before any measured work."""
    if base_runner is None:
        return
    warmup_samples = int(args.throughput_warmup_samples)
    if warmup_samples < 0:
        raise ValueError("--throughput-warmup-samples must be >= 0")
    warmup_records = indexed_records[:warmup_samples]
    if warmup_records:
        logger.info(
            "[%s] warming up DSpark and base model with %d sample(s)",
            dataset,
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


def _log_dataset_progress(
    *,
    dataset: str,
    processed: int,
    total: int,
    stats: EvalStats,
    elapsed: float,
    paired: bool,
    base_elapsed_s: float,
    base_total_output_tokens: int,
) -> None:
    """Report a caller-timed snapshot without reading the clock or changing stats."""
    out_tps = stats.total_output_tokens / elapsed if elapsed else 0.0
    if not paired:
        logger.info(
            "[%s] %d/%d samples | out_tok=%d | tok/s=%.2f | acc_len=%.3f",
            dataset,
            processed,
            total,
            stats.total_output_tokens,
            out_tps,
            stats.acceptance_length,
        )
    else:
        base_tps = base_total_output_tokens / base_elapsed_s if base_elapsed_s else 0.0
        speedup = out_tps / base_tps if base_tps else 0.0
        logger.info(
            "[%s] %d/%d samples | DSpark=%.2f tok/s | "
            "base=%.2f tok/s | speedup=%.3fx | acc_len=%.3f",
            dataset,
            processed,
            total,
            out_tps,
            base_tps,
            speedup,
            stats.acceptance_length,
        )


def _evaluate_dataset(
    *,
    path: Path,
    runner: DSparkOfflineRunner,
    base_runner: BaseModelOfflineRunner | None,
    args: argparse.Namespace,
    stop_token_ids: list[int] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = _dataset_id(path, args.datasets_root)
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
            dataset,
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

    _warmup_dataset(
        dataset=dataset,
        path=path,
        indexed_records=indexed_records,
        runner=runner,
        base_runner=base_runner,
        args=args,
        stop_token_ids=stop_token_ids,
    )

    # Keep the wall-clock boundary before progress construction. Paired runs
    # instead accumulate only the synchronized generation times below.
    start_time = time.perf_counter()
    iterator = indexed_records
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(
            iterator,
            total=len(indexed_records),
            desc=dataset,
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
            _log_dataset_progress(
                dataset=dataset,
                processed=processed,
                total=len(indexed_records),
                stats=stats,
                elapsed=elapsed,
                paired=base_runner is not None,
                base_elapsed_s=base_elapsed_s,
                base_total_output_tokens=base_total_output_tokens,
            )

    if base_runner is None:
        stats.elapsed_s = time.perf_counter() - start_time
    row = _summary_row(dataset, len(indexed_records), stats)
    if base_runner is not None:
        _eval_reporting.add_base_speedup_metrics(
            row,
            base_elapsed_s=base_elapsed_s,
            base_total_output_tokens=base_total_output_tokens,
        )
    return row, artifacts


def _worker_command(
    args: argparse.Namespace,
    *,
    dataset_path: Path,
    shard_index: int,
    num_shards: int,
    output_dir: Path,
) -> list[str]:
    return _eval_parallel.worker_command(
        args,
        entrypoint=Path(__file__),
        dataset_path=dataset_path,
        shard_index=shard_index,
        num_shards=num_shards,
        output_dir=output_dir,
    )


def run_ascend_data_parallel(args: argparse.Namespace) -> None:
    _eval_parallel.run_ascend_data_parallel(args, entrypoint=Path(__file__))


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


def _validate_target_cache_support(verifier_model: str, target_backend="hf") -> dict:
    from transformers import PretrainedConfig  # noqa: PLC0415

    target_config, _ = PretrainedConfig.get_config_dict(verifier_model)
    is_dsv4 = target_config.get("model_type") == "deepseek_v4"
    if is_dsv4 and target_backend != "dsv4-vllm":
        raise NotImplementedError(
            "DSV4 cannot use this runner's DynamicCache.crop rollback. "
            "Select --target-backend dsv4-vllm with a running DSV4 HS service."
        )
    if target_backend == "dsv4-vllm" and not is_dsv4:
        raise ValueError("The dsv4-vllm backend requires a DSV4 target")
    return target_config


def _write_backend_metadata(args, report):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "dsv4_verification_mode", "reference")
    metadata = {
        "target_backend": "dsv4-vllm",
        "verification_mode": mode,
        "verification": (
            "full-prefix-block-recompute"
            if mode == "block"
            else "full-prefix-per-position-recompute"
        ),
        "probability_source": (
            "target_native_full_vocabulary_logprobs_packet"
            if mode == "block"
            else "target_api_full_vocabulary_logprobs"
        ),
        "online_speedup_benchmark": False,
        "target_model": report["model_path"],
        "checkpoint_signature": report["checkpoint_signature"],
        "hidden_states_path": str(Path(args.hidden_states_path).resolve()),
        "hs_transport": "http" if getattr(args, "hs_http_endpoint", None) else "file",
        "hs_http_endpoint": getattr(args, "hs_http_endpoint", None),
        "max_model_len": args.dsv4_max_model_len,
        "temperature": args.temperature,
        "top_p": 1.0,
        "top_k": "disabled",
        "acceptance_length": "1 + accepted_draft_tokens / proposals",
        "position_accept_rates": "accepted_prefix_count / proposed_count",
    }
    with (args.output_dir / "eval_backend.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)


def run(args: argparse.Namespace) -> None:
    # Close service clients on failed setup, transport errors and interrupts too.
    with ExitStack() as resources:
        _run(args, resources)


def _prepare_hs_http(args, target_backend):
    endpoint = getattr(args, "hs_http_endpoint", None)
    if not endpoint:
        return None
    from speculators_dsv4.hs_http import (  # noqa: PLC0415
        validate_endpoint,
        validate_token,
    )

    if target_backend != "dsv4-vllm":
        raise ValueError("--hs-http-endpoint requires --target-backend dsv4-vllm")
    args.hs_http_endpoint = validate_endpoint(endpoint)
    validate_token(os.environ.get("DSV4_HS_HTTP_TOKEN"))
    if not args.hidden_states_path:
        args.hidden_states_path = args.output_dir / "target-hs-downloads"
    return args.hs_http_endpoint


@dataclass(frozen=True)
class _TargetSetup:
    """Validated target metadata; no loaded model or service client."""

    backend: str
    target_config: dict
    report: dict | None
    hs_http_endpoint: str | None


@dataclass(frozen=True)
class _LoadedEvaluation:
    """References shared by runner construction and evaluation, without ownership."""

    target_model: Any
    draft_model: Any
    tokenizer: Any
    draft_config: Any


def _prepare_target_backend(args: argparse.Namespace) -> _TargetSetup:
    """Validate transport/cache contracts and publish metadata before dispatch."""
    target_backend = getattr(args, "target_backend", "hf")
    hs_http_endpoint = _prepare_hs_http(args, target_backend)
    target_config = _validate_target_cache_support(args.verifier_model, target_backend)
    report = None
    if target_backend == "dsv4-vllm":
        from speculators_dsv4.contract import inspect_checkpoint  # noqa: PLC0415

        if not args.vllm_endpoint or not args.hidden_states_path:
            raise ValueError("DSV4 needs --vllm-endpoint and --hidden-states-path")
        if args.measure_base_speedup:
            raise ValueError(
                "DSV4 recompute evaluates acceptance, not online base speedup"
            )
        if args.dtype != "bfloat16":
            raise ValueError("DSV4 evaluation requires --dtype bfloat16")
        report = inspect_checkpoint(args.verifier_model)
        _write_backend_metadata(args, report)
    return _TargetSetup(target_backend, target_config, report, hs_http_endpoint)


def _run(args: argparse.Namespace, resources: ExitStack) -> None:
    setup = _prepare_target_backend(args)
    if (
        getattr(args, "ascend_devices", None)
        and getattr(args, "worker_shard_index", None) is None
    ):
        run_ascend_data_parallel(args)
        return
    models = _load_evaluation_models(args, setup, resources)
    _log_loaded_draft(models.draft_model, models.draft_config)
    _evaluate_loaded_models(args, models)


def _load_evaluation_models(
    args: argparse.Namespace, setup: _TargetSetup, resources: ExitStack
) -> _LoadedEvaluation:
    """Load models only in a single-device process, after parent dispatch."""
    global torch, DynamicCache  # noqa: PLW0603 -- Existing decoding helpers use these.

    import torch as torch_module  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from transformers import DynamicCache as DynamicCacheClass  # noqa: PLC0415

    from speculators.model import SpeculatorModel  # noqa: PLC0415

    torch = torch_module
    DynamicCache = DynamicCacheClass
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"

    target_model = None
    if setup.backend == "hf":
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

    draft_model, draft_config = _load_evaluation_draft(
        args, setup, target_model, device, model_class=SpeculatorModel
    )
    if setup.backend == "dsv4-vllm":
        target_model, tokenizer = _load_dsv4_target(args, setup, draft_model, resources)
    return _LoadedEvaluation(target_model, draft_model, tokenizer, draft_config)


def _load_evaluation_draft(
    args: argparse.Namespace,
    setup: _TargetSetup,
    target_model,
    device,
    *,
    model_class,
):
    """Bind target identity and overrides before loading/casting draft weights.

    The caller imports the model class before seeding, retaining the original
    import/RNG order. Keep precision resolution after config and vocabulary checks.
    """
    draft_config = _load_draft_config(args.draft_model)
    if setup.backend == "dsv4-vllm":
        from speculators_dsv4.eval_contract import bind_draft_verifier  # noqa: PLC0415

        bind_draft_verifier(draft_config, setup.report)
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
    # Use the loaded target's dtype, including the dtype resolved by HF's "auto".
    # Older supported Transformers versions otherwise load the draft in FP32.
    draft_dtype = torch.bfloat16 if setup.backend == "dsv4-vllm" else target_model.dtype
    draft_model = model_class.from_pretrained(
        args.draft_model,
        config=draft_config,
        d2t=d2t,
        t2d=t2d,
        torch_dtype=draft_dtype,
    )
    # from_pretrained also refreshes borrowed verifier weights before returning.
    draft_model = draft_model.to(device=device, dtype=draft_dtype).eval()
    _ensure_loaded_vocab_mappings(draft_model, args)
    return draft_model, draft_config


def _load_dsv4_target(
    args: argparse.Namespace, setup: _TargetSetup, draft_model, resources: ExitStack
):
    """Register each client immediately; the run's ExitStack owns both lifetimes."""
    from urllib.parse import urlsplit, urlunsplit  # noqa: PLC0415

    import openai  # noqa: PLC0415

    from speculators_dsv4.offline import DSV4OfflineTarget  # noqa: PLC0415
    from speculators_dsv4.tokenizer import DSV4ServerTokenizer  # noqa: PLC0415

    client = openai.OpenAI(
        base_url=args.vllm_endpoint,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        timeout=args.target_request_timeout,
        max_retries=0,
    )
    resources.callback(client.close)
    target_model = DSV4OfflineTarget(
        draft_model,
        setup.report,
        hidden_states_path=args.hidden_states_path,
        client=client,
        model_name=args.served_model_name,
        max_model_len=args.dsv4_max_model_len,
        timeout=args.target_request_timeout,
        keep_hidden_states=args.keep_target_hs,
        verification_mode=args.dsv4_verification_mode,
        hs_http_endpoint=setup.hs_http_endpoint,
        hs_http_token=os.environ.get("DSV4_HS_HTTP_TOKEN"),
    )
    endpoint = urlsplit(args.vllm_endpoint)
    root_path = endpoint.path.rstrip("/").removesuffix("/v1")
    tokenizer_client = openai.OpenAI(
        base_url=urlunsplit((endpoint.scheme, endpoint.netloc, root_path, "", "")),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        timeout=args.target_request_timeout,
        max_retries=0,
    )
    resources.callback(tokenizer_client.close)
    tokenizer = DSV4ServerTokenizer(
        tokenizer_client,
        target_model.model_name,
        target_model.generation_config.eos_token_id,
        vocab_size=setup.target_config["vocab_size"],
    )
    logger.warning(
        "DSV4 %s verification uses full-prefix recomputation and native target "
        "probabilities. Reported elapsed time is NOT online speculative speed.",
        args.dsv4_verification_mode,
    )
    return target_model, tokenizer


def _log_loaded_draft(draft_model, draft_config) -> None:
    """Validate the sequential head and report the loaded architecture."""
    if draft_model.correction_head is not None:
        if not _is_preprojection_correction(draft_model):
            raise RuntimeError(
                "Loaded checkpoint does not use the native causal CorrectionHead"
            )
        sequential_head = (
            f"correction:{draft_config.correction_output_mode}"
            f"+markov:{draft_config.markov_head_type}"
            if draft_model.markov_head is not None
            else f"correction:{draft_config.correction_output_mode}"
        )
    elif draft_model.markov_head is not None:
        sequential_head = f"markov:{draft_config.markov_head_type}"
    else:
        sequential_head = "none"
    logger.info(
        "Loaded %s | block_size=%d sample_from_anchor=%s "
        "max_proposal_tokens=%d sequential_head=%s lm_head_fusion=%s "
        "dflash2_conv=%s dflash2_selector=%s",
        draft_config.speculators_model_type,
        int(draft_model.block_size),
        bool(draft_config.sample_from_anchor),
        speculative_slots_for_draft(draft_model),
        sequential_head,
        bool(getattr(draft_config, "correction_lm_head_fusion", False)),
        bool(getattr(draft_config, "dflash2_dynamic_conv", False)),
        bool(getattr(draft_config, "dflash2_candidate_selector", False)),
    )
    logger.info(
        "Draft implementation: %s",
        sys.modules[type(draft_model).__module__].__file__,
    )


def _evaluate_loaded_models(
    args: argparse.Namespace, models: _LoadedEvaluation
) -> None:
    """Construct runners and execute the unchanged dataset/reporting pipeline."""
    target_model, draft_model, tokenizer = (
        models.target_model,
        models.draft_model,
        models.tokenizer,
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
            artifacts_by_dataset[row["dataset"]] = artifacts
    _write_outputs(args.output_dir, rows, artifacts_by_dataset)
    logger.info("Wrote results to %s", args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline DSpark/MMuse evaluation on JSONL data.",
    )
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-backend", choices=["hf", "dsv4-vllm"], default="hf")
    parser.add_argument("--vllm-endpoint", default=None)
    parser.add_argument("--hidden-states-path", type=Path, default=None)
    parser.add_argument(
        "--hs-http-endpoint",
        default=None,
        help="Optional authenticated HS sidecar URL (not the vLLM /v1 URL). "
        "Use DSV4_HS_HTTP_TOKEN; --hidden-states-path becomes "
        "a local download directory.",
    )
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--dsv4-max-model-len", type=int, default=4096)
    parser.add_argument(
        "--dsv4-verification-mode", choices=["reference", "block"], default="reference"
    )
    parser.add_argument("--target-request-timeout", type=float, default=120.0)
    parser.add_argument("--keep-target-hs", action="store_true")
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
            "DSpark output-throughput speedup."
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
