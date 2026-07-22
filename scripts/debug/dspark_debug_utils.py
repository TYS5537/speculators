#!/usr/bin/env python3
"""Shared helpers for DSpark debug probes."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

log = logging.getLogger("dspark_debug")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_eval_impl(torch):
    path = repo_root() / "scripts" / "evaluate" / "dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("dspark_offline_eval", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.torch = torch
    from transformers import DynamicCache  # noqa: PLC0415

    module.DynamicCache = DynamicCache
    return module


def dtype_of(torch, name: str):
    return "auto" if name == "auto" else getattr(torch, name)


def token_text(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return "n/a"
    return repr(tokenizer.decode([int(token_id)], skip_special_tokens=False))


def load_vocab_maps(torch, args):
    paths = []
    if getattr(args, "d2t_path", None) or getattr(args, "t2d_path", None):
        if not (args.d2t_path and args.t2d_path):
            raise ValueError("--d2t-path and --t2d-path must be passed together.")
        paths.append((args.d2t_path, args.t2d_path, "explicit"))
    draft_model = getattr(args, "draft_model", None)
    if draft_model:
        paths.append(
            (
                Path(draft_model) / "d2t.npy",
                Path(draft_model) / "t2d.npy",
                "draft",
            )
        )
    data_path = getattr(args, "data_path", None)
    if data_path:
        paths.append((Path(data_path) / "d2t.npy", Path(data_path) / "t2d.npy", "data"))
    for d2t_path, t2d_path, source in paths:
        if d2t_path.exists() and t2d_path.exists():
            import numpy as np  # noqa: PLC0415

            log.info("loading vocab maps from %s: %s %s", source, d2t_path, t2d_path)
            return torch.from_numpy(np.load(d2t_path)), torch.from_numpy(
                np.load(t2d_path)
            )
    return None, None


def sample_indices(torch, dataset_len: int, start: int, count: int, randomize: bool):
    available = list(range(start, dataset_len))
    if len(available) <= count:
        return available
    if not randomize:
        return available[:count]
    perm = torch.randperm(len(available))[:count].tolist()
    return [available[i] for i in perm]


def valid_anchor_positions(torch, loss_mask, block_size: int) -> Any:
    valid = loss_mask.bool().clone()
    if valid.numel() <= block_size:
        return torch.empty(0, dtype=torch.long, device=valid.device)
    valid[0] = False
    valid[-block_size:] = False
    return torch.nonzero(valid, as_tuple=False).view(-1).long()


def choose_anchor(torch, loss_mask, block_size: int, requested: int | None) -> int:
    candidates = valid_anchor_positions(torch, loss_mask, block_size)
    if requested is not None:
        if not 0 <= requested < loss_mask.numel() or not bool(
            loss_mask[requested].item()
        ):
            raise ValueError("--anchor-position must point to a loss_mask=1 token")
        if requested == 0:
            raise ValueError("--anchor-position must be > 0 for verifier replay")
        if requested > loss_mask.numel() - block_size - 1:
            raise ValueError("--anchor-position must leave one full block available")
        return int(requested)
    if candidates.numel() == 0:
        raise ValueError("sample has no valid loss_mask anchor leaving a full block")
    picked = torch.randint(candidates.numel(), (1,), device=candidates.device)
    return int(candidates[int(picked.item())].item())


def load_jsonl_record(path: Path, index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no == index:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{index + 1}: expected JSON object")
                return item
    raise IndexError(f"{path} has no record index {index}")


def log_token_window(tokenizer, label: str, ids, max_tokens: int) -> None:
    ids_list = ids.detach().cpu().view(-1).tolist()
    tail = ids_list[-max_tokens:]
    log.info("%s_ids=%s", label, tail)
    log.info("%s_text=%r", label, tokenizer.decode(tail, skip_special_tokens=False))


def map_draft_to_target(eval_impl, draft, draft_id: int) -> int:
    return eval_impl._draft_ids_to_target_ids(draft, [int(draft_id)])[0]


def topk_rows(torch, eval_impl, draft, logits, probs, tokenizer, top_k: int):
    k = min(int(top_k), probs.shape[-1])
    top_probs, top_draft_ids = probs.float().topk(k, dim=-1)
    rows = []
    for rank in range(k):
        draft_id = int(top_draft_ids[0, 0, rank].item())
        target_id = map_draft_to_target(eval_impl, draft, draft_id)
        rows.append(
            SimpleNamespace(
                rank=rank + 1,
                draft_id=draft_id,
                target_id=target_id,
                prob=float(top_probs[0, 0, rank].item()),
                logit=float(logits[0, 0, draft_id].float().item()),
                text=token_text(tokenizer, target_id),
            )
        )
    return rows


def log_sample_from_anchor_contract(eval_impl, draft, anchor: int) -> None:
    log.info(
        (
            "alignment_contract sample_from_anchor=%s block_size=%d "
            "first_draft_slot=%d proposal_tokens=%d"
        ),
        bool(draft.config.sample_from_anchor),
        int(draft.block_size),
        eval_impl.first_draft_slot_for_draft(draft),
        eval_impl.speculative_slots_for_draft(draft),
    )
    for slot in range(int(draft.block_size)):
        if slot < eval_impl.first_draft_slot_for_draft(draft):
            log.info("slot=%d role=anchor_or_masked target_pos=%d", slot, anchor + slot)
            continue
        log.info(
            "slot=%d role=draft predicts_token_at_position=%d",
            slot,
            eval_impl.target_position_for_slot(draft, anchor, slot),
        )
