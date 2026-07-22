#!/usr/bin/env python3
"""Probe one random DSpark anchor from a preprocessed Arrow training dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

from dspark_debug_utils import (
    choose_anchor,
    dtype_of,
    load_eval_impl,
    load_vocab_maps,
    log_sample_from_anchor_contract,
    log_token_window,
    map_draft_to_target,
    sample_indices,
    token_text,
    topk_rows,
    valid_anchor_positions,
)

log = logging.getLogger("arrow_anchor_probe")


def _load_hidden_state_sample(torch, data, sample_index: int, hidden_states_path: Path):
    from safetensors.torch import load_file

    row = data[int(sample_index)]
    input_ids = torch.as_tensor(row["input_ids"], dtype=torch.long)
    loss_mask = torch.as_tensor(row["loss_mask"], dtype=torch.bool)
    hs_file = hidden_states_path / f"hs_{sample_index}.safetensors"
    if not hs_file.exists():
        raise FileNotFoundError(
            f"Hidden state file not found: {hs_file}. Pass --hidden-states-path or "
            "run data generation/cache first."
        )
    loaded = load_file(hs_file)
    token_ids = loaded["token_ids"].long()
    if not torch.equal(token_ids.cpu(), input_ids.cpu()):
        raise ValueError(
            f"{hs_file} token_ids do not match Arrow input_ids for sample "
            f"{sample_index}"
        )
    raw_hidden = loaded["hidden_states"]
    return SimpleNamespace(
        row=row,
        input_ids=token_ids,
        loss_mask=loss_mask,
        hidden_states=raw_hidden[:, :-1].flatten(1),
        verifier_last_hidden_states=raw_hidden[:, -1],
        raw_hidden_shape=tuple(raw_hidden.shape),
        hidden_file=hs_file,
    )


def _apply_markov(torch, draft, hidden, base_logits, input_ids, anchor: int):
    block = int(draft.block_size)
    hidden_blocks = hidden.view(1, block, -1)
    base = base_logits.view(1, block, -1)
    block_tokens = input_ids[:, anchor : anchor + block]
    if draft.config.sample_from_anchor:
        prev_token_ids = block_tokens
    else:
        prev_token_ids = torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)
    if draft.markov_head is None:
        bias = torch.zeros_like(base)
    else:
        bias = draft.markov_head.block_bias(
            prev_token_ids=prev_token_ids,
            hidden_states=hidden_blocks,
        )
    return base, bias, base + bias, prev_token_ids


def _training_anchor_forward(torch, draft, sample, anchor: int, device):
    seq_len = int(sample.input_ids.numel())
    input_ids = sample.input_ids.to(device).unsqueeze(0)
    loss_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    # Force select_anchors(max_anchors=1) to use exactly this anchor.  The real
    # sample loss_mask is printed per slot below; do not feed the full block mask
    # here or the backbone may randomly select anchor+1, anchor+2, ...
    loss_mask[0, anchor] = True
    document_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        hidden, base_logits, targets, aligned_mask, anchored_idx = (
            draft._backbone_forward(
                sample.hidden_states.to(device).unsqueeze(0),
                input_ids,
                loss_mask,
                sample.verifier_last_hidden_states.to(device).unsqueeze(0),
                document_ids,
                position_ids,
                max_anchors=1,
            )
        )
    base, bias, final, prev_token_ids = _apply_markov(
        torch,
        draft,
        hidden,
        base_logits,
        input_ids,
        anchor,
    )
    return SimpleNamespace(
        hidden=hidden,
        base=base,
        bias=bias,
        final=final,
        targets=targets.view(1, int(draft.block_size), -1),
        mask=aligned_mask.view(1, int(draft.block_size)),
        real_loss_mask=sample.loss_mask[anchor : anchor + int(draft.block_size)],
        anchored_idx=anchored_idx.view(1, int(draft.block_size)),
        prev_token_ids=prev_token_ids,
    )


def _print_training_slots(
    torch,
    eval_impl,
    tokenizer,
    draft,
    replay,
    sample,
    anchor,
    top_k,
):
    first_slot = eval_impl.first_draft_slot_for_draft(draft)
    for slot in range(int(draft.block_size)):
        target_pos = eval_impl.target_position_for_slot(draft, anchor, slot)
        target_draft_id = int(torch.argmax(replay.targets[0, slot]).item())
        target_id = map_draft_to_target(eval_impl, draft, target_draft_id)
        pred_draft_id = int(torch.argmax(replay.final[0, slot]).item())
        pred_id = map_draft_to_target(eval_impl, draft, pred_draft_id)
        gt_id = (
            int(sample.input_ids[target_pos].item())
            if 0 <= target_pos < sample.input_ids.numel()
            else None
        )
        target_real_mask = (
            bool(sample.loss_mask[target_pos].item())
            if 0 <= target_pos < sample.loss_mask.numel()
            else False
        )
        log.info(
            (
                "train_slot=%d active=%s anchor_block_index=%d prev_id=%d "
                "target_pos=%s gt_id=%s gt_text=%s target_top1=%d target_text=%s "
                "pred_top1=%d pred_text=%s forced_anchor_mask=%.1f "
                "block_input_loss_mask=%s target_token_loss_mask=%s"
            ),
            slot,
            slot >= first_slot,
            int(replay.anchored_idx[0, slot].item()),
            int(replay.prev_token_ids[0, slot].item()),
            target_pos,
            gt_id,
            token_text(tokenizer, gt_id) if gt_id is not None else "n/a",
            target_id,
            token_text(tokenizer, target_id),
            pred_id,
            token_text(tokenizer, pred_id),
            float(replay.mask[0, slot].item()),
            bool(replay.real_loss_mask[slot].item()),
            target_real_mask,
        )
        if slot >= first_slot:
            probs = eval_impl.logits_to_probs(
                replay.final[:, slot : slot + 1, :],
                0.0,
            )
            for row in topk_rows(
                torch,
                eval_impl,
                draft,
                replay.final[:, slot : slot + 1, :],
                probs,
                tokenizer,
                top_k,
            ):
                log.info(
                    "  train_slot=%d rank=%d draft_id=%d target_id=%d prob=%.6g "
                    "logit=%.6g text=%s",
                    slot,
                    row.rank,
                    row.draft_id,
                    row.target_id,
                    row.prob,
                    row.logit,
                    row.text,
                )


def _print_offline_proposal(torch, eval_impl, tokenizer, runner, sample, anchor, top_k):
    device = runner.device
    input_ids = sample.input_ids.to(device).unsqueeze(0)
    prefix = input_ids[:, :anchor]
    position_ids = torch.arange(
        input_ids.shape[1] + runner.max_proposal_tokens + 2,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    past = eval_impl.DynamicCache()
    with torch.inference_mode():
        out = runner.target_model(
            input_ids=prefix,
            position_ids=position_ids[:, :anchor],
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
        )
        context = runner._init_context(initial_output=out)
        proposal = runner._propose(
            context=context,
            output_ids=input_ids,
            position_ids=position_ids,
            start=anchor,
        )
        verification = eval_impl.verify_draft_tokens(
            target_model=runner.target_model,
            proposal=proposal,
            position_ids=position_ids,
            start=anchor,
            past_key_values_target=past,
            temperature=0.0,
            max_proposal_tokens=runner.max_proposal_tokens,
            current_token_ids=input_ids[:, anchor : anchor + 1],
        )

    log.info(
        "offline_proposal verify_input_ids=%s accepted=%d/%d next_token=%d %s",
        proposal.verify_input_ids[0].detach().cpu().tolist(),
        int(verification.accepted_draft_tokens),
        int(proposal.draft_token_count),
        int(verification.next_token[0].item()),
        token_text(tokenizer, int(verification.next_token[0].item())),
    )
    if verification.accept_probs is not None:
        log.info(
            "offline_accept_probs=%s",
            verification.accept_probs.float()[0].tolist(),
        )
        log.info(
            "offline_support_accept_rates=%s",
            verification.support_accept_rates.float()[0].tolist(),
        )

    first_slot = eval_impl.first_draft_slot_for_draft(runner.draft_model)
    hidden, base_logits = runner._single_anchor_backbone(
        context.target_hidden_states,
        input_ids,
        anchor,
    )
    prev = input_ids[:, anchor]
    for token_idx in range(runner.max_proposal_tokens):
        slot = first_slot + token_idx
        logits = base_logits[:, slot : slot + 1, :]
        if runner.draft_model.markov_head is not None:
            logits = logits + runner.draft_model.markov_head.block_bias(
                prev_token_ids=prev.reshape(1, 1),
                hidden_states=hidden[:, slot : slot + 1, :],
            )
        probs = eval_impl.logits_to_probs(logits, 0.0)
        proposed_id = int(proposal.verify_input_ids[0, token_idx + 1].item())
        target_pos = eval_impl.target_position_for_slot(
            runner.draft_model,
            anchor,
            slot,
        )
        gt_id = int(sample.input_ids[target_pos].item())
        log.info(
            "offline_slot=%d token_idx=%d target_pos=%d gt_id=%d %s proposed_id=%d %s",
            slot,
            token_idx,
            target_pos,
            gt_id,
            token_text(tokenizer, gt_id),
            proposed_id,
            token_text(tokenizer, proposed_id),
        )
        for row in topk_rows(
            torch,
            eval_impl,
            runner.draft_model,
            logits,
            probs,
            tokenizer,
            top_k,
        ):
            log.info(
                "  offline_slot=%d rank=%d draft_id=%d target_id=%d prob=%.6g "
                "logit=%.6g text=%s",
                slot,
                row.rank,
                row.draft_id,
                row.target_id,
                row.prob,
                row.logit,
                row.text,
            )
        prev = torch.tensor([[proposed_id]], dtype=torch.long, device=device)


def run(args):
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from speculators.models.dspark.core import DSparkDraftModel

    torch.manual_seed(args.seed)
    eval_impl = load_eval_impl(torch)
    device = torch.device(args.device)
    data = load_from_disk(args.data_path)
    indices = sample_indices(
        torch,
        len(data),
        args.sample_start,
        args.num_samples,
        args.random_samples,
    )
    sample_index = int(indices[0])
    hidden_states_path = args.hidden_states_path or (
        Path(args.data_path) / "hidden_states"
    )
    sample = _load_hidden_state_sample(torch, data, sample_index, hidden_states_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model,
        trust_remote_code=args.trust_remote_code,
    )

    cfg = DSparkDraftModel.config_class.from_pretrained(args.draft_model)
    if args.sample_from_anchor is not None:
        cfg.sample_from_anchor = args.sample_from_anchor == "true"
    if args.draft_attn_impl != "auto":
        cfg.transformer_layer_config._attn_implementation = args.draft_attn_impl
    d2t, t2d = load_vocab_maps(torch, args)
    draft = DSparkDraftModel.from_pretrained(
        args.draft_model,
        config=cfg,
        d2t=d2t,
        t2d=t2d,
    ).to(device).eval()
    target = AutoModelForCausalLM.from_pretrained(
        args.verifier_model,
        torch_dtype=dtype_of(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()
    runner = eval_impl.DSparkOfflineRunner(
        target,
        draft,
        tokenizer,
        SimpleNamespace(temperature=0.0, max_new_tokens=args.max_new_tokens),
    )

    candidates = valid_anchor_positions(torch, sample.loss_mask, int(draft.block_size))
    anchor = choose_anchor(
        torch,
        sample.loss_mask,
        int(draft.block_size),
        args.anchor_position,
    )
    log.info("sample_index=%d hidden_file=%s", sample_index, sample.hidden_file)
    log.info(
        "seq_len=%d loss_tokens=%d valid_anchors=%d raw_hidden_shape=%s "
        "train_hidden_shape=%s",
        int(sample.input_ids.numel()),
        int(sample.loss_mask.sum().item()),
        int(candidates.numel()),
        sample.raw_hidden_shape,
        tuple(sample.hidden_states.shape),
    )
    log.info(
        "anchor=%d anchor_id=%d anchor_text=%s",
        anchor,
        int(sample.input_ids[anchor].item()),
        token_text(tokenizer, int(sample.input_ids[anchor].item())),
    )
    log.info(
        "decoded_sample_prefix=%r",
        tokenizer.decode(sample.input_ids[: args.print_tokens]),
    )
    log_token_window(
        tokenizer,
        "anchor_window",
        sample.input_ids[
            max(0, anchor - args.context_tokens) : anchor + args.context_tokens
        ],
        args.context_tokens * 2,
    )
    log_sample_from_anchor_contract(eval_impl, draft, anchor)

    replay = _training_anchor_forward(torch, draft, sample, anchor, device)
    _print_training_slots(
        torch,
        eval_impl,
        tokenizer,
        draft,
        replay,
        sample,
        anchor,
        args.top_k,
    )
    _print_offline_proposal(
        torch,
        eval_impl,
        tokenizer,
        runner,
        sample,
        anchor,
        args.top_k,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--verifier-model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--hidden-states-path", type=Path, default=None)
    p.add_argument("--sample-start", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--random-samples", action="store_true")
    p.add_argument("--anchor-position", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--context-tokens", type=int, default=16)
    p.add_argument("--print-tokens", type=int, default=256)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--draft-attn-impl",
        choices=["auto", "simple_flex_attention", "sdpa", "eager"],
        default="auto",
    )
    p.add_argument("--d2t-path", type=Path, default=None)
    p.add_argument("--t2d-path", type=Path, default=None)
    p.add_argument("--sample-from-anchor", choices=["true", "false"], default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
