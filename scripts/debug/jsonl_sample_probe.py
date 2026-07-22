#!/usr/bin/env python3
"""Print and probe one sample from a real JSONL benchmark dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from dspark_debug_utils import (
    dtype_of,
    load_eval_impl,
    load_jsonl_record,
    load_vocab_maps,
    log_sample_from_anchor_contract,
    log_token_window,
    token_text,
    topk_rows,
)

log = logging.getLogger("jsonl_sample_probe")


def _format_prompt(eval_impl, tokenizer, record, args):
    prompt_args = SimpleNamespace(
        enable_thinking=args.enable_thinking,
        raw_prompt_mode=args.raw_prompt_mode,
    )
    return eval_impl._prompt_from_record(
        record,
        tokenizer,
        source=f"{args.dataset}:{args.sample_index + 1}",
        args=prompt_args,
    )


def _initial_anchor(torch, target, input_ids, eval_impl):
    position_ids = torch.arange(
        input_ids.shape[1] + 1,
        device=input_ids.device,
    ).unsqueeze(0)
    past = eval_impl.DynamicCache()
    with torch.inference_mode():
        out = target(
            input_ids=input_ids,
            position_ids=position_ids[:, : input_ids.shape[1]],
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
        )
    anchor_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    return out, past, anchor_token


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from speculators.models.dspark.core import DSparkDraftModel

    torch.manual_seed(args.seed)
    eval_impl = load_eval_impl(torch)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model,
        trust_remote_code=args.trust_remote_code,
    )
    record = load_jsonl_record(args.dataset, args.sample_index)
    prompt = _format_prompt(eval_impl, tokenizer, record, args)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

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
        SimpleNamespace(
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        ),
    )

    target_out, past, anchor_token = _initial_anchor(
        torch,
        target,
        input_ids,
        eval_impl,
    )
    output_ids = torch.empty(
        (1, input_ids.shape[1] + runner.max_proposal_tokens + 2),
        dtype=torch.long,
        device=device,
    )
    output_ids[:, : input_ids.shape[1]] = input_ids
    anchor = input_ids.shape[1]
    output_ids[:, anchor : anchor + 1] = anchor_token
    position_ids = torch.arange(
        output_ids.shape[1],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    context = runner._init_context(initial_output=target_out)

    with torch.inference_mode():
        proposal = runner._propose(
            context=context,
            output_ids=output_ids,
            position_ids=position_ids,
            start=anchor,
        )
        verification = eval_impl.verify_draft_tokens(
            target_model=target,
            proposal=proposal,
            position_ids=position_ids,
            start=anchor,
            past_key_values_target=past,
            temperature=args.temperature,
            max_proposal_tokens=runner.max_proposal_tokens,
            current_token_ids=output_ids[:, anchor : anchor + 1],
            stop_token_ids=eval_impl.resolve_stop_token_ids(target, tokenizer),
        )

    log.info("dataset=%s sample_index=%d", args.dataset, args.sample_index)
    log.info("raw_record=%s", json.dumps(record, ensure_ascii=False)[: args.raw_chars])
    log.info(
        "prompt_chars=%d prompt_tokens=%d enable_thinking=%s raw_prompt_mode=%s",
        len(prompt),
        input_ids.shape[1],
        args.enable_thinking,
        args.raw_prompt_mode,
    )
    log.info("prompt_preview=%r", prompt[: args.prompt_chars])
    log_token_window(tokenizer, "prompt_tail", input_ids[0], args.prompt_tail_tokens)
    log.info(
        "anchor_pos=%d anchor_id=%d anchor_text=%s",
        anchor,
        int(anchor_token[0, 0].item()),
        token_text(tokenizer, int(anchor_token[0, 0].item())),
    )
    log_sample_from_anchor_contract(eval_impl, draft, anchor)
    log.info(
        "proposal_ids=%s accepted=%d/%d next_token=%d %s",
        proposal.verify_input_ids[0].detach().cpu().tolist(),
        int(verification.accepted_draft_tokens),
        int(proposal.draft_token_count),
        int(verification.next_token[0].item()),
        token_text(tokenizer, int(verification.next_token[0].item())),
    )
    if verification.accept_probs is not None:
        log.info(
            "accept_probs=%s",
            verification.accept_probs.detach().float()[0].tolist(),
        )
        log.info(
            "support_accept_rates=%s",
            verification.support_accept_rates.detach().float()[0].tolist(),
        )

    hidden, base_logits = runner._single_anchor_backbone(
        context.target_hidden_states,
        output_ids,
        anchor,
    )
    prev = output_ids[:, anchor]
    first_slot = eval_impl.first_draft_slot_for_draft(draft)
    for token_idx in range(runner.max_proposal_tokens):
        slot = first_slot + token_idx
        logits = base_logits[:, slot : slot + 1, :]
        if draft.markov_head is not None:
            logits = logits + draft.markov_head.block_bias(
                prev_token_ids=prev.reshape(1, 1),
                hidden_states=hidden[:, slot : slot + 1, :],
            )
        probs = eval_impl.logits_to_probs(logits, args.temperature)
        proposed = int(proposal.verify_input_ids[0, token_idx + 1].item())
        log.info(
            "slot=%d token_idx=%d proposed_id=%d proposed_text=%s",
            slot,
            token_idx,
            proposed,
            token_text(tokenizer, proposed),
        )
        for row in topk_rows(
            torch,
            eval_impl,
            draft,
            logits,
            probs,
            tokenizer,
            args.top_k,
        ):
            log.info(
                (
                    "  slot=%d rank=%d draft_id=%d target_id=%d prob=%.6g "
                    "logit=%.6g text=%s"
                ),
                slot,
                row.rank,
                row.draft_id,
                row.target_id,
                row.prob,
                row.logit,
                row.text,
            )
        prev = torch.tensor([[proposed]], dtype=torch.long, device=device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--verifier-model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--raw-chars", type=int, default=2000)
    p.add_argument("--prompt-chars", type=int, default=2000)
    p.add_argument("--prompt-tail-tokens", type=int, default=64)
    p.add_argument(
        "--enable-thinking",
        choices=["false", "true", "default"],
        default="false",
    )
    p.add_argument(
        "--raw-prompt-mode",
        choices=["auto", "chat_template", "raw"],
        default="auto",
    )
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
