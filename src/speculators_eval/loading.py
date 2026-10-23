"""Lazy target and drafter setup; importing this module loads no model runtime."""

from __future__ import annotations

import argparse  # noqa: TC003 -- Runtime type hints.
import json
import os
from contextlib import ExitStack  # noqa: TC003 -- Runtime type hints.
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TargetSetup:
    """Validated target metadata; no loaded model or service client."""

    backend: str
    target_config: dict
    report: dict | None
    hs_http_endpoint: str | None


@dataclass(frozen=True)
class LoadedEvaluation:
    """References shared by runner construction and evaluation, without ownership."""

    target_model: Any
    draft_model: Any
    tokenizer: Any
    draft_config: Any


def resolve_draft_attn_impl(device: str, draft_attn_impl: str) -> str | None:
    if draft_attn_impl != "auto":
        return draft_attn_impl
    if str(device).startswith("npu"):
        return "sdpa"
    return None


def parse_bool_override(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected boolean value, got {value}")


def prepare_hs_http(args, target_backend):
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


def write_backend_metadata(args, report):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "dsv4_verification_mode", "reference")
    metadata = {
        "target_backend": "dsv4-vllm",
        "verification_mode": mode,
        "verification": (
            "checked-native-greedy-trajectory-replay"
            if mode == "replay"
            else "immutable-kv-snapshot-accepted-tail-replay"
            if getattr(args, "dsv4_kv_reuse", False)
            else "full-prefix-block-recompute"
            if mode == "block"
            else "full-prefix-per-position-recompute"
        ),
        "probability_source": (
            "cached-native-greedy-tokens-no-off-trajectory-probabilities"
            if mode == "replay"
            else "target_native_full_vocabulary_head_packet"
            if mode == "block"
            else "target_api_full_vocabulary_logprobs"
        ),
        "block_output_policy": getattr(args, "dsv4_block_output", "auto"),
        "diagnostic_profile": getattr(args, "dsv4_profile", False),
        "target_kv_reuse": getattr(args, "dsv4_kv_reuse", False),
        "draft_kv_reuse": getattr(args, "draft_kv_reuse", False),
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
    if mode == "replay":
        metadata.update(
            replay_cache=str(args.dsv4_replay_cache.resolve()),
            replay_cache_tag=args.dsv4_replay_cache_tag,
            replay_audit_samples_per_worker=args.dsv4_replay_audit_samples,
            probability_diagnostics_available=False,
            timing_scope=(
                "cache preparation + local replay + optional live audits; "
                "NOT serving speed"
            ),
            experimental=True,
        )
    with (args.output_dir / "eval_backend.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)


def load_evaluation_models(
    args: argparse.Namespace,
    setup: TargetSetup,
    resources: ExitStack,
    *,
    set_runtime,
    load_draft,
    load_dsv4_target,
) -> LoadedEvaluation:
    """Load models only in a single-device process, after parent dispatch."""
    import torch as torch_module  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from transformers import DynamicCache as DynamicCacheClass  # noqa: PLC0415

    from speculators.model import SpeculatorModel  # noqa: PLC0415

    set_runtime(torch_module, DynamicCacheClass)
    torch = torch_module
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
                attn_implementation=getattr(args, "target_attn_impl", "sdpa"),
                trust_remote_code=args.trust_remote_code,
            )
            .to(device)
            .eval()
        )

    draft_model, draft_config = load_draft(
        args, setup, target_model, device, model_class=SpeculatorModel
    )
    if setup.backend == "dsv4-vllm":
        target_model, tokenizer = load_dsv4_target(args, setup, draft_model, resources)
    return LoadedEvaluation(target_model, draft_model, tokenizer, draft_config)


def load_evaluation_draft(
    args: argparse.Namespace,
    setup: TargetSetup,
    target_model,
    device,
    *,
    model_class,
    torch,
    load_draft_config,
    parse_bool_override,
    resolve_draft_attn_impl,
    load_vocab_mapping_tensors,
    ensure_loaded_vocab_mappings,
):
    """Bind target identity and overrides before loading/casting draft weights.

    The caller imports the model class before seeding, retaining the original
    import/RNG order. Keep precision resolution after config and vocabulary checks.
    """
    draft_config = load_draft_config(args.draft_model)
    if setup.backend == "dsv4-vllm":
        from speculators_dsv4.eval_contract import bind_draft_verifier  # noqa: PLC0415

        bind_draft_verifier(draft_config, setup.report)
    else:
        # Borrow embeddings/head/norm from the same target that verifies tokens,
        # including when its local path differs from the training checkpoint.
        draft_config.speculators_config.verifier.name_or_path = args.verifier_model
    sample_from_anchor = parse_bool_override(args.sample_from_anchor)
    if sample_from_anchor is not None:
        draft_config.sample_from_anchor = sample_from_anchor
    draft_attn_impl = resolve_draft_attn_impl(args.device, args.draft_attn_impl)
    if draft_attn_impl is not None:
        draft_config.transformer_layer_config._attn_implementation = draft_attn_impl  # noqa: SLF001 -- Transformers backend selection.
    d2t, t2d = load_vocab_mapping_tensors(
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
    ensure_loaded_vocab_mappings(draft_model, args)
    return draft_model, draft_config


def load_dsv4_target(
    args: argparse.Namespace,
    setup: TargetSetup,
    draft_model,
    resources: ExitStack,
    *,
    logger,
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
    target_class = DSV4OfflineTarget
    target_options = {}
    if args.dsv4_verification_mode == "replay":
        from speculators_dsv4.replay import DSV4GreedyReplayTarget  # noqa: PLC0415

        target_class = DSV4GreedyReplayTarget
        target_options = {
            "replay_cache": args.dsv4_replay_cache,
            "replay_tag": args.dsv4_replay_cache_tag,
            "audit_samples": args.dsv4_replay_audit_samples,
        }
    if getattr(args, "dsv4_kv_reuse", False):
        from speculators_dsv4.cached_target import DSV4CachedTarget  # noqa: PLC0415

        target_class = DSV4CachedTarget
    target_model = target_class(
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
        profile=getattr(args, "dsv4_profile", False),
        **target_options,
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
    if args.dsv4_verification_mode == "replay":
        logger.warning(
            "Experimental DSV4 greedy trajectory replay: acceptance counts only; "
            "no target probability diagnostics. Timings include cache preparation "
            "and optional live audits, NOT online speculative speed."
        )
    else:
        logger.warning(
            "DSV4 %s verification uses %s and native target "
            "probabilities. Reported elapsed time is NOT online speculative speed.",
            args.dsv4_verification_mode,
            "host KV snapshots with accepted-tail replay"
            if getattr(args, "dsv4_kv_reuse", False)
            else "full-prefix recomputation",
        )
    return target_model, tokenizer
