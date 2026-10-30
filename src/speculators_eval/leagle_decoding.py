"""Single-chain EAGLE decoding with greedy or rejection-sampling verification."""

from types import SimpleNamespace

import torch
from transformers import DynamicCache, GenerationConfig

from speculators.models.leagle import LEagleDraftModel
from speculators.models.leagle.sparse_candidates import SparseCandidateLogits
from speculators.proposals.sampling import (
    logits_to_probs,
    sample_from_logits,
    sample_from_probs,
    validate_temperature,
)
from speculators_eval.leagle_target import VerifiedStaticCache, selected_hidden_target


def resolve_stop_token_ids(target_model, tokenizer) -> list[int]:
    eos = getattr(target_model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        return []
    return [eos] if isinstance(eos, int) else list(dict.fromkeys(eos))


def _stop_tensor(tokens, stop_token_ids):
    if isinstance(stop_token_ids, torch.Tensor):
        return stop_token_ids
    return tokens.new_tensor([] if stop_token_ids is None else stop_token_ids)


def _contains_stop(tokens, stop_token_ids):
    stops = _stop_tensor(tokens, stop_token_ids)
    return stops.numel() > 0 and torch.isin(tokens, stops).any().item()


def _statistics_to_host(accept_probs, support_rates):
    """Transfer the two tiny diagnostic vectors once per completed request."""
    if not accept_probs or not isinstance(accept_probs[0], torch.Tensor):
        return accept_probs, support_rates
    lengths = [values.numel() for values in accept_probs]
    rows = torch.stack((torch.cat(accept_probs), torch.cat(support_rates))).tolist()
    result = [[], []]
    start = 0
    for length in lengths:
        for destination, row in zip(result, rows, strict=True):
            destination.append(row[start : start + length])
        start += length
    return result


def _features(output, layer_ids):
    return torch.cat([output.hidden_states[index] for index in layer_ids], dim=-1)


def _response(
    output_ids,
    input_length,
    proposal_lengths,
    accepted_lengths,
    accept_prob_lists=None,
    support_accept_rate_lists=None,
):
    accept_prob_lists, support_accept_rate_lists = _statistics_to_host(
        accept_prob_lists or [], support_accept_rate_lists or []
    )
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=input_length,
        num_output_tokens=output_ids.shape[1] - input_length,
        proposal_lengths=proposal_lengths,
        accepted_draft_lengths=accepted_lengths,
        accept_prob_lists=accept_prob_lists or [],
        support_accept_rate_lists=support_accept_rate_lists or [],
    )


def _accepted_prefix(draft_ids, target_ids, stop_token_ids):
    matches = draft_ids.eq(target_ids[:, : draft_ids.shape[1]])
    accepted = int(matches.long().cumprod(dim=-1).sum().item())
    return _truncate_accepted_stop(draft_ids, accepted, stop_token_ids)


def _truncate_accepted_stop(draft_ids, accepted, stop_token_ids):
    stop_ids = _stop_tensor(draft_ids, stop_token_ids)
    if accepted and stop_ids.numel():
        stops = torch.isin(draft_ids[0, :accepted], stop_ids).nonzero(as_tuple=True)[0]
        if stops.numel():
            return int(stops[0].item()) + 1, True
    return accepted, False


def _draft_probabilities(draft_model, logits, temperature, target_vocab_size):
    probs = logits_to_probs(logits, temperature)
    if not draft_model.use_draft_vocab:
        if probs.shape[-1] != target_vocab_size:
            raise ValueError("Full draft and target vocabularies must match")
        return probs
    # q is zero outside the draft vocabulary. In particular, rejection may
    # select a target token that the pruned draft could never have proposed.
    ids = torch.arange(probs.shape[-1], device=probs.device) + draft_model.d2t
    expanded = probs.new_zeros(*probs.shape[:-1], target_vocab_size)
    expanded.index_copy_(-1, ids.long(), probs)
    return expanded


def _proposal_statistics(draft_model, logits, temperature, target_probs, draft_ids):
    count = draft_ids.shape[1]
    if isinstance(logits, SparseCandidateLogits):
        if logits.vocab_size != target_probs.shape[-1]:
            raise ValueError("Sparse proposal IDs must use the target vocabulary")
        probs = logits_to_probs(logits.values, temperature)
        selected = (probs * (logits.indices == draft_ids.unsqueeze(-1))).sum(-1)
        supported_p = target_probs[:, :count].gather(-1, logits.indices)
        return (
            (logits.indices, probs),
            selected,
            torch.minimum(supported_p, probs).sum(-1),
        )
    probs = _draft_probabilities(
        draft_model, logits, temperature, target_probs.shape[-1]
    )
    return (
        probs,
        probs.gather(-1, draft_ids.unsqueeze(-1)).squeeze(-1),
        torch.minimum(target_probs[:, :count], probs).sum(-1),
    )


def _rejection_residual(target_probs, draft_probs, position):
    if isinstance(draft_probs, tuple):
        indices, probs = draft_probs
        # q=0 outside the candidate set. Retain all target tokens in p-q,
        # including tokens absent from the candidate AND reduced vocabularies.
        return (
            target_probs.clone()
            .scatter_add_(-1, indices[:, position], -probs[:, position])
            .clamp_min_(0)
        )
    return (target_probs - draft_probs[:, position]).clamp_min(0)


def _verify_sampled(
    draft_model,
    draft_ids,
    draft_logits,
    target_logits,
    temperature,
    stop_token_ids,
    *,
    defer_statistics=False,
):
    """Verify sampled q tokens against p and draw the corrected continuation."""
    count = draft_ids.shape[1]
    target_probs = logits_to_probs(target_logits, temperature)
    accept_probs = target_probs.new_empty((1, 0))
    support_rates = target_probs.new_empty((1, 0))
    accepted = 0
    if count:
        if draft_logits is None:
            raise ValueError("Sampled proposals must return their conditional logits")
        draft_probs, selected_q, support_rates = _proposal_statistics(
            draft_model, draft_logits, temperature, target_probs, draft_ids
        )
        if not torch.all(torch.isfinite(selected_q) & (selected_q > 0)):
            raise ValueError("Sampled draft tokens must have positive finite q")
        selected_p = (
            target_probs[:, :count].gather(-1, draft_ids.unsqueeze(-1)).squeeze(-1)
        )
        accept_probs = (selected_p / selected_q).clamp(max=1.0)
        accepted = int(
            (torch.rand_like(accept_probs) < accept_probs)
            .long()
            .cumprod(-1)
            .sum()
            .item()
        )
    accepted, stopped = _truncate_accepted_stop(draft_ids, accepted, stop_token_ids)
    effective_count = accepted if stopped else count
    next_token = None
    if not stopped:
        if accepted < count:
            # multinomial normalizes max(p-q, 0); no epsilon/fallback that would
            # silently change the target distribution on rejected proposals.
            residual = _rejection_residual(
                target_probs[:, accepted], draft_probs, accepted
            )
            next_token = sample_from_probs(residual).unsqueeze(-1)
        else:
            next_token = sample_from_probs(target_probs[:, count]).unsqueeze(-1)
    probabilities = accept_probs[0, :effective_count]
    support = support_rates[0, :effective_count]
    return (
        accepted,
        stopped,
        next_token,
        probabilities if defer_statistics else probabilities.tolist(),
        support if defer_statistics else support.tolist(),
    )


@torch.inference_mode()
def generate_speculative(
    target_model,
    draft_model,
    input_ids,
    *,
    spec_tokens=7,
    max_new_tokens=512,
    stop_token_ids=None,
    propose=None,
    temperature=0.0,
    draft_kv_reuse=True,
):
    """Generate one request with optional selected-state capture on the target."""
    selected = getattr(draft_model, "selective_target_hidden_states", False)
    with selected_hidden_target(
        target_model,
        draft_model.target_layer_ids if selected else (),
        enabled=selected,
    ) as target:
        return _generate_speculative(
            target,
            draft_model,
            input_ids,
            spec_tokens=spec_tokens,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            propose=propose,
            temperature=temperature,
            draft_kv_reuse=draft_kv_reuse,
        )


def _generate_speculative(  # noqa: C901 -- Explicit acceptance, EOS and KV rollback cases.
    target_model,
    draft_model,
    input_ids,
    *,
    spec_tokens=7,
    max_new_tokens=512,
    stop_token_ids=None,
    propose=None,
    temperature=0.0,
    draft_kv_reuse=True,
):
    """Verify draft proposals, retaining only the accepted target KV/features.

    The target prefill supplies an anchor token. At every subsequent round the
    target cache/features exclude that anchor; verification consumes the anchor
    and its draft continuation. The next anchor is the rejection replacement or
    the all-accepted bonus token. With draft_kv_reuse, a request-local cache
    retains only target-conditioned prefix states, never synthetic rollout KV.

    Statistics follow the DSpark offline evaluator, including the final target-
    only round (zero proposed tokens) and truncation at an accepted EOS.
    Positive temperatures sample both models and use exact rejection sampling;
    zero retains the original top-1 path without consuming sampling RNG.
    """
    validate_temperature(temperature)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:  # noqa: PLR2004
        raise ValueError("Expected one nonempty, unpadded input sequence")
    if spec_tokens < 1 or max_new_tokens < 0:
        raise ValueError("spec_tokens must be positive and max_new_tokens nonnegative")
    if target_model.training or draft_model.training:
        raise ValueError("Both models must be in eval mode")
    stop_token_ids = _stop_tensor(input_ids, stop_token_ids)
    if getattr(draft_model, "reuse_rollout_buffers", False) and (
        not draft_kv_reuse or propose is not None
    ):
        raise ValueError(
            "Rollout buffer reuse requires the native cached LEAGLE proposer"
        )
    input_length = input_ids.shape[1]
    proposal_lengths, accepted_lengths = [], []
    accept_prob_lists, support_accept_rate_lists = [], []
    if max_new_tokens == 0:
        return _response(input_ids.clone(), input_length, [], [])

    max_length = input_length + max_new_tokens
    output_ids = input_ids.new_empty((1, max_length))
    output_ids[:, :input_length] = input_ids
    positions = torch.arange(max_length, device=input_ids.device).unsqueeze(0)
    cache = (
        VerifiedStaticCache(target_model.config, max_length)
        if getattr(draft_model, "static_target_kv", False)
        else DynamicCache(config=target_model.config)
    )
    output = target_model(
        input_ids=input_ids,
        position_ids=positions[:, :input_length],
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    output_ids[:, input_length : input_length + 1] = sample_from_logits(
        output.logits[:, -1:], temperature
    )
    end = input_length + 1
    if max_new_tokens == 1 or _contains_stop(
        output_ids[:, input_length:end], stop_token_ids
    ):
        return _response(output_ids[:, :end], input_length, [], [])

    native_leagle = propose is None and isinstance(draft_model, LEagleDraftModel)
    if draft_kv_reuse and max_new_tokens > 2:  # noqa: PLR2004 -- Anchor + bonus need no draft.
        from speculators_eval.eagle_cache import cached_proposal  # noqa: PLC0415

        propose = cached_proposal(draft_model, propose, capacity=max_length)
    initial_features = _features(output, draft_model.target_layer_ids)
    context = initial_features.new_empty((1, max_length, initial_features.shape[-1]))
    context[:, :input_length] = initial_features
    del output, initial_features

    while end < max_length:
        start = end - 1  # Known anchor, not yet processed by the target.
        count = min(spec_tokens, max_length - end - 1)  # Reserve a bonus token.
        anchor = output_ids[:, start:end]
        # Keep compatibility with existing greedy-only proposal callbacks.
        proposal_kwargs = {"temperature": temperature} if temperature else {}
        if native_leagle and temperature == 0:
            proposal_kwargs["return_logits"] = False
        draft_logits = None
        if count and propose is not None:
            draft_ids, draft_logits = propose(
                draft_model,
                hidden_states=context[:, :start],
                input_ids=output_ids[:, :end],
                position_ids=positions[:, :start],
                num_tokens=count,
                **proposal_kwargs,
            )
        elif count:
            draft_ids, draft_logits = draft_model.propose(
                hidden_states=context[:, :start],
                anchor_token=anchor,
                position_ids=positions[:, :start],
                num_tokens=count,
                **proposal_kwargs,
            )
        else:
            draft_ids = input_ids.new_empty((1, 0))
        verify_ids = torch.cat([anchor, draft_ids], dim=1)
        output = target_model(
            input_ids=verify_ids,
            position_ids=positions[:, start : start + count + 1],
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
        if temperature > 0:
            accepted, stopped, next_token, accept_probs, support_rates = (
                _verify_sampled(
                    draft_model,
                    draft_ids,
                    draft_logits,
                    output.logits,
                    temperature,
                    stop_token_ids,
                    defer_statistics=True,
                )
            )
            accept_prob_lists.append(accept_probs)
            support_accept_rate_lists.append(support_rates)
        else:
            target_ids = output.logits.argmax(dim=-1)
            accepted, stopped = _accepted_prefix(draft_ids, target_ids, stop_token_ids)
            next_token = target_ids[:, accepted : accepted + 1]
        proposal_lengths.append(accepted if stopped else count)
        accepted_lengths.append(accepted)
        output_ids[:, end : end + accepted] = draft_ids[:, :accepted]
        end += accepted
        if stopped:
            break

        output_ids[:, end : end + 1] = next_token
        end += 1
        # The anchor and accepted proposals are now verified. Discard the
        # rejected suffix before another round can observe its KV or features.
        committed = accepted + 1
        cache.crop(start + committed)
        if end >= max_length or _contains_stop(
            output_ids[:, end - 1 : end], stop_token_ids
        ):
            break
        context[:, start : start + committed] = _features(
            output, draft_model.target_layer_ids
        )[:, :committed]
        del output

    return _response(
        output_ids[:, :end],
        input_length,
        proposal_lengths,
        accepted_lengths,
        accept_prob_lists,
        support_accept_rate_lists,
    )


def generate_greedy(
    target_model,
    draft_model,
    input_ids,
    *,
    spec_tokens=7,
    max_new_tokens=512,
    stop_token_ids=None,
    propose=None,
    draft_kv_reuse=True,
):
    """Compatibility entrypoint for the original deterministic protocol."""
    return generate_speculative(
        target_model,
        draft_model,
        input_ids,
        spec_tokens=spec_tokens,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        propose=propose,
        temperature=0.0,
        draft_kv_reuse=draft_kv_reuse,
    )


@torch.inference_mode()
def check_greedy_output(
    target_model, input_ids, response, max_new_tokens, stop_token_ids
):
    """Audit against independent HF generation, with ordinary greedy settings."""
    if max_new_tokens == 0:
        expected = input_ids
    else:
        expected = target_model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=GenerationConfig(
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=stop_token_ids or None,
                pad_token_id=stop_token_ids[0] if stop_token_ids else 0,
                use_cache=True,
            ),
        )
    if not torch.equal(expected, response.output_ids):
        raise RuntimeError(
            "Speculative output differs from target-only greedy generation; "
            "check target numerical precision and attention backend"
        )


class LEagleOfflineRunner:
    def __init__(self, target_model, draft_model, tokenizer, args):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.args = args
        self.device = next(target_model.parameters()).device
        self.propose = None
        if draft_model.config.speculators_model_type == "eagle3":
            from speculators_eval.eagle3_decoding import propose_eagle3  # noqa: PLC0415

            self.propose = propose_eagle3

    def generate_one(self, prompt, stop_token_ids):
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        limit = self.target_model.config.max_position_embeddings
        if input_ids.shape[1] + self.args.max_new_tokens > limit:
            raise ValueError(
                f"Prompt + max_new_tokens exceeds target context limit {limit}"
            )
        response = generate_speculative(
            self.target_model,
            self.draft_model,
            input_ids,
            spec_tokens=self.args.spec_tokens,
            max_new_tokens=self.args.max_new_tokens,
            stop_token_ids=stop_token_ids,
            propose=self.propose,
            temperature=self.args.temperature,
            draft_kv_reuse=not getattr(self.args, "no_draft_kv_reuse", False),
        )
        if self.args.check_greedy:
            check_greedy_output(
                self.target_model,
                input_ids,
                response,
                self.args.max_new_tokens,
                stop_token_ids,
            )
        return response
