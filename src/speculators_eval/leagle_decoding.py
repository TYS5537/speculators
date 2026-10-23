"""Single-chain greedy LEAGLE decoding with a cached Hugging Face verifier."""

from types import SimpleNamespace

import torch
from transformers import DynamicCache, GenerationConfig


def resolve_stop_token_ids(target_model, tokenizer) -> list[int]:
    eos = getattr(target_model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        return []
    return [eos] if isinstance(eos, int) else list(dict.fromkeys(eos))


def _contains_stop(tokens, stop_token_ids):
    return (
        bool(stop_token_ids)
        and torch.isin(tokens, tokens.new_tensor(stop_token_ids)).any().item()
    )


def _features(output, layer_ids):
    return torch.cat([output.hidden_states[index] for index in layer_ids], dim=-1)


def _response(output_ids, input_length, proposal_lengths, accepted_lengths):
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=input_length,
        num_output_tokens=output_ids.shape[1] - input_length,
        proposal_lengths=proposal_lengths,
        accepted_draft_lengths=accepted_lengths,
        accept_prob_lists=[],
        support_accept_rate_lists=[],
    )


def _accepted_prefix(draft_ids, target_ids, stop_token_ids):
    matches = draft_ids.eq(target_ids[:, : draft_ids.shape[1]])
    accepted = int(matches.long().cumprod(dim=-1).sum().item())
    if accepted and stop_token_ids:
        stops = torch.isin(
            draft_ids[0, :accepted], draft_ids.new_tensor(stop_token_ids)
        ).nonzero(as_tuple=True)[0]
        if stops.numel():
            return int(stops[0].item()) + 1, True
    return accepted, False


@torch.inference_mode()
def generate_greedy(  # noqa: C901 -- Explicit acceptance, EOS and KV rollback cases.
    target_model,
    draft_model,
    input_ids,
    *,
    spec_tokens=7,
    max_new_tokens=512,
    stop_token_ids=None,
    propose=None,
):
    """Verify top-1 proposals, retaining only the accepted target KV/features.

    The target prefill supplies an anchor token. At every subsequent round the
    target cache/features exclude that anchor; verification consumes the anchor
    and its draft continuation. The next anchor is the rejection replacement or
    the all-accepted bonus token. Draft caches remain local to ``propose``.

    Statistics follow the DSpark offline evaluator, including the final target-
    only round (zero proposed tokens) and truncation at an accepted EOS.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:  # noqa: PLR2004
        raise ValueError("Expected one nonempty, unpadded input sequence")
    if spec_tokens < 1 or max_new_tokens < 0:
        raise ValueError("spec_tokens must be positive and max_new_tokens nonnegative")
    if target_model.training or draft_model.training:
        raise ValueError("Both models must be in eval mode")
    input_length = input_ids.shape[1]
    proposal_lengths, accepted_lengths = [], []
    if max_new_tokens == 0:
        return _response(input_ids.clone(), input_length, [], [])

    max_length = input_length + max_new_tokens
    output_ids = input_ids.new_empty((1, max_length))
    output_ids[:, :input_length] = input_ids
    positions = torch.arange(max_length, device=input_ids.device).unsqueeze(0)
    cache = DynamicCache(config=target_model.config)
    output = target_model(
        input_ids=input_ids,
        position_ids=positions[:, :input_length],
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    output_ids[:, input_length : input_length + 1] = output.logits[:, -1].argmax(
        dim=-1, keepdim=True
    )
    end = input_length + 1
    if max_new_tokens == 1 or _contains_stop(
        output_ids[:, input_length:end], stop_token_ids
    ):
        return _response(output_ids[:, :end], input_length, [], [])

    initial_features = _features(output, draft_model.target_layer_ids)
    context = initial_features.new_empty((1, max_length, initial_features.shape[-1]))
    context[:, :input_length] = initial_features
    del output, initial_features

    while end < max_length:
        start = end - 1  # Known anchor, not yet processed by the target.
        count = min(spec_tokens, max_length - end - 1)  # Reserve a bonus token.
        anchor = output_ids[:, start:end]
        if count and propose is not None:
            draft_ids, _ = propose(
                draft_model,
                hidden_states=context[:, :start],
                input_ids=output_ids[:, :end],
                position_ids=positions[:, :start],
                num_tokens=count,
            )
        elif count:
            draft_ids, _ = draft_model.propose(
                hidden_states=context[:, :start],
                anchor_token=anchor,
                position_ids=positions[:, :start],
                num_tokens=count,
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
        target_ids = output.logits.argmax(dim=-1)
        accepted, stopped = _accepted_prefix(draft_ids, target_ids, stop_token_ids)
        proposal_lengths.append(accepted if stopped else count)
        accepted_lengths.append(accepted)
        output_ids[:, end : end + accepted] = draft_ids[:, :accepted]
        end += accepted
        if stopped:
            break

        output_ids[:, end : end + 1] = target_ids[:, accepted : accepted + 1]
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
        output_ids[:, :end], input_length, proposal_lengths, accepted_lengths
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
        response = generate_greedy(
            self.target_model,
            self.draft_model,
            input_ids,
            spec_tokens=self.args.spec_tokens,
            max_new_tokens=self.args.max_new_tokens,
            stop_token_ids=stop_token_ids,
            propose=self.propose,
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
