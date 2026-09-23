"""DSV4 verification using stateless vLLM full-prefix requests.

No target KV state is cropped. Probabilities come from the target's complete
vocabulary, not from a reconstructed/local teacher head. This is intentionally
an acceptance evaluator, NOT an online speculative-throughput implementation.
"""

from __future__ import annotations

import json
import math
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from speculators_dsv4 import HS_FORMAT
from speculators_dsv4.block_protocol import BLOCK_PROTOCOL_VERSION, BLOCK_REQUEST_KEY
from speculators_dsv4.contract import make_manifest, validate_layers
from speculators_dsv4.eval_contract import read_eval_manifest

_LOGPROB_POSITIVE_TOLERANCE = 1e-6
_TOKEN_BATCH_NDIM = 2


def parse_full_logprobs(top_logprobs, vocab_size):
    """Require exactly one raw log probability for every vocabulary token."""
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    if not isinstance(top_logprobs, dict) or len(top_logprobs) != vocab_size:
        raise ValueError(
            "Target must return full-vocabulary logprobs, not a truncated top-k. "
            "Set the server --max-logprobs to the vocabulary size (or -1)."
        )
    values = [None] * vocab_size
    for key, value in top_logprobs.items():
        if not isinstance(key, str) or not re.fullmatch(r"token_id:(0|[1-9]\d*)", key):
            raise ValueError(
                "Expected token_id:N keys; enable return_tokens_as_token_ids"
            )
        token = int(key.split(":", 1)[1])
        if not 0 <= token < vocab_size or values[token] is not None:
            raise ValueError("Target logprobs contain duplicate/out-of-range token IDs")
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError("Target logprobs must be numeric")
        logprob = float(value)
        if (
            math.isnan(logprob)
            or logprob == math.inf
            or logprob > _LOGPROB_POSITIVE_TOLERANCE
        ):
            raise ValueError("Target logprobs contain invalid values")
        values[token] = logprob
    if any(value is None for value in values):
        raise ValueError("Target logprobs are missing vocabulary token IDs")
    total = math.fsum(math.exp(value) for value in values)
    if not math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-5):
        raise ValueError(
            f"Target full-vocabulary probabilities sum to {total}, not 1. "
            "Use --logprobs-mode raw_logprobs and neutral sampling settings."
        )
    return values


class TokenHistoryCache:
    """Only committed/input token IDs; never a DeepSeek compressed-state cache."""

    def __init__(self):
        self.tokens = []

    def get_seq_length(self):
        return len(self.tokens)

    def crop(self, length):
        if type(length) is not int or not 0 <= length <= len(self.tokens):
            raise ValueError("Token-history crop must be within the current prefix")
        del self.tokens[length:]


def request_hidden_file(handle, directory, request_id):
    """Accept only this request's direct child file, never cached training HS."""
    if not isinstance(handle, str) or not handle:
        raise ValueError("Target response is missing its hidden-states file")
    directory = Path(directory).resolve()
    path = Path(handle)
    pattern = rf"cmpl-{re.escape(request_id)}-0(?:-[0-9a-f]{{8}})?\.safetensors"
    if (
        not path.is_absolute()
        or path.is_symlink()
        or path.resolve().parent != directory
        or re.fullmatch(pattern, path.name) is None
        or Path(str(path) + ".lock").is_symlink()
    ):
        raise ValueError(
            "HS handle does not belong to this evaluation request/directory"
        )
    return path.resolve()


class DSV4OfflineTarget:
    def __init__(
        self,
        draft_model,
        report,
        *,
        hidden_states_path,
        client,
        model_name,
        max_model_len,
        timeout=120.0,
        keep_hidden_states=False,
        verification_mode="reference",
        hs_http_endpoint=None,
        hs_http_token=None,
    ):
        import torch  # noqa: PLC0415

        if draft_model.config.target_hidden_state_format != HS_FORMAT:
            raise ValueError("DSV4 evaluation requires a draft trained with DSV4 HS")
        configured_target = draft_model.config.speculators_config.verifier.name_or_path
        if Path(configured_target).resolve() != Path(report["model_path"]).resolve():
            raise ValueError("Draft checkpoint and evaluation target paths differ")
        parameter = next(draft_model.parameters())
        if parameter.dtype != torch.bfloat16:
            raise ValueError("DSV4 evaluation requires a BF16 draft")
        self.draft_model = draft_model
        self.device = parameter.device
        self._initialize_contract(
            report,
            draft_model.target_layer_ids,
            hidden_states_path=hidden_states_path,
            client=client,
            model_name=model_name,
            max_model_len=max_model_len,
            timeout=timeout,
            keep_hidden_states=keep_hidden_states,
            verification_mode=verification_mode,
            hs_http_endpoint=hs_http_endpoint,
            hs_http_token=hs_http_token,
        )

    @classmethod
    def from_contract(
        cls,
        report,
        layer_ids,
        *,
        hidden_states_path,
        client,
        model_name,
        max_model_len,
        timeout=120.0,
        keep_hidden_states=False,
        verification_mode="reference",
        hs_http_endpoint=None,
        hs_http_token=None,
    ):
        """Create strict CPU transport for teacher diagnostics without a drafter."""
        import torch  # noqa: PLC0415

        target = cls.__new__(cls)
        target.draft_model = None
        target.device = torch.device("cpu")
        target._initialize_contract(  # noqa: SLF001 -- Same-class factory.
            report,
            layer_ids,
            hidden_states_path=hidden_states_path,
            client=client,
            model_name=model_name,
            max_model_len=max_model_len,
            timeout=timeout,
            keep_hidden_states=keep_hidden_states,
            verification_mode=verification_mode,
            hs_http_endpoint=hs_http_endpoint,
            hs_http_token=hs_http_token,
        )
        return target

    def _initialize_contract(
        self,
        report,
        layer_ids,
        *,
        hidden_states_path,
        client,
        model_name,
        max_model_len,
        timeout,
        keep_hidden_states,
        verification_mode,
        hs_http_endpoint,
        hs_http_token,
    ):
        self.layer_ids = list(layer_ids)
        validate_layers(self.layer_ids)
        self.hidden_states_path = Path(hidden_states_path).resolve()
        manifest = make_manifest(report, self.layer_ids)
        self.http_transfer = None
        if hs_http_endpoint:
            from speculators_dsv4.hs_http import HttpHiddenStates  # noqa: PLC0415

            self.http_transfer = HttpHiddenStates(
                hs_http_endpoint,
                hs_http_token,
                self.hidden_states_path,
                timeout=timeout,
            )
            remote_manifest = self.http_transfer.validate_manifest(manifest)
        else:
            remote_manifest = read_eval_manifest(self.hidden_states_path, manifest)
        self.packet_layer_ids = [*self.layer_ids, manifest["teacher_hs_id"]]
        if max_model_len <= 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Invalid target context limit or request timeout")
        # vLLM defaults to the SERVER's model path, not the evaluator's local copy.
        if model_name is None:
            model_name = remote_manifest.get("model_path")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Target model alias must be a nonempty string")
        if verification_mode not in ("reference", "block"):
            raise ValueError("DSV4 verification mode must be reference or block")
        self.vocab_size = int(report["config"]["vocab_size"])
        self.hidden_size = int(report["config"]["hidden_size"])
        self.client = client
        self.model_name = model_name
        self.max_model_len = int(max_model_len)
        self.timeout = timeout
        self.keep_hidden_states = keep_hidden_states
        self.verification_mode = verification_mode
        self.num_target_requests = 0
        generation = dict(report["config"])
        generation_path = Path(report["model_path"]) / "generation_config.json"
        if generation_path.exists():
            generation.update(json.loads(generation_path.read_text(encoding="utf-8")))
        self.generation_config = SimpleNamespace(
            eos_token_id=generation.get("eos_token_id")
        )

    def parameters(self):
        if self.draft_model is None:
            raise RuntimeError(
                "Contract-only teacher transport has no draft parameters"
            )
        return self.draft_model.parameters()

    @staticmethod
    def new_cache():
        return TokenHistoryCache()

    def validate_request_budget(
        self, prompt_length, max_new_tokens, max_proposal_tokens
    ):
        del max_proposal_tokens  # Retained for the evaluator's shared target API.
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        # The evaluator truncates each proposal to leave room for one bonus token.
        # Its longest target prefix is one token short of the output limit; the
        # API output token used to obtain next-token logprobs fills that last slot.
        needed = prompt_length + max_new_tokens
        if needed > self.max_model_len:
            raise ValueError(
                f"DSV4 evaluation needs up to {needed} target positions including "
                f"the API output token; configured limit is {self.max_model_len}. "
                "Increase both server and --dsv4-max-model-len, or shorten the request."
            )

    def _validate_response(self, response, request_id):
        if response.id != f"cmpl-{request_id}" or len(response.choices) != 1:
            raise ValueError("Target response request ID/choice count mismatch")
        if getattr(response, "model", None) != self.model_name:
            raise ValueError("Target response model alias mismatch")

    def _new_request_id(self):
        return (
            self.http_transfer.new_request_id()
            if self.http_transfer is not None
            else uuid4().hex
        )

    @contextmanager
    def _artifact(self, handle, request_id, *, block=False, download=True):
        if self.http_transfer is not None:
            with self.http_transfer.artifact(
                handle, request_id, keep=self.keep_hidden_states, download=download
            ) as path:
                yield path
            return
        path = request_hidden_file(handle, self.hidden_states_path, request_id)
        lock_path = str(path) + ".lock"
        if Path(lock_path).exists():
            if block:
                raise ValueError(
                    "Target block packet must be complete without a writer lock"
                )
            from hs_connectors.transfer import wait_for_lock  # noqa: PLC0415

            # On timeout leave files intact, rather than racing the writer.
            wait_for_lock(lock_path, timeout=self.timeout)
        try:
            yield path
        finally:
            if not self.keep_hidden_states and path.exists():
                if block:
                    path.unlink()
                else:
                    from hs_connectors import FileTransfer  # noqa: PLC0415

                    FileTransfer(self.hidden_states_path).delete(str(path))

    def _request(self, prefix, need_hidden):
        import torch  # noqa: PLC0415

        request_id = self._new_request_id()
        self.num_target_requests += 1
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prefix,
            max_tokens=1,
            n=1,
            echo=False,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=self.vocab_size,
            timeout=self.timeout,
            extra_headers={"X-Request-Id": request_id},
            extra_body={
                "request_id": request_id,
                "top_k": 0,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "ignore_eos": True,
                "skip_special_tokens": False,
                "add_special_tokens": False,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
            },
        )
        self._validate_response(response, request_id)
        transfer_params = getattr(response, "kv_transfer_params", None) or {}
        with self._artifact(
            transfer_params.get("hidden_states_path"),
            request_id,
            download=need_hidden,
        ) as path:
            choice = response.choices[0]
            if getattr(choice, "prompt_token_ids", None) != prefix:
                raise ValueError("Target changed/truncated the requested token prefix")
            top = getattr(getattr(choice, "logprobs", None), "top_logprobs", None)
            if not isinstance(top, list) or len(top) != 1:
                raise ValueError("Target must return one full next-token distribution")
            logprobs = parse_full_logprobs(top[0], self.vocab_size)
            if not need_hidden:
                return logprobs, None
            if self.http_transfer is not None:
                from safetensors.torch import load_file  # noqa: PLC0415

                payload = load_file(str(path), device="cpu")
            else:
                from hs_connectors import FileTransfer  # noqa: PLC0415

                payload = FileTransfer(self.hidden_states_path).get_generated(str(path))
            if payload is None or payload["token_ids"].tolist() != prefix:
                raise ValueError(
                    "Target HS file is missing or its token IDs do not match"
                )
            hidden = payload["hidden_states"]
            expected = (len(prefix), len(self.layer_ids) + 1, self.hidden_size)
            if tuple(hidden.shape) != expected:
                raise ValueError(
                    f"Expected target HS shape {expected}, got {hidden.shape}"
                )
            if (
                hidden.dtype != torch.bfloat16
                or not torch.isfinite(hidden).all().item()
            ):
                raise ValueError("Target hidden states must be finite BF16 tensors")
            return logprobs, hidden

    def _validate_block_packet(self, packet, prefix, logits_start, hidden_start):
        import torch  # noqa: PLC0415

        expected_fields = {
            "token_ids",
            "verification_metadata",
            "layer_ids",
            "logprobs",
            "hidden_states",
        }
        if set(packet) != expected_fields:
            raise ValueError("Target block packet has missing or unsupported fields")
        identities = {
            "token_ids": prefix,
            "verification_metadata": [
                BLOCK_PROTOCOL_VERSION,
                len(prefix),
                logits_start,
                hidden_start,
            ],
            "layer_ids": self.packet_layer_ids,
        }
        for name, expected in identities.items():
            value = packet[name]
            if (
                value.dtype != torch.int64
                or tuple(value.shape) != (len(expected),)
                or value.tolist() != expected
            ):
                raise ValueError(f"Target block packet {name} identity mismatch")
        logprobs = packet["logprobs"]
        expected_logits = (len(prefix) - logits_start, self.vocab_size)
        if logprobs.dtype != torch.float32 or tuple(logprobs.shape) != expected_logits:
            raise ValueError(
                f"Target block logprobs must be float32 with shape {expected_logits}"
            )
        if (
            torch.isnan(logprobs).any()
            or (logprobs > _LOGPROB_POSITIVE_TOLERANCE).any()
        ):
            raise ValueError(
                "Target block logprobs contain NaN, +inf or positive values"
            )
        totals = logprobs.double().exp().sum(-1)
        if not torch.isclose(
            totals, torch.ones_like(totals), rtol=1e-3, atol=1e-5
        ).all():
            raise ValueError("Target block full-vocabulary probabilities must sum to 1")
        hidden = packet["hidden_states"]
        expected_hidden = (
            len(prefix) - hidden_start,
            len(self.packet_layer_ids),
            self.hidden_size,
        )
        if (
            hidden.dtype != torch.bfloat16
            or tuple(hidden.shape) != expected_hidden
            or not torch.isfinite(hidden).all()
        ):
            raise ValueError(
                "Target block hidden states must be finite BF16 with shape "
                f"{expected_hidden}"
            )
        return logprobs, hidden

    def _request_block(self, prefix, *, logits_start, hidden_start):
        from safetensors.torch import load_file  # noqa: PLC0415

        request_id = self._new_request_id()
        self.num_target_requests += 1
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prefix,
            max_tokens=1,
            n=1,
            echo=False,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            timeout=self.timeout,
            extra_headers={"X-Request-Id": request_id},
            extra_body={
                "request_id": request_id,
                "top_k": 0,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "ignore_eos": True,
                "skip_special_tokens": False,
                "add_special_tokens": False,
                "return_token_ids": True,
                "kv_transfer_params": {
                    BLOCK_REQUEST_KEY: {
                        "version": BLOCK_PROTOCOL_VERSION,
                        "logits_start": logits_start,
                        "hidden_start": hidden_start,
                    }
                },
            },
        )
        self._validate_response(response, request_id)
        transfer_params = getattr(response, "kv_transfer_params", None)
        if not isinstance(transfer_params, dict):
            raise ValueError("Target did not return DSV4 block transfer parameters")
        version = transfer_params.get("dsv4_block_verify_version")
        if type(version) is not int or version != BLOCK_PROTOCOL_VERSION:
            # A legacy service may still own an asynchronous HS writer. Only a
            # confirmed block response promises that it is safe to remove its file.
            raise ValueError("Target did not confirm the DSV4 block protocol version")
        with self._artifact(
            transfer_params.get("hidden_states_path"),
            request_id,
            block=True,
        ) as path:
            if getattr(response.choices[0], "prompt_token_ids", None) != prefix:
                raise ValueError("Target changed/truncated the requested token prefix")
            return self._validate_block_packet(
                load_file(str(path), device="cpu"), prefix, logits_start, hidden_start
            )

    def __call__(
        self,
        *,
        input_ids,
        position_ids,
        past_key_values,
        use_cache=True,
        output_hidden_states=False,
    ):
        if not isinstance(past_key_values, TokenHistoryCache) or not use_cache:
            raise ValueError("DSV4 target requires its token-history cache")
        if (
            input_ids.ndim != _TOKEN_BATCH_NDIM
            or input_ids.shape[0] != 1
            or input_ids.shape[1] == 0
        ):
            raise ValueError("DSV4 target supports one nonempty sequence")
        old_length = past_key_values.get_seq_length()
        new_tokens = input_ids[0].tolist()
        if position_ids.tolist() != [
            list(range(old_length, old_length + len(new_tokens)))
        ]:
            raise ValueError(
                "Target positions do not match committed token-history length"
            )
        if any(
            type(token) is not int or not 0 <= token < self.vocab_size
            for token in new_tokens
        ):
            raise ValueError("Target input token outside vocabulary")
        prefix = [*past_key_values.tokens, *new_tokens]
        if len(prefix) + 1 > self.max_model_len:
            raise ValueError(
                "Target prefix exceeds context limit including one API token"
            )
        if self.verification_mode == "block":
            return self._forward_block(
                prefix, old_length, past_key_values, output_hidden_states
            )
        return self._forward_reference(
            prefix, old_length, past_key_values, output_hidden_states
        )

    def _forward_reference(self, prefix, old_length, cache, output_hidden_states):
        import torch  # noqa: PLC0415

        # Prefill consumers only use the last logit row, but need all prompt HS.
        positions = (
            [len(prefix)] if old_length == 0 else range(old_length + 1, len(prefix) + 1)
        )
        rows = []
        hidden = None
        for length in positions:
            need_hidden = output_hidden_states and length == len(prefix)
            row, hidden = self._request(prefix[:length], need_hidden=need_hidden)
            rows.append(row)
        logits = torch.tensor(rows, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        if logits.shape[-1] != self.vocab_size:
            raise ValueError("Target verification requires the full target vocabulary")
        states = None
        if output_hidden_states:
            if hidden is None:
                raise ValueError("Target verification did not return hidden states")
            states = {
                layer: hidden[old_length:, slot, :].unsqueeze(0).to(self.device)
                for slot, layer in enumerate(self.layer_ids)
            }
        # Commit only after every service call/validation succeeds.
        cache.tokens = prefix
        return SimpleNamespace(logits=logits, hidden_states=states)

    def _forward_block(self, prefix, old_length, cache, output_hidden_states):
        # The same suffix used by reference verification, with one target request.
        # A fresh prefill only needs the final probability row but all prompt HS.
        logprobs, hidden = self._request_block(
            prefix,
            logits_start=old_length if old_length else len(prefix) - 1,
            hidden_start=old_length if output_hidden_states else len(prefix),
        )
        logits = logprobs.unsqueeze(0).to(self.device)
        states = None
        if output_hidden_states:
            states = {
                layer: hidden[:, slot, :].unsqueeze(0).to(self.device)
                for slot, layer in enumerate(self.layer_ids)
            }
        # No automatic fallback, and no cache commit before complete validation.
        cache.tokens = prefix
        return SimpleNamespace(logits=logits, hidden_states=states)
