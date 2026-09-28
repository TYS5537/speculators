"""Content-keyed, immutable greedy traces; never cache speculative candidates."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file, save_file

SCHEMA_VERSION = 1


def trace_identity(target, prompt, max_new_tokens, stop_ids, tag):
    manifest = dict(target.manifest)
    manifest.pop("model_path", None)  # Same weights may live on another host.
    return {
        "schema_version": SCHEMA_VERSION,
        "target": manifest,
        "layer_ids": target.layer_ids,
        "vocab_size": target.vocab_size,
        "hidden_size": target.hidden_size,
        "max_model_len": target.max_model_len,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "stop_token_ids": sorted(set(stop_ids or [])),
        "sampling": "greedy-neutral-ignore-eos-then-trim-v1",
        "extraction": "native-block-argmax-checked-v1",
        "cache_tag": tag,
    }


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_trace(tensors, identity):
    if set(tensors) != {"tokens", "hidden_states"}:
        raise ValueError("Replay trace has missing/extra tensors")
    tokens, hidden = tensors["tokens"], tensors["hidden_states"]
    prompt = identity["prompt"]
    if (
        tokens.dtype != torch.int64
        or tokens.ndim != 1
        or not len(prompt) < len(tokens) <= len(prompt) + identity["max_new_tokens"]
        or tokens[: len(prompt)].tolist() != prompt
        or not ((tokens >= 0) & (tokens < identity["vocab_size"])).all()
    ):
        raise ValueError("Replay trace token identity/length/range mismatch")
    generated = tokens[len(prompt) :].tolist()
    stops = identity["stop_token_ids"]
    if any(token in stops for token in generated[:-1]) or (
        len(generated) < identity["max_new_tokens"] and generated[-1] not in stops
    ):
        raise ValueError("Replay trace ended without EOS or contains post-EOS tokens")
    expected = (len(tokens) - 1, len(identity["layer_ids"]), identity["hidden_size"])
    if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != expected:
        raise ValueError("Replay trace hidden-state shape/dtype mismatch")
    if not torch.isfinite(hidden).all():
        raise ValueError("Replay trace contains nonfinite hidden states")
    return tensors


class GreedyTraceCache:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    def path(self, identity):
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return self.directory / hashlib.sha256(encoded).hexdigest()

    def load(self, identity):
        directory = self.path(identity)
        if not directory.exists():
            return None
        metadata = json.loads((directory / "trace.json").read_text(encoding="utf-8"))
        path = directory / "trace.safetensors"
        if metadata.get("identity") != identity or metadata.get("sha256") != _digest(
            path
        ):
            raise ValueError(f"Replay cache identity/checksum mismatch: {directory}")
        return validate_trace(load_file(str(path), device="cpu"), identity)

    def publish(self, identity, tensors):
        validate_trace(tensors, identity)
        self.directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".trace-", dir=self.directory) as temporary:
            directory = Path(temporary)
            path = directory / "trace.safetensors"
            save_file(
                {name: value.contiguous() for name, value in tensors.items()}, path
            )
            (directory / "trace.json").write_text(
                json.dumps({"identity": identity, "sha256": _digest(path)}, indent=2),
                encoding="utf-8",
            )
            try:
                directory.rename(self.path(identity))
            except OSError:
                # Another worker may have completed the same immutable prompt.
                # Never overwrite it, and never hide an unrelated IO failure.
                existing = self.load(identity)
                if existing is None:
                    raise
                if not torch.equal(existing["tokens"], tensors["tokens"]):
                    raise ValueError(
                        "Concurrent greedy generations disagreed"
                    ) from None
                return existing
        return tensors
