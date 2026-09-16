"""Strict text-only training data encoding through the target's V4 renderer.

The vLLM 0.26.0 renderer/encoder is the authority, not a guessed chat template:
https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/tokenizers/deepseek_v4_encoding.py
Its default drops historical reasoning, so each assistant turn is a separate
sample. We compare that turn's prompt and completed token prefixes exactly;
we never splice masks from different renderings of a multi-turn conversation.
No checkpoint Python code is loaded, and no responses are generated.
"""

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from speculators_dsv4.contract import MANIFEST, inspect_checkpoint
from speculators_dsv4.tokenizer import DSV4ServerTokenizer

DATA_MANIFEST = "dspark_dsv4_data.json"
DATA_FORMAT = "dspark_dsv4_training_tokens_v1"
ENCODING = "vllm-0.26.0-deepseek_v4-per-assistant-prefix-v1"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
)


def tokenizer_fingerprint(model_path):
    """Hash declarative tokenizer assets only; never import remote model code."""
    root = Path(model_path).resolve(strict=True)
    if not (root / "tokenizer.json").is_file():
        raise ValueError("DSV4 preprocessing requires the checkpoint tokenizer.json")
    result = {}
    for name in TOKENIZER_FILES:
        path = root / name
        if path.is_file():
            # HF snapshots legitimately use symlinked immutable blob storage.
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def data_identity(report):
    return {
        "format": DATA_FORMAT,
        "model_path": report["model_path"],
        "checkpoint_signature": report["checkpoint_signature"],
        "tokenizer_files": tokenizer_fingerprint(report["model_path"]),
        "vocab_size": report["config"]["vocab_size"],
    }


def validate_data_manifest(path, report):
    """Fail closed on unproven/foreign token data, including a Qwen Arrow set."""
    path = Path(path)
    if path.is_dir():
        path /= DATA_MANIFEST
    if not path.is_file():
        raise ValueError(f"Missing DSV4 training data contract: {path}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in data_identity(report).items():
        if actual.get(key) != expected:
            raise ValueError(f"DSV4 training data identity mismatch: {key}")
    if actual.get("encoding") != ENCODING:
        raise ValueError("Unsupported DSV4 training data encoding contract")
    if type(actual.get("enable_thinking")) is not bool:
        raise ValueError("DSV4 data contract must record explicit thinking mode")
    if actual.get("mask_policy") != "current-assistant-continuation-including-eos":
        raise ValueError("Unsupported DSV4 loss mask contract")
    return actual


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ValueError("Refusing to redirect tokenizer requests or credentials")


class TokenizationClient:
    """Small bounded, no-proxy HTTP client; auth is read from the environment."""

    def __init__(self, endpoint, timeout=120):
        parts = urlsplit(endpoint)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
            or parts.path.rstrip("/")
        ):
            raise ValueError("DSV4 tokenizer endpoint must be the server root URL")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("DSV4 tokenizer timeout must be finite and positive")
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "VLLM_API_KEY"
        )

    def request(self, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(  # noqa: S310 -- Constructor restricts HTTP(S) roots.
            self.endpoint + path,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            payload = response.read(16 * 1024 * 1024 + 1)
        if len(payload) > 16 * 1024 * 1024:
            raise ValueError("DSV4 tokenizer response exceeds the size limit")
        result = json.loads(payload)
        if not isinstance(result, dict):
            raise ValueError("DSV4 tokenizer response must be a JSON object")
        return result

    def post(self, path, *, cast_to, body):
        if cast_to is not dict:
            raise TypeError("TokenizationClient only supports dictionary responses")
        return self.request(path, body)


def _normalize_messages(value):  # noqa: C901
    if not isinstance(value, list) or not value:
        raise ValueError("DSV4 requires non-empty messages/conversations")
    messages = []
    expect = "user"
    allowed = {
        "role",
        "from",
        "content",
        "value",
        "reasoning",
        "reasoning_content",
        "thinking",
    }
    for index, turn in enumerate(value):
        if not isinstance(turn, dict):
            raise ValueError("Each DSV4 conversation turn must be an object")
        if any(value is not None for key, value in turn.items() if key not in allowed):
            raise ValueError(
                "DSV4 training supports plain text only, "
                "without tools or extra turn fields"
            )
        role = turn.get("role") or turn.get("from")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        if not (role == "system" and index == 0):
            if role != expect:
                raise ValueError(
                    "DSV4 expects optional initial system, "
                    "then alternating user/assistant turns"
                )
            expect = "assistant" if role == "user" else "user"
        content = turn.get("content")
        if content is None:
            content = turn.get("value") or ""
        if not isinstance(content, str):
            raise ValueError("DSV4 preprocessing does not support multimodal content")
        reasoning_fields = [
            turn[key]
            for key in ("reasoning", "reasoning_content", "thinking")
            if turn.get(key) is not None
        ]
        if any(not isinstance(item, str) for item in reasoning_fields):
            raise ValueError("DSV4 reasoning must be a string")
        if len(set(reasoning_fields)) > 1:
            raise ValueError("Conflicting DSV4 reasoning/thinking fields")
        reasoning = reasoning_fields[0] if reasoning_fields else ""
        if reasoning and role != "assistant":
            raise ValueError("Only assistant turns may carry reasoning")
        # Inline reasoning is ambiguous (a quote/code block may contain tags).
        # Demand structured reasoning rather than silently double-wrapping it.
        if role == "assistant" and ("<think>" in content or "</think>" in content):
            raise ValueError("Move assistant thinking into a separate reasoning field")
        messages.append(
            {
                "role": role,
                "content": content,
                **({"reasoning": reasoning} if reasoning else {}),
            }
        )
    if not any(turn["role"] == "assistant" for turn in messages):
        raise ValueError("DSV4 training requires at least one assistant response")
    return messages


class DSV4TrainingEncoder:
    def __init__(self, tokenizer, *, enable_thinking=False):
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    def encode(self, conversation):
        messages = _normalize_messages(conversation)
        if not self.enable_thinking and any(turn.get("reasoning") for turn in messages):
            raise ValueError("Reasoning would be discarded; specify --enable-thinking")
        rows = []
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            options = {
                "enable_thinking": self.enable_thinking,
                "thinking": self.enable_thinking,
                "drop_thinking": True,
                "reasoning_effort": None,
            }
            prefix = self.tokenizer.apply_chat_template(messages[:index], **options)
            completed = self.tokenizer.apply_chat_template(
                messages[: index + 1], add_generation_prompt=False, **options
            )
            if len(completed) <= len(prefix) or completed[: len(prefix)] != prefix:
                raise ValueError(
                    "DSV4 renderer changed the prompt token prefix; "
                    "refusing an unsafe loss mask"
                )
            eos = self.tokenizer.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            if completed[-1] not in eos:
                raise ValueError(
                    "DSV4 completed assistant turn does not end in checkpoint EOS"
                )
            rows.append(
                {
                    "input_ids": completed,
                    "loss_mask": [0] * len(prefix)
                    + [1] * (len(completed) - len(prefix)),
                    "seq_len": len(completed),
                }
            )
        return rows


def connect_encoder(args, report):
    """Validate the intended server/checkpoint/renderer before sending user data."""
    if (
        not args.dsv4_tokenizer_endpoint
        or not args.dsv4_served_model_name
        or not args.dsv4_hs_manifest
    ):
        raise ValueError(
            "Raw DSV4 data needs --dsv4-tokenizer-endpoint, "
            "--dsv4-served-model-name and --dsv4-hs-manifest"
        )
    manifest_path = Path(args.dsv4_hs_manifest)
    if manifest_path.is_dir():
        manifest_path /= MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("model_path", "checkpoint_signature"):
        if manifest.get(key) != report[key]:
            raise ValueError(f"DSV4 target HS manifest mismatch: {key}")
    client = TokenizationClient(
        args.dsv4_tokenizer_endpoint, args.dsv4_tokenizer_timeout
    )
    if client.request("/version").get("version", "").split("+")[0] != "0.26.0":
        raise ValueError("DSV4 data encoding requires vLLM 0.26.0")
    models = client.request("/v1/models").get("data", [])
    model = next(
        (item for item in models if item.get("id") == args.dsv4_served_model_name), None
    )
    if model is None or model.get("root") != report["model_path"]:
        raise ValueError(
            "DSV4 tokenizer server does not serve the requested checkpoint path"
        )
    info = client.request("/tokenizer_info")
    tokenizer_class = str(info.get("tokenizer_class", ""))
    # vLLM wraps its tokenizer using these two official class-name prefixes.
    tokenizer_class = tokenizer_class.removeprefix("TokenizerPool").removeprefix(
        "Cached"
    )
    if not tokenizer_class.startswith("DSV4"):
        raise ValueError(
            "Enable --tokenizer-mode deepseek_v4 and "
            "--enable-tokenizer-info-endpoint on the target"
        )
    eos = report["config"].get("eos_token_id")
    if eos is None:
        raise ValueError("Checkpoint must declare eos_token_id")
    tokenizer = DSV4ServerTokenizer(
        client,
        args.dsv4_served_model_name,
        eos,
        vocab_size=report["config"]["vocab_size"],
    )
    # A declarative tokenizer.json can be safely read by the Rust tokenizer.
    # Compare multilingual/special-token probes to reject an overridden tokenizer.
    from tokenizers import Tokenizer  # noqa: PLC0415

    local = Tokenizer.from_file(str(Path(report["model_path"]) / "tokenizer.json"))
    local.no_truncation()
    local.no_padding()
    probe = (
        "DSV4 tokenizer identity: 中文 English 42\n"
        "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜>"
        "<think></think><｜end▁of▁sentence｜>"
    )
    if (
        tokenizer._tokenize(  # noqa: SLF001 -- Shared validated raw-token protocol.
            {
                "model": args.dsv4_served_model_name,
                "prompt": probe,
                "add_special_tokens": False,
            }
        )
        != local.encode(probe, add_special_tokens=False).ids
    ):
        raise ValueError("DSV4 server tokenizer differs from checkpoint tokenizer.json")
    return DSV4TrainingEncoder(tokenizer, enable_thinking=bool(args.enable_thinking))


def _validated_row(row, vocab_size, max_length):
    ids, mask = row["input_ids"], row["loss_mask"]
    if (
        not isinstance(ids, list)
        or not isinstance(mask, list)
        or not ids
        or len(ids) != len(mask)
    ):
        raise ValueError(
            "DSV4 pre-tokenized rows need equally sized, "
            "non-empty input_ids/loss_mask lists"
        )
    if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
        raise ValueError("DSV4 row contains invalid/out-of-vocabulary token IDs")
    if any(type(bit) is not int or bit not in (0, 1) for bit in mask):
        raise ValueError("DSV4 loss_mask must contain only integer 0/1")
    if mask[0] != 0:
        raise ValueError(
            "DSV4 loss_mask must mask the first token (no causal predecessor)"
        )
    if 1 in mask and 0 in mask[mask.index(1) :]:
        raise ValueError("DSV4 per-assistant data requires one trainable suffix")
    ids, mask = ids[:max_length], mask[:max_length]
    return {"input_ids": ids, "loss_mask": mask, "seq_len": len(ids)}


def prepare_dsv4_dataset(args, token_freq_path):  # noqa: C901
    """Return a training-ready Dataset plus its explicit encoding contract."""
    from datasets import Features, List, Value, concatenate_datasets  # noqa: PLC0415

    from speculators.data_generation.preprocessing import (  # noqa: PLC0415
        load_raw_dataset,
    )
    from speculators.train.vocab_mapping import (  # noqa: PLC0415
        save_token_frequency_distribution,
    )

    if args.trust_remote_code or args.assistant_pattern:
        raise ValueError(
            "DSV4 preprocessing does not use remote code or assistant regexes"
        )
    if args.seq_length <= 1 or (args.max_samples is not None and args.max_samples <= 0):
        raise ValueError("Use seq-length >= 2 and a positive max-samples")
    if args.minimum_valid_tokens is not None and args.minimum_valid_tokens < 0:
        raise ValueError("minimum-valid-tokens must be nonnegative")
    report = inspect_checkpoint(args.model)
    identity = data_identity(report)
    encoder = None
    source_contract = None
    thinking_modes = set()
    processed = []
    features = Features(
        {
            "input_ids": List(Value("int64")),
            "loss_mask": List(Value("int64")),
            "seq_len": Value("int64"),
        }
    )
    for source in args.data:
        raw, normalize = load_raw_dataset(source)
        raw = raw.shuffle(seed=args.seed)
        if args.max_samples is not None:
            raw = raw.select(range(min(len(raw), 3 * args.max_samples)))
        if normalize is not None:
            raw = raw.map(
                normalize, num_proc=args.num_preprocessing_workers, keep_in_memory=True
            )
        pretokenized = {"input_ids", "loss_mask"} <= set(raw.column_names)
        if pretokenized:
            path = args.dsv4_source_manifest or Path(source).parent / DATA_MANIFEST
            if Path(source).is_dir():
                path = args.dsv4_source_manifest or Path(source) / DATA_MANIFEST
            source_contract = validate_data_manifest(path, report)
            thinking_modes.add(source_contract["enable_thinking"])
            if (
                args.enable_thinking is not None
                and source_contract["enable_thinking"] != args.enable_thinking
            ):
                raise ValueError(
                    "Requested thinking mode disagrees with pre-tokenized DSV4 data"
                )
        elif encoder is None:
            encoder = connect_encoder(args, report)
        if not pretokenized:
            thinking_modes.add(encoder.enable_thinking)
        if len(thinking_modes) != 1:
            raise ValueError("Cannot mix DSV4 datasets with different thinking modes")

        def encode_batch(batch, pretokenized=pretokenized, encoder=encoder):
            result = {"input_ids": [], "loss_mask": [], "seq_len": []}
            for values in zip(*batch.values(), strict=True):
                example = dict(zip(batch, values, strict=True))
                if (
                    example.get("tools")
                    or example.get("images")
                    or example.get("videos")
                    or example.get("audio")
                ):
                    raise ValueError(
                        "DSV4 training preprocessing supports text-only "
                        "conversations without tools"
                    )
                if pretokenized:
                    rows = [example]
                else:
                    conversation = example.get("messages") or example.get(
                        "conversations"
                    )
                    rows = encoder.encode(conversation)
                for raw_row in rows:
                    row = _validated_row(
                        raw_row, identity["vocab_size"], args.seq_length
                    )
                    # A fully truncated answer provides no training signal.
                    if sum(row["loss_mask"]) < max(1, args.minimum_valid_tokens or 0):
                        continue
                    for key, value in row.items():
                        result[key].append(value)
            return result

        processed.append(
            raw.map(
                encode_batch,
                batched=True,
                batch_size=64,
                remove_columns=raw.column_names,
                features=features,
                keep_in_memory=True,
                new_fingerprint=uuid.uuid4().hex,
            )
        )
    dataset = concatenate_datasets(processed).shuffle(seed=args.seed)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))
    if not len(dataset) and not args.allow_empty_output:
        raise ValueError("No samples remain after DSV4 preprocessing")
    dataset.set_format(type="torch")
    if Path(token_freq_path).exists():
        raise ValueError(
            "DSV4 token frequency file already exists; "
            "use a fresh path to avoid stale counts"
        )
    save_token_frequency_distribution(dataset, token_freq_path)
    thinking = next(iter(thinking_modes))
    metadata = {
        **identity,
        "encoding": ENCODING,
        "mask_policy": "current-assistant-continuation-including-eos",
        "enable_thinking": thinking,
        "drop_historical_thinking": True,
        "max_length": args.seq_length,
        "seed": args.seed,
        "row_count": len(dataset),
        "input_sources": list(args.data),
        "pretokenized_source": source_contract,
    }
    return dataset, metadata
