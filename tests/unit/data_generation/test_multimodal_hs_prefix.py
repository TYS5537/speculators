"""Real HS/data methods with a mocked service, never a truncated image request."""

import ast
import asyncio
import bisect
import importlib.util
import json
import logging
import re
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from jinja2 import Template
from packaging.version import Version
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).parents[3]


def _module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _definitions(relative, names, namespace):
    source = ROOT / relative
    tree = ast.parse("from __future__ import annotations")
    tree.body.extend(
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    )
    exec(compile(tree, str(source), "exec"), namespace)  # noqa: S102


@pytest.fixture
def api(monkeypatch):
    # Import the real network client and CLI module. Only optional SDK/storage
    # imports are stubs; request construction and response validation run as-is.
    completion = type("Completion", (), {})
    chat_completion = type("ChatCompletion", (), {})
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(Client=object, AsyncClient=object, OpenAI=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai.types.chat",
        SimpleNamespace(
            ChatCompletion=chat_completion, ChatCompletionMessageParam=dict
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai.types.completion",
        SimpleNamespace(Completion=completion),
    )
    offline = _module("prefix_offline", "src/speculators/data_generation/offline.py")
    client = _module("prefix_client", "src/speculators/data_generation/vllm_client.py")
    namespace = {
        "torch": torch,
        "Dataset": torch.utils.data.Dataset,
        "warnings": warnings,
        "cast": cast,
        "DEFAULT_REQUEST_TIMEOUT": 120,
        "DEFAULT_MAX_RETRIES": 0,
        "check_hidden_states": offline.check_hidden_states,
        "align_hidden_states": offline.align_hidden_states,
        "InvalidResponseError": client.InvalidResponseError,
        "generate_hidden_states": client.generate_hidden_states,
    }
    _definitions(
        "src/speculators/train/data.py",
        {
            "BaseDataset",
            "ArrowDataset",
            "_has_multimodal_content",
            "build_client_item",
        },
        namespace,
    )
    data = ModuleType("prefix_data")
    data.build_client_item = namespace["build_client_item"]
    for name, module in (
        ("speculators.data_generation.offline", offline),
        ("speculators.data_generation.vllm_client", client),
        ("speculators.train.data", data),
        ("speculators.train.logger", SimpleNamespace(setup_root_logger=Mock())),
        ("datasets", SimpleNamespace(load_from_disk=Mock())),
        ("tqdm", SimpleNamespace(tqdm=object)),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    script = _module("prefix_cli", "scripts/data_generation_offline.py")
    return SimpleNamespace(
        offline=offline, client=client, data=namespace, script=script
    )


def _row():
    return {
        "input_ids": torch.tensor([10, 20]),
        "loss_mask": torch.tensor([0, 1]),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.invalid/image"},
                    },
                    {
                        "type": "text",
                        "text": "Complete original message, never shortened",
                    },
                ],
            }
        ],
    }


def _payload(tokens=(10, 20, 30, 40)):
    return {
        "token_ids": torch.tensor(tokens),
        "hidden_states": torch.arange(len(tokens) * 12, dtype=torch.float32).reshape(
            len(tokens), 3, 4
        ),
    }


def _service(api, handle, tokens, *, asynchronous=False, text=False):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        response = api.client.Completion() if text else api.client.ChatCompletion()
        if text:
            response.choices = [SimpleNamespace(prompt_token_ids=tokens)]
        else:
            response.prompt_token_ids = tokens
        response.kv_transfer_params = {"hidden_states_path": str(handle)}
        return response

    async def async_create(**kwargs):
        return create(**kwargs)

    endpoint = SimpleNamespace(create=async_create if asynchronous else create)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=endpoint), completions=endpoint
    ), calls


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("tokens", [[10, 20], [10, 20, 30, 40]])
def test_service_keeps_complete_media_and_accepts_only_right_extension(
    api, asynchronous, tokens
):
    service, calls = _service(api, "generated", tokens, asynchronous=asynchronous)
    item = api.data["build_client_item"](_row())
    if asynchronous:
        handle = asyncio.run(
            api.client.generate_hidden_states_async(
                service, "target", item, max_retries=0
            )
        )
    else:
        handle = api.client.generate_hidden_states(
            service, "target", item, max_retries=0
        )
    assert handle == "generated"
    assert calls[0]["messages"] is item["messages"]
    assert "prompt" not in calls[0]
    assert "truncate_prompt_tokens" not in calls[0]["extra_body"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("tokens", [[10], [20, 30], [10, 99, 30], [0, 10, 20]])
def test_response_mismatch_or_left_truncation_is_rejected(api, asynchronous, tokens):
    service, _ = _service(api, "generated", tokens, asynchronous=asynchronous)
    item = api.data["build_client_item"](_row())
    if asynchronous:
        with pytest.raises(api.client.InvalidResponseError, match="mismatch"):
            asyncio.run(
                api.client.generate_hidden_states_async(
                    service, "target", item, max_retries=0
                )
            )
    else:
        with pytest.raises(api.client.InvalidResponseError, match="mismatch"):
            api.client.generate_hidden_states(service, "target", item, max_retries=0)


def test_plain_text_still_requires_exact_response_length(api):
    service, _ = _service(api, "generated", [10, 20, 30], text=True)
    with pytest.raises(api.client.InvalidResponseError, match="mismatch"):
        api.client.generate_hidden_states(
            service, "target", {"input_ids": [10, 20]}, max_retries=0
        )


def _dataset(api, service, payload, *, cached=False):
    dataset = api.data["ArrowDataset"].__new__(api.data["ArrowDataset"])
    dataset.data = [_row()]
    dataset.client = service
    dataset.model = "target"
    dataset.request_timeout = None
    dataset.max_retries = 0
    dataset.start_file_idx = 0
    dataset.pretokenized_text_only = False
    dataset.on_missing = "generate"
    dataset.on_generate = "cache"
    dataset.transfer = SimpleNamespace(
        get_cached=Mock(return_value=payload if cached else None),
        get_generated=Mock(return_value=payload),
        cache=Mock(),
        delete=Mock(),
    )
    return dataset


@pytest.mark.parametrize("cached", [False, True])
def test_online_and_cached_states_share_exact_token_prefix(api, cached):
    full = _payload()
    service, calls = _service(api, "generated", full["token_ids"].tolist())
    dataset = _dataset(api, service, full, cached=cached)
    row = dataset._get_raw_data(0)
    assert row["input_ids"].tolist() == [10, 20]
    assert row["loss_mask"].tolist() == [0, 1]
    assert torch.equal(row["hidden_states"], full["hidden_states"][:2, :-1].flatten(1))
    assert torch.equal(
        row["verifier_last_hidden_states"], full["hidden_states"][:2, -1]
    )
    assert full["hidden_states"].shape == (4, 3, 4)
    assert len(calls) == int(not cached)


def test_online_full_response_cache_is_reused_on_the_next_read(api):
    full = _payload()
    service, calls = _service(api, "generated", full["token_ids"].tolist())
    dataset = _dataset(api, service, full)

    def cache(_handle, _index):
        # FileTransfer caches the original complete response, not the views.
        dataset.transfer.get_cached.return_value = full

    dataset.transfer.cache.side_effect = cache
    first = dataset._get_raw_data(0)
    second = dataset._get_raw_data(0)
    assert len(calls) == 1
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert second["input_ids"].tolist() == [10, 20]
    assert full["token_ids"].tolist() == [10, 20, 30, 40]


@pytest.mark.parametrize("failure", ["response", "states", "shape", "nan"])
def test_online_invalid_media_payload_raises_instead_of_silently_skipping(api, failure):
    full = _payload()
    service_tokens = full["token_ids"].tolist()
    if failure == "response":
        service_tokens[0] = 99
    elif failure == "states":
        full["token_ids"][0] = 99
    elif failure == "shape":
        full["hidden_states"] = full["hidden_states"][:-1]
    else:
        full["hidden_states"][-1, 0, 0] = torch.nan
    service, _ = _service(api, "generated", service_tokens)
    dataset = _dataset(api, service, full)
    with pytest.raises((ValueError, api.client.InvalidResponseError)):
        dataset._get_raw_data(0)
    dataset.transfer.cache.assert_not_called()


@pytest.mark.parametrize("validate_outputs", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_offline_worker_publishes_synchronously_sliced_states(
    api, tmp_path, validate_outputs, valid
):
    full = _payload()
    source = tmp_path / "response.safetensors"
    if not valid:
        full["token_ids"][0] = 99
    save_file(full, source)
    destination = tmp_path / "cache"
    destination.mkdir()
    service, calls = _service(api, source, [10, 20, 30, 40], asynchronous=True)
    skipped = []

    async def run():
        queue = asyncio.Queue()
        queue.put_nowait(api.data["build_client_item"](_row()) | {"idx": 0})
        queue.put_nowait(None)
        await api.script.worker(
            service,
            "target",
            queue,
            SimpleNamespace(update=Mock()),
            asyncio.Semaphore(1),
            asyncio.Semaphore(1),
            destination,
            validate_outputs,
            None,
            0,
            False,
            skipped,
            asyncio.Event(),
            None,
        )
        await queue.join()

    asyncio.run(run())
    assert calls[0]["messages"] == _row()["messages"]
    assert skipped == ([] if valid else [0])
    path = destination / "hs_0.safetensors"
    assert path.exists() == valid
    assert source.exists() != valid
    if valid:
        cached = load_file(path)
        assert cached["token_ids"].tolist() == [10, 20]
        assert torch.equal(cached["hidden_states"], full["hidden_states"][:2])
        assert api.offline.validate_existing_hidden_states(
            destination, [_row()], [0]
        ) == [0]


def test_existing_complete_media_cache_is_read_without_destructive_rewrite(
    api, tmp_path
):
    path = tmp_path / "hs_0.safetensors"
    save_file(_payload(), path)
    before = path.read_bytes()
    assert api.offline.validate_existing_hidden_states(tmp_path, [_row()], [0]) == [0]
    assert path.read_bytes() == before


def test_preprocessing_keeps_the_same_right_prefix_and_complete_messages():
    processor_type = type("ProcessorMixin", (), {})
    conversation = _row()["messages"]
    namespace = {
        "torch": torch,
        "ProcessorMixin": processor_type,
        "log": logging.getLogger("prefix-preprocessing"),
        "_normalize_conversation": lambda value: value,
        "_parse_conv_tools": lambda *_: None,
        "_get_input_ids_loss_mask": lambda *_a, **_kw: (
            [10, 20, 30, 40],
            torch.tensor([0, 1, 1, 1]),
        ),
        "_adapt_conv_for_vllm": lambda value: value,
    }
    _definitions(
        "src/speculators/data_generation/preprocessing.py",
        {"_preprocess_batch"},
        namespace,
    )
    output = namespace["_preprocess_batch"](
        {"conversations": [conversation]}, processor_type(), 2, None
    )
    assert output["input_ids"][0].tolist() == [10, 20]
    assert output["loss_mask"][0].tolist() == [0, 1]
    assert output["messages"][0] is conversation


class _TemplateProcessor:
    """Service-boundary double with parameter-sensitive Jinja rendering.

    Character IDs make the rendered prompt inspectable without downloading a
    tokenizer. Both ends run this same template; response IDs are never fixed.
    The final-message option deliberately removes the final end-of-turn token,
    matching the Transformers Chat Completions contract.
    """

    template = Template(
        "{% for message in messages %}<{{ message.role }}>"
        "{% if message.content is string %}{{ message.content }}{% else %}"
        "{% for part in message.content %}"
        "{% if part.type in ['image', 'image_url'] %}<image>"
        "{% else %}{{ part.text }}{% endif %}{% endfor %}{% endif %}"
        "{% if not (loop.last and continue_final_message) %}<eot>{% endif %}"
        "{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    )

    @classmethod
    def render(cls, messages, **kwargs):
        return cls.template.render(
            messages=messages,
            continue_final_message=kwargs.get("continue_final_message", False),
            add_generation_prompt=kwargs.get("add_generation_prompt", False),
        )

    def apply_chat_template(self, messages, **kwargs):
        text = self.render(messages, **kwargs)
        limit = kwargs.get("processor_kwargs", {}).get("max_length", len(text))
        text = text[:limit]
        return {
            "input_ids": [list(text.encode())],
            "offset_mapping": [[(i, i + 1) for i in range(len(text))]],
        }

    @staticmethod
    def decode(ids):
        return bytes(ids).decode()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("max_length", [256, 40], ids=["short", "right-prefix"])
@pytest.mark.parametrize("list_content", [False, True])
def test_template_end_markers_match_preprocessing(
    api, asynchronous, max_length, list_content
):
    namespace = {
        "torch": torch,
        "ProcessorMixin": _TemplateProcessor,
        "Version": Version,
        "TRANSFORMERS_VERSION": "5.13.0",
        "cast": cast,
        "re": re,
        "bisect": bisect,
        "json": json,
        "Path": Path,
        "log": logging.getLogger("template-parity"),
    }
    _definitions(
        "src/speculators/data_generation/preprocessing.py",
        {
            "_normalize_conversation",
            "_adapt_part_for_hf",
            "_adapt_turn_for_hf",
            "_adapt_conv_for_hf",
            "_adapt_part_for_vllm",
            "_adapt_turn_for_vllm",
            "_adapt_conv_for_vllm",
            "_thinking_template_kwargs",
            "_get_input_ids_loss_mask",
            "_create_loss_mask_from_offsets",
            "_parse_conv_tools",
            "_preprocess_batch",
        },
        namespace,
    )
    answer = "answer " * 12
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://example.invalid/image"},
                {"type": "text", "text": "Q"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}] if list_content else answer,
        },
    ]
    output = namespace["_preprocess_batch"](
        {"conversations": [conversation]},
        _TemplateProcessor(),
        max_length,
        r"<assistant>(.*?)<eot>",
    )
    item = api.data["build_client_item"](
        {key: values[0] for key, values in output.items()}
    )
    rendered = []

    def create(**kwargs):
        prompt = _TemplateProcessor.render(kwargs["messages"], **kwargs["extra_body"])
        rendered.append(prompt)
        response = api.client.ChatCompletion()
        response.prompt_token_ids = list(prompt.encode())
        response.kv_transfer_params = {"hidden_states_path": "rendered-response"}
        return response

    async def async_create(**kwargs):
        return create(**kwargs)

    service = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=async_create if asynchronous else create)
        )
    )
    if asynchronous:
        handle = asyncio.run(
            api.client.generate_hidden_states_async(
                service, "target", item, max_retries=0
            )
        )
    else:
        handle = api.client.generate_hidden_states(
            service, "target", item, max_retries=0
        )
    assert handle == "rendered-response"
    assert rendered[0].endswith("<eot>")
    assert item["input_ids"] == list(rendered[0].encode())[:max_length]
    assert (len(item["input_ids"]) == len(rendered[0])) == (max_length == 256)
    full = _payload(list(rendered[0].encode()))
    prefix = api.offline.align_hidden_states(full, item["input_ids"], allow_prefix=True)
    assert prefix["hidden_states"].shape[0] == len(item["input_ids"])
