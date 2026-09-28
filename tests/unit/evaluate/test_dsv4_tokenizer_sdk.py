"""Exercise the real SDK response parser without a server, model, or API key."""

# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.

import json

import httpx
import openai
import pytest

from speculators_dsv4.tokenizer import DSV4ServerTokenizer


@pytest.fixture
def sdk_tokenizer():
    requests = []
    payload = {"tokens": [1, 5, 9], "count": 3}

    def reply(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    with openai.OpenAI(
        api_key="fixture-key",
        base_url="http://target.invalid/",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(reply), trust_env=False),
    ) as client:
        yield (
            DSV4ServerTokenizer(client, "fixture", 2, vocab_size=128),
            requests,
            payload,
        )


@pytest.mark.parametrize("thinking", [False, True])
def test_successful_tokenize_response_is_parsed_by_real_sdk(sdk_tokenizer, thinking):
    tokenizer, requests, _ = sdk_tokenizer
    messages = [{"role": "user", "content": "hello"}]

    assert tokenizer.apply_chat_template(messages, enable_thinking=thinking) == [
        1,
        5,
        9,
    ]

    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "http://target.invalid/tokenize"
    assert json.loads(requests[0].content) == {
        "model": "fixture",
        "messages": messages,
        "add_generation_prompt": True,
        "add_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {},
        {"tokens": []},
        {"tokens": [True]},
        {"tokens": [128]},
        {"tokens": [1], "count": 2},
        {"tokens": [1], "count": True},
    ],
)
def test_sdk_preserves_values_for_token_validation(sdk_tokenizer, invalid_payload):
    tokenizer, requests, payload = sdk_tokenizer
    payload.clear()
    payload.update(invalid_payload)

    with pytest.raises(ValueError, match="DSV4"):
        tokenizer.apply_chat_template([{"role": "user", "content": "hello"}])

    assert len(requests) == 1
