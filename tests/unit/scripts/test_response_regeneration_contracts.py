"""Compatibility-script ordering, payload, retry and failure contracts; no network."""

import asyncio
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from speculators.data_generation import vllm_client


@pytest.fixture
def regen():
    path = Path(__file__).parents[3] / "scripts/response_regeneration/script.py"
    spec = importlib.util.spec_from_file_location("regen_contract_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call(name="lookup", call_id="generated-call"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _response(index=0, *, tool_calls=None, content="answer"):
    return {
        "prompt_token_ids": [10 + index, 20 + index],
        "choices": [
            {
                "message": {"content": content, "tool_calls": tool_calls},
                "token_ids": [30 + index, 40 + index],
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"completion_tokens": 2},
    }


def _item(**overrides):
    return {
        "idx": 31,
        "primary_id": "conversation",
        "turns": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        **overrides,
    }


def _run(regen, item, responses, *, samples=None, params=None, thinking=None):
    samples = [] if samples is None else samples
    sent = []
    pending = iter(responses)

    async def post(payload):
        sent.append(copy.deepcopy(payload))
        value = next(pending)
        if isinstance(value, BaseException):
            raise value
        return value

    async def run():
        return await regen.regenerate_conversation(
            post,
            item,
            model="target",
            max_tokens=64,
            endpoint="fixture-endpoint",
            sampling_params={} if params is None else params,
            samples=samples,
            enable_thinking=thinking,
        )

    return SimpleNamespace(run=run, sent=sent, samples=samples)


@pytest.mark.parametrize("thinking", [None, False, True])
@pytest.mark.parametrize(
    "template", ["absent", None, {"custom": 7, "enable_thinking": "old"}, ["invalid"]]
)
def test_sampling_merge_keeps_owned_fields_metadata_and_input_immutability(
    regen, thinking, template
):
    params = {
        "model": "ignored-model",
        "messages": ["ignored-message"],
        "max_tokens": 1,
        "return_token_ids": False,
        "temperature": 0.6,
    }
    if template != "absent":
        params["chat_template_kwargs"] = copy.deepcopy(template)
    original = copy.deepcopy(params)
    existing = {"already": "committed"}
    case = _run(
        regen,
        _item(),
        [_response()],
        samples=[existing],
        params=params,
        thinking=thinking,
    )

    if isinstance(template, list) and thinking is not None:
        with pytest.raises(
            ValueError, match="chat_template_kwargs must be a JSON object"
        ):
            asyncio.run(case.run())
        assert case.sent == []
        assert case.samples == [existing]
    else:
        assert asyncio.run(case.run()) is False
        payload = case.sent[0]
        assert payload["model"] == "target"
        assert payload["messages"] == _item()["turns"]
        assert payload["max_tokens"] == 64
        assert payload["return_token_ids"] is True
        expected = copy.deepcopy(original)
        if thinking is not None:
            expected["chat_template_kwargs"] = {
                **(expected.get("chat_template_kwargs") or {}),
                "enable_thinking": thinking,
            }
        sample = case.samples[1]
        assert case.samples[0] is existing
        assert sample["id"] == "conversation_gen1"
        assert sample["metadata"]["sampling_params"] == expected
        assert sample["metadata"]["sampling_params"] is not params
        assert payload.get("chat_template_kwargs") == expected.get(
            "chat_template_kwargs"
        )
    assert params == original


@pytest.mark.parametrize(
    "row_tools", [None, [], [{"type": "function", "function": {"name": "lookup"}}]]
)
def test_row_tools_override_sampling_options_only_when_present(regen, row_tools):
    params = {"tools": ["sampling-tools"], "tool_choice": "none"}
    case = _run(regen, _item(tools=row_tools), [_response()], params=params)

    assert asyncio.run(case.run()) is False

    assert case.sent[0]["tools"] == (row_tools or params["tools"])
    assert case.sent[0]["tool_choice"] == ("auto" if row_tools else "none")
    assert case.samples[0]["metadata"]["sampling_params"] == params


def test_tool_chain_spans_turns_and_preserves_boundary_snapshots(regen):
    turns = _item()["turns"] + [
        {"role": "system", "content": "later-system"},
        {"role": "user", "content": "second-question"},
    ]
    item = _item(
        turns=turns,
        tool_results=[
            ("first-result", ["lookup"]),
            ({"raw": "second-result"}, []),
            (None, ["finish"]),
        ],
    )
    original = copy.deepcopy(item)
    calls = [_call(), _call(call_id=None), _call("finish", call_id="")]
    case = _run(
        regen,
        item,
        [
            _response(0, tool_calls=[calls[0]], content=None),
            _response(1, tool_calls=[calls[1]], content="thinking"),
            _response(2),
            _response(3, tool_calls=[calls[2]], content=""),
            _response(4),
        ],
    )

    assert asyncio.run(case.run()) is False

    assert len(case.sent) == len(case.samples) == 5
    assert [len(sample["conversations"]) for sample in case.samples] == [
        3,
        5,
        7,
        10,
        12,
    ]
    assert case.sent[1]["messages"][-1] == {
        "role": "tool",
        "content": "first-result",
        "tool_call_id": "generated-call",
    }
    assert case.sent[2]["messages"][-1] == {
        "role": "tool",
        "content": {"raw": "second-result"},
    }
    assert case.sent[3]["messages"][-2:] == turns[-2:]
    assert case.sent[4]["messages"][-1] == {"role": "tool", "content": None}
    assert case.samples[0]["conversations"][-1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [calls[0]],
    }
    for index, sample in enumerate(case.samples):
        assert sample["id"] == f"conversation_gen{index}"
        assert sample["primary_id"] == "conversation"
        assert sample["input_ids"] == [10 + index, 20 + index, 30 + index, 40 + index]
        assert sample["loss_mask"] == [0, 0, 1, 1]
        assert sample["metadata"]["idx"] == 31
        assert sample["metadata"]["endpoint"] == "fixture-endpoint"
    assert item == original


@pytest.mark.parametrize(
    ("calls", "results"),
    [
        ([_call()], []),
        ([_call(), _call()], [("unused", [])]),
        ([_call("different")], [("unused", ["lookup"])]),
        ([{"function": None}], [("unused", ["lookup"])]),
    ],
)
def test_unpairable_calls_commit_the_call_then_truncate_all_remaining_turns(
    regen, calls, results
):
    item = _item(
        turns=[
            {"role": "user", "content": "first"},
            {"role": "user", "content": "must-not-run"},
        ],
        tool_results=results,
    )
    original = copy.deepcopy(item)
    case = _run(regen, item, [_response(tool_calls=calls, content=None)])

    assert asyncio.run(case.run()) is True

    assert len(case.sent) == len(case.samples) == 1
    assert case.samples[0]["metadata"]["is_tool_call"] is True
    assert case.samples[0]["conversations"][-1]["tool_calls"] == calls
    assert item == original


@pytest.mark.parametrize("turns", [[], [{"role": "system", "content": "only-system"}]])
def test_empty_or_system_only_conversation_never_posts(regen, turns):
    case = _run(regen, _item(turns=turns), [])
    assert asyncio.run(case.run()) is False
    assert case.sent == case.samples == []


@pytest.mark.parametrize("after_commit", [False, True])
@pytest.mark.parametrize(
    "failure_kind", ["network", "permanent", "cancel", "empty", "tokens", "schema"]
)
def test_failed_generation_preserves_partial_samples_without_conversation_retry(
    regen, after_commit, failure_kind
):
    failures = {
        "network": RuntimeError("network failure"),
        "permanent": vllm_client.InvalidResponseError("permanent failure"),
        "cancel": asyncio.CancelledError("cancelled request"),
        "empty": _response(content=""),
        "tokens": {**_response(), "prompt_token_ids": []},
        "schema": {"choices": []},
    }
    failure = failures[failure_kind]
    responses = [_response(tool_calls=[_call()], content=None)] if after_commit else []
    responses.append(failure)
    case = _run(regen, _item(tool_results=[("cached", ["lookup"])]), responses)
    expected = (
        type(failure)
        if isinstance(failure, BaseException)
        else (IndexError if failure_kind == "schema" else ValueError)
    )

    with pytest.raises(expected) as caught:
        asyncio.run(case.run())

    if isinstance(failure, BaseException):
        assert caught.value is failure
    assert len(case.samples) == int(after_commit)
    assert len(case.sent) == int(after_commit) + 1
    if after_commit:
        assert case.samples[0]["id"] == "conversation_gen0"
        assert case.samples[0]["conversations"][-1]["tool_calls"] == [_call()]
        assert case.sent[-1]["messages"][-1]["content"] == "cached"


class _Session:
    """Fake HTTP context managers; every attempt advances a deterministic clock."""

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.now = 0.0
        self.sent = []
        self.backoffs = []

    async def sleep(self, delay):
        self.backoffs.append(delay)
        self.now += delay

    def post(self, endpoint, json):
        self.now += 1.0
        self.sent.append(copy.deepcopy(json))
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return _Response(outcome)


class _Response:
    def __init__(self, outcome):
        self.ok = isinstance(outcome, dict)
        self.status = 200 if self.ok else outcome
        self.payload = outcome

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return "fixture HTTP failure"

    async def json(self):
        return self.payload


def _worker_fixture(regen, monkeypatch, outcomes, *, results=None, retries=2):
    session = _Session(outcomes)
    monkeypatch.setattr(vllm_client, "RETRY_BACKOFF_BASE", 2)
    monkeypatch.setattr(vllm_client.asyncio, "sleep", session.sleep)
    monkeypatch.setattr(
        regen, "time", SimpleNamespace(perf_counter=lambda: session.now)
    )
    stats = {
        "ok": 0,
        "errors": 0,
        "truncated": 0,
        "requests": 0,
        "completion_tokens": 0,
        "total_request_s": 0.0,
        "start_time": 0.0,
    }
    queue = asyncio.Queue()
    queue.put_nowait(
        _item(tool_results=[("cached", ["lookup"])] if results is None else results)
    )
    output, errors, progress = io.StringIO(), io.StringIO(), Mock()
    args = SimpleNamespace(
        model="target",
        max_tokens=64,
        max_retries=retries,
        sampling_params={"temperature": 0.6},
        enable_thinking=False,
    )

    async def run(*, cancellation=False):
        if not cancellation:
            queue.put_nowait(None)
        try:
            await regen.worker(
                session,
                queue,
                args,
                output,
                errors,
                "fixture-endpoint",
                progress,
                stats,
            )
        finally:
            # A finally/task_done regression must fail promptly rather than hang.
            await asyncio.wait_for(queue.join(), timeout=1.0)

    return SimpleNamespace(
        run=run,
        session=session,
        stats=stats,
        output=output,
        errors=errors,
        progress=progress,
    )


@pytest.mark.parametrize(
    (
        "mode",
        "attempts",
        "requests",
        "latency",
        "backoffs",
        "ok",
        "truncated",
        "completed",
    ),
    [
        ("retry-success", 4, 2, 10.0, [2, 4], 1, 0, 2),
        ("retry-exhausted", 3, 0, 0.0, [2, 4], 0, 0, 0),
        ("permanent-after-call", 2, 1, 1.0, [], 0, 0, 1),
        ("invalid-generation", 2, 2, 2.0, [], 0, 0, 1),
        ("truncated", 1, 1, 1.0, [], 1, 1, 1),
        ("connection-retry", 3, 2, 5.0, [2], 1, 0, 2),
    ],
)
def test_worker_retries_only_requests_and_writes_only_completed_conversations(
    regen,
    monkeypatch,
    mode,
    attempts,
    requests,
    latency,
    backoffs,
    ok,
    truncated,
    completed,
):
    call, final = _response(tool_calls=[_call()], content=None), _response(1)
    outcomes = {
        "retry-success": [503, 429, call, final],
        "retry-exhausted": [503, 503, 503],
        "permanent-after-call": [call, 404],
        "invalid-generation": [call, _response(content="")],
        "truncated": [call],
        "connection-retry": [ConnectionError("connection failed"), call, final],
    }[mode]
    case = _worker_fixture(
        regen, monkeypatch, outcomes, results=[] if truncated else None
    )

    asyncio.run(case.run())

    assert len(case.session.sent) == attempts
    assert case.session.backoffs == backoffs
    assert case.stats["requests"] == requests
    assert case.stats["total_request_s"] == latency
    assert case.stats["completion_tokens"] == (2 * completed if ok else 0)
    assert case.stats["ok"] == ok
    assert case.stats["truncated"] == truncated
    assert case.stats["errors"] == 1 - ok
    case.progress.update.assert_called_once_with(1)
    case.progress.set_postfix.assert_called_once()
    rows = [json.loads(line) for line in case.output.getvalue().splitlines()]
    if ok:
        assert len(rows) == completed
        assert [row["id"] for row in rows] == [
            f"conversation_gen{i}" for i in range(completed)
        ]
        assert case.errors.getvalue() == ""
    else:
        assert rows == []
        error = json.loads(case.errors.getvalue())
        assert error["metadata"]["generations_completed"] == completed
        assert error["id"] == "conversation"
    assert all(
        payload["chat_template_kwargs"] == {"enable_thinking": False}
        for payload in case.session.sent
    )
    if backoffs:
        assert case.session.sent[0] == case.session.sent[1]


@pytest.mark.parametrize("after_commit", [False, True])
def test_worker_cancellation_propagates_without_retry_or_output_but_finishes_queue_item(
    regen, monkeypatch, after_commit
):
    cancellation = asyncio.CancelledError("cancel worker")
    outcomes = [_response(tool_calls=[_call()], content=None)] if after_commit else []
    outcomes.append(cancellation)
    case = _worker_fixture(regen, monkeypatch, outcomes)

    with pytest.raises(asyncio.CancelledError) as caught:
        asyncio.run(case.run(cancellation=True))

    assert caught.value is cancellation
    assert case.output.getvalue() == case.errors.getvalue() == ""
    assert len(case.session.sent) == int(after_commit) + 1
    assert case.session.backoffs == []
    assert case.stats["errors"] == case.stats["ok"] == case.stats["truncated"] == 0
    assert case.stats["requests"] == int(after_commit)
    case.progress.update.assert_called_once_with(1)
