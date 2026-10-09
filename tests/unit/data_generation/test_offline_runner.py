"""Both offline entrypoint policies retain their request and cache contracts."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from speculators.data_generation import offline_runner as runner


def _payload():
    return {
        "token_ids": torch.tensor([1, 2]),
        "hidden_states": torch.ones(2, 3, 4, dtype=torch.bfloat16),
    }


def _row(*, media=False, continuation=None):
    row = {"input_ids": torch.tensor([1, 2])}
    if media:
        row["messages"] = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    if continuation is not None:
        row["continue_final_message"] = continuation
    return row


def _args(directory, **overrides):
    return SimpleNamespace(
        **(
            {
                "model": "target",
                "endpoint": "http://fixture/v1",
                "preprocessed_data": str(directory),
                "output": str(directory / "cache"),
                "max_samples": None,
                "concurrency": 2,
                "validate_outputs": True,
                "request_timeout": 23,
                "max_retries": 3,
                "fail_on_error": False,
                "max_consecutive_errors": None,
                "world_size": 1,
                "rank": 0,
            }
            | overrides
        )
    )


class _Progress:
    def __init__(self, **_kwargs):
        self.count = 0
        self.postfixes = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def update(self, count):
        self.count += count

    def set_postfix(self, postfix, **_kwargs):
        self.postfixes.append(postfix)


def _service(monkeypatch, model_ids=("target",)):
    models = SimpleNamespace(
        list=AsyncMock(
            return_value=SimpleNamespace(
                data=[SimpleNamespace(id=name) for name in model_ids]
            )
        )
    )

    class Client:
        async def __aenter__(self):
            return SimpleNamespace(models=models)

        async def __aexit__(self, *_args):
            return False

    factory = Mock(return_value=Client())
    monkeypatch.setattr(runner.openai, "AsyncOpenAI", factory)
    progress = _Progress()
    monkeypatch.setattr(runner, "tqdm", Mock(return_value=progress))
    return factory, progress


@pytest.mark.parametrize("legacy", [True, False])
def test_validated_text_retains_each_entrypoints_publication_policy(tmp_path, legacy):
    source, target = tmp_path / "server.safetensors", tmp_path / "hs_0.safetensors"
    payload = _payload()
    save_file(payload, source, metadata={"fixture": "original server artifact"})
    original = source.read_bytes()
    runner.validate_and_publish(
        str(source), target, {"input_ids": [1, 2]}, True, legacy=legacy
    )
    assert not source.exists()
    assert all(
        torch.equal(load_file(target)[key], tensor) for key, tensor in payload.items()
    )
    with safe_open(target, framework="pt") as artifact:
        assert artifact.metadata() == (
            {"fixture": "original server artifact"} if legacy else None
        )
    if legacy:
        assert target.read_bytes() == original


@pytest.mark.parametrize("legacy", [True, False])
def test_resume_rank_partition_message_flags_and_sdk_retries(
    tmp_path, monkeypatch, legacy
):
    args = _args(tmp_path, world_size=2)
    cache = tmp_path / "cache"
    cache.mkdir()
    save_file(_payload(), cache / "hs_0.safetensors")
    preserved = (cache / "hs_0.safetensors").read_bytes()
    (cache / "hs_1.safetensors").write_bytes(b"bad local artifact")
    (cache / "hs_4.safetensors").write_bytes(b"other rank")
    dataset = [_row(), _row(media=True), _row(media=True, continuation=True)] * 2
    factory, progress = _service(monkeypatch)
    calls = []
    in_flight = maximum = 0

    async def generate(_client, model, item, **kwargs):
        nonlocal in_flight, maximum
        calls.append((model, item.copy(), kwargs))
        in_flight += 1
        maximum = max(maximum, in_flight)
        await asyncio.sleep(0)
        source = tmp_path / f"server-{item['idx']}.safetensors"
        save_file(_payload(), source)
        in_flight -= 1
        return str(source)

    monkeypatch.setattr(runner, "generate_hidden_states_async", generate)
    asyncio.run(runner.run_generation(args, dataset, legacy=legacy))
    factory.assert_called_once_with(
        base_url=args.endpoint, api_key="EMPTY", max_retries=0
    )
    assert sorted(item["idx"] for _, item, _ in calls) == [1, 2]
    for model, item, kwargs in calls:
        assert model == "target"
        assert kwargs == {"timeout": 23, "max_retries": 3}
        assert item["messages"] == dataset[item["idx"]]["messages"]
        if item["idx"] == 2:
            assert item["continue_final_message"] is True
        elif legacy:
            assert item["continue_final_message"] is False
        else:
            assert "continue_final_message" not in item
    assert maximum <= args.concurrency
    assert progress.count == 2
    assert bool(progress.postfixes) is not legacy
    assert (cache / "hs_0.safetensors").read_bytes() == preserved
    assert (cache / "hs_4.safetensors").read_bytes() == b"other rank"
    assert (
        next(cache.glob("hs_1.safetensors.invalid-*")).read_bytes()
        == b"bad local artifact"
    )
    assert sorted(path.name for path in cache.glob("hs_*.safetensors")) == [
        "hs_0.safetensors",
        "hs_1.safetensors",
        "hs_2.safetensors",
        "hs_4.safetensors",
    ]


@pytest.mark.parametrize("legacy", [True, False])
def test_consecutive_failures_abort_a_full_queue_without_hanging(
    tmp_path, monkeypatch, legacy
):
    args = _args(tmp_path, concurrency=1, max_consecutive_errors=1)
    _service(monkeypatch)
    calls = []

    async def fail(_client, _model, item, **_kwargs):
        calls.append(item["idx"])
        raise RuntimeError("target unavailable")

    monkeypatch.setattr(runner, "generate_hidden_states_async", fail)

    async def run():
        await asyncio.wait_for(
            runner.run_generation(args, [_row()] * 50, legacy=legacy), timeout=3
        )

    with pytest.raises(RuntimeError, match="1 consecutive samples"):
        asyncio.run(run())
    assert len(calls) < 50
    assert not list((tmp_path / "cache").glob("hs_*.safetensors"))


@pytest.mark.parametrize("legacy", [True, False])
def test_success_resets_shared_failure_counter(tmp_path, monkeypatch, legacy):
    calls = []

    async def generate(_client, _model, item, **_kwargs):
        calls.append(item["idx"])
        if item["idx"] != 1:
            raise ValueError("bad response")
        source = tmp_path / "server.safetensors"
        save_file(_payload(), source)
        return str(source)

    monkeypatch.setattr(runner, "generate_hidden_states_async", generate)
    progress = _Progress()
    skipped = []

    async def run():
        queue = asyncio.Queue()
        for index in range(3):
            queue.put_nowait({"idx": index, "input_ids": [1, 2]})
        queue.put_nowait(None)
        await runner.worker(
            None,
            "target",
            queue,
            progress,
            asyncio.Semaphore(1),
            asyncio.Semaphore(1),
            tmp_path,
            True,
            None,
            0,
            False,
            skipped,
            asyncio.Event(),
            runner.FailureTracker(2),
            legacy=legacy,
        )
        await asyncio.wait_for(queue.join(), timeout=1)

    asyncio.run(run())
    assert calls == [0, 1, 2]
    assert skipped == [0, 2]
    assert progress.count == 3
    assert (tmp_path / "hs_1.safetensors").is_file()


@pytest.mark.parametrize(
    ("legacy", "error"), [(True, IndexError), (False, RuntimeError)]
)
def test_empty_model_list_retains_entrypoint_error(
    tmp_path, monkeypatch, legacy, error
):
    _service(monkeypatch, model_ids=())
    with pytest.raises(error):
        asyncio.run(runner.run_generation(_args(tmp_path), [_row()], legacy=legacy))


@pytest.mark.parametrize("legacy", [True, False])
def test_cancellation_during_validation_preserves_publication_boundary(
    tmp_path, monkeypatch, legacy
):
    source = tmp_path / "server.safetensors"
    save_file(_payload(), source)
    entered, release = threading.Event(), threading.Event()
    validate = runner._validated_payload

    def paused_validation(*args, **kwargs):
        entered.set()
        if not release.wait(timeout=3):
            raise TimeoutError("test validation was never released")
        return validate(*args, **kwargs)

    monkeypatch.setattr(runner, "_validated_payload", paused_validation)
    monkeypatch.setattr(
        runner, "generate_hidden_states_async", AsyncMock(return_value=str(source))
    )

    async def run():
        queue = asyncio.Queue()
        queue.put_nowait({"idx": 0, "input_ids": [1, 2]})
        task = asyncio.create_task(
            runner.worker(
                None,
                "target",
                queue,
                _Progress(),
                asyncio.Semaphore(1),
                asyncio.Semaphore(1),
                tmp_path,
                True,
                None,
                0,
                False,
                [],
                asyncio.Event(),
                None,
                legacy=legacy,
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(queue.join(), timeout=1)
        finally:
            release.set()

    # asyncio.run waits for the released executor thread to finish, including any
    # publication that the CLI intentionally performs in that same thread.
    asyncio.run(run())
    assert source.exists() is legacy
    assert (tmp_path / "hs_0.safetensors").exists() is not legacy
