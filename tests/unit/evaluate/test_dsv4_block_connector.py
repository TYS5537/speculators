"""CPU tests for the server connector: real tensors/files, isolated vLLM stubs."""

# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path
from threading import Barrier, Lock
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from speculators_dsv4 import HS_FORMAT
from speculators_dsv4 import offline as offline_backend

ROOT = Path(__file__).resolve().parents[3]


class _ConnectorBase:
    def __init__(self, vllm_config, role, kv_cache_config):
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self._connector_metadata = None

    def has_connector_metadata(self):
        return self._connector_metadata is not None

    def bind_connector_metadata(self, metadata):
        self._connector_metadata = metadata

    def _get_connector_metadata(self):
        return self._connector_metadata


class _Metadata:
    pass


class _SupportsHMA:
    pass


class _TPGroup:
    def __init__(self):
        self.flags = []
        self.remote_error = False

    def all_reduce(self, tensor):
        self.flags.append(tensor.detach().clone())
        if self.remote_error and len(self.flags) == 2:
            return torch.ones_like(tensor)
        return tensor


@pytest.fixture
def connector_module(monkeypatch):
    transfer = ModuleType("vllm.distributed.kv_transfer")
    transfer.has_kv_transfer_group = lambda: False
    transfer.get_kv_transfer_group = lambda: None
    base = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    base.KVConnectorBase_V1 = _ConnectorBase
    base.KVConnectorMetadata = _Metadata
    base.KVConnectorWorkerMetadata = _Metadata
    base.SupportsHMA = _SupportsHMA
    parallel = ModuleType("vllm.distributed.parallel_state")
    group = _TPGroup()
    parallel.get_tensor_model_parallel_rank = lambda: 0
    parallel.get_tp_group = lambda: group
    parallel.get_dp_group = lambda: pytest.fail("DP1 must not use a DP collective")
    for name, module in (
        (transfer.__name__, transfer),
        (base.__name__, base),
        (parallel.__name__, parallel),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT / "src/speculators_dsv4/block_connector.py"
    spec = importlib.util.spec_from_file_location("dsv4_block_connector_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.test_group = group
    return module


def _config(directory, *, max_num_seqs=1, dp_size=1, dp_rank=0, **extra):
    settings = {"shared_storage_path": str(directory), **extra}
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            data_parallel_size_local=dp_size,
            data_parallel_rank=dp_rank,
            enable_expert_parallel=dp_size > 1,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        kv_transfer_config=SimpleNamespace(get_from_extra_config=settings.get),
        speculative_config=SimpleNamespace(
            method="extract_hidden_states",
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(eagle_aux_hidden_state_layer_ids=[1, 11, 43]),
            ),
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(vocab_size=5, hidden_size=4),
        ),
    )


def _request(*, tokens=None, logits_start=2, hidden_start=1):
    return SimpleNamespace(
        req_id="cmpl-test-0",
        prompt_token_ids=list(tokens if tokens is not None else [1, 2, 3, 4]),
        num_computed_tokens=0,
        prompt_embeds=None,
        mm_features=[],
        lora_request=None,
        sampling_params=SimpleNamespace(
            max_tokens=1,
            n=1,
            logprobs=None,
            prompt_logprobs=None,
            extra_args={
                "kv_transfer_params": {
                    "dsv4_block_verify": {
                        "version": 1,
                        "logits_start": logits_start,
                        "hidden_start": hidden_start,
                    },
                },
            },
        ),
    )


def _schedule(request):
    return _schedule_batch([request])


@pytest.fixture
def cached_connector_module(connector_module, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "speculators_dsv4.block_connector", connector_module
    )
    path = ROOT / "src/speculators_dsv4/cached_connector.py"
    spec = importlib.util.spec_from_file_location("dsv4_cached_connector_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.base_module = connector_module
    return module


def _cached_request(
    key, *, read=None, tokens=None, logits_start=2, hidden_start=2, release=()
):
    request = _request(
        tokens=tokens, logits_start=logits_start, hidden_start=hidden_start
    )
    request.req_id = request.request_id = "cmpl-" + key + "-0"
    request.sampling_params.extra_args["kv_transfer_params"][
        "dsv4_block_verify"
    ].update(
        version=3,
        output_mode="logprobs",
        profile=False,
        cache={"read": read, "write": key, "release": list(release)},
    )
    return request


@pytest.fixture
def replay_connector_module(connector_module, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "speculators_dsv4.block_connector", connector_module
    )
    path = ROOT / "src/speculators_dsv4/replay_connector.py"
    spec = importlib.util.spec_from_file_location("dsv4_replay_connector_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _generation_request():
    request = _request(tokens=[1, 2])
    request.req_id = request.request_id = "cmpl-native-0"
    request.sampling_params.temperature = 0.0
    request.sampling_params.max_tokens = 20
    request.sampling_params.extra_args = {
        "kv_transfer_params": {"dsv4_greedy_trace": {"version": 1}}
    }
    return request


def test_replay_native_decode_has_no_artifacts(replay_connector_module, tmp_path):
    module = replay_connector_module
    connector = module.DSV4ReplayConnector(_config(tmp_path), "scheduler", None)
    request = _generation_request()
    metadata = connector.build_connector_meta(_schedule(request))
    assert metadata.requests == []
    for length in (1, 2):
        metadata = connector.build_connector_meta(
            SimpleNamespace(
                scheduled_new_reqs=[],
                num_scheduled_tokens={request.req_id: length},
            )
        )
        connector.bind_connector_metadata(metadata)
        model = SimpleNamespace(compute_logits=lambda _: pytest.fail("No export head"))
        hidden = torch.zeros(length, 4, dtype=torch.bfloat16)
        connector.capture(
            model,
            torch.ones(length).long(),
            torch.arange(length),
            (hidden, [hidden] * 3),
        )
    assert not list(tmp_path.iterdir())
    assert connector.request_finished(request, [])[1] == {
        "dsv4_greedy_trace": {"version": 1}
    }
    assert not connector._generating


def test_replay_mixed_layout_extracts_only_verification_rows(
    replay_connector_module, tmp_path
):
    connector = replay_connector_module.DSV4ReplayConnector(
        _config(tmp_path, max_num_seqs=2), "scheduler", None
    )
    generation = _generation_request()
    block = _request(tokens=[3, 4], logits_start=1, hidden_start=0)
    metadata = connector.build_connector_meta(_schedule_batch([block, generation]))
    # Native worker packs generation first despite the scheduler's list order.
    metadata.forward_layout = [(generation.req_id, 0, 2), (block.req_id, 2, 4)]
    connector.bind_connector_metadata(metadata)
    hidden = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    rows = []

    def head(value):
        rows.append(value.clone())
        return torch.zeros(value.shape[0], 5)

    connector.capture(
        SimpleNamespace(compute_logits=head),
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([0, 1, 0, 1]),
        (hidden, [hidden] * 3),
    )
    torch.testing.assert_close(rows[0], hidden[3:4])
    packet = load_file(str(tmp_path / f"{block.req_id}.safetensors"))
    torch.testing.assert_close(packet["hidden_states"][:, 0], hidden[2:4])
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize(
    "fault", ["temperature", "protocol", "budget", "unknown", "resume"]
)
def test_replay_scheduler_fails_closed(replay_connector_module, tmp_path, fault):
    connector = replay_connector_module.DSV4ReplayConnector(
        _config(tmp_path), "scheduler", None
    )
    request = _generation_request()
    schedule = _schedule(request)
    if fault == "temperature":
        request.sampling_params.temperature = 1.0
    elif fault == "protocol":
        request.sampling_params.extra_args["kv_transfer_params"]["dsv4_greedy_trace"][
            "version"
        ] = True
    elif fault == "budget":
        schedule.num_scheduled_tokens[request.req_id] = 1
    elif fault == "unknown":
        schedule.scheduled_new_reqs = []
    else:
        connector.build_connector_meta(schedule)
    with pytest.raises(ValueError):
        connector.build_connector_meta(schedule)


def _allocated(*groups):
    return SimpleNamespace(
        blocks=tuple(
            tuple(
                SimpleNamespace(is_null=index is None, block_id=index or 0)
                for index in group
            )
            for group in groups
        )
    )


def test_cached_scheduler_worker_round_trip_and_suffix_packet(
    cached_connector_module, tmp_path
):
    module = cached_connector_module
    spec = type("AscendSlidingWindowMLASpec", (), {"page_size_bytes": 32})()
    config = SimpleNamespace(
        num_blocks=16,
        kv_cache_groups=[SimpleNamespace(layer_names=["state"], kv_cache_spec=spec)],
    )
    scheduler = module.DSV4CachedBlockVerifyConnector(
        _config(tmp_path, max_num_seqs=2), "scheduler", config
    )
    worker = module.DSV4CachedBlockVerifyConnector(
        _config(tmp_path, max_num_seqs=2), "worker", config
    )
    state = torch.arange(128, dtype=torch.float32).reshape(16, 8)
    worker.register_kv_caches({"state": [state]})
    expected_state = state[1].clone()
    first = _cached_request("a" * 32, tokens=[1, 2], logits_start=1, hidden_start=0)
    assert scheduler.get_num_new_matched_tokens(first, 0) == (0, False)
    scheduler.update_state_after_alloc(first, _allocated([1]), 0)
    metadata = scheduler.build_connector_meta(_schedule(first))
    worker.bind_connector_metadata(metadata)
    worker.start_load_kv()
    worker.check_ready("cpu")
    normalized = torch.ones(2, 4, dtype=torch.bfloat16)
    head_rows = []

    def head(value):
        head_rows.append(value.clone())
        return torch.arange(5).float().expand(value.shape[0], -1)

    model = SimpleNamespace(compute_logits=head)
    worker.capture(
        model, torch.tensor([1, 2]), torch.arange(2), (normalized, [normalized] * 3)
    )
    scheduler.update_connector_output(
        SimpleNamespace(kv_connector_worker_meta=worker.build_connector_worker_meta())
    )
    scheduler.request_finished(first, None)
    state.fill_(99)  # Freed vLLM pages may immediately be reused by other requests.
    second = _cached_request("b" * 32, read="a" * 32, release=["a" * 32])
    assert scheduler.get_num_new_matched_tokens(second, 0) == (2, False)
    second.num_computed_tokens = 2
    scheduler.update_state_after_alloc(second, _allocated([7, 8]), 2)
    scheduled = _schedule(second)
    scheduled.num_scheduled_tokens[second.req_id] = 2
    metadata = scheduler.build_connector_meta(scheduled)
    worker.bind_connector_metadata(metadata)
    worker.start_load_kv()
    worker.check_ready("cpu")
    assert torch.equal(state[7], expected_state)
    assert state[8].eq(99).all()  # New page must not receive stale ancestor data.
    worker.capture(
        model,
        torch.tensor([3, 4]),
        torch.tensor([2, 3]),
        (normalized, [normalized + i for i in range(3)]),
    )
    packet = load_file(metadata.requests[0].filename)
    assert packet["token_ids"].tolist() == [1, 2, 3, 4]
    assert packet["hidden_states"].shape == (2, 3, 4)
    assert packet["logprobs"].shape == (2, 5)
    assert packet["kv_reuse_metadata"].tolist() == [2, 1, 0]
    assert [rows.shape[0] for rows in head_rows] == [1, 2]
    scheduler.update_connector_output(
        SimpleNamespace(kv_connector_worker_meta=worker.build_connector_worker_meta())
    )
    assert set(scheduler._snapshots) == {"b" * 32}
    # Feedback removes an evicted handle, so the NEXT scheduled request really
    # computes the full prefix rather than trying to load an absent snapshot.
    worker.shutdown()
    scheduler.update_connector_output(
        SimpleNamespace(kv_connector_worker_meta=worker.build_connector_worker_meta())
    )
    assert scheduler.get_num_new_matched_tokens(second, 0) == (0, False)


def test_cached_snapshot_availability_requires_all_tp_ranks(cached_connector_module):
    cls = cached_connector_module.SnapshotAvailability
    assert cls({"a", "b"}).aggregate(cls({"b", "c"})).keys == {"b"}


def test_stateless_server_rejects_snapshot_protocol(ready):
    request = _cached_request("a" * 32)
    with pytest.raises(ValueError, match="--dsv4-kv-reuse"):
        ready.connector.build_connector_meta(_schedule(request))


@pytest.mark.parametrize("case", ["missing", "prefix", "output", "native_apc"])
def test_snapshot_scheduler_rejects_unsafe_cache_reuse(
    cached_connector_module, tmp_path, case
):
    connector = cached_connector_module.DSV4CachedBlockVerifyConnector(
        _config(tmp_path), "scheduler", None
    )
    connector._snapshots["a" * 32] = (1, 2)
    request = _cached_request("b" * 32, read="a" * 32)
    if case == "missing":
        with pytest.raises(ValueError, match="allocation metadata"):
            connector.build_connector_meta(_schedule(request))
        return
    if case == "prefix":
        request.prompt_token_ids[0] = 0
    elif case == "output":
        request.sampling_params.extra_args["kv_transfer_params"]["dsv4_block_verify"][
            "hidden_start"
        ] = 0
    with pytest.raises(ValueError, match="prefix|output rows|prefix caching"):
        connector.get_num_new_matched_tokens(request, int(case == "native_apc"))


def test_restore_failure_is_synchronized_before_forward(
    cached_connector_module, tmp_path
):
    module = cached_connector_module
    worker = module.DSV4CachedBlockVerifyConnector(_config(tmp_path), "worker", None)
    metadata = module.base_module.BlockMetadata(
        [
            module.CachedBlockRequest(
                "x", "unused", [1, 2], 1, 1, computed_tokens=1, cache_read="missing"
            )
        ]
    )
    worker.bind_connector_metadata(metadata)
    worker.start_load_kv()  # Capture local failure without stranding peers.
    with pytest.raises(RuntimeError, match="restore failed"):
        worker.check_ready("cpu")


def _schedule_batch(requests):
    return SimpleNamespace(
        scheduled_new_reqs=requests,
        num_scheduled_tokens={
            request.req_id: len(request.prompt_token_ids) for request in requests
        },
    )


@pytest.fixture
def ready(connector_module, tmp_path):
    connector = connector_module.DSV4BlockVerifyConnector(
        _config(tmp_path),
        "worker",
        None,
    )
    request = _request()
    metadata = connector.build_connector_meta(_schedule(request))
    connector.bind_connector_metadata(metadata)
    normalized = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
    auxiliary = [normalized + 40 * (index + 1) for index in range(3)]
    # Poison padded rows so accidental inclusion cannot silently pass.
    normalized[4:] = torch.nan
    for value in auxiliary:
        value[4:] = torch.nan
    calls = []
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [4.0, 2.0, 0.0, 1.0, 3.0]])

    def compute_logits(hidden):
        calls.append(hidden.clone())
        return logits.clone()

    return SimpleNamespace(
        module=connector_module,
        connector=connector,
        request=request,
        metadata=metadata,
        model=SimpleNamespace(compute_logits=compute_logits),
        input_ids=torch.tensor([1, 2, 3, 4, 0, 0]),
        positions=torch.arange(6),
        output=(normalized, auxiliary),
        calls=calls,
        logits=logits,
        path=Path(metadata.requests[0].filename),
    )


def _capture(ready):
    ready.connector.capture(ready.model, ready.input_ids, ready.positions, ready.output)


def _batch_ready(
    module, directory, lengths=(4, 6), *, dp_size=1, dp_rank=0, extended=True
):
    connector = module.DSV4BlockVerifyConnector(
        _config(directory, max_num_seqs=3, dp_size=dp_size, dp_rank=dp_rank),
        "worker",
        None,
    )
    requests = []
    for index, length in enumerate(lengths):
        request = _request(
            tokens=[index + 1] * length,
            logits_start=max(0, length - 3),
            hidden_start=index % length,
        )
        request.req_id = f"cmpl-batch-dp{dp_rank}-{index}"
        if extended:
            request.sampling_params.extra_args["kv_transfer_params"][
                "dsv4_block_verify"
            ].update(
                version=2,
                output_mode="greedy" if index % 2 else "logprobs",
                profile=False,
            )
        requests.append(request)
    metadata = connector.build_connector_meta(_schedule_batch(requests))
    # Native input preparation may reorder equal-length or unequal-length inputs.
    # The two NaN tail rows must never appear in any packet or head projection.
    tokens, positions, hidden, layout = [], [], [], []
    for index in reversed(range(len(requests))):
        request = requests[index]
        length = lengths[index]
        layout.append((request.req_id, len(tokens), len(tokens) + length))
        tokens.extend(request.prompt_token_ids)
        positions.extend(range(length))
        hidden.append(torch.arange(length * 4).reshape(-1, 4).float() / 8 + index)
    metadata.forward_layout = layout
    connector.bind_connector_metadata(metadata)
    normalized = torch.cat([*hidden, torch.full((2, 4), torch.nan)]).to(torch.bfloat16)
    auxiliary = [normalized + offset for offset in (10, 20, 30)]
    weights = torch.arange(20).reshape(4, 5).float() / 40
    calls = []

    def head(value):
        calls.append(value.clone())
        return value.float() @ weights

    return SimpleNamespace(
        module=module,
        connector=connector,
        requests=requests,
        metadata=metadata,
        input_ids=torch.tensor(tokens + [0, 0]),
        positions=torch.tensor(positions + [0, 0]),
        output=(normalized, auxiliary),
        model=SimpleNamespace(compute_logits=head),
        weights=weights,
        calls=calls,
    )


def _assert_batch_packets(ready):
    by_id = {request.request_id: request for request in ready.metadata.requests}
    for request_id, start, end in ready.metadata.forward_layout:
        request = by_id[request_id]
        packet = load_file(request.filename)
        assert packet["token_ids"].tolist() == request.token_ids
        assert packet["verification_metadata"].tolist() == [
            request.version,
            len(request.token_ids),
            request.logits_start,
            request.hidden_start,
        ]
        expected = torch.log_softmax(
            ready.output[0][start + request.logits_start : end].float() @ ready.weights,
            -1,
        )
        if request.output_mode == "greedy":
            assert torch.equal(packet["greedy_token_ids"], expected.argmax(-1))
            assert "logprobs" not in packet
        else:
            torch.testing.assert_close(packet["logprobs"], expected)
        torch.testing.assert_close(
            packet["hidden_states"],
            torch.stack(
                [
                    value[start + request.hidden_start : end]
                    for value in ready.output[1]
                ],
                1,
            ),
        )


@pytest.mark.parametrize("lengths", [(4, 6), (4, 4), (2, 7, 5)])
@pytest.mark.parametrize("extended", [False, True])
def test_batch_reorders_and_slices_each_request_with_one_head(
    connector_module, tmp_path, lengths, extended
):
    ready = _batch_ready(connector_module, tmp_path, lengths, extended=extended)
    _capture(ready)
    _assert_batch_packets(ready)
    assert len(ready.calls) == 1
    assert ready.calls[0].shape == (sum(min(3, length) for length in lengths), 4)
    assert torch.isfinite(ready.calls[0]).all()
    assert len(connector_module.test_group.flags) == 2
    for request in reversed(ready.metadata.requests):
        _, handle = ready.connector.request_finished(
            SimpleNamespace(request_id=request.request_id), []
        )
        assert handle["hidden_states_path"] == request.filename
    assert ready.connector._requests == {}


def test_batch_identical_prefixes_still_use_native_request_ids(
    connector_module, tmp_path
):
    ready = _batch_ready(connector_module, tmp_path, (4, 4))
    ready.metadata.requests[1].token_ids = [1] * 4
    ready.input_ids[:8] = 1
    _capture(ready)
    _assert_batch_packets(ready)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unknown", "gap", "size"])
def test_bad_batch_layout_fails_before_head(connector_module, tmp_path, failure):
    ready = _batch_ready(connector_module, tmp_path)
    layout = ready.metadata.forward_layout
    if failure == "missing":
        ready.metadata.forward_layout = None
    elif failure == "duplicate":
        layout[1] = (layout[0][0], 6, 10)
    elif failure == "unknown":
        layout[0] = ("unknown-request", 0, 6)
    elif failure == "gap":
        layout[1] = (layout[1][0], 7, 11)
    else:
        layout[0] = (layout[0][0], 0, 5)
    with pytest.raises(RuntimeError, match="packet export failed") as exc_info:
        _capture(ready)
    assert "layout" in str(exc_info.value.__cause__)
    assert not ready.calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["input_ids", "positions"])
def test_batch_validates_later_prefix_before_any_head_or_publication(
    connector_module, tmp_path, field
):
    ready = _batch_ready(connector_module, tmp_path)
    getattr(ready, field)[6] += 1
    with pytest.raises(RuntimeError, match="packet export failed"):
        _capture(ready)
    assert not ready.calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["capacity", "invalid_second", "duplicate"])
def test_scheduler_rejects_entire_invalid_batch_without_registering_ids(
    connector_module, tmp_path, failure
):
    connector = connector_module.DSV4BlockVerifyConnector(
        _config(tmp_path, max_num_seqs=2), "scheduler", None
    )
    requests = [_request() for _ in range(3 if failure == "capacity" else 2)]
    for index, request in enumerate(requests):
        request.req_id = f"cmpl-{index}"
    if failure == "invalid_second":
        requests[1].num_computed_tokens = 1
    elif failure == "duplicate":
        requests[1].req_id = requests[0].req_id
    with pytest.raises(ValueError):
        connector.build_connector_meta(_schedule_batch(requests))
    assert connector._requests == {}


def test_worker_layout_hook_uses_post_prepare_order_and_is_scoped_and_idempotent(
    connector_module, monkeypatch
):
    calls = []

    class Runner:
        def _prepare_inputs(self, schedule, value, *, extra):
            calls.append((schedule, value, extra))
            self.input_batch = SimpleNamespace(req_ids=["second", "first"])
            self.query_start_loc = SimpleNamespace(np=torch.tensor([0, 6, 10, 999]))
            return "native-result"

    def get_runner(name):
        assert name == "vllm_ascend.worker.model_runner_v1"
        return SimpleNamespace(NPUModelRunner=Runner)

    monkeypatch.setattr(connector_module, "import_module", get_runner)
    connector_module.install_worker_block_layout()
    wrapped = Runner._prepare_inputs
    connector_module.install_worker_block_layout()
    assert Runner._prepare_inputs is wrapped
    metadata = connector_module.BlockMetadata()
    for current in (metadata, SimpleNamespace(), None):
        schedule = SimpleNamespace(kv_connector_metadata=current)
        assert Runner()._prepare_inputs(schedule, 7, extra=8) == "native-result"
        assert calls[-1] == (schedule, 7, 8)
        if current is not None and current is not metadata:
            assert not hasattr(current, "forward_layout")
    assert metadata.forward_layout == [("second", 0, 6), ("first", 6, 10)]


@pytest.mark.parametrize("profile_flags", [(True, True), (False, True)])
def test_batch_profiles_share_forward_and_head_time_without_multiplying_work(
    connector_module, monkeypatch, tmp_path, profile_flags
):
    ready = _batch_ready(connector_module, tmp_path)
    for request, enabled in zip(ready.metadata.requests, profile_flags, strict=True):
        request.profile = enabled
    monkeypatch.setattr(connector_module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(
        connector_module, "get_kv_transfer_group", lambda: ready.connector
    )
    profiler = connector_module.block_forward_profiler("cpu")
    assert profiler.enabled
    profiler.record("server_forward", 6.0)
    # head, then two packet preparations: durations 2, 1, 1 seconds.
    clock = iter([0.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    monkeypatch.setattr(
        "speculators_eval.profiling.time.perf_counter", lambda: next(clock)
    )
    ready.connector.capture(
        ready.model, ready.input_ids, ready.positions, ready.output, profiler=profiler
    )
    for request in ready.metadata.requests:
        packet = load_file(request.filename)
        if request.profile:
            assert packet["server_timings"].tolist() == [3.0, 1.0, 1.0]
        else:
            assert "server_timings" not in packet


def test_scheduler_copies_prefix_and_returns_versioned_handle(ready):
    ready.request.prompt_token_ids[0] = 4
    assert ready.metadata.requests[0].token_ids == [1, 2, 3, 4]
    request = SimpleNamespace(request_id=ready.request.req_id)
    delayed, handle = ready.connector.request_finished_all_groups(request, ([], []))
    assert delayed is False
    assert handle == {
        "hidden_states_path": str(ready.path),
        "dsv4_block_verify_version": 1,
    }
    assert ready.connector.request_finished(request, []) == (False, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_token_ids", []),
        ("prompt_token_ids", None),
        ("prompt_token_ids", [True, 2, 3, 4]),
        ("prompt_token_ids", [5, 2, 3, 4]),
        ("prompt_token_ids", [-1, 2, 3, 4]),
        ("prompt_token_ids", [1.0, 2, 3, 4]),
        ("num_computed_tokens", 1),
        ("prompt_embeds", torch.zeros(4, 4)),
        ("mm_features", [object()]),
        ("lora_request", object()),
        ("req_id", "../outside"),
        ("sampling_params", None),
    ],
)
def test_scheduler_rejects_non_full_token_prefix(ready, field, value):
    request = _request()
    scheduled = _schedule(request)
    setattr(request, field, value)
    if field == "req_id":
        scheduled.num_scheduled_tokens = {value: 4}
    with pytest.raises(ValueError):
        ready.connector.build_connector_meta(scheduled)


@pytest.mark.parametrize(("field", "value"), [("max_tokens", 2), ("n", 2)])
def test_scheduler_requires_one_output_and_one_sequence(ready, field, value):
    request = _request()
    setattr(request.sampling_params, field, value)
    with pytest.raises(ValueError, match="max_tokens=1 and n=1"):
        ready.connector.build_connector_meta(_schedule(request))


@pytest.mark.parametrize("field", ["logprobs", "prompt_logprobs"])
def test_scheduler_rejects_http_probability_materialization(ready, field):
    request = _request()
    setattr(request.sampling_params, field, 5)
    with pytest.raises(ValueError, match="not HTTP logprobs"):
        ready.connector.build_connector_meta(_schedule(request))


def test_scheduler_rejects_duplicate_inflight_id(ready):
    with pytest.raises(ValueError, match="Duplicate in-flight"):
        ready.connector.build_connector_meta(_schedule(_request()))


@pytest.mark.parametrize(
    "params",
    [
        None,
        [],
        {"other": {}},
        {"dsv4_block_verify": {"version": 2, "logits_start": 2, "hidden_start": 1}},
        {"dsv4_block_verify": {"version": 1, "logits_start": -1, "hidden_start": 1}},
        {"dsv4_block_verify": {"version": 1, "logits_start": 4, "hidden_start": 1}},
        {"dsv4_block_verify": {"version": 1, "logits_start": True, "hidden_start": 1}},
        {"dsv4_block_verify": {"version": 1, "logits_start": 2, "hidden_start": 5}},
        {"dsv4_block_verify": {"version": 1, "logits_start": 2}},
        {"dsv4_block_verify": {}, "hidden_states_path": "outside.safetensors"},
    ],
)
def test_scheduler_rejects_missing_or_bad_protocol(ready, params):
    request = _request()
    request.sampling_params.extra_args = {"kv_transfer_params": params}
    with pytest.raises(ValueError):
        ready.connector.build_connector_meta(_schedule(request))


def test_scheduler_rejects_chunks_cached_work_and_multiple_requests(ready):
    request = _request()
    scheduled = _schedule(request)
    scheduled.num_scheduled_tokens[request.req_id] = 3
    with pytest.raises(ValueError, match="unchunked"):
        ready.connector.build_connector_meta(scheduled)
    scheduled.scheduled_new_reqs = []
    with pytest.raises(ValueError, match="fresh full-prefix"):
        ready.connector.build_connector_meta(scheduled)
    scheduled.scheduled_new_reqs = [request, _request()]
    with pytest.raises(ValueError, match="distinct full-prefix"):
        ready.connector.build_connector_meta(scheduled)
    scheduled.scheduled_new_reqs = []
    scheduled.num_scheduled_tokens = {}
    assert ready.connector.build_connector_meta(scheduled).requests == []


def test_connector_requires_positive_capacity_and_storage(connector_module, tmp_path):
    with pytest.raises(ValueError, match="max-num-seqs"):
        connector_module.DSV4BlockVerifyConnector(
            _config(tmp_path, max_num_seqs=0),
            "worker",
            None,
        )
    with pytest.raises(ValueError, match="storage directory"):
        connector_module.DSV4BlockVerifyConnector(
            _config(tmp_path, shared_storage_path=""),
            "worker",
            None,
        )


def test_connector_rechecks_block_topology_without_launcher(connector_module, tmp_path):
    config = _config(tmp_path, dp_size=2)
    config.parallel_config.enable_expert_parallel = False
    with pytest.raises(ValueError, match="expert-parallel"):
        connector_module.DSV4BlockVerifyConnector(config, "worker", None)
    with pytest.raises(ValueError, match="single-host"):
        connector_module.DSV4BlockVerifyConnector(
            _config(tmp_path, dp_size=4), "worker", None
        )


def test_connector_never_loads_or_defers_kv(ready):
    assert ready.connector.get_num_new_matched_tokens(None, 0) == (0, False)
    ready.connector.update_state_after_alloc(None, None, 0)
    with pytest.raises(ValueError, match="external KV"):
        ready.connector.update_state_after_alloc(None, None, 1)
    ready.connector.start_load_kv(None)
    ready.connector.wait_for_layer_load("anything")
    ready.connector.save_kv_layer(None, None, None)
    ready.connector.wait_for_save()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_connector_requires_positive_integer_row_limit(
    connector_module, tmp_path, limit
):
    with pytest.raises(ValueError, match="max_verify_rows"):
        connector_module.DSV4BlockVerifyConnector(
            _config(tmp_path, max_verify_rows=limit),
            "worker",
            None,
        )


def test_scheduler_caps_head_rows_not_prefix_length(connector_module, tmp_path):
    connector = connector_module.DSV4BlockVerifyConnector(
        _config(tmp_path, max_verify_rows=2),
        "worker",
        None,
    )
    request = _request(tokens=[1] * 129, logits_start=126, hidden_start=0)
    with pytest.raises(ValueError, match="exceeds 2"):
        connector.build_connector_meta(_schedule(request))
    request = _request(tokens=[1] * 129, logits_start=127, hidden_start=0)
    assert (
        connector.build_connector_meta(_schedule(request)).requests[0].logits_start
        == 127
    )


def test_capture_projects_only_normalized_suffix_and_real_packet_roundtrip(ready):
    _capture(ready)
    assert len(ready.calls) == 1
    torch.testing.assert_close(ready.calls[0], ready.output[0][2:4])
    assert not torch.equal(ready.calls[0], ready.output[1][-1][2:4])
    packet = load_file(str(ready.path))
    assert packet["verification_metadata"].tolist() == [1, 4, 2, 1]
    assert packet["token_ids"].tolist() == [1, 2, 3, 4]
    assert packet["layer_ids"].tolist() == [1, 11, 43]
    assert packet["logprobs"].dtype == torch.float32
    assert packet["hidden_states"].dtype == torch.bfloat16
    assert packet["hidden_states"].shape == (3, 3, 4)
    torch.testing.assert_close(
        packet["logprobs"],
        torch.log_softmax(ready.logits.float(), dim=-1),
    )
    torch.testing.assert_close(
        packet["hidden_states"],
        torch.stack([value[1:4] for value in ready.output[1]], 1),
    )
    assert list(ready.path.parent.iterdir()) == [ready.path]
    assert len(ready.module.test_group.flags) == 2
    assert ready.module.test_group.flags[0].dtype == torch.int32
    assert ready.module.test_group.flags[0].item() == 0


def test_nonroot_also_enters_native_head_before_rank_gate(ready, monkeypatch):
    events = []

    def head(hidden):
        events.append("head")
        torch.testing.assert_close(hidden, ready.output[0][2:4])

    def rank():
        events.append("rank")
        return 1

    ready.model.compute_logits = head
    monkeypatch.setattr(ready.module, "get_tensor_model_parallel_rank", rank)
    _capture(ready)
    assert events.index("head") < events.index("rank")
    assert not ready.path.exists()
    assert len(ready.module.test_group.flags) == 2


def test_capture_skips_warmup_and_rejects_wrong_metadata(ready):
    ready.connector.bind_connector_metadata(None)
    _capture(ready)
    ready.connector.bind_connector_metadata(ready.module.BlockMetadata())
    _capture(ready)
    assert not ready.calls
    ready.connector.bind_connector_metadata(object())
    with pytest.raises(RuntimeError, match="packet export failed") as exc_info:
        _capture(ready)
    assert "metadata" in str(exc_info.value.__cause__)


def test_completed_request_is_not_reexported_by_following_dummy_forward(ready):
    _capture(ready)
    packet = ready.path.read_bytes()
    # Mirrors vLLM finalization: cleared metadata precedes an idle dummy forward.
    ready.connector.bind_connector_metadata(None)
    ready.connector.capture(
        ready.model,
        torch.tensor([0]),
        torch.tensor([0]),
        (ready.output[0][:1], [value[:1] for value in ready.output[1]]),
    )
    assert len(ready.calls) == 1
    assert ready.path.read_bytes() == packet


@pytest.mark.parametrize("field", ["input_ids", "positions"])
def test_capture_rejects_prefix_position_mismatch_before_head(ready, field):
    value = getattr(ready, field).clone()
    value[0] += 1
    setattr(ready, field, value)
    with pytest.raises(RuntimeError, match="packet export failed") as exc_info:
        _capture(ready)
    assert "tokens/positions" in str(exc_info.value.__cause__)
    assert not ready.calls


@pytest.mark.parametrize("case", ["dtype", "width", "length", "layer_count"])
def test_capture_rejects_bad_hidden_shape_or_dtype(ready, case):
    normalized, auxiliary = ready.output
    auxiliary = list(auxiliary)
    if case == "dtype":
        auxiliary[0] = auxiliary[0].float()
    elif case == "width":
        auxiliary[0] = auxiliary[0][:, :3]
    elif case == "length":
        auxiliary[0] = auxiliary[0][:3]
    else:
        auxiliary.pop()
    ready.output = (normalized, auxiliary)
    with pytest.raises(RuntimeError, match="packet export failed") as exc_info:
        _capture(ready)
    assert "shape/dtype" in str(exc_info.value.__cause__)
    assert not ready.calls


@pytest.mark.parametrize("case", ["nan_logits", "infinite_logits", "nan_hidden"])
def test_capture_rejects_nonfinite_export(ready, case):
    if case == "nan_logits":
        ready.logits[0, 0] = torch.nan
    elif case == "infinite_logits":
        ready.logits[0] = -torch.inf
    else:
        ready.output[1][0][1, 0] = torch.nan
    with pytest.raises(RuntimeError, match="packet export failed") as exc_info:
        _capture(ready)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "nonfinite" in str(exc_info.value.__cause__)
    assert ready.module.test_group.flags[-1].item() == 1
    assert not ready.path.exists()


@pytest.mark.parametrize("value", [None, torch.zeros(1, 5), torch.zeros(2, 4)])
def test_capture_rejects_missing_or_truncated_vocab(ready, value):
    ready.model.compute_logits = lambda _: value
    with pytest.raises(RuntimeError, match="packet export failed") as exc_info:
        _capture(ready)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "logits shape" in str(exc_info.value.__cause__)
    assert not ready.path.exists()


def test_atomic_packet_publish_and_collision_never_overwrite(ready, monkeypatch):
    original_save = ready.module.save_file
    checked = []

    def save(tensors, temporary):
        assert not ready.path.exists()
        assert Path(temporary).parent == ready.path.parent
        original_save(tensors, temporary)
        checked.append(temporary)

    monkeypatch.setattr(ready.module, "save_file", save)
    _capture(ready)
    assert len(checked) == 1
    original_bytes = ready.path.read_bytes()
    with pytest.raises(FileExistsError):
        ready.module._save_packet({"x": torch.ones(1)}, str(ready.path))
    assert ready.path.read_bytes() == original_bytes
    assert list(ready.path.parent.iterdir()) == [ready.path]


def test_failed_packet_write_removes_only_temporary_file(ready, monkeypatch):
    def fail_save(tensors, filename):
        raise OSError("simulated full disk")

    monkeypatch.setattr(ready.module, "save_file", fail_save)
    with pytest.raises(OSError, match="full disk"):
        ready.module._save_packet({"x": torch.ones(1)}, str(ready.path))
    assert not ready.path.exists()
    assert list(ready.path.parent.iterdir()) == []


def test_atomic_publish_collision_race_preserves_other_artifact(ready, monkeypatch):
    original_save = ready.module.save_file

    def race_save(tensors, temporary):
        original_save(tensors, temporary)
        original_save({"other_request": torch.tensor([7])}, str(ready.path))

    monkeypatch.setattr(ready.module, "save_file", race_save)
    with pytest.raises(FileExistsError):
        ready.module._save_packet({"new_request": torch.tensor([8])}, str(ready.path))
    assert load_file(str(ready.path))["other_request"].tolist() == [7]
    assert list(ready.path.parent.iterdir()) == [ready.path]


def test_root_io_failure_is_collectively_reported(ready, monkeypatch):
    def fail_write(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(ready.module, "_save_packet", fail_write)
    with pytest.raises(RuntimeError, match="packet export failed") as exc_info:
        _capture(ready)
    assert isinstance(exc_info.value.__cause__, OSError)
    assert ready.module.test_group.flags[-1].item() == 1


def test_nonroot_receives_root_failure_after_native_head(ready, monkeypatch):
    monkeypatch.setattr(ready.module, "get_tensor_model_parallel_rank", lambda: 1)
    ready.module.test_group.remote_error = True
    with pytest.raises(RuntimeError, match="packet export failed") as exc_info:
        _capture(ready)
    assert exc_info.value.__cause__ is None
    assert len(ready.calls) == 1
    assert ready.module.test_group.flags[0].item() == 0
    assert not ready.path.exists()


def test_export_dispatch_requires_correct_connector(ready, monkeypatch):
    ready.module.export_block(None, None, None, None)
    monkeypatch.setattr(ready.module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(ready.module, "get_kv_transfer_group", object)
    with pytest.raises(ValueError, match="dedicated connector"):
        ready.module.export_block(None, None, None, None)
    monkeypatch.setattr(ready.module, "get_kv_transfer_group", lambda: ready.connector)
    ready.module.export_block(
        ready.model, ready.input_ids, ready.positions, ready.output
    )
    assert ready.path.exists()


class _Collective:
    """Bounded CPU rendezvous: catches skipped/misordered TP/DP calls, not HCCL."""

    def __init__(self):
        self.barrier = Barrier(2, timeout=10)
        self.lock = Lock()
        self.values = []

    def all_reduce(self, value):
        with self.lock:
            self.values.append(value.clone())
        leader = self.barrier.wait()
        result = torch.stack(self.values).sum(0).to(value.dtype)
        self.barrier.wait()
        if leader == 0:
            self.values.clear()
        self.barrier.wait()
        return result


@pytest.mark.parametrize("fault_rank", [None, (0, 1), (1, 0)])
def test_cached_dp2_restore_barrier_includes_idle_engine(
    cached_connector_module, monkeypatch, tmp_path, fault_rank
):
    module = cached_connector_module
    current = ContextVar("cached_rank")
    tp_groups, dp_groups = (
        [_Collective(), _Collective()],
        [_Collective(), _Collective()],
    )
    monkeypatch.setattr(module, "get_tp_group", lambda: tp_groups[current.get()[0]])
    monkeypatch.setattr(module, "get_dp_group", lambda: dp_groups[current.get()[1]])

    def worker(dp, tp):
        current.set((dp, tp))
        connector = module.DSV4CachedBlockVerifyConnector(
            _config(tmp_path, dp_size=2, dp_rank=dp), "worker", None
        )
        # DP1 is idle (no metadata) but must still rendezvous before native MoE.
        if (dp, tp) == fault_rank:
            connector._load_error = RuntimeError("rank-local copy failure")
        try:
            connector.check_ready("cpu")
        except RuntimeError as exc:
            return str(exc)
        return "ok"

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, dp, tp) for dp in range(2) for tp in range(2)]
        results = [future.result(timeout=30) for future in futures]
    assert all(
        ("restore failed" in result) if fault_rank else result == "ok"
        for result in results
    )


def test_mixed_cached_and_uncached_batch_respects_worker_reordering(
    cached_connector_module, tmp_path
):
    module = cached_connector_module
    connector = module.DSV4CachedBlockVerifyConnector(
        _config(tmp_path, max_num_seqs=2), "worker", None
    )
    connector._store = SimpleNamespace(save=lambda *args: True)
    cached = module.CachedBlockRequest(
        "cached",
        str(tmp_path / "cached.safetensors"),
        [1, 2, 3, 4],
        2,
        2,
        version=3,
        computed_tokens=2,
        cache_write="b" * 32,
    )
    fresh = module.base_module.BlockRequest(
        "fresh", str(tmp_path / "fresh.safetensors"), [2, 1, 0], 2, 0
    )
    metadata = module.base_module.BlockMetadata(
        [cached, fresh], forward_layout=[("fresh", 0, 3), ("cached", 3, 5)]
    )
    connector.bind_connector_metadata(metadata)
    hidden = torch.arange(20).reshape(5, 4).to(torch.bfloat16)
    rows = []

    def head(value):
        rows.append(value.clone())
        return value.float() @ torch.arange(20).reshape(4, 5).float() / 100

    connector.capture(
        SimpleNamespace(compute_logits=head),
        torch.tensor([2, 1, 0, 3, 4]),
        torch.tensor([0, 1, 2, 2, 3]),
        (hidden, [hidden] * 3),
    )
    assert torch.equal(rows[0], hidden[2:])
    cache_packet, fresh_packet = load_file(cached.filename), load_file(fresh.filename)
    assert torch.equal(
        cache_packet["hidden_states"], torch.stack([hidden[3:]] * 3, dim=1)
    )
    assert torch.equal(
        fresh_packet["hidden_states"], torch.stack([hidden[:3]] * 3, dim=1)
    )
    assert cache_packet["verification_metadata"].tolist() == [3, 4, 2, 2]
    assert "kv_reuse_metadata" not in fresh_packet


@pytest.mark.parametrize(
    ("lengths", "fault"),
    [
        (((4, 6), (3, 7, 5)), None),
        (((4, 6), ()), None),
        (((), (3, 7)), None),
        (((4, 6), (3, 7, 5)), "layout"),
        (((4, 6), ()), "layout"),
        (((4, 6), (3, 7)), "pack"),
        (((4, 6), (3, 7, 5)), "write"),
        (((4, 6), ()), "write"),
    ],
)
def test_dp2_batched_verification_with_uneven_and_idle_engines(  # noqa: C901
    connector_module, monkeypatch, tmp_path, lengths, fault
):
    module = connector_module
    current_rank = ContextVar("batched_worker_rank")
    tp_groups, dp_groups = (
        [_Collective(), _Collective()],
        [_Collective(), _Collective()],
    )
    monkeypatch.setattr(
        module, "get_tp_group", lambda: tp_groups[current_rank.get()[0]]
    )
    monkeypatch.setattr(
        module, "get_dp_group", lambda: dp_groups[current_rank.get()[1]]
    )
    monkeypatch.setattr(
        module, "get_tensor_model_parallel_rank", lambda: current_rank.get()[1]
    )

    original_pack = module._pack_verification_rows

    def pack_rows(batch, normalized):
        if fault == "pack" and current_rank.get() == (0, 1):
            raise RuntimeError("simulated packing allocation failure")
        return original_pack(batch, normalized)

    monkeypatch.setattr(module, "_pack_verification_rows", pack_rows)

    def worker(dp_rank, tp_rank):
        current_rank.set((dp_rank, tp_rank))
        ready = _batch_ready(
            module, tmp_path, lengths[dp_rank], dp_size=2, dp_rank=dp_rank
        )
        if fault == "layout" and (dp_rank, tp_rank) == (0, 1):
            ready.metadata.forward_layout.reverse()
        if fault == "write" and (dp_rank, tp_rank) == (0, 0):
            original = ready.connector._write_output

            def write_packet(request, *args, **kwargs):
                if request.request_id.endswith("-0"):
                    raise OSError("second packet write failed")
                return original(request, *args, **kwargs)

            ready.connector._write_output = write_packet

        def head(hidden):
            ready.calls.append(hidden.clone())
            shard = slice(tp_rank * 2, tp_rank * 2 + 2)
            partial = hidden[:, shard].float() @ ready.weights[shard]
            logits = tp_groups[dp_rank].all_reduce(partial)
            return logits if tp_rank == 0 else None

        ready.model.compute_logits = head
        error = None
        try:
            _capture(ready)
        except RuntimeError as exc:
            error = exc
        return ready, dp_rank, tp_rank, error

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, dp, tp) for dp in range(2) for tp in range(2)]
        workers = [future.result(timeout=30) for future in futures]
    for ready, dp_rank, tp_rank, error in workers:
        if fault:
            assert isinstance(error, RuntimeError)
            assert "packet export failed" in str(error)
        else:
            assert error is None
            if tp_rank == 0:
                _assert_batch_packets(ready)
        projected = bool(lengths[dp_rank]) and not (
            fault in {"layout", "pack"} and dp_rank == 0
        )
        assert len(ready.calls) == int(projected)
        if projected:
            assert ready.calls[0].shape[0] == sum(min(3, n) for n in lengths[dp_rank])
    if not fault:
        assert len(list(tmp_path.iterdir())) == sum(map(len, lengths))


def _run_dp2_round(  # noqa: C901 -- Simulate independent rank/engine failure cases.
    module, monkeypatch, directory, lengths, *, fault=None, extended=False
):
    """Four concurrent CPU workers emulate TP2 x DP2 with separate TP/DP groups."""
    current_rank = ContextVar("block_worker_rank")
    tp_groups = [_Collective(), _Collective()]
    dp_groups = [_Collective(), _Collective()]
    monkeypatch.setattr(
        module, "get_tp_group", lambda: tp_groups[current_rank.get()[0]]
    )
    monkeypatch.setattr(
        module, "get_dp_group", lambda: dp_groups[current_rank.get()[1]]
    )
    monkeypatch.setattr(
        module, "get_tensor_model_parallel_rank", lambda: current_rank.get()[1]
    )
    weights = torch.arange(20).reshape(4, 5).float() / 40

    def worker(dp_rank, tp_rank):
        current_rank.set((dp_rank, tp_rank))
        connector = module.DSV4BlockVerifyConnector(
            _config(directory, dp_size=2, dp_rank=dp_rank), "worker", None
        )
        length = lengths[dp_rank]
        metadata = None
        tokens = torch.ones(length or 1, dtype=torch.int64)
        normalized = (
            torch.arange((length or 1) * 4).reshape(-1, 4).to(torch.bfloat16) / 8
        )
        auxiliary = [normalized + offset for offset in (10, 20, 30)]
        if length:
            request = _request(tokens=tokens.tolist(), logits_start=length // 2)
            if extended:
                request.sampling_params.extra_args["kv_transfer_params"][
                    "dsv4_block_verify"
                ].update(
                    version=2,
                    output_mode="greedy" if dp_rank == 0 else "logprobs",
                    profile=True,
                )
            request.req_id = (
                "cmpl-collision-0" if fault == "collision" else f"cmpl-dp{dp_rank}-0"
            )
            metadata = connector.build_connector_meta(_schedule(request))
        elif fault == "empty_metadata":
            metadata = connector.build_connector_meta(
                SimpleNamespace(scheduled_new_reqs=[], num_scheduled_tokens={})
            )
        connector.bind_connector_metadata(metadata)
        if fault == "validation" and (dp_rank, tp_rank) == (0, 1):
            tokens[0] = 4
        if fault == "ownership" and dp_rank == 0:
            metadata.data_parallel_rank = 1
        if fault == "write" and (dp_rank, tp_rank) == (0, 0):

            def fail_write(*_args, **_kwargs):
                raise OSError("simulated DP0 full disk")

            connector._write_output = fail_write
        calls = []

        def head(hidden):
            calls.append(hidden.shape[0])
            # Real CPU sharded projection plus a simulated TP collective. Unequal
            # DP row counts must never enter the same head collective.
            shard = slice(tp_rank * 2, tp_rank * 2 + 2)
            partial = hidden[:, shard].float() @ weights[shard]
            logits = tp_groups[dp_rank].all_reduce(partial)
            return logits if tp_rank == 0 else None

        error = None
        try:
            connector.capture(
                SimpleNamespace(compute_logits=head),
                tokens,
                torch.arange(len(tokens)),
                (normalized, auxiliary),
            )
        except RuntimeError as exc:
            error = exc
        return SimpleNamespace(
            dp_rank=dp_rank,
            tp_rank=tp_rank,
            error=error,
            calls=calls,
            normalized=normalized,
            auxiliary=auxiliary,
            metadata=metadata,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, dp, tp) for dp in range(2) for tp in range(2)]
        return [future.result(timeout=30) for future in futures], weights


@pytest.mark.parametrize(
    ("lengths", "fault"),
    [
        ((4, 6), None),
        ((4, 0), None),
        ((0, 6), None),
        ((0, 0), None),
        ((4, 0), "empty_metadata"),
    ],
)
@pytest.mark.parametrize("extended", [False, True])
def test_dp2_uneven_and_idle_engines_keep_packets_and_native_logits_local(
    connector_module, monkeypatch, tmp_path, lengths, fault, extended
):
    workers, weights = _run_dp2_round(
        connector_module, monkeypatch, tmp_path, lengths, fault=fault, extended=extended
    )
    for worker in workers:
        assert worker.error is None
        length = lengths[worker.dp_rank]
        assert worker.calls == ([length - length // 2] if length else [])
        if not length or worker.tp_rank:
            continue
        request = worker.metadata.requests[0]
        assert worker.metadata.data_parallel_rank == worker.dp_rank
        packet = load_file(request.filename)
        assert packet["token_ids"].tolist() == [1] * length
        expected = torch.log_softmax(
            worker.normalized[request.logits_start :].float() @ weights, -1
        )
        if extended and worker.dp_rank == 0:
            assert torch.equal(packet["greedy_token_ids"], expected.argmax(-1))
            assert "logprobs" not in packet
        else:
            torch.testing.assert_close(packet["logprobs"], expected)
        if extended:
            assert packet["server_timings"].shape == (3,)
        torch.testing.assert_close(
            packet["hidden_states"],
            torch.stack([value[1:] for value in worker.auxiliary], 1),
        )
    assert len(list(tmp_path.iterdir())) == sum(length > 0 for length in lengths)


@pytest.mark.parametrize(
    ("fault", "lengths"),
    [
        ("write", (4, 6)),
        ("write", (4, 0)),
        ("validation", (4, 6)),
        ("ownership", (4, 6)),
        ("collision", (4, 6)),
    ],
)
def test_dp2_failure_reaches_both_engines_including_idle_peers(
    connector_module, monkeypatch, tmp_path, fault, lengths
):
    workers, _ = _run_dp2_round(
        connector_module, monkeypatch, tmp_path, lengths, fault=fault
    )
    assert all(isinstance(worker.error, RuntimeError) for worker in workers)
    assert all("packet export failed" in str(worker.error) for worker in workers)
    assert any(worker.error.__cause__ is not None for worker in workers)
    if fault in {"validation", "ownership"}:
        # A non-root TP validation error prevents BOTH peers entering the head.
        assert all(not worker.calls for worker in workers if worker.dp_rank == 0)
    if fault == "collision":
        assert len(list(tmp_path.iterdir())) == 1
        assert any(
            isinstance(worker.error.__cause__, FileExistsError) for worker in workers
        )


class _TinyDraft(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.target_layer_ids = [1, 11]
        self.config = SimpleNamespace(
            target_hidden_state_format=HS_FORMAT,
            speculators_config=SimpleNamespace(
                verifier=SimpleNamespace(name_or_path=str(model_path)),
            ),
        )


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("profile", [False, True])
def test_real_connector_packet_to_offline_target_prefill_and_suffix(
    connector_module,
    tmp_path,
    monkeypatch,
    compact,
    profile,
):
    """Only HTTP/model execution is faked; both transport ends use real code."""
    connector = connector_module.DSV4BlockVerifyConnector(
        _config(tmp_path),
        "worker",
        None,
    )
    model_path = tmp_path / "checkpoint"
    draft = _TinyDraft(model_path)
    report = {
        "model_path": str(model_path),
        "checkpoint_signature": "test-checkpoint",
        "config": {
            "model_type": "deepseek_v4",
            "hidden_size": 4,
            "num_hidden_layers": 43,
            "vocab_size": 5,
            "eos_token_id": 4,
        },
    }
    monkeypatch.setattr(
        offline_backend, "read_eval_manifest", lambda _path, expected: expected
    )
    recorded = []

    def create(**kwargs):
        prefix = kwargs["prompt"]
        body = kwargs["extra_body"]
        request_id = body["request_id"]
        assert "logprobs" not in kwargs
        assert "prompt_logprobs" not in body
        request = _request(tokens=prefix)
        request.req_id = f"cmpl-{request_id}-0"
        request.sampling_params.max_tokens = kwargs["max_tokens"]
        request.sampling_params.n = kwargs["n"]
        request.sampling_params.extra_args = {
            "kv_transfer_params": body["kv_transfer_params"],
        }
        metadata = connector.build_connector_meta(_schedule(request))
        connector.bind_connector_metadata(metadata)
        normalized = torch.arange(len(prefix), dtype=torch.float32).view(-1, 1)
        normalized = normalized.expand(-1, 4).to(torch.bfloat16)
        auxiliary = [normalized + 10 * (index + 1) for index in range(3)]

        def head(hidden):
            recorded.append(
                (list(prefix), hidden.shape[0], metadata.requests[0].hidden_start)
            )
            weights = torch.arange(5, dtype=torch.float32).view(1, -1)
            return hidden[:, :1].float() * weights

        connector.capture(
            SimpleNamespace(compute_logits=head),
            torch.tensor(prefix),
            torch.arange(len(prefix)),
            (normalized, auxiliary),
        )
        _, handle = connector.request_finished(
            SimpleNamespace(request_id=request.req_id),
            [],
        )
        return SimpleNamespace(
            id=f"cmpl-{request_id}",
            model="test-target",
            choices=[SimpleNamespace(prompt_token_ids=list(prefix))],
            kv_transfer_params=handle,
        )

    target = offline_backend.DSV4OfflineTarget(
        draft,
        report,
        hidden_states_path=tmp_path,
        client=SimpleNamespace(completions=SimpleNamespace(create=create)),
        model_name="test-target",
        max_model_len=64,
        verification_mode="block",
        profile=profile,
    )
    if compact:
        target.configure_evaluation(
            temperature=0.0, requires_target_logits=False, block_output="auto"
        )
    cache = target.new_cache()
    prefill = target(
        input_ids=torch.tensor([[1, 2, 3]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        past_key_values=cache,
        output_hidden_states=True,
    )
    suffix = target(
        input_ids=torch.tensor([[4, 0]]),
        position_ids=torch.tensor([[3, 4]]),
        past_key_values=cache,
        output_hidden_states=True,
    )
    assert recorded == [([1, 2, 3], 1, 0), ([1, 2, 3, 4, 0], 2, 3)]
    assert target.num_target_requests == 2
    assert cache.tokens == [1, 2, 3, 4, 0]
    if compact:
        assert prefill.logits is suffix.logits is None
        assert prefill.greedy_token_ids.tolist() == [[4]]
        assert suffix.greedy_token_ids.tolist() == [[4, 4]]
    else:
        assert prefill.logits.shape == (1, 1, 5)
        assert suffix.logits.shape == (1, 2, 5)
        torch.testing.assert_close(
            prefill.logits[0, 0], torch.log_softmax(torch.arange(5).float() * 2, 0)
        )
        torch.testing.assert_close(
            suffix.logits[0],
            torch.log_softmax(torch.tensor([[3.0], [4.0]]) * torch.arange(5), -1),
        )
    assert prefill.hidden_states[1].shape == (1, 3, 4)
    torch.testing.assert_close(
        suffix.hidden_states[1],
        torch.tensor([[[13.0] * 4, [14.0] * 4]], dtype=torch.bfloat16),
    )
    assert list(tmp_path.iterdir()) == []


def test_compact_argmax_preserves_logsoftmax_rounding_ties(ready):
    ready.logits.copy_(torch.tensor([[0.0, 1e-8, 0.0, 0.0, 0.0]]).expand(2, -1))
    assert ready.logits.argmax(-1).tolist() == [1, 1]
    expected = ready.logits.log_softmax(-1).argmax(-1)
    assert expected.tolist() == [0, 0]
    ready.metadata.requests[0].version = 2
    ready.metadata.requests[0].output_mode = "greedy"
    _capture(ready)
    packet = load_file(str(ready.path))
    assert torch.equal(packet["greedy_token_ids"], expected)
    assert "logprobs" not in packet


def test_compact_packet_reduces_vocabulary_payload_without_changing_hs(ready):
    ready.connector._vocab_size = 129280
    logits = torch.zeros(2, 129280)
    logits[:, 128000] = 1.0
    ready.model.compute_logits = lambda hidden: logits
    _capture(ready)
    full = load_file(str(ready.path))
    full_bytes = ready.path.stat().st_size
    compact_path = ready.path.with_name("cmpl-compact-0.safetensors")
    ready.metadata.requests[0].filename = str(compact_path)
    ready.metadata.requests[0].version = 2
    ready.metadata.requests[0].output_mode = "greedy"
    _capture(ready)
    compact = load_file(str(compact_path))
    assert compact["greedy_token_ids"].tolist() == [128000, 128000]
    assert torch.equal(compact["hidden_states"], full["hidden_states"])
    assert compact_path.stat().st_size < full_bytes / 100
