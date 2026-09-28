"""Forward timing is opt-in, sampled, rank-safe and outside checkpoint state."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from speculators.models.mmuse import profiling


@pytest.mark.parametrize("interval", [None, "0"])
def test_disabled_profiling_does_not_touch_clock_device_or_model(monkeypatch, interval):
    if interval is None:
        monkeypatch.delenv("MMUSE_PROFILE_FORWARD", raising=False)
    else:
        monkeypatch.setenv("MMUSE_PROFILE_FORWARD", interval)
    model = torch.nn.Linear(2, 2)
    attributes = vars(model).copy()
    factory = Mock(side_effect=AssertionError("Disabled profiling must be a no-op"))
    monkeypatch.setattr(profiling, "_ForwardProfile", factory)
    for _ in range(3):
        profile = profiling.start_forward_profile(model, torch.device("cpu"))
        profile.mark("backbone")
        profile.finish()
    factory.assert_not_called()
    assert vars(model) == attributes


def test_sampling_excludes_validation_and_is_not_saved_in_checkpoint(monkeypatch):
    monkeypatch.setenv("MMUSE_PROFILE_FORWARD", "3")
    model = torch.nn.Linear(2, 2)
    original_keys = model.state_dict().keys()
    factory = Mock()
    monkeypatch.setattr(profiling, "_ForwardProfile", factory)
    device = torch.device("cpu")
    assert profiling.start_forward_profile(model, device) is factory.return_value
    assert profiling.start_forward_profile(model, device) is profiling._NO_PROFILE
    model.eval()
    for _ in range(4):
        assert profiling.start_forward_profile(model, device) is profiling._NO_PROFILE
    model.train()
    assert profiling.start_forward_profile(model, device) is profiling._NO_PROFILE
    assert profiling.start_forward_profile(model, device) is factory.return_value
    assert [call.args for call in factory.call_args_list] == [(device, 0), (device, 3)]
    assert model.state_dict().keys() == original_keys


def test_disabled_profiling_does_not_add_a_compile_graph_break(monkeypatch):
    monkeypatch.delenv("MMUSE_PROFILE_FORWARD", raising=False)
    model = torch.nn.Linear(2, 2)

    def forward(value):
        profile = profiling.start_forward_profile(model, value.device)
        result = model(value)
        profile.mark("backbone")
        profile.finish()
        return result

    value = torch.ones(1, 2)
    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(value), forward(value), rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["cpu", "cuda", "npu"])
@pytest.mark.parametrize("rank", [0, 1])
def test_stage_timings_synchronize_and_only_rank_zero_logs(
    monkeypatch, caplog, backend, rank
):
    monkeypatch.setenv("MMUSE_PROFILE_FORWARD", "1")
    clock = Mock(side_effect=[10.0, 10.125, 10.375])
    monkeypatch.setattr(profiling, "time", SimpleNamespace(perf_counter=clock))
    synchronize = Mock()
    if backend == "cuda":
        monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    elif backend == "npu":
        monkeypatch.setattr(
            torch, backend, SimpleNamespace(synchronize=synchronize), raising=False
        )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    device = SimpleNamespace(type=backend)
    model = torch.nn.Linear(2, 2)
    with caplog.at_level(logging.INFO, logger=profiling.__name__):
        profile = profiling.start_forward_profile(model, device)
        profile.mark("backbone")
        profile.mark("correction_markov")
        profile.finish()
    assert profile.timings == {"backbone": 125.0, "correction_markov": 250.0}
    assert clock.call_count == 3
    assert synchronize.call_count == (0 if backend == "cpu" else 3)
    if backend != "cpu":
        synchronize.assert_called_with(device)
    if rank == 0:
        assert "call=0 backbone_ms=125.00 correction_markov_ms=250.00" in caplog.text
        assert "total_ms=375.00 (synchronized sample)" in caplog.text
    else:
        assert not caplog.records


def test_sync_failure_does_not_abort_training_or_publish_false_timings(
    monkeypatch, caplog
):
    monkeypatch.setenv("MMUSE_PROFILE_FORWARD", "1")
    synchronize = Mock(side_effect=[None, RuntimeError("device sync failed")])
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=synchronize), raising=False
    )
    model = torch.nn.Linear(2, 2)
    with caplog.at_level(logging.INFO, logger=profiling.__name__):
        profile = profiling.start_forward_profile(model, SimpleNamespace(type="npu"))
        profile.mark("backbone")
        profile.mark("correction_markov")
        profile.finish()
    assert synchronize.call_count == 2
    assert "device sync failed" in caplog.text
    assert "total_ms=" not in caplog.text


@pytest.mark.parametrize("interval", ["-1", "invalid"])
def test_invalid_interval_fails_before_updating_model(monkeypatch, interval):
    monkeypatch.setenv("MMUSE_PROFILE_FORWARD", interval)
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError):
        profiling.start_forward_profile(model, torch.device("cpu"))
    assert not hasattr(model, "_mmuse_profile_forward_calls")
