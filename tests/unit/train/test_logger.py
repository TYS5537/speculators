import json
import logging

import pytest

from speculators.train.logger import IsRank0Filter, TensorBoardHandler


def _record(msg="msg", **extra):
    record = logging.LogRecord(
        "speculators", logging.INFO, __file__, 0, msg, None, None
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


@pytest.fixture
def clean_rank_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def test_global_rank0_filter_passes_only_global_rank0(monkeypatch, clean_rank_env):
    # Multi-node: a non-zero global rank that happens to be local_rank 0
    # must still be filtered out (the bug this guards against).
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert IsRank0Filter().filter(_record()) is False

    monkeypatch.setenv("RANK", "0")
    assert IsRank0Filter().filter(_record()) is True


def test_override_bypasses_filter(clean_rank_env, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    assert IsRank0Filter().filter(_record(override_rank0_filter=True)) is True


@pytest.fixture
def tensorboard_handler(tmp_path):
    pytest.importorskip("tensorboard")
    handler = TensorBoardHandler(log_dir=tmp_path, run_name="run")
    try:
        yield handler
    finally:
        handler.close()


def test_tensorboard_keeps_reading_metrics_after_hparams(tensorboard_handler, tmp_path):
    event_accumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator"
    )
    handler = tensorboard_handler
    handler.handle(_record({"lr": 0.0006}, hparams=True))
    handler.flush()
    reader = event_accumulator.EventAccumulator(str(tmp_path / "run"))
    # Start reading before the first training update, exactly like a live server.
    reader.Reload()
    for step, loss in enumerate([0.75, 0.5, 0.25], start=1):
        handler.handle(_record({"train": {"loss": loss}}, step=step))
        handler.flush()
        reader.Reload()
        assert "train/loss" in reader.Tags()["scalars"]
        events = reader.Scalars("train/loss")
        assert [event.step for event in events] == list(range(1, step + 1))
        assert events[-1].value == pytest.approx(loss)
    assert len(list((tmp_path / "run").glob("events.out.tfevents.*"))) == 1


@pytest.mark.parametrize("step", [None, 7])
def test_tensorboard_preserves_hparams_in_one_file(tensorboard_handler, tmp_path, step):
    event_file_loader = pytest.importorskip(
        "tensorboard.backend.event_processing.event_file_loader"
    )
    plugin_data = pytest.importorskip("tensorboard.plugins.hparams.plugin_data_pb2")
    handler = tensorboard_handler
    handler.handle(
        _record(
            {
                "optimizer": {"lr": 0.0006, "enabled": True},
                "layers": [1, 9, 17],
                "precision": "bfloat16",
                "optional": None,
            },
            hparams=True,
            step=step,
        )
    )
    # Configuration metadata must be visible without waiting for the flush timer.
    files = list((tmp_path / "run").glob("events.out.tfevents.*"))
    assert len(files) == 1
    records = {}
    for event in event_file_loader.EventFileLoader(str(files[0])).Load():
        for value in event.summary.value:
            if value.metadata.plugin_data.plugin_name != "hparams":
                continue
            data = plugin_data.HParamsPluginData.FromString(
                value.metadata.plugin_data.content
            )
            records[data.WhichOneof("data")] = data
            assert event.step == (0 if step is None else step)
    assert set(records) == {"experiment", "session_start_info", "session_end_info"}
    params = records["session_start_info"].session_start_info.hparams
    assert params["optimizer/lr"].number_value == pytest.approx(0.0006)
    # PyTorch treats bool as numeric here; preserve its existing serialization.
    assert params["optimizer/enabled"].number_value == 1.0
    assert params["layers"].string_value == json.dumps([1, 9, 17])
    assert params["precision"].string_value == "bfloat16"
    assert "optional" not in params


def test_tensorboard_nonzero_rank_creates_no_files(
    tensorboard_handler, tmp_path, monkeypatch, clean_rank_env
):
    monkeypatch.setenv("RANK", "1")
    tensorboard_handler.addFilter(IsRank0Filter())
    tensorboard_handler.handle(_record({"lr": 0.0006}, hparams=True))
    tensorboard_handler.handle(_record({"loss": 0.5}, step=1))
    tensorboard_handler.flush()
    assert not list(tmp_path.rglob("events.out.tfevents.*"))
