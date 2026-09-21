"""CPU training trajectories with real process RNG and num_workers=0 data reads."""

# The production snapshot deliberately supports these legacy process-global RNGs.
# ruff: noqa: NPY002

import copy
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).parents[3]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def support():
    # Reuse the established actual-Trainer AST shim without copying or changing it.
    helpers = _load_module(
        "rng_supervision_helpers",
        Path(__file__).with_name("test_trainer_supervision.py"),
    )
    sampler = _load_module(
        "rng_resume_sampler",
        ROOT / "src/speculators/train/distributed_batch_sampler.py",
    )
    return SimpleNamespace(
        module=helpers.trainer_module.__wrapped__(),
        make=helpers._make_trainer,
        sampler=sampler.MultipackDistributedBatchSamplerV2,
    )


@pytest.fixture(autouse=True)
def preserve_rng():
    original = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    yield
    random.setstate(original[0])
    np.random.set_state(original[1])
    torch.set_rng_state(original[2])


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _InterruptionError(Exception):
    pass


class _NoisyRows(Dataset):
    def __init__(self, count, empty_ids=()):
        self.count = count
        self.empty_ids = set(empty_ids)
        self.read_ids = []

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        index = int(index)
        self.read_ids.append(index)
        return {
            "row_id": torch.tensor(index),
            "noise": torch.tensor(
                [random.random(), np.random.random(), torch.rand(()).item()]
            ),
            "loss_mask": torch.tensor([index not in self.empty_ids]),
            "document_ids": torch.tensor([0]),
        }


class _FallbackSampler(Sampler):
    """Same row order as Multipack, with no private fast-skip/cache interface."""

    def __init__(self, source):
        self.source = source

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        return iter(self.source)


class _RandomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(3))
        self.trace = []
        self.stop_first_validation = False

    def forward(self, row_id, noise, loss_mask, document_ids):
        del document_ids
        # This stands in for the model's random anchor/noise selection, and must
        # share the real global torch RNG with data reads and iterator setup.
        anchor_draw = torch.rand(3)
        loss = (self.weight * noise + anchor_draw).square().mean()
        self.trace.append(
            {
                "training": self.training,
                "row_id": row_id.detach().clone(),
                "noise": noise.detach().clone(),
                "anchor": anchor_draw.detach().clone(),
                "loss": loss.detach().clone(),
            }
        )
        if not self.training and self.stop_first_validation:
            self.stop_first_validation = False
            raise _InterruptionError
        return (
            None,
            loss,
            {
                "loss_sum": loss.detach().clone(),
                "loss_total": torch.tensor(1.0),
                "supervision_total": loss_mask.sum().float(),
            },
        )


def _make_trainer(support, path, *, fast=True, empty=False, epochs=1, validation=False):
    trainer = support.make(support.module, path, [])
    trainer.model = _RandomModel()
    trainer.config = trainer.config._replace(num_epochs=epochs)
    trainer.optimizers = [torch.optim.AdamW(trainer.model.parameters(), lr=0.01)]
    trainer.schedulers = [
        torch.optim.lr_scheduler.StepLR(trainer.optimizers[0], 1, 0.9)
    ]
    sampler = support.sampler(
        batch_max_length=1, lengths=[1] * 6, num_replicas=1, rank=0
    )
    order = [int(batch[0]) for batch in sampler]
    # Interrupt after an empty second batch; another empty batch remains ahead.
    rows = _NoisyRows(6, (order[1], order[4]) if empty else ())
    trainer.train_loader = DataLoader(
        rows,
        batch_sampler=sampler if fast else _FallbackSampler(sampler),
        num_workers=0,
    )
    trainer.val_loader = (
        DataLoader(_NoisyRows(2), batch_size=1, num_workers=0) if validation else None
    )
    trainer.checkpointer = SimpleNamespace(
        path=path,
        prev_path=path / "interrupted",
        previous_epoch=0,
        mark_epoch_finalized=Mock(),
    )
    trainer._resume_validation_epoch = None
    trainer._pending_rng_states = None
    trainer._rng_restore_pending = False
    trainer._resume_rng_after_loader = False
    trainer._prepare_validation_checkpoint = Mock()
    trainer.maybe_update_best = Mock()
    return trainer


def _snapshot(trainer):
    rng = trainer._checkpoint_rng_states("interrupted", 0)
    trainer._save_training_state("interrupted", 0, rng_states=rng)
    return {
        "progress": json.loads(
            (trainer.checkpointer.prev_path / "training_state.json").read_text()
        ),
        "model": copy.deepcopy(trainer.model.state_dict()),
        "optimizer": copy.deepcopy(trainer.optimizers[0].state_dict()),
        "scheduler": copy.deepcopy(trainer.schedulers[0].state_dict()),
    }


def _resume(trainer, saved):
    trainer.model.load_state_dict(saved["model"])
    trainer.optimizers[0].load_state_dict(saved["optimizer"])
    trainer.schedulers[0].load_state_dict(saved["scheduler"])
    trainer.current_epoch = trainer.checkpointer.previous_epoch + 1
    trainer._restore_training_progress(saved["progress"])
    trainer.global_step = trainer._resume_global_step
    # Real initialization, logging or checkpoint loading may consume randomness
    # after progress is parsed. A pending restore must undo this only at run time.
    for _ in range(7):
        random.random()
        np.random.random()
        torch.rand(3)


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for current, wanted in zip(actual, expected, strict=True):
            _assert_equal(current, wanted)
    else:
        assert actual == expected


def _assert_finished_matches(support, actual, expected, expected_rng):
    _assert_equal(actual.model.state_dict(), expected.model.state_dict())
    _assert_equal(
        actual.optimizers[0].state_dict(), expected.optimizers[0].state_dict()
    )
    _assert_equal(
        actual.schedulers[0].state_dict(), expected.schedulers[0].state_dict()
    )
    assert actual.global_step == expected.global_step
    assert support.module.capture_rng_states("cpu") == expected_rng
    assert not actual._rng_restore_pending


@pytest.mark.parametrize("fast", [False, True], ids=["fallback", "fast"])
@pytest.mark.parametrize("empty", [False, True], ids=["supervised", "empty-batches"])
def test_mid_epoch_replays_data_model_and_optimizer_random_trajectory(
    support, tmp_path, fast, empty
):
    _seed(314)
    expected = _make_trainer(support, tmp_path / "full", fast=fast, empty=empty)
    expected.run_training()
    expected_rng = support.module.capture_rng_states("cpu")

    _seed(314)
    interrupted = _make_trainer(support, tmp_path / "cut", fast=fast, empty=empty)
    record = interrupted._record_checkpoint_position

    def stop_after_second(epoch, local_step, **kwargs):
        record(epoch, local_step, **kwargs)
        if local_step == 2:
            raise _InterruptionError

    interrupted._record_checkpoint_position = stop_after_second
    with pytest.raises(_InterruptionError):
        interrupted.run_training()
    saved = _snapshot(interrupted)
    _assert_equal(interrupted.model.trace, expected.model.trace[:2])
    assert saved["progress"]["local_step"] == 2
    assert not saved["progress"]["epoch_complete"]
    assert saved["progress"]["global_step"] == (2 if empty else 3)

    _seed(999)
    resumed = _make_trainer(support, tmp_path / "resume", fast=fast, empty=empty)
    _resume(resumed, saved)
    assert resumed._resume_rng_after_loader
    resumed.run_training()

    _assert_equal(resumed.model.trace, expected.model.trace[2:])
    _assert_finished_matches(support, resumed, expected, expected_rng)
    expected_ids = expected.train_loader.dataset.read_ids
    assert resumed.train_loader.dataset.read_ids == (
        expected_ids[2:] if fast else expected_ids
    )


@pytest.mark.parametrize("phase", ["validation", "finalized"])
def test_phase_resume_replays_validation_and_next_epoch(support, tmp_path, phase):
    _seed(271)
    expected = _make_trainer(support, tmp_path / "full", epochs=2, validation=True)
    expected.run_training()
    expected_rng = support.module.capture_rng_states("cpu")

    _seed(271)
    interrupted = _make_trainer(support, tmp_path / "cut", epochs=2, validation=True)
    if phase == "validation":
        interrupted.model.stop_first_validation = True
    else:
        record = interrupted._record_checkpoint_position

        def stop_after_finalized(epoch, local_step, **kwargs):
            record(epoch, local_step, **kwargs)
            if epoch == 0 and kwargs.get("finalized"):
                raise _InterruptionError

        interrupted._record_checkpoint_position = stop_after_finalized
    with pytest.raises(_InterruptionError):
        interrupted.run_training()
    saved = _snapshot(interrupted)
    assert saved["progress"]["epoch_complete"]
    assert saved["progress"]["epoch_finalized"] == (phase == "finalized")

    _seed(999)
    resumed = _make_trainer(support, tmp_path / "resume", epochs=2, validation=True)
    _resume(resumed, saved)
    assert not resumed._resume_rng_after_loader
    resumed.run_training()

    # Pending validation replays its whole epoch from pre-validation RNG, even
    # when the interrupted validation already consumed a batch and model draws.
    offset = 6 if phase == "validation" else 8
    _assert_equal(resumed.model.trace, expected.model.trace[offset:])
    _assert_finished_matches(support, resumed, expected, expected_rng)
