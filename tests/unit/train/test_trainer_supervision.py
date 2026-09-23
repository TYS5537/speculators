"""CPU regressions for empty supervision and best-checkpoint progress.

Load the production trainer definitions without model/backend imports; the actual
training, validation and checkpoint methods run with tiny CPU models below.
"""

import ast
import copy
import importlib.util
import json
import logging
import math
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal, NamedTuple
from unittest.mock import ANY, Mock

import pytest
import torch


@pytest.fixture
def trainer_module():
    root = Path(__file__).parents[3]
    rng_spec = importlib.util.spec_from_file_location(
        "supervision_rng", root / "src/speculators/train/rng.py"
    )
    rng_module = importlib.util.module_from_spec(rng_spec)
    rng_spec.loader.exec_module(rng_module)
    module = ModuleType("supervision_cpu_trainer")
    module.__dict__.update(
        torch=torch,
        dist=SimpleNamespace(ReduceOp=torch.distributed.ReduceOp),
        json=json,
        logging=logging,
        math=math,
        time=time,
        islice=islice,
        Path=Path,
        Literal=Literal,
        NamedTuple=NamedTuple,
        root_logger=logging.getLogger("supervision-test"),
        metric_logger=Mock(),
        MIN_STEP_PCT=0.25,
        _VAL_SYNC_INTERVAL=50,
        # Recovery collectives are covered by test_data_recovery; these tests
        # isolate supervision accounting with no recovery metadata present.
        BatchRecoveryCoordinator=lambda _phase: SimpleNamespace(
            consume=lambda batch, **_kwargs: None
        ),
        with_graceful_shutdown=lambda: lambda function: function,
        _rank0_only=lambda function: function,
        TrainingInterruptedError=RuntimeError,
        capture_rng_states=rng_module.capture_rng_states,
        restore_rng_states=rng_module.restore_rng_states,
    )
    sources = {
        "src/speculators/train/utils.py": {"normalize_counted_metrics"},
        "src/speculators/train/trainer.py": {
            "_synchronize_device",
            "_all_reduce_metrics",
            "_StepTimer",
            "TrainerConfig",
            "Trainer",
        },
    }
    for source, names in sources.items():
        tree = ast.parse((root / source).read_text(encoding="utf-8"))
        selected = [node for node in tree.body if getattr(node, "name", None) in names]
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        tree = ast.fix_missing_locations(
            ast.Module(body=[future, *selected], type_ignores=[])
        )
        exec(compile(tree, str(root / source), "exec"), module.__dict__)  # noqa: S102
    return module


class _Loader(list):
    batch_sampler = SimpleNamespace()


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.backward_grads = []
        self.awaiting_backward = False
        self.weight.register_hook(self._record_backward)

    def _record_backward(self, gradient):
        self.backward_grads.append(gradient.detach().clone())
        self.awaiting_backward = False

    def forward(self, loss_mask, document_ids, aligned_count=None):
        del document_ids
        if self.training and torch.is_grad_enabled():
            assert not self.awaiting_backward, "forward must be paired with backward"
            self.awaiting_backward = True
        # This can include an auxiliary objective even on an empty batch. The
        # trainer must zero it, not merely assume the model already did so.
        loss = self.weight.square()
        metrics = {
            "loss_sum": loss.detach().clone(),
            "loss_total": torch.tensor(1.0),
            "supervision_total": (
                loss_mask.sum() if aligned_count is None else aligned_count
            ).float(),
            "plain_probe": torch.tensor(3.0),
        }
        return None, loss, metrics


def _batch(mask=1, aligned_count=None):
    batch = {
        "loss_mask": torch.full((1, 2), mask, dtype=torch.float32),
        "document_ids": torch.zeros((1, 2), dtype=torch.long),
    }
    if aligned_count is not None:
        batch["aligned_count"] = torch.tensor(float(aligned_count))
    return batch


def _make_trainer(module, tmp_path, batches, *, distributed=False):
    trainer = module.Trainer.__new__(module.Trainer)
    trainer.model = _TinyModel()
    trainer.config = module.TrainerConfig(
        lr=0.01,
        num_epochs=2,
        save_path=str(tmp_path),
        log_freq=100,
        hidden_states_dtype=torch.bfloat16,
    )
    trainer.train_loader = _Loader(batches)
    trainer.val_loader = _Loader(batches)
    trainer.local_rank = "cpu"
    trainer.device_type = "cpu"
    trainer.rank = 1  # Avoid rendering progress bars, including in single-device tests.
    trainer.is_distributed = distributed
    trainer.current_epoch = 0
    trainer.global_step = 1
    trainer._resume_local_step = 0
    trainer._ssal_curriculum = False
    trainer.optimizers = [torch.optim.AdamW(trainer.model.parameters(), lr=0.01)]
    trainer.schedulers = [
        torch.optim.lr_scheduler.StepLR(trainer.optimizers[0], 1, 0.9)
    ]
    return trainer


def _set_collectives(module, values, world_size=2):
    remaining = iter(values)

    def all_reduce(tensor, **_kwargs):
        value = next(remaining)
        if isinstance(value, list):
            tensor.copy_(torch.tensor(value, dtype=tensor.dtype))
        else:
            tensor.fill_(value)

    module.dist.all_reduce = Mock(side_effect=all_reduce)
    module.dist.get_world_size = lambda: world_size
    return module.dist.all_reduce


@pytest.mark.parametrize("distributed", [False, True])
def test_globally_empty_batch_preserves_momentum_weights_and_progress(
    trainer_module, tmp_path, distributed
):
    trainer = _make_trainer(
        trainer_module, tmp_path, [_batch(aligned_count=0)], distributed=distributed
    )
    if distributed:
        _set_collectives(trainer_module, [0])
    opt = trainer.optimizers[0]
    trainer.model.weight.square().backward()
    opt.step()
    opt.zero_grad()
    before = trainer.model.weight.detach().clone()
    state = copy.deepcopy(opt.state[trainer.model.weight])
    scheduler_state = trainer.schedulers[0].state_dict()
    trainer.model.backward_grads.clear()

    trainer.train_epoch(0)

    torch.testing.assert_close(trainer.model.weight, before, rtol=0, atol=0)
    for key, value in state.items():
        torch.testing.assert_close(opt.state[trainer.model.weight][key], value)
    assert trainer.schedulers[0].state_dict() == scheduler_state
    assert trainer.global_step == 1
    assert len(trainer.model.backward_grads) == 1
    assert trainer.model.backward_grads[0].item() == 0
    assert trainer.model.weight.grad is None
    assert not trainer.model.awaiting_backward


def test_empty_then_valid_batch_completes_both_backward_lifecycles(
    trainer_module, tmp_path
):
    trainer = _make_trainer(trainer_module, tmp_path, [_batch(0), _batch(1)])
    trainer.train_epoch(0)
    assert len(trainer.model.backward_grads) == 2
    assert trainer.global_step == 2
    assert trainer.optimizers[0].state[trainer.model.weight]["step"].item() == 1


@pytest.mark.parametrize("local_active", [False, True])
def test_one_active_rank_keeps_all_ranks_in_backward_and_optimizer_step(
    trainer_module, tmp_path, local_active
):
    trainer = _make_trainer(
        trainer_module, tmp_path, [_batch(int(local_active))], distributed=True
    )
    _set_collectives(trainer_module, [1])
    trainer.train_epoch(0)
    assert trainer.global_step == 2
    assert trainer.optimizers[0].state[trainer.model.weight]["step"].item() == 1
    # Before the real DDP/FSDP gradient average, the valid rank contributes 2x
    # its normal 0.5 gradient; the empty rank contributes zero, not an early exit.
    assert trainer.model.backward_grads[0].item() == (1.0 if local_active else 0.0)


def test_all_valid_training_retains_existing_numerical_updates(
    trainer_module, tmp_path
):
    trainer = _make_trainer(trainer_module, tmp_path, [_batch(), _batch()])
    expected = _TinyModel()
    optimizer = torch.optim.AdamW(expected.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
    for _ in range(2):
        optimizer.zero_grad()
        expected.weight.square().backward()
        torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    trainer.train_epoch(0)
    torch.testing.assert_close(trainer.model.weight, expected.weight, rtol=0, atol=0)
    assert trainer.schedulers[0].state_dict() == scheduler.state_dict()
    assert trainer.global_step == 3


@pytest.mark.parametrize("active_ranks", [1, 2])
def test_training_logs_average_only_supervised_ranks(
    trainer_module, tmp_path, active_ranks
):
    trainer = _make_trainer(trainer_module, tmp_path, [_batch()], distributed=True)
    trainer.config = trainer.config._replace(log_freq=1)
    _set_collectives(
        trainer_module,
        [
            active_ranks,
            [
                0.0625 * active_ranks,
                active_ranks,
                2 * active_ranks,
                3 * active_ranks,
                0,
                1,
            ],
        ],
    )
    trainer.train_epoch(0)
    logged = trainer_module.metric_logger.info.call_args.args[0]["train"]
    assert logged == {"loss": 0.0625, "plain_probe": 3.0, "error_records": 0.0}


def test_supervision_count_overrides_input_mask_and_legacy_falls_back(
    trainer_module, tmp_path
):
    trainer = _make_trainer(trainer_module, tmp_path, [])
    device = torch.device("cpu")
    assert trainer._supervision_status(_batch(), {"supervision_total": 0}, device) == (
        False,
        0,
    )
    assert trainer._supervision_status(_batch(0), {}, device) == (False, 0)
    assert trainer._supervision_status(_batch(1), {}, device) == (True, 1)
    assert trainer._supervision_status({}, {}, device) == (True, 1)


def test_validation_excludes_empty_batch_from_loss_and_plain_metric(
    trainer_module, tmp_path
):
    trainer = _make_trainer(trainer_module, tmp_path, [_batch(), _batch(0)])
    metrics = trainer.val_epoch(0)
    assert metrics == {"loss_epoch": 0.0625, "plain_probe_epoch": 3.0}


def test_validation_excludes_empty_ranks_without_skipping_collectives(
    trainer_module, tmp_path
):
    trainer = _make_trainer(
        trainer_module, tmp_path, [_batch(), _batch(0)], distributed=True
    )
    # First batch has two valid ranks; second has only the remote rank. Each
    # active batch packs loss_sum, loss_total, supervision_total, plain_probe.
    collective = _set_collectives(trainer_module, [2, [1, 2, 4, 6], 1, [9, 1, 2, 6]])
    metrics = trainer.val_epoch(0)
    assert metrics["loss_epoch"] == pytest.approx(10 / 3)
    assert metrics["plain_probe_epoch"] == 4
    assert collective.call_count == 4


@pytest.mark.parametrize("distributed", [False, True])
def test_wholly_unsupervised_validation_cannot_replace_best_checkpoint(
    trainer_module, tmp_path, distributed
):
    trainer = _make_trainer(
        trainer_module, tmp_path, [_batch(0)], distributed=distributed
    )
    if distributed:
        collective = _set_collectives(trainer_module, [0])
    trainer.best_val_loss = 1.0
    trainer.checkpointer = Mock()
    result = trainer.val_epoch(0)
    assert result is None
    trainer.maybe_update_best(0, result)
    assert trainer.best_val_loss == 1.0
    assert trainer.checkpointer.mock_calls == []
    if distributed:
        assert collective.call_count == 1


def test_empty_validation_loader_returns_no_metrics(trainer_module, tmp_path):
    trainer = _make_trainer(trainer_module, tmp_path, [])
    assert trainer.val_epoch(0) is None


def test_best_checkpoint_saves_and_restores_training_progress(trainer_module, tmp_path):
    trainer = _make_trainer(trainer_module, tmp_path, [_batch()])
    trainer.config = trainer.config._replace(save_best=True)
    trainer.global_step = 900
    trainer.best_val_loss = float("inf")
    trainer.checkpointer = Mock(
        path=tmp_path, previous_epoch=2, prev_path=tmp_path / "2"
    )
    trainer.checkpointer.load_best_val_loss.return_value = 0.0625
    trainer.checkpointer.checkpoint_transaction.side_effect = lambda *a, **k: (
        nullcontext()
    )
    trainer.maybe_save_checkpoint(2)
    trainer.checkpointer.save_checkpoint.assert_not_called()

    trainer.maybe_update_best(2, {"loss_epoch": 0.0625})

    state = json.loads((tmp_path / "2" / "training_state.json").read_text())
    assert state.pop("rng_states")["world_size"] == 1
    assert state == {
        "epoch": 2,
        "local_step": 0,
        "global_step": 900,
        "epoch_complete": True,
        "epoch_finalized": False,
    }
    trainer.checkpointer.save_scheduler_state_dict.assert_called_once_with(
        trainer.schedulers, 2
    )
    trainer.resume_from_checkpoint = True
    trainer.global_step = 0
    trainer.setup_trainer()
    # The bundle is saved before the epoch bookkeeping is finalized by the loop.
    assert trainer.current_epoch == 2
    assert trainer._resume_validation_epoch == 2
    assert trainer.global_step == 900
    assert trainer._resume_local_step == 0


def _make_best_trainer(module, tmp_path, save_best, best_loss=0.25):
    trainer = _make_trainer(module, tmp_path, [_batch()])
    trainer.config = trainer.config._replace(save_best=save_best)
    trainer.best_val_loss = best_loss
    trainer.checkpointer = Mock()
    trainer.checkpointer.checkpoint_transaction.side_effect = lambda *a, **k: (
        nullcontext()
    )
    trainer._save_training_state = Mock()
    return trainer


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("save_best", [False, True])
@pytest.mark.parametrize("best_loss", [float("inf"), 0.25])
def test_nonfinite_validation_cannot_mutate_best_checkpoint(
    trainer_module, tmp_path, caplog, *, loss, save_best, best_loss
):
    trainer = _make_best_trainer(trainer_module, tmp_path, save_best, best_loss)

    with caplog.at_level(logging.WARNING, logger="supervision-test"):
        trainer.maybe_update_best(2, {"loss_epoch": loss})

    assert trainer.best_val_loss == best_loss
    assert trainer.checkpointer.mock_calls == []
    trainer._save_training_state.assert_not_called()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert str(loss) in caplog.records[0].getMessage()


@pytest.mark.parametrize("save_best", [False, True])
@pytest.mark.parametrize("best_loss", [float("inf"), 0.25])
def test_finite_improvement_keeps_existing_best_checkpoint_behavior(
    trainer_module, tmp_path, save_best, best_loss
):
    trainer = _make_best_trainer(trainer_module, tmp_path, save_best, best_loss)
    metrics = {"loss_epoch": 0.125}

    trainer.maybe_update_best(2, metrics)

    assert trainer.best_val_loss == 0.125
    trainer.checkpointer.save_val_metrics.assert_called_once_with(2, metrics)
    trainer.checkpointer.update_best_symlink.assert_called_once_with(2)
    if save_best:
        trainer.checkpointer.save_checkpoint.assert_called_once_with(
            trainer.model, trainer.optimizers, 2
        )
        trainer.checkpointer.save_scheduler_state_dict.assert_called_once_with(
            trainer.schedulers, 2
        )
        trainer._save_training_state.assert_called_once_with(2, 0, rng_states=ANY)
        trainer.checkpointer.cleanup_keep_only_best.assert_called_once_with(
            best_epoch=2
        )
    else:
        trainer.checkpointer.save_checkpoint.assert_not_called()
        trainer.checkpointer.save_scheduler_state_dict.assert_not_called()
        trainer._save_training_state.assert_not_called()
        trainer.checkpointer.cleanup_keep_only_best.assert_not_called()


@pytest.mark.parametrize("save_best", [False, True])
@pytest.mark.parametrize("loss", [0.25, 0.5])
def test_finite_nonimprovement_keeps_best_untouched(
    trainer_module, tmp_path, save_best, loss
):
    trainer = _make_best_trainer(trainer_module, tmp_path, save_best)

    trainer.maybe_update_best(2, {"loss_epoch": loss})

    assert trainer.best_val_loss == 0.25
    assert trainer.checkpointer.mock_calls == []
    trainer._save_training_state.assert_not_called()


@pytest.mark.parametrize("save_best", [False, True])
@pytest.mark.parametrize("metrics", [None, {}, {"accuracy_epoch": 1.0}])
def test_absent_validation_loss_keeps_best_untouched(
    trainer_module, tmp_path, save_best, metrics
):
    trainer = _make_best_trainer(trainer_module, tmp_path, save_best)

    trainer.maybe_update_best(2, metrics)

    assert trainer.best_val_loss == 0.25
    assert trainer.checkpointer.mock_calls == []
    trainer._save_training_state.assert_not_called()


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("save_best", [False, True])
def test_finite_improvement_after_anomaly_can_still_update_best(
    trainer_module, tmp_path, loss, save_best
):
    trainer = _make_best_trainer(trainer_module, tmp_path, save_best)
    trainer.maybe_update_best(2, {"loss_epoch": loss})
    assert trainer.checkpointer.mock_calls == []

    metrics = {"loss_epoch": 0.125}
    trainer.maybe_update_best(3, metrics)

    assert trainer.best_val_loss == 0.125
    trainer.checkpointer.save_val_metrics.assert_called_once_with(3, metrics)
    trainer.checkpointer.update_best_symlink.assert_called_once_with(3)
    if save_best:
        trainer.checkpointer.cleanup_keep_only_best.assert_called_once_with(
            best_epoch=3
        )
        trainer._save_training_state.assert_called_once_with(3, 0, rng_states=ANY)
    else:
        trainer.checkpointer.cleanup_keep_only_best.assert_not_called()
        trainer._save_training_state.assert_not_called()
