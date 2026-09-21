"""Real CPU serialization and fault-injection tests for checkpoint publication."""

import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from . import test_optimizer_checkpoint_counters as counters
from . import test_trainer_supervision as supervision


@pytest.fixture
def checkpoint_module():
    return counters.checkpoint_module.__wrapped__()


@pytest.fixture
def trainer_module():
    return supervision.trainer_module.__wrapped__()


@pytest.fixture
def shutdown_module():
    path = Path(__file__).parents[3] / "src/speculators/train/graceful_shutdown.py"
    spec = importlib.util.spec_from_file_location("checkpoint_test_shutdown", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DiskModel(counters._TinyModel):
    def save_pretrained(self, path, *, state_dict):
        super().save_pretrained(path, state_dict=state_dict)
        # The test loader uses torch.load; production validates the actual model
        # artifact's presence, without needing the Transformers dependency here.
        torch.save(state_dict, path / "model.safetensors")


def _components():
    model = _DiskModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
    return model, optimizer, scheduler


def _step(model, optimizer, scheduler):
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()


def _save(
    cp, model, optimizer, scheduler, *, label=0, step=1, complete=False, finalized=None
):
    with cp.checkpoint_transaction(label, has_scheduler=scheduler is not None):
        cp.save_checkpoint(model, optimizer, label)
        if scheduler is not None:
            cp.save_scheduler_state_dict(scheduler, label)
        state = {
            "epoch": 0,
            "local_step": step,
            "global_step": step,
            "epoch_complete": complete,
        }
        if finalized is not None:
            state["epoch_finalized"] = finalized
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            (cp.path / str(label) / "training_state.json").write_text(json.dumps(state))


@pytest.mark.parametrize("failure", ["model", "optimizer", "scheduler", "progress"])
def test_failed_overwrite_keeps_old_complete_bundle(
    checkpoint_module,
    tmp_path,
    monkeypatch,
    failure,
):
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    model, opt, sched = _components()
    _step(model, opt, sched)
    _save(cp, model, opt, sched)
    previous = {p.name: p.read_bytes() for p in (tmp_path / "0").iterdir()}
    _step(model, opt, sched)
    save_pretrained, torch_save, write_text = (
        model.save_pretrained,
        torch.save,
        Path.write_text,
    )

    def write_model(path, **kwargs):
        save_pretrained(path, **kwargs)
        if failure == "model":
            raise OSError("injected model failure")

    def save_payload(payload, path, **kwargs):
        if Path(path).name == f"{failure}_state_dict.pt":
            raise OSError("injected payload failure")
        return torch_save(payload, path, **kwargs)

    def write_progress(path, text, **kwargs):
        if failure == "progress" and path.name == "training_state.json":
            raise OSError("injected progress failure")
        return write_text(path, text, **kwargs)

    monkeypatch.setattr(model, "save_pretrained", write_model)
    monkeypatch.setattr(torch, "save", save_payload)
    monkeypatch.setattr(Path, "write_text", write_progress)
    with pytest.raises(OSError, match="injected"):
        _save(cp, model, opt, sched, step=2)
    assert cp.path == tmp_path
    restored = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    assert restored.prev_path == tmp_path / "0"
    assert {p.name: p.read_bytes() for p in restored.prev_path.iterdir()} == previous
    assert list(tmp_path.glob(".pending-*"))


def test_publish_rename_failure_rolls_back_and_gap_remains_recoverable(
    checkpoint_module,
    tmp_path,
    monkeypatch,
):
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    model, opt, sched = _components()
    _step(model, opt, sched)
    _save(cp, model, opt, sched)
    original = Path.rename
    observed_gap = []

    def fail_publish(path, target):
        if path.parent.name.startswith(".pending-"):
            scanned = checkpoint_module.SingleGPUCheckpointer(tmp_path)
            observed_gap.append(scanned.prev_path.name)
            assert scanned.previous_epoch == 0
            assert scanned.prev_path.name.startswith(".previous-0-")
            raise OSError("publish failed")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", fail_publish)
    with pytest.raises(OSError, match="publish failed"):
        _save(cp, model, opt, sched, step=2)
    assert len(observed_gap) == 1
    restored = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    assert restored.prev_path == tmp_path / "0"
    assert (
        json.loads((restored.prev_path / "training_state.json").read_text())[
            "global_step"
        ]
        == 1
    )


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("step", [0, 3])
def test_interrupted_auto_resume_and_next_numeric_save(
    checkpoint_module,
    trainer_module,
    tmp_path,
    complete,
    step,
):
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    model, opt, sched = _components()
    for _ in range(step):
        _step(model, opt, sched)
    _save(cp, model, opt, sched, label="interrupted", step=step, complete=complete)
    restored = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    trainer = supervision._make_trainer(trainer_module, tmp_path, [])
    trainer.checkpointer = restored
    trainer.resume_from_checkpoint = True
    trainer.setup_trainer()
    assert trainer.current_epoch == int(complete)
    assert trainer.global_step == step
    assert trainer._resume_local_step == (0 if complete else step)
    restored.load_optimizer_state_dict(model, opt)
    restored.load_scheduler_state_dict(sched)
    assert sched.last_epoch == step
    interrupted_bytes = (tmp_path / "interrupted/optimizer_state_dict.pt").read_bytes()
    _step(model, opt, sched)
    _save(restored, model, opt, sched, step=step + 1)
    assert (
        tmp_path / "interrupted/optimizer_state_dict.pt"
    ).read_bytes() == interrupted_bytes
    assert checkpoint_module.SingleGPUCheckpointer(tmp_path).prev_path == tmp_path / "0"


@pytest.mark.parametrize(
    "damage", ["model", "config", "optimizer", "scheduler", "state", "manifest"]
)
def test_invalid_complete_bundle_falls_back_to_previous_generation(
    checkpoint_module,
    tmp_path,
    damage,
):
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    model, opt, sched = _components()
    _step(model, opt, sched)
    _save(cp, model, opt, sched)
    _step(model, opt, sched)
    _save(cp, model, opt, sched, step=2)
    names = {
        "model": "model.safetensors",
        "config": "config.json",
        "optimizer": "optimizer_state_dict.pt",
        "scheduler": "scheduler_state_dict.pt",
        "state": "training_state.json",
        "manifest": cp.COMMIT_FILENAME,
    }
    (tmp_path / "0" / names[damage]).write_text("" if damage != "manifest" else "{}")
    restored = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    assert restored.prev_path.name.startswith(".previous-0-")
    assert (
        json.loads((restored.prev_path / "training_state.json").read_text())[
            "global_step"
        ]
        == 1
    )


def test_old_numeric_checkpoint_is_still_supported(checkpoint_module, tmp_path):
    (tmp_path / "3").mkdir()
    (tmp_path / "interrupted").mkdir()
    (tmp_path / ".pending-ignored/99").mkdir(parents=True)
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    assert cp.previous_epoch == 3
    assert cp.prev_path == tmp_path / "3"


def test_missing_model_prevents_publication(checkpoint_module, tmp_path):
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    model = counters._TinyModel()
    opt = torch.optim.AdamW(model.parameters())
    with pytest.raises(FileNotFoundError):
        _save(cp, model, opt, None)
    assert not (tmp_path / "0").exists()
    assert checkpoint_module.SingleGPUCheckpointer(tmp_path).previous_epoch == -1


def test_sharded_model_is_complete_and_restorable(
    checkpoint_module, tmp_path, monkeypatch
):
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    model, opt, sched = _components()
    _step(model, opt, sched)

    def save_shards(path, *, state_dict):
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}")
        weights = {}
        for index, (name, tensor) in enumerate(state_dict.items()):
            shard = f"model-{index:05d}.safetensors"
            torch.save({name: tensor}, path / shard)
            weights[name] = shard
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weights})
        )

    @contextmanager
    def open_shard(path, **_kwargs):
        tensors = torch.load(path, weights_only=True)
        yield SimpleNamespace(keys=tensors.keys, get_tensor=tensors.__getitem__)

    model.save_pretrained = save_shards
    monkeypatch.setattr(checkpoint_module, "safe_open", open_shard, raising=False)
    _save(cp, model, opt, sched)
    restored = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    target = _DiskModel()
    restored.load_model_state_dict(target)
    assert torch.equal(target.matrix, model.matrix.bfloat16().float())
    assert torch.equal(target.bias, model.bias.bfloat16().float())
    (tmp_path / "0/model-00001.safetensors").unlink()
    assert checkpoint_module.SingleGPUCheckpointer(tmp_path).previous_epoch == -1


@pytest.mark.parametrize(
    "shard", ["../outside.safetensors", "/outside.safetensors", "model.bin"]
)
def test_shard_index_cannot_escape_bundle(checkpoint_module, tmp_path, shard):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": shard}})
    )
    with pytest.raises(ValueError, match="shard path"):
        checkpoint_module.load_safetensors_state_dict(
            tmp_path / "model.safetensors", "cpu"
        )


def test_repeated_save_keeps_one_backup_and_best_cleanup_preserves_it(
    checkpoint_module,
    tmp_path,
    monkeypatch,
):
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    model, opt, sched = _components()
    for step in range(1, 4):
        _step(model, opt, sched)
        _save(cp, model, opt, sched, step=step)
    assert len(list(tmp_path.glob(".previous-0-*"))) == 1
    monkeypatch.setattr(cp, "best_path", lambda: tmp_path / "checkpoint_best")
    cp.cleanup_keep_only_best(0)
    assert len(list(tmp_path.glob(".previous-0-*"))) == 1
    assert cp._epoch_checkpoint_path(0) == tmp_path / "0"


@pytest.mark.parametrize(
    "damage", ["config", "manifest", "missing_manifest", "missing_directory"]
)
def test_resumed_backup_survives_replacement_of_invalid_current(
    checkpoint_module,
    tmp_path,
    damage,
):
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    model, opt, sched = _components()
    for step in (1, 2):
        _step(model, opt, sched)
        _save(cp, model, opt, sched, step=step)
    if damage == "missing_directory":
        shutil.rmtree(tmp_path / "0")
    elif damage == "missing_manifest":
        (tmp_path / "0" / cp.COMMIT_FILENAME).unlink()
    else:
        name = "config.json" if damage == "config" else cp.COMMIT_FILENAME
        (tmp_path / "0" / name).write_text("")
    restored = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    valid_backup = restored.prev_path
    assert valid_backup.name.startswith(".previous-0-")
    _step(model, opt, sched)
    _save(restored, model, opt, sched, step=3)
    assert list(tmp_path.glob(".previous-0-*")) == [valid_backup]
    assert restored._checkpoint_candidate(valid_backup) is not None
    # Losing the new current generation must still expose the same valid backup.
    (tmp_path / "0/config.json").write_text("")
    assert checkpoint_module.SingleGPUCheckpointer(tmp_path).prev_path == valid_backup


def _phase_trainer(checkpoint_module, trainer_module, path, *, save_best):
    trainer = supervision._make_trainer(trainer_module, path, [supervision._batch()])
    trainer.config = trainer.config._replace(num_epochs=1, save_best=save_best)
    trainer.best_val_loss = float("inf")
    trainer.model.dtype = torch.float32
    trainer.checkpointer = checkpoint_module.SingleGPUCheckpointer(path)
    # Windows symlink privileges are not needed to exercise actual bundle I/O,
    # validation, metadata finalization, or best-checkpoint selection calls.
    trainer.checkpointer.update_best_symlink = Mock()

    def save_model(directory, *, state_dict):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text("{}")
        torch.save(state_dict, directory / "model.safetensors")

    @contextmanager
    def open_weights(filename, **_kwargs):
        tensors = torch.load(filename, weights_only=True)
        yield SimpleNamespace(keys=tensors.keys, get_tensor=tensors.__getitem__)

    trainer.model.save_pretrained = save_model
    checkpoint_module.safe_open = open_weights
    return trainer


def _restore_phase_trainer(checkpoint_module, trainer_module, path, *, save_best):
    trainer = _phase_trainer(
        checkpoint_module, trainer_module, path, save_best=save_best
    )
    trainer.resume_from_checkpoint = True
    trainer.setup_trainer()
    trainer.checkpointer.load_model_state_dict(trainer.model)
    trainer.checkpointer.load_optimizer_state_dict(trainer.model, trainer.optimizers)
    trainer.checkpointer.load_scheduler_state_dict(trainer.schedulers)
    return trainer


def _arm_interrupt(trainer, location):
    if location == "last_step":
        original = trainer.optimizers[0].step

        def step(*args, **kwargs):
            result = original(*args, **kwargs)
            signal.raise_signal(signal.SIGINT)
            return result

        step._wrapped_by_lr_sched = True
        trainer.optimizers[0].step = step
    else:
        original = trainer.model.forward

        def forward(*args, **kwargs):
            result = original(*args, **kwargs)
            if not trainer.model.training:
                signal.raise_signal(signal.SIGINT)
            return result

        trainer.model.forward = forward


@pytest.mark.parametrize("save_best", [False, True])
@pytest.mark.parametrize("location", ["last_step", "validation"])
def test_final_epoch_interrupt_resumes_validation_without_retraining(
    checkpoint_module,
    trainer_module,
    shutdown_module,
    tmp_path,
    *,
    save_best,
    location,
):
    trainer_module.TrainingInterruptedError = shutdown_module.TrainingInterruptedError
    run = shutdown_module.with_graceful_shutdown()(trainer_module.Trainer.run_training)
    trainer = _phase_trainer(
        checkpoint_module, trainer_module, tmp_path, save_best=save_best
    )
    _arm_interrupt(trainer, location)
    run(trainer)
    state = json.loads((tmp_path / "interrupted/training_state.json").read_text())
    assert state["epoch_complete"] is True
    assert state["epoch_finalized"] is False
    resumed = _restore_phase_trainer(
        checkpoint_module, trainer_module, tmp_path, save_best=save_best
    )
    assert resumed.current_epoch == resumed._resume_validation_epoch == 0
    before = resumed.model.weight.detach().clone()
    run(resumed)
    assert resumed.global_step == 2
    assert resumed.schedulers[0].last_epoch == 1
    assert resumed.optimizers[0].state[resumed.model.weight]["step"].item() == 1
    assert not resumed.model.backward_grads
    assert torch.equal(before, resumed.model.weight)
    resumed.checkpointer.update_best_symlink.assert_called_once_with(0)
    selected = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    assert selected.prev_path in (tmp_path / "0", tmp_path / "interrupted")
    assert selected._checkpoint_candidate(tmp_path / "0") is not None
    assert (
        json.loads((selected.prev_path / "training_state.json").read_text())[
            "epoch_finalized"
        ]
        is True
    )
    final = _restore_phase_trainer(
        checkpoint_module, trainer_module, tmp_path, save_best=save_best
    )
    assert final.current_epoch == 1
    final.val_epoch = Mock()
    run(final)
    final.val_epoch.assert_not_called()


def test_periodic_prevalidation_checkpoint_resumes_and_finalizes(
    checkpoint_module,
    trainer_module,
    tmp_path,
):
    trainer = _phase_trainer(
        checkpoint_module, trainer_module, tmp_path, save_best=False
    )
    trainer.train_epoch(0)
    trainer.maybe_save_checkpoint(0)
    restored = _restore_phase_trainer(
        checkpoint_module, trainer_module, tmp_path, save_best=False
    )
    assert restored.current_epoch == restored._resume_validation_epoch == 0
    restored.run_training()
    assert not restored.model.backward_grads
    assert restored.global_step == 2
    final = _restore_phase_trainer(
        checkpoint_module, trainer_module, tmp_path, save_best=False
    )
    assert final.current_epoch == 1
    assert final._resume_validation_epoch is None


def test_second_interrupt_during_resumed_validation_remains_pending(
    checkpoint_module,
    trainer_module,
    shutdown_module,
    tmp_path,
):
    trainer_module.TrainingInterruptedError = shutdown_module.TrainingInterruptedError
    run = shutdown_module.with_graceful_shutdown()(trainer_module.Trainer.run_training)
    trainer = _phase_trainer(
        checkpoint_module, trainer_module, tmp_path, save_best=True
    )
    for _ in range(2):
        _arm_interrupt(trainer, "validation")
        run(trainer)
        trainer = _restore_phase_trainer(
            checkpoint_module, trainer_module, tmp_path, save_best=True
        )
        assert trainer.current_epoch == trainer._resume_validation_epoch == 0
        assert trainer.global_step == 2
    run(trainer)
    assert not trainer.model.backward_grads
    assert (
        _restore_phase_trainer(
            checkpoint_module, trainer_module, tmp_path, save_best=True
        ).current_epoch
        == 1
    )


@pytest.mark.parametrize("source", ["numeric", "interrupted"])
def test_validation_retry_keeps_best_metrics_attached_to_same_weights(
    checkpoint_module,
    trainer_module,
    tmp_path,
    source,
):
    first = _phase_trainer(checkpoint_module, trainer_module, tmp_path, save_best=False)
    first.checkpointer.mark_epoch_finalized = Mock(
        side_effect=OSError("finalize failed")
    )
    with pytest.raises(OSError, match="finalize failed"):
        first.run_training()
    old_metrics = (tmp_path / "0/val_metrics.json").read_bytes()
    old_model = (tmp_path / "0/model.safetensors").read_bytes()
    old_manifest = (tmp_path / "0" / first.checkpointer.COMMIT_FILENAME).read_bytes()
    if source == "interrupted":
        first._save_checkpoint_bundle("interrupted", 0)
    resumed = _phase_trainer(
        checkpoint_module, trainer_module, tmp_path, save_best=False
    )
    # The logical best pointer still references epoch zero; no Windows symlink
    # permission is needed to test actual metric loading and snapshot identities.
    resumed.checkpointer.read_best_epoch = lambda: 0
    resumed.resume_from_checkpoint = True
    resumed.setup_trainer()
    resumed.checkpointer.load_model_state_dict(resumed.model)
    resumed.checkpointer.load_optimizer_state_dict(resumed.model, resumed.optimizers)
    resumed.checkpointer.load_scheduler_state_dict(resumed.schedulers)
    old_loss = resumed.best_val_loss
    resumed.val_epoch = Mock(return_value={"loss_epoch": old_loss + 0.1})
    resumed.run_training()
    assert (tmp_path / "0/val_metrics.json").read_bytes() == old_metrics
    assert (tmp_path / "0/model.safetensors").read_bytes() == old_model
    assert (
        tmp_path / "0" / resumed.checkpointer.COMMIT_FILENAME
    ).read_bytes() == old_manifest
    assert resumed.checkpointer.load_best_val_loss() == old_loss
    resumed.checkpointer.update_best_symlink.assert_not_called()
    later = _phase_trainer(checkpoint_module, trainer_module, tmp_path, save_best=False)
    later.checkpointer.read_best_epoch = lambda: 0
    later.resume_from_checkpoint = True
    later.setup_trainer()
    assert later.best_val_loss == old_loss


@pytest.mark.parametrize("difference", ["weights", "config", "step", "missing"])
def test_validation_reuse_rejects_different_or_incomplete_snapshot(
    checkpoint_module,
    tmp_path,
    difference,
):
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    model, opt, sched = _components()
    _step(model, opt, sched)
    _save(cp, model, opt, sched, complete=True, finalized=False)
    if difference == "weights":
        with torch.no_grad():
            model.matrix.add_(1)
    _save(
        cp,
        model,
        opt,
        sched,
        label="interrupted",
        step=2 if difference == "step" else 1,
        complete=True,
        finalized=False,
    )
    if difference == "config":
        (tmp_path / "interrupted/config.json").write_text(
            '{"dtype": "bfloat16", "different": true}'
        )
    elif difference == "missing":
        (tmp_path / "0/model.safetensors").unlink()
    restored = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    assert restored.prev_path == tmp_path / "interrupted"
    assert not restored.can_reuse_validation_checkpoint(
        0, 2 if difference == "step" else 1
    )


class _MemoryBestPointer:
    """Only symlink creation is simulated; all snapshot files remain real."""

    def __init__(self):
        self.target = None

    def is_symlink(self):
        return self.target is not None

    def readlink(self):
        return self.target


def _install_best_pointer(checkpointer, pointer):
    checkpointer.best_path = lambda: pointer

    def set_target(target):
        pointer.target = Path(target.name)

    checkpointer._set_best_target = set_target


def test_replacing_best_preserves_its_exact_weights_and_metrics(
    checkpoint_module,
    tmp_path,
):
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    pointer = _MemoryBestPointer()
    _install_best_pointer(cp, pointer)
    model, opt, sched = _components()
    _step(model, opt, sched)
    _save(cp, model, opt, sched, complete=True, finalized=False)
    cp.save_val_metrics(0, {"loss_epoch": 0.1})
    cp.update_best_symlink(0)
    best_weights = (tmp_path / "0/model.safetensors").read_bytes()
    for step in (2, 3):
        _step(model, opt, sched)
        _save(cp, model, opt, sched, step=step, complete=True, finalized=False)
        assert pointer.target.name.startswith(".previous-0-")
        assert cp.read_best_epoch() == 0
        assert cp.load_best_val_loss() == 0.1
        assert (
            tmp_path / pointer.target / "model.safetensors"
        ).read_bytes() == best_weights
        assert not (tmp_path / "0/val_metrics.json").exists()
    # The newest recovery generation and the older best are separately retained.
    assert len(list(tmp_path.glob(".previous-0-*"))) == 2
    cp.cleanup_keep_only_best(0)
    assert cp.load_best_val_loss() == 0.1
    restored = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    _install_best_pointer(restored, pointer)
    assert restored.prev_path == tmp_path / "0"
    assert restored.load_best_val_loss() == 0.1
    cp.save_val_metrics(0, {"loss_epoch": 0.05})
    cp.update_best_symlink(0)
    assert pointer.target == Path("0")
    assert cp.load_best_val_loss() == 0.05
    cp._prune_recovery_generations("0")
    assert len(list(tmp_path.glob(".previous-0-*"))) == 1


@pytest.mark.parametrize("failure", ["best_pointer", "publish"])
def test_failed_best_generation_replacement_remains_recoverable(
    checkpoint_module,
    tmp_path,
    monkeypatch,
    failure,
):
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    pointer = _MemoryBestPointer()
    _install_best_pointer(cp, pointer)
    model, opt, sched = _components()
    _step(model, opt, sched)
    _save(cp, model, opt, sched, complete=True, finalized=False)
    cp.save_val_metrics(0, {"loss_epoch": 0.1})
    cp.update_best_symlink(0)
    original = Path.rename

    def fail_publish(path, destination):
        if path.parent.name.startswith(".pending-"):
            raise OSError("injected best publication failure")
        return original(path, destination)

    if failure == "best_pointer":
        monkeypatch.setattr(
            cp,
            "_set_best_target",
            Mock(side_effect=OSError("injected best pointer failure")),
        )
    else:
        monkeypatch.setattr(Path, "rename", fail_publish)
    _step(model, opt, sched)
    with pytest.raises(OSError, match="injected best"):
        _save(cp, model, opt, sched, step=2, complete=True, finalized=False)
    restored = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    _install_best_pointer(restored, pointer)
    assert restored._checkpoint_candidate(restored.prev_path) is not None
    assert restored.load_best_val_loss() == 0.1
    assert restored._best_link_target().exists()
    state = json.loads((restored.prev_path / "training_state.json").read_text())
    assert state["global_step"] == 1


def test_unrestored_pending_rng_is_preserved_by_another_interrupt(
    checkpoint_module,
    trainer_module,
    tmp_path,
):
    trainer = _phase_trainer(
        checkpoint_module, trainer_module, tmp_path, save_best=False
    )
    trainer._pending_rng_states = trainer_module.capture_rng_states("cpu")
    trainer._rng_restore_pending = True
    trainer._checkpoint_position = {
        "epoch": 0,
        "local_step": 2,
        "epoch_complete": False,
        "epoch_finalized": False,
    }
    torch.rand(11)
    before_save = torch.get_rng_state().clone()
    trainer._save_checkpoint_bundle("interrupted", 0)
    saved = json.loads((tmp_path / "interrupted/training_state.json").read_text())
    assert saved["rng_states"] == trainer._pending_rng_states
    assert torch.equal(torch.get_rng_state(), before_save)


@pytest.mark.parametrize("label", [0, "interrupted"])
def test_pending_validation_snapshots_use_prevalidation_rng(
    checkpoint_module,
    trainer_module,
    tmp_path,
    label,
):
    trainer = _phase_trainer(
        checkpoint_module, trainer_module, tmp_path, save_best=False
    )
    trainer._validation_rng_states = trainer_module.capture_rng_states("cpu")
    trainer._record_checkpoint_position(0, 0, complete=True)
    torch.rand(11)
    trainer._save_checkpoint_bundle(label, 0)
    saved = json.loads((tmp_path / str(label) / "training_state.json").read_text())
    assert saved["rng_states"] == trainer._validation_rng_states
    assert saved["epoch_finalized"] is False


def test_finalized_interrupt_uses_current_not_prevalidation_rng(
    checkpoint_module,
    trainer_module,
    tmp_path,
):
    trainer = _phase_trainer(
        checkpoint_module, trainer_module, tmp_path, save_best=False
    )
    trainer._validation_rng_states = trainer_module.capture_rng_states("cpu")
    torch.rand(11)
    current = trainer_module.capture_rng_states("cpu")
    trainer._record_checkpoint_position(0, 0, complete=True, finalized=True)
    trainer._save_checkpoint_bundle("interrupted", 0)
    saved = json.loads((tmp_path / "interrupted/training_state.json").read_text())
    assert saved["rng_states"] == current
    assert saved["rng_states"] != trainer._validation_rng_states
    assert saved["epoch_finalized"] is True


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_phase_update_leaves_original_manifest_restorable(
    checkpoint_module,
    tmp_path,
    monkeypatch,
    failure,
):
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    model, opt, sched = _components()
    _step(model, opt, sched)
    _save(cp, model, opt, sched, complete=True, finalized=False)
    method = "write_text" if failure == "write" else "replace"
    original = getattr(Path, method)

    def fail_phase(path, *args, **kwargs):
        if path.name.startswith(".training-state-"):
            raise OSError("injected phase write failure")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, method, fail_phase)
        with pytest.raises(OSError, match="phase write"):
            cp.mark_epoch_finalized(0, 1)
    assert checkpoint_module.SingleGPUCheckpointer(tmp_path).prev_path == tmp_path / "0"
    assert (
        json.loads((tmp_path / "0/training_state.json").read_text())["epoch_finalized"]
        is False
    )
    cp.mark_epoch_finalized(0, 1)
    assert checkpoint_module.SingleGPUCheckpointer(tmp_path).prev_path == tmp_path / "0"
    assert (
        json.loads((tmp_path / "0/training_state.json").read_text())["epoch_finalized"]
        is True
    )
    assert (
        json.loads((tmp_path / "0" / cp.COMMIT_FILENAME).read_text())["training_state"][
            "epoch_finalized"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("field", "value"), [("global_step", 999), ("local_step", 999), ("epoch", 999)]
)
def test_phase_finalization_cannot_bypass_immutable_progress_check(
    checkpoint_module,
    tmp_path,
    field,
    value,
):
    cp = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    model, opt, sched = _components()
    _save(cp, model, opt, sched, complete=True, finalized=False)
    cp.mark_epoch_finalized(0, 1)
    path = tmp_path / "0/training_state.json"
    state = json.loads(path.read_text())
    state[field] = value
    path.write_text(json.dumps(state))
    assert checkpoint_module.SingleGPUCheckpointer(tmp_path).previous_epoch == -1


def test_interrupt_during_first_optimizer_waits_for_all_updates(
    checkpoint_module,
    trainer_module,
    shutdown_module,
    tmp_path,
):
    trainer = supervision._make_trainer(
        trainer_module, tmp_path, [supervision._batch()] * 3
    )

    # Supply a simple serializer for the scalar supervision test model.
    def save_model(path, *, state_dict):
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}")
        torch.save(state_dict, path / "model.safetensors")

    trainer.model.save_pretrained = save_model
    second = torch.nn.Parameter(torch.tensor(0.5))
    trainer.optimizers.append(torch.optim.AdamW([second], lr=0.1))
    trainer.schedulers.append(
        torch.optim.lr_scheduler.StepLR(trainer.optimizers[1], 1, 0.9)
    )
    trainer.checkpointer = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    trainer.val_loader = None
    trainer.best_val_loss = float("inf")
    trainer_module.TrainingInterruptedError = shutdown_module.TrainingInterruptedError
    first_step = trainer.optimizers[0].step
    original_handlers = (
        signal.getsignal(signal.SIGINT),
        signal.getsignal(signal.SIGTERM),
    )

    def interrupted_step(*args, **kwargs):
        value = first_step(*args, **kwargs)
        second.grad = torch.ones_like(second)
        signal.raise_signal(signal.SIGINT)
        return value

    trainer.optimizers[0].step = interrupted_step
    trainer.optimizers[0].step._wrapped_by_lr_sched = True
    shutdown_module.with_graceful_shutdown()(trainer_module.Trainer.run_training)(
        trainer
    )
    assert trainer.global_step == 2
    assert [s.last_epoch for s in trainer.schedulers] == [1, 1]
    assert trainer.optimizers[1].state[second]["step"].item() == 1
    state = json.loads((tmp_path / "interrupted/training_state.json").read_text())
    assert state.pop("rng_states")["world_size"] == 1
    assert state == {
        "epoch": 0,
        "local_step": 1,
        "global_step": 2,
        "epoch_complete": False,
        "epoch_finalized": False,
    }
    assert original_handlers == (
        signal.getsignal(signal.SIGINT),
        signal.getsignal(signal.SIGTERM),
    )


def test_generic_sampler_resume_discards_completed_batches(trainer_module, tmp_path):
    trainer = supervision._make_trainer(
        trainer_module, tmp_path, [supervision._batch()] * 4
    )
    trainer._resume_local_step = 3
    trainer.global_step = 3
    trainer.train_epoch(0)
    assert trainer.global_step == 4
    assert len(trainer.model.backward_grads) == 1
    assert trainer._checkpoint_position == {
        "epoch": 0,
        "local_step": 4,
        "epoch_complete": True,
        "epoch_finalized": False,
    }


@pytest.mark.parametrize("raises", [False, True])
def test_shutdown_restores_handlers_on_every_exit(shutdown_module, raises):
    original = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    owner = SimpleNamespace()

    def body(_self):
        if raises:
            raise ValueError("ordinary failure")
        return 12

    wrapped = shutdown_module.with_graceful_shutdown()(body)
    if raises:
        with pytest.raises(ValueError, match="ordinary failure"):
            wrapped(owner)
    else:
        assert wrapped(owner) == 12
    assert original == (
        signal.getsignal(signal.SIGINT),
        signal.getsignal(signal.SIGTERM),
    )


def test_signal_duplicates_force_exit_and_watchdog(shutdown_module, monkeypatch):
    handler = shutdown_module.GracefulShutdownHandler(timeout=23)
    handler._owner_pid = os.getpid()
    timer = Mock()
    monkeypatch.setattr(shutdown_module.threading, "Timer", Mock(return_value=timer))
    forced_exit = Mock()
    monkeypatch.setattr(shutdown_module.os, "_exit", forced_exit)
    monotonic = Mock(side_effect=[10.0, 10.1, 11.1])
    monkeypatch.setattr(shutdown_module.time, "monotonic", monotonic)
    handler._handler(signal.SIGINT, None)
    handler._handler(signal.SIGINT, None)
    assert handler.interrupted
    forced_exit.assert_not_called()
    handler._handler(signal.SIGINT, None)
    forced_exit.assert_called_once_with(128 + signal.SIGINT)
    callback = shutdown_module.threading.Timer.call_args.args[1]
    callback()
    assert forced_exit.call_args.args == (1,)
    handler.cancel_watchdog()
    timer.cancel.assert_called_once()


@pytest.mark.parametrize("kind", ["single", "distributed"])
def test_two_rank_checkpoint_failure_and_coordinated_stop(tmp_path, kind):
    """Real Gloo collectives: rank-zero I/O failure and rank-one stop agree."""
    script = textwrap.dedent("""
        import json, sys, datetime
        from pathlib import Path
        from types import SimpleNamespace
        sys.path[:] = json.loads(sys.argv[1])
        import torch
        import torch.distributed as dist
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
            set_optimizer_state_dict,
        )
        from tests.unit.train.test_checkpoint_transactions import (
            _components, _step, _save,
        )
        from tests.unit.train import test_optimizer_checkpoint_counters as counters
        from tests.unit.train import test_trainer_supervision as supervision
        rank, port, root = int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
        dist.init_process_group(
            'gloo', init_method='tcp://127.0.0.1:'+port, rank=rank, world_size=2,
            timeout=datetime.timedelta(seconds=20),
        )
        try:
            mod = counters.checkpoint_module.__wrapped__()
            mod.dist, mod.get_rank = dist, dist.get_rank
            mod.is_distributed = lambda: True
            mod.StateDictOptions = StateDictOptions
            mod.get_model_state_dict = get_model_state_dict
            mod.get_optimizer_state_dict = get_optimizer_state_dict
            mod.set_optimizer_state_dict = set_optimizer_state_dict
            cls = (mod.SingleGPUCheckpointer if sys.argv[5] == 'single'
                   else mod.DistributedCheckpointer)
            cp = cls(root)
            model, opt, sched = _components()
            _step(model, opt, sched)
            _save(cp, model, opt, sched)
            restored = cls(root)
            restored.load_optimizer_state_dict(model, opt)
            restored.load_scheduler_state_dict(sched)
            assert all(state['step'].item() == 1 for state in opt.state.values())
            original_save = model.save_pretrained
            def fail_save(*a, **kw):
                original_save(*a, **kw)
                raise OSError('rank-zero simulated write error')
            model.save_pretrained = fail_save
            try:
                _save(cp, model, opt, sched, step=2)
            except RuntimeError as error:
                assert 'rank-zero simulated write error' in str(error)
            else:
                raise AssertionError('every rank must see write failure')
            assert mod.SingleGPUCheckpointer(root).prev_path == root/'0'
            model.save_pretrained = original_save
            _save(cp, model, opt, sched, complete=True, finalized=False)
            replace = Path.replace
            def fail_phase(path, *a, **kw):
                if path.name.startswith('.training-state-'):
                    raise OSError('rank-zero phase write error')
                return replace(path, *a, **kw)
            Path.replace = fail_phase
            try:
                cp.mark_epoch_finalized(0, 1)
            except RuntimeError as error:
                assert 'rank-zero phase write error' in str(error)
            else:
                raise AssertionError('every rank must see phase write failure')
            finally:
                Path.replace = replace
            assert mod.SingleGPUCheckpointer(root).prev_path == root/'0'
            cp.mark_epoch_finalized(0, 1)
            progress = json.loads((root/'0/training_state.json').read_text())
            assert progress['epoch_finalized'] is True
            assert mod.SingleGPUCheckpointer(root).prev_path == root/'0'
            tm = supervision.trainer_module.__wrapped__()
            tm.dist = dist
            trainer = supervision._make_trainer(tm, root, [], distributed=True)
            trainer._shutdown_handler = SimpleNamespace(interrupted=rank == 1)
            try:
                trainer._record_checkpoint_position(0, 5, complete=False)
            except RuntimeError as error:
                assert 'complete training-step boundary' in str(error)
            else:
                raise AssertionError('every rank must stop')
            print('rank', rank, 'passed', flush=True)
        finally:
            dist.destroy_process_group()
    """)
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    processes = [
        subprocess.Popen(  # noqa: S603 -- Fixed test script, arguments are local test paths.
            [
                sys.executable,
                "-B",
                "-c",
                script,
                json.dumps(sys.path),
                str(rank),
                str(port),
                str(tmp_path),
                kind,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            output, _ = process.communicate(timeout=35)
            assert process.returncode == 0, output
            assert "passed" in output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
