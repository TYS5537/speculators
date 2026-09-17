"""Runtime checkpointing must be enabled before placement/distributed wrapping."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from speculators.model import SpeculatorModel
from speculators.train.trainer import Trainer, TrainerConfig


def _trainer(model, *, enabled=False, strategy="single", resume=False):
    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.config = TrainerConfig(
        lr=1e-4,
        num_epochs=1,
        save_path="unused",
        fsdp_shard=strategy == "fsdp",
        activation_checkpointing=enabled,
    )
    trainer.local_rank = "cpu"
    trainer.is_distributed = strategy != "single"
    trainer.resume_from_checkpoint = resume
    trainer.checkpointer = SimpleNamespace(
        previous_epoch=0 if resume else -1, load_model_state_dict=Mock()
    )
    trainer._setup_model_ddp = Mock()
    trainer._setup_model_fsdp = Mock()
    return trainer


@pytest.mark.parametrize("strategy", ["single", "ddp", "fsdp"])
@pytest.mark.parametrize("resume", [False, True])
def test_enable_before_placement_or_wrapping(monkeypatch, strategy, resume):
    monkeypatch.setattr(SpeculatorModel, "verify_training_compatible", lambda _: None)
    events = []
    model = SimpleNamespace(
        set_activation_checkpointing=lambda enabled: events.append(
            ("enabled", enabled)
        ),
        to=lambda device: events.append(("placed", device)),
    )
    trainer = _trainer(model, enabled=True, strategy=strategy, resume=resume)
    trainer._setup_model_ddp.side_effect = lambda load: events.append(("ddp", load))
    trainer._setup_model_fsdp.side_effect = lambda load: events.append(("fsdp", load))

    trainer.setup_model()

    assert events[0] == ("enabled", True)
    if strategy == "single":
        assert events[1] == ("placed", "cpu")
        assert trainer.checkpointer.load_model_state_dict.call_count == int(resume)
    else:
        assert events[1] == (strategy, resume)


def test_default_config_does_not_enable_or_require_checkpointing(monkeypatch):
    monkeypatch.setattr(SpeculatorModel, "verify_training_compatible", lambda _: None)
    model = SimpleNamespace(to=Mock())
    trainer = _trainer(model)
    assert (
        TrainerConfig(
            lr=1e-4, num_epochs=1, save_path="unused"
        ).activation_checkpointing
        is False
    )

    trainer.setup_model()

    model.to.assert_called_once_with("cpu")
    assert not hasattr(model, "activation_checkpointing")


def test_unsupported_model_fails_before_placement(monkeypatch):
    monkeypatch.setattr(SpeculatorModel, "verify_training_compatible", lambda _: None)
    model = SimpleNamespace(to=Mock())
    trainer = _trainer(model, enabled=True)

    with pytest.raises(ValueError, match="only supported by.*DFlash/DSpark"):
        trainer.setup_model()

    model.to.assert_not_called()
