"""Runtime sharing is applied before wrapping and checked after checkpoint load."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from speculators.train import distributed
from tests.unit.models.test_leagle import make_model

from .test_activation_checkpointing_setup import _trainer


@pytest.mark.parametrize("resume", [False, True])
def test_trainer_validates_shared_head_after_loading(resume):
    model = make_model(attention="sdpa")
    trainer = _trainer(model, resume=resume)
    trainer.config = trainer.config._replace(leagle_share_output_head=True)

    def load_checkpoint(loaded):
        assert loaded.lm_head.weight is loaded.verifier_lm_head.weight
        with torch.no_grad():
            loaded.lm_head.weight.add_(1)

    trainer.checkpointer.load_model_state_dict.side_effect = load_checkpoint
    if resume:
        with pytest.raises(ValueError, match="differs"):
            trainer.setup_model()
    else:
        trainer.setup_model()
        assert model.lm_head.weight is model.verifier_lm_head.weight
        assert not model.lm_head.weight.requires_grad


@pytest.mark.parametrize(
    "flag", ["leagle_share_output_head", "leagle_fsdp_keep_forward_params"]
)
def test_runtime_optimizations_reject_eagle(flag):
    model = make_model()
    model.config.speculators_model_type = "eagle3"
    trainer = _trainer(model, strategy="fsdp")
    trainer.config = trainer.config._replace(**{flag: True})
    with pytest.raises(ValueError, match="LEAGLE model"):
        trainer.setup_model()


def test_forward_retention_requires_fsdp_in_trainer():
    trainer = _trainer(make_model())
    trainer.config = trainer.config._replace(leagle_fsdp_keep_forward_params=True)
    with pytest.raises(ValueError, match="requires FSDP"):
        trainer.setup_model()


@pytest.mark.parametrize("keep", [False, True])
def test_fsdp_retention_affects_decoder_forward_only(keep, monkeypatch):
    wrap = Mock()
    monkeypatch.setattr(distributed, "fully_shard", wrap)
    model = SimpleNamespace(layers=[object(), object()])
    distributed.apply_fully_sharded(model, keep_forward_params=keep)
    assert wrap.call_count == 3
    for call in wrap.call_args_list[:-1]:
        if keep:
            assert call.kwargs["reshard_after_forward"] is False
        else:
            assert "reshard_after_forward" not in call.kwargs
    assert "reshard_after_forward" not in wrap.call_args_list[-1].kwargs
