"""CPU integration for saved draft identities and resume ordering."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train import trainer as trainer_module
from speculators.train.trainer import Trainer, TrainerConfig
from speculators_dsv4 import HS_FORMAT
from speculators_dsv4.contract import DEFAULT_LAYERS, make_manifest


def _saved_config(tmp_path):
    contract = make_manifest(
        {"model_path": str(tmp_path / "target"), "checkpoint_signature": "a" * 64},
        DEFAULT_LAYERS,
    )
    contract["runtime_quantization"] = {"method": "ascend"}
    return {
        "target_training_contract": contract,
        "target_hidden_state_format": HS_FORMAT,
        "aux_hidden_state_layer_ids": list(DEFAULT_LAYERS),
        "speculators_config": {"verifier": {"name_or_path": contract["model_path"]}},
    }


def test_dspark_config_roundtrip_preserves_identity(tmp_path):
    saved = _saved_config(tmp_path)
    config = DSparkSpeculatorConfig(
        **{key: value for key, value in saved.items() if key != "speculators_config"},
        speculators_config=SpeculatorsConfig(
            algorithm="dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=7)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=saved["target_training_contract"]["model_path"],
                architectures=["DeepSeekV4ForCausalLM"],
            ),
        ),
    )
    config.save_pretrained(tmp_path / "saved")
    restored = DSparkSpeculatorConfig.from_pretrained(tmp_path / "saved")
    assert restored.target_training_contract == saved["target_training_contract"]
    assert (
        restored.to_dict()["target_training_contract"]
        == config.target_training_contract
    )


def _isolate_trainer(monkeypatch, *, distributed=False, remote_error=None):
    hooks = {}
    for name in (
        "setup_trainer",
        "setup_model",
        "setup_optimizer",
        "_init_loss_curricula",
    ):
        hooks[name] = Mock()
        monkeypatch.setattr(Trainer, name, hooks[name])
    monkeypatch.setattr(trainer_module, "is_distributed", lambda: distributed)
    monkeypatch.setattr(trainer_module, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_module, "get_local_rank", lambda: 0)
    monkeypatch.setattr(
        trainer_module.torch.accelerator,
        "current_accelerator",
        lambda: SimpleNamespace(type="cpu"),
    )

    def gather(errors, error):
        errors[:] = [error, remote_error]

    process_group = SimpleNamespace(
        get_world_size=lambda: 2, all_gather_object=Mock(side_effect=gather)
    )
    monkeypatch.setattr(trainer_module, "dist", process_group)
    return hooks, process_group


def _model(saved):
    return SimpleNamespace(
        config=SimpleNamespace(
            target_hidden_state_format=saved["target_hidden_state_format"],
            to_dict=lambda: copy.deepcopy(saved),
        )
    )


@pytest.mark.parametrize(
    ("distributed", "fsdp"), [(False, False), (True, False), (True, True)]
)
def test_auto_resume_rejects_stale_identity_before_loading(
    tmp_path, monkeypatch, distributed, fsdp
):
    saved = _saved_config(tmp_path)
    stale = copy.deepcopy(saved)
    stale["target_training_contract"]["checkpoint_signature"] = "b" * 64
    epoch = tmp_path / "0"
    epoch.mkdir()
    (epoch / "config.json").write_text(json.dumps(stale), encoding="utf-8")
    hooks, process_group = _isolate_trainer(monkeypatch, distributed=distributed)
    config = TrainerConfig(
        lr=1e-4,
        num_epochs=2,
        save_path=str(tmp_path),
        resume_from_checkpoint=True,
        fsdp_shard=fsdp,
    )
    with pytest.raises(ValueError, match="checkpoint_signature"):
        Trainer(_model(saved), config, [])
    for hook in hooks.values():
        hook.assert_not_called()
    assert process_group.all_gather_object.call_count == int(distributed)


@pytest.mark.parametrize("fsdp", [False, True])
def test_other_rank_invalid_identity_aborts_before_ddp_or_fsdp_setup(
    tmp_path, monkeypatch, fsdp
):
    saved = _saved_config(tmp_path)
    hooks, process_group = _isolate_trainer(
        monkeypatch,
        distributed=True,
        remote_error="ValueError: invalid rank-local manifest",
    )
    config = TrainerConfig(
        lr=1e-4, num_epochs=2, save_path=str(tmp_path), fsdp_shard=fsdp
    )
    with pytest.raises(ValueError, match="rank 1: ValueError"):
        Trainer(_model(saved), config, [])
    for hook in hooks.values():
        hook.assert_not_called()
    process_group.all_gather_object.assert_called_once()


def test_valid_auto_resume_continues_setup(tmp_path, monkeypatch):
    saved = _saved_config(tmp_path)
    epoch = tmp_path / "2"
    epoch.mkdir()
    (epoch / "config.json").write_text(json.dumps(saved), encoding="utf-8")
    hooks, _ = _isolate_trainer(monkeypatch)
    config = TrainerConfig(
        lr=1e-4, num_epochs=4, save_path=str(tmp_path), resume_from_checkpoint=True
    )
    trainer = Trainer(_model(saved), config, [])
    assert trainer.checkpointer.previous_epoch == 2
    for hook in hooks.values():
        hook.assert_called_once()


def test_no_resume_does_not_accept_old_weights_but_allows_fresh_run(
    tmp_path, monkeypatch
):
    saved = _saved_config(tmp_path)
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "config.json").write_text("{}", encoding="utf-8")
    hooks, _ = _isolate_trainer(monkeypatch)
    config = TrainerConfig(lr=1e-4, num_epochs=2, save_path=str(tmp_path))
    Trainer(_model(saved), config, [])
    for hook in hooks.values():
        hook.assert_called_once()


def test_qwen_resume_does_not_enter_dsv4_identity_path(tmp_path, monkeypatch):
    (tmp_path / "0").mkdir()
    hooks, process_group = _isolate_trainer(monkeypatch, distributed=True)
    model = SimpleNamespace(
        config=SimpleNamespace(
            target_hidden_state_format="standard",
            to_dict=Mock(side_effect=AssertionError("Qwen should not be inspected")),
        )
    )
    config = TrainerConfig(
        lr=1e-4, num_epochs=2, save_path=str(tmp_path), resume_from_checkpoint=True
    )
    Trainer(model, config, [])
    model.config.to_dict.assert_not_called()
    process_group.all_gather_object.assert_not_called()
    for hook in hooks.values():
        hook.assert_called_once()
