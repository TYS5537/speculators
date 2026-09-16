"""Exercise the smoke wrapper against the production CPU training lifecycle.

Only accelerator placement, model-class registration validation for the tiny
fixture, and Windows-only convenience symlinks are adapted. Training/validation
loops, gradient clipping, Muon + AdamW, linear schedulers, BF16 safetensors,
optimizer/scheduler/training-state restoration are the production implementation.
This is not an A3, target-HS, distributed, or full DSpark architecture test.
"""

import json
import os
import signal
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train import checkpointer as checkpointer_module
from speculators.train import trainer as trainer_module
from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.trainer import Trainer, TrainerConfig
from speculators_dsv4 import HS_FORMAT
from speculators_dsv4.contract import DEFAULT_LAYERS, make_manifest
from speculators_dsv4.training_smoke import (
    SmokeSettings,
    make_smoke_trainer,
    state_probe,
)


class TinyTrainingModel(PreTrainedModel):
    config_class = DSparkSpeculatorConfig

    def __init__(self, config):
        super().__init__(config)
        self.proj = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        for name in ("embed_tokens", "lm_head", "verifier_lm_head", "verifier_norm"):
            module = torch.nn.Linear(4, 4, bias=False)
            module.weight.requires_grad_(False)
            torch.nn.init.constant_(module.weight, 0.125)
            self.add_module(name, module)

    def forward(self, hidden_states, document_ids):
        output = self.norm(self.proj(hidden_states))
        target = torch.arange(4, device=output.device, dtype=torch.float32) / 4
        loss = (output.float() - target).square().mean()
        count = (document_ids >= 0).sum().float()
        metrics = {
            "loss_sum": loss.detach() * count,
            "loss_total": count,
        }
        return output.argmax(dim=-1), loss, metrics


class CPUTrainer(Trainer):
    def setup_model(self):
        # The tiny fixture is a PreTrainedModel, not a registered SpeculatorModel.
        # Mirror only single-device placement with CPU; actual checkpoint loading
        # remains the same production call. Do not override any optimizer/loop.
        self.local_rank = "cpu"
        self.device_type = "cpu"
        self.model.to("cpu")
        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_model_state_dict(self.model)


@pytest.fixture
def cpu_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpointer_module, "get_current_device", lambda: "cpu")
    monkeypatch.setattr(trainer_module, "is_distributed", lambda: False)
    monkeypatch.setattr(trainer_module, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_module, "get_local_rank", lambda: 0)
    if os.name == "nt":
        # The test account cannot create Windows symlinks. These are descriptive
        # aliases only; auto-resume still reads the real numbered checkpoint.
        def skip_alias(path, target, target_is_directory=False):
            assert path.is_relative_to(tmp_path)
            assert path.name.startswith("epoch") or path.name == "checkpoint_best"

        monkeypatch.setattr(Path, "symlink_to", skip_alias)
    # Production graceful-shutdown decorators install handlers; restore the
    # process's original handlers after this unit-test fixture finishes.
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in handlers.items():
        signal.signal(sig, handler)


def _model_config(tmp_path):
    contract = make_manifest(
        {"model_path": str(tmp_path / "target"), "checkpoint_signature": "a" * 64},
        DEFAULT_LAYERS,
    )
    contract["runtime_quantization"] = {"method": "ascend"}
    return DSparkSpeculatorConfig(
        target_training_contract=contract,
        target_hidden_state_format=HS_FORMAT,
        aux_hidden_state_layer_ids=list(DEFAULT_LAYERS),
        speculators_config=SpeculatorsConfig(
            algorithm="dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=7)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=contract["model_path"],
                architectures=["DeepSeekV4ForCausalLM"],
            ),
        ),
    )


def _make_trainer(tmp_path, phase):
    torch.manual_seed(35 if phase == "fresh" else 93)
    model = TinyTrainingModel(_model_config(tmp_path))
    rows = [
        {
            "hidden_states": torch.arange(12).reshape(3, 4).float() / (10 + index),
            "document_ids": torch.zeros(3, dtype=torch.long),
        }
        for index in range(4)
    ]
    loader = DataLoader(rows, batch_size=1, shuffle=False)
    config = TrainerConfig(
        lr=0.003,
        muon_lr=0.02,
        num_epochs=10,
        save_path=str(tmp_path / "checkpoints"),
        optimizer="muon",
        scheduler_type="linear",
        hidden_states_dtype=torch.bfloat16,
    )
    settings = SmokeSettings(
        phase=phase,
        report_dir=tmp_path / "reports",
        train_batches=2,
        val_batches=1,
        recipe_sha256="same-fixture-recipe",
    )
    smoke_class = make_smoke_trainer(CPUTrainer, settings)
    return smoke_class(model, config, loader, loader)


def test_production_train_validate_save_and_resume(tmp_path, cpu_platform):
    first = _make_trainer(tmp_path, "fresh")
    assert isinstance(first.checkpointer, SingleGPUCheckpointer)
    assert [type(opt).__name__ for opt in first.optimizers] == ["Muon", "AdamW"]
    assert len(first.schedulers) == 2
    first.run_training()

    epoch = tmp_path / "checkpoints" / "0"
    saved_config = json.loads((epoch / "config.json").read_text())
    assert (
        saved_config["target_training_contract"]
        == first.model.config.target_training_contract
    )
    for filename in (
        "model.safetensors",
        "optimizer_state_dict.pt",
        "scheduler_state_dict.pt",
        "training_state.json",
        "val_metrics.json",
    ):
        assert (epoch / filename).is_file()
    fresh = json.loads((tmp_path / "reports" / "fresh.rank-0.json").read_text())
    assert fresh["status"] == "passed"
    assert fresh["train_forward_calls"] == fresh["gradient_steps"] == 2
    assert fresh["validation_forward_calls"] == 1
    assert fresh["validation_metrics"]["loss_epoch"] > 0

    second = _make_trainer(tmp_path, "resume")
    assert second.current_epoch == 1
    assert second.global_step == 2
    assert state_probe(second, torch) == fresh["final"]
    assert [opt.param_groups[0]["lr"] for opt in second.optimizers] == [0.01, 0.0015]
    second.run_training()
    resumed = json.loads((tmp_path / "reports" / "resume.rank-0.json").read_text())
    assert resumed["status"] == "passed"
    assert resumed["initial"] == fresh["final"]
    assert resumed["final"]["global_step"] == 4
    assert resumed["train_forward_calls"] == resumed["gradient_steps"] == 2
    assert resumed["validation_forward_calls"] == 1
    assert [opt.param_groups[0]["lr"] for opt in second.optimizers] == [0.0, 0.0]
    assert (tmp_path / "checkpoints" / "1" / "model.safetensors").is_file()


@pytest.mark.parametrize("corruption", ["optimizer", "scheduler"])
def test_smoke_rejects_corrupted_production_resume_state(
    tmp_path, cpu_platform, corruption
):
    first = _make_trainer(tmp_path, "fresh")
    first.run_training()
    path = tmp_path / "checkpoints" / "0" / f"{corruption}_state_dict.pt"
    saved = torch.load(path, weights_only=True, map_location="cpu")
    if corruption == "optimizer":
        next(iter(saved[0]["state"].values()))["momentum_buffer"].add_(1)
    else:
        saved[0]["_last_lr"] = [0.001]
    torch.save(saved, path)
    with pytest.raises(ValueError, match="did not restore optimizers"):
        _make_trainer(tmp_path, "resume")
    assert not (tmp_path / "reports" / "resume.rank-0.json").exists()
