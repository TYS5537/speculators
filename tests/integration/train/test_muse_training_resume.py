"""Real Muse CPU training, transactional checkpoint restoration and continuation.

Only device placement and deterministic local verifier weights are adapted. The
registered Muse model, Trainer loop, AdamW, LR scheduler and SingleGPUCheckpointer
are production implementations, not AST extracts or replacement training loops.
BF16 parameters/moments make the production BF16 checkpoint serialization lossless
so the interrupted and uninterrupted CPU trajectories can be compared exactly.

This test needs no downloads and can run with ``pytest --noconftest``. It does not
exercise validation/best-checkpoint selection, real target hidden-state export,
accelerators, distributed training or FP32-to-BF16 checkpoint quantization.
"""

# The trainer intentionally snapshots these process-global generators.
# This is a self-contained pytest module, not an importable test-helper package.
# ruff: noqa: NPY002, INP001

import copy
import json
import os
import random
import signal
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorModelConfig
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.muse import MuseDraftModel, MuseSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train import checkpointer as checkpointer_module
from speculators.train import trainer as trainer_module
from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.rng import capture_rng_states
from speculators.train.trainer import Trainer, TrainerConfig


class _CPUTrainer(Trainer):
    def setup_model(self):
        self.local_rank = "cpu"
        self.device_type = "cpu"
        # Retain real registration validation and checkpoint loading.
        super().setup_model()


@pytest.fixture
def cpu_training(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpointer_module, "get_current_device", lambda: "cpu")
    monkeypatch.setattr(trainer_module, "is_distributed", lambda: False)
    monkeypatch.setattr(trainer_module, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_module, "get_local_rank", lambda: 0)
    if os.name == "nt":
        # Exercise production alias-failure handling; pretending a symlink was
        # created would incorrectly bypass filesystem semantics.
        def unavailable_alias(path, target, target_is_directory=False):
            del target, target_is_directory
            assert path.is_relative_to(tmp_path)
            assert path.name.startswith("epoch")
            raise OSError("Descriptive checkpoint symlinks unavailable in CPU test")

        monkeypatch.setattr(Path, "symlink_to", unavailable_alias)
    original_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    original_threads = torch.get_num_threads()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    torch.set_num_threads(1)
    yield
    random.setstate(original_rng[0])
    np.random.set_state(original_rng[1])
    torch.set_rng_state(original_rng[2])
    torch.set_num_threads(original_threads)
    for sig, handler in handlers.items():
        signal.signal(sig, handler)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _config(mode):
    transformer = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        layer_types=["full_attention"] * 2,
    )
    transformer._attn_implementation = "eager"
    enhanced = mode != "baseline"
    return MuseSpeculatorConfig(
        transformer_layer_config=transformer,
        draft_vocab_size=32,
        block_size=3,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=4,
        enable_correction_head=enhanced,
        correction_output_mode=mode if enhanced else "hidden",
        correction_hidden_size=16,
        correction_num_heads=4,
        correction_rank=8,
        dflash_gated_layer_fusion=enhanced,
        speculators_config=SpeculatorsConfig(
            algorithm="muse",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None, architectures=["Qwen3ForCausalLM"]
            ),
        ),
    )


def _make_model(config):
    # Attention implementation is a runtime setting, not saved architecture.
    config.transformer_layer_config._attn_implementation = "eager"
    model = MuseDraftModel(config).to(dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(904)
    with torch.no_grad():
        embeddings = torch.randn(32, 16, generator=generator) * 0.15
        projection = torch.randn(32, 16, generator=generator) * 0.15
        model.embed_tokens.weight.copy_(embeddings)
        model.lm_head.weight.copy_(projection)
        # These verifier tensors are intentionally not saved in draft checkpoints;
        # every process receives the same fixed, finite external target fixture.
        model.verifier_lm_head.weight.copy_(projection)
        model.verifier_norm.weight.fill_(1)
    for name in ("embed_tokens", "lm_head", "verifier_lm_head", "verifier_norm"):
        assert not getattr(model, name).weight.requires_grad
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    return model


def _loader():
    generator = torch.Generator().manual_seed(306)
    rows = [
        {
            "hidden_states": torch.randn(12, 32, generator=generator).bfloat16(),
            "input_ids": torch.randint(1, 32, (12,), generator=generator),
            "loss_mask": torch.ones(12),
            "verifier_last_hidden_states": torch.randn(
                12, 16, generator=generator
            ).bfloat16(),
            "document_ids": torch.zeros(12, dtype=torch.long),
        }
        for _ in range(2)
    ]
    # num_workers=0 keeps data/model RNG in the production checkpoint's scope.
    return DataLoader(rows, batch_size=1, shuffle=False, num_workers=0)


def _trainer(path, config, *, epochs, resume=False):
    train_kwargs, _ = MuseDraftModel.get_trainer_kwargs(max_anchors=2, block_size=3)
    return _CPUTrainer(
        _make_model(config),
        TrainerConfig(
            lr=0.005,
            num_epochs=epochs,
            save_path=str(path),
            resume_from_checkpoint=resume,
            optimizer="adamw",
            weight_decay=0.0,
            scheduler_type="linear",
            # The first process runs one epoch, but both processes use the same
            # complete-run schedule; loading scheduler state does not save lambda.
            scheduler_total_steps=4,
            scheduler_warmup_steps=0,
            hidden_states_dtype=torch.bfloat16,
            train_call_kwargs=train_kwargs,
            log_freq=100,
        ),
        _loader(),
    )


def _observe(model):
    losses, gradient_norms, handles = [], {}, []

    def record_forward(_module, _args, output):
        losses.append(output[1].detach().clone())

    handles.append(model.register_forward_hook(record_forward))
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            gradient_norms[name] = []

            def record_gradient(gradient, name=name):
                gradient_norms[name].append(gradient.detach().float().norm().item())

            handles.append(parameter.register_hook(record_gradient))
    return losses, gradient_norms, handles


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_equal(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_equal(left, right)
    else:
        assert actual == expected


def _assert_frozen_unchanged(model, initial_weights, frozen_names):
    for name in frozen_names:
        _assert_equal(model.state_dict()[name], initial_weights[name])


@pytest.mark.parametrize("mode", ["baseline", "hidden", "logits"])
def test_muse_train_save_resume_matches_uninterrupted(tmp_path, cpu_training, mode):
    _seed(71)
    reference = _trainer(tmp_path / "reference", _config(mode), epochs=2)
    initial_weights = copy.deepcopy(reference.model.state_dict())
    frozen_names = [
        name
        for name, parameter in reference.model.named_parameters()
        if not parameter.requires_grad
    ]
    reference_losses, _, reference_handles = _observe(reference.model)
    reference.run_training()
    reference_rng = capture_rng_states("cpu")
    for handle in reference_handles:
        handle.remove()
    assert reference.global_step == 4
    assert len(reference_losses) == 4

    _seed(71)
    first = _trainer(tmp_path / "resumed", _config(mode), epochs=1)
    _assert_equal(first.model.state_dict(), initial_weights)
    first_losses, gradients, handles = _observe(first.model)
    first.run_training()
    for handle in handles:
        handle.remove()
    assert isinstance(first.checkpointer, SingleGPUCheckpointer)
    assert type(first.model) is MuseDraftModel
    assert [type(optimizer) for optimizer in first.optimizers] == [torch.optim.AdamW]
    assert len(first.schedulers) == 1
    assert first.global_step == 2
    _assert_equal(first_losses, reference_losses[:2])
    assert all(torch.isfinite(loss) and loss > 0 for loss in first_losses)
    assert all(np.isfinite(values).all() for values in gradients.values())
    _assert_frozen_unchanged(first.model, initial_weights, frozen_names)

    updated_names = ["fc.weight", "confidence_head.proj.weight"]
    updated_names += (
        ["markov_head.markov_w2.weight"]
        if mode == "baseline"
        else ["correction_head.correction_up.weight", "layer_fusion_gate"]
    )
    for name in updated_names:
        assert any(value > 0 for value in gradients[name]), name
        changed = not torch.equal(first.model.state_dict()[name], initial_weights[name])
        assert changed, name

    checkpoint = tmp_path / "resumed" / "0"
    for filename in (
        "config.json",
        "model.safetensors",
        "optimizer_state_dict.pt",
        "scheduler_state_dict.pt",
        "training_state.json",
        "checkpoint_complete.json",
    ):
        assert (checkpoint / filename).is_file(), filename
    saved_config = json.loads((checkpoint / "config.json").read_text())
    assert saved_config["speculators_model_type"] == "muse"
    assert saved_config["speculators_config"]["algorithm"] == "muse"
    assert saved_config["enable_correction_head"] == (mode != "baseline")
    assert saved_config["dflash_gated_layer_fusion"] == (mode != "baseline")
    assert saved_config["correction_output_mode"] == (
        "hidden" if mode == "baseline" else mode
    )
    progress = json.loads((checkpoint / "training_state.json").read_text())
    assert progress["epoch_complete"]
    assert progress["epoch_finalized"]
    assert progress["epoch"] == 0
    assert progress["global_step"] == 2
    assert progress["rng_states"] == capture_rng_states("cpu")
    disk_weights = load_file(checkpoint / "model.safetensors")
    assert "verifier_lm_head.weight" not in disk_weights
    assert "verifier_norm.weight" not in disk_weights
    for name, value in disk_weights.items():
        _assert_equal(value, first.model.state_dict()[name])

    # A new random initialization must be overwritten by checkpoint loading, and
    # the trainer must restore RNG just before continuing real model execution.
    _seed(999)
    restored_config = SpeculatorModelConfig.from_pretrained(
        checkpoint, local_files_only=True
    )
    assert type(restored_config) is MuseSpeculatorConfig
    second = _trainer(tmp_path / "resumed", restored_config, epochs=2, resume=True)
    assert second.current_epoch == 1
    assert second.global_step == 2
    assert second.checkpointer.prev_path == checkpoint
    _assert_equal(second.model.state_dict(), first.model.state_dict())
    _assert_equal(second.optimizers[0].state_dict(), first.optimizers[0].state_dict())
    _assert_equal(second.schedulers[0].state_dict(), first.schedulers[0].state_dict())
    assert second.optimizers[0].param_groups[0]["lr"] == 0.0025
    for state in second.optimizers[0].state.values():
        assert state["step"].dtype == torch.float32
        assert state["step"].item() == 2
    resumed_losses, _, handles = _observe(second.model)
    second.run_training()
    for handle in handles:
        handle.remove()

    assert second.global_step == reference.global_step == 4
    assert second.schedulers[0].last_epoch == 4
    assert second.optimizers[0].param_groups[0]["lr"] == 0.0
    for state in second.optimizers[0].state.values():
        assert state["step"].item() == 4
    assert any(
        not torch.equal(second.model.state_dict()[name], first.model.state_dict()[name])
        for name in updated_names
    )
    _assert_equal(resumed_losses, reference_losses[2:])
    _assert_equal(second.model.state_dict(), reference.model.state_dict())
    _assert_equal(
        second.optimizers[0].state_dict(), reference.optimizers[0].state_dict()
    )
    _assert_equal(
        second.schedulers[0].state_dict(), reference.schedulers[0].state_dict()
    )
    _assert_frozen_unchanged(second.model, initial_weights, frozen_names)
    assert capture_rng_states("cpu") == reference_rng
    assert not second._rng_restore_pending
    assert (tmp_path / "resumed" / "1" / "checkpoint_complete.json").is_file()
