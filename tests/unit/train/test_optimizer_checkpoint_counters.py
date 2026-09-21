"""CPU checkpoint regressions that keep optimizer counters out of BF16 conversion."""

import ast
import copy
import filecmp
import functools
import json
import logging
import shutil
import time
import uuid
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.utils._pytree as pytree
from torch.nn.parallel import DistributedDataParallel


@pytest.fixture
def checkpoint_module():
    # Run the actual production methods without importing Transformers/verifiers.
    path = Path(__file__).parents[3] / "src/speculators/train/checkpointer.py"
    module = ModuleType("optimizer_counter_checkpointer")
    module.__dict__.update(
        torch=torch,
        pytree=pytree,
        functools=functools,
        filecmp=filecmp,
        json=json,
        shutil=shutil,
        time=time,
        uuid=uuid,
        contextmanager=contextmanager,
        Path=Path,
        abstractmethod=abstractmethod,
        DistributedDataParallel=DistributedDataParallel,
        StateDictOptions=SimpleNamespace,
        logger=logging.getLogger("optimizer-counter-tests"),
        dist=SimpleNamespace(barrier=Mock()),
        get_rank=lambda: 0,
        is_distributed=lambda: False,
        get_current_device=lambda: "cpu",
    )
    source = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in source.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    source = ast.fix_missing_locations(
        ast.Module(body=[future, *definitions], type_ignores=[])
    )
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


class _TinyModel(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.ones(4, 4, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.ones(4, dtype=dtype))
        self.saved_weights = None

    @property
    def dtype(self):
        return self.matrix.dtype

    def save_pretrained(self, path, *, state_dict):
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps({"dtype": "float32"}))
        self.saved_weights = state_dict


def _optimizers(model, mixed):
    if mixed:
        return [
            torch.optim.Muon([model.matrix], lr=0.02),
            torch.optim.AdamW([model.bias], lr=6e-5),
        ]
    return [torch.optim.AdamW(model.parameters(), lr=6e-5)]


def _saved_step_values(payload):
    states = payload if isinstance(payload, list) else [payload]
    saved_steps = []
    for state_dict in states:
        for state in state_dict["state"].values():
            for key, tensor in state.items():
                assert tensor.dtype == (
                    torch.float32 if key == "step" else torch.bfloat16
                )
            if "step" in state:
                saved_steps.append(state["step"].item())
    return saved_steps


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("load_dtype", [torch.float32, torch.bfloat16])
def test_real_adam_257_step_roundtrip_and_next_update(
    checkpoint_module, tmp_path, mixed, load_dtype
):
    model = _TinyModel()
    optimizers = _optimizers(model, mixed)
    for _ in range(257):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        for optimizer in optimizers:
            optimizer.step()
    if not mixed:
        # Also cover the longer-run value from the reported failure.
        optimizers[0].state[model.bias]["step"].fill_(16626)
    expected_steps = [257] if mixed else [257, 16626]

    checkpointer = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    checkpointer.save_checkpoint(model, optimizers if mixed else optimizers[0], 0)
    payload = torch.load(tmp_path / "0/optimizer_state_dict.pt", weights_only=True)
    assert isinstance(payload, list) == mixed
    assert _saved_step_values(payload) == expected_steps
    assert all(
        tensor.dtype == torch.bfloat16 for tensor in model.saved_weights.values()
    )
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())

    restored_model = _TinyModel(load_dtype)
    restored_optimizers = _optimizers(restored_model, mixed)
    restored_checkpointer = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    restored_checkpointer.load_optimizer_state_dict(
        restored_model,
        restored_optimizers if mixed else restored_optimizers[0],
    )
    restored_steps = [
        state["step"]
        for optimizer in restored_optimizers
        for state in optimizer.state.values()
        if "step" in state
    ]
    assert [step.item() for step in restored_steps] == expected_steps
    assert all(step.dtype == torch.float32 for step in restored_steps)
    for parameter in restored_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    for optimizer in restored_optimizers:
        optimizer.step()
    assert [step.item() for step in restored_steps] == [
        step + 1 for step in expected_steps
    ]


@pytest.mark.parametrize("container", [list, tuple])
def test_optimizer_conversion_preserves_list_and_tuple_structure(
    checkpoint_module, container
):
    source = container(
        {"state": {0: {"step": torch.tensor(float(step)), "exp_avg": torch.ones(2)}}}
        for step in [257, 16626]
    )
    converted = checkpoint_module.convert_optimizer_state_dtype(source, torch.bfloat16)
    assert isinstance(converted, container)
    for before, after in zip(source, converted, strict=True):
        assert after["state"][0]["step"].dtype == torch.float32
        torch.testing.assert_close(
            after["state"][0]["step"], before["state"][0]["step"]
        )
        assert after["state"][0]["exp_avg"].dtype == torch.bfloat16
        assert before["state"][0]["exp_avg"].dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
def test_higher_precision_or_integer_counters_are_not_downcast(
    checkpoint_module, dtype
):
    step = torch.tensor(2**24 + 1, dtype=dtype)
    source = {
        "state.layer.weight.step": step,
        "state.layer.weight.exp_avg": torch.ones(2),
    }
    converted = checkpoint_module.convert_optimizer_state_dtype(source, torch.bfloat16)
    assert converted["state.layer.weight.step"].dtype == dtype
    assert converted["state.layer.weight.step"].item() == 2**24 + 1
    assert converted["state.layer.weight.exp_avg"].dtype == torch.bfloat16


@pytest.mark.parametrize("shape", ["nested", "flat"])
@pytest.mark.parametrize("rank", [0, 1])
def test_distributed_save_and_all_rank_load_preserve_counters(
    checkpoint_module, tmp_path, shape, rank
):
    model = _TinyModel()
    states = {
        name: {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.ones_like(parameter),
            "exp_avg_sq": torch.ones_like(parameter),
        }
        for (name, parameter), step in zip(
            model.named_parameters(), [257, 16626], strict=True
        )
    }
    payload = (
        {"state": states, "param_groups": []}
        if shape == "nested"
        else {
            f"state.{name}.{key}": value
            for name, state in states.items()
            for key, value in state.items()
        }
    )
    checkpoint_module.get_model_state_dict = lambda model, **_kwargs: model.state_dict()
    checkpoint_module.get_optimizer_state_dict = Mock(
        side_effect=lambda *_args, **_kwargs: copy.deepcopy(payload)
    )
    checkpointer = checkpoint_module.DistributedCheckpointer(tmp_path)
    checkpointer.save_checkpoint(model, _optimizers(model, False), 0)
    on_disk = torch.load(tmp_path / "0/optimizer_state_dict.pt", weights_only=True)

    def tensor_from(state_dict, name, key):
        if shape == "nested":
            return state_dict["state"][name][key]
        return state_dict[f"state.{name}.{key}"]

    for name, expected in ["matrix", 257], ["bias", 16626]:
        assert tensor_from(on_disk, name, "step").item() == expected
        assert tensor_from(on_disk, name, "step").dtype == torch.float32
        assert tensor_from(on_disk, name, "exp_avg").dtype == torch.bfloat16

    restored_model = _TinyModel(torch.bfloat16)
    optimizer = torch.optim.AdamW(restored_model.parameters(), lr=6e-5)

    def restore_state(model, optimizers, state_dict, **_kwargs):
        assert optimizers == [optimizer]
        for name, parameter in model.named_parameters():
            optimizer.state[parameter] = {
                key: tensor_from(state_dict, name, key).clone()
                for key in ["step", "exp_avg", "exp_avg_sq"]
            }

    # Simulate the DCP broadcast consumer on each rank; conversion and checkpoint
    # disk I/O remain the production save/load methods, without requiring GPUs.
    checkpoint_module.get_rank = lambda: rank
    checkpoint_module.set_optimizer_state_dict = Mock(side_effect=restore_state)
    restored = checkpoint_module.DistributedCheckpointer(tmp_path)
    restored.load_optimizer_state_dict(restored_model, optimizer)
    checkpoint_module.set_optimizer_state_dict.assert_called_once()
    assert checkpoint_module.dist.barrier.call_count == 1
    for parameter, expected in zip(
        restored_model.parameters(), [257, 16626], strict=True
    ):
        assert optimizer.state[parameter]["step"].item() == expected
        assert optimizer.state[parameter]["step"].dtype == torch.float32
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert [optimizer.state[p]["step"].item() for p in restored_model.parameters()] == [
        258,
        16627,
    ]


def test_legacy_rounded_counter_is_promoted_but_not_fabricated(
    checkpoint_module, tmp_path
):
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for state in optimizer.state.values():
        state["step"].fill_(257)
    legacy = checkpoint_module.convert_float_dtype(
        optimizer.state_dict(), torch.bfloat16
    )
    directory = tmp_path / "0"
    directory.mkdir()
    torch.save(legacy, directory / "optimizer_state_dict.pt")

    restored_model = _TinyModel(torch.bfloat16)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=6e-5)
    checkpointer = checkpoint_module.SingleGPUCheckpointer(tmp_path)
    checkpointer.load_optimizer_state_dict(restored_model, restored_optimizer)
    for state in restored_optimizer.state.values():
        assert state["step"].dtype == torch.float32
        assert (
            state["step"].item() == 256
        )  # The old file already lost the original 257.
    for parameter in restored_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    restored_optimizer.step()
    assert all(
        state["step"].item() == 257 for state in restored_optimizer.state.values()
    )
