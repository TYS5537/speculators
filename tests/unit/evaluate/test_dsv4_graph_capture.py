"""Real-tensor teacher lifetime tests; optional NPUGraph test is not simulated.

The native backend is a tiny module, not the target model. CPU tests exercise the
production bridge and its actual PyTorch hook/clone behavior without graph replay.
The optional NPU test captures that whole bridge with the real graph API; it is
not an integration test for vLLM scheduling, attention kernels or HS transport.
"""

# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.

import gc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
LAYERS = (1, 11, 21, 30, 40, 43)


class _InplaceNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.fail = False

    def forward(self, hidden):
        self.calls += 1
        hidden.add_(100)
        if self.fail:
            raise RuntimeError("test norm failure after in-place update")
        return hidden


class _NativeModel(nn.Module):
    def __init__(self, *, vllm_config, prefix):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = nn.Module()
        self.model.norm = _InplaceNorm()
        self.forward_calls = 0
        self.failure = None
        self.skip_hook = False

    def set_aux_hidden_state_layers(self, layers):
        self.layers = tuple(layers)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        self.forward_calls += 1
        if self.failure == "before_norm":
            raise RuntimeError("test failure before norm")
        teacher = input_ids * 2
        normalized = (
            self.model.norm.forward(teacher)
            if self.skip_hook
            else self.model.norm(teacher)
        )
        if self.failure == "after_norm":
            raise RuntimeError("test failure after norm")
        auxiliary = [input_ids + layer for layer in self.layers]
        if self.failure == "invalid_auxiliary":
            auxiliary.pop()
        return normalized, auxiliary


def _runtime_config(*, graph=False):
    metadata = {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "hc_mult": 4,
        "vocab_size": 129280,
    }
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(**metadata, to_dict=lambda: dict(metadata)),
            enforce_eager=not graph,
            dtype=torch.bfloat16,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False, async_scheduling=graph
        ),
        speculative_config=SimpleNamespace(method="extract_hidden_states"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False),
            mode=0,
            cudagraph_mode="FULL_DECODE_ONLY" if graph else "NONE",
        ),
    )


@pytest.fixture
def bridge_class(monkeypatch):
    native = ModuleType("vllm_ascend.models.deepseek_v4")
    native.AscendDeepseekV4ForCausalLM = _NativeModel
    ascend_config = ModuleType("vllm_ascend.ascend_config")
    ascend_config.get_ascend_config = lambda: SimpleNamespace(
        enable_flashcomm1=False, enable_dsa_cp=False
    )
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setitem(sys.modules, ascend_config.__name__, ascend_config)
    path = ROOT / "src/speculators_dsv4/ascend.py"
    spec = importlib.util.spec_from_file_location("dsv4_real_tensor_bridge_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "version",
        {"vllm": "0.26.0", "vllm-ascend": "0.26.0rc1"}.__getitem__,
    )
    monkeypatch.setattr(module, "install_worker_cache_compatibility", Mock())
    monkeypatch.setattr(module, "install_worker_graph_compatibility", Mock())
    return module.SpeculatorsDeepseekV4ForCausalLM


def _make_model(bridge_class, *, graph=False):
    model = bridge_class(vllm_config=_runtime_config(graph=graph))
    model.set_aux_hidden_state_layers(LAYERS)
    return model


def _assert_output(output, inputs):
    normalized, auxiliary = output
    expected = inputs.detach()
    torch.testing.assert_close(normalized, expected * 2 + 100, rtol=0, atol=0)
    assert len(auxiliary) == len(LAYERS)
    for index, layer in enumerate(LAYERS[:-1]):
        torch.testing.assert_close(auxiliary[index], expected + layer, rtol=0, atol=0)
    torch.testing.assert_close(auxiliary[-1], expected * 2, rtol=0, atol=0)
    assert auxiliary[-1].dtype == torch.bfloat16
    assert not auxiliary[-1].requires_grad
    if inputs.numel():
        assert auxiliary[-1].data_ptr() != normalized.data_ptr()
        assert auxiliary[-1].data_ptr() != inputs.data_ptr()


@pytest.mark.parametrize("graph_configured", [False, True])
def test_teacher_clone_survives_inplace_norm_and_attribute_cleanup(
    bridge_class, graph_configured
):
    # Graph-configured CPU calls are still plain forward calls, not graph tests.
    model = _make_model(bridge_class, graph=graph_configured)
    inputs = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8).requires_grad_()
    original = inputs.detach().clone()
    output = model(inputs, torch.arange(2))
    _assert_output(output, original)
    assert model._teacher_pre_norm is None
    assert model.forward_calls == model.model.norm.calls == 1
    with torch.no_grad():
        inputs.fill_(-10)
    _assert_output(output, original)


@pytest.mark.parametrize("graph_configured", [False, True])
def test_returned_teachers_keep_their_own_values_across_forwards(
    bridge_class, graph_configured
):
    model = _make_model(bridge_class, graph=graph_configured)
    retained = []
    for rows, value in ((2, 1), (2, 7), (0, 0), (3, -3), (1, 4)):
        inputs = torch.full((rows, 8), value, dtype=torch.bfloat16)
        output = model(inputs, torch.arange(rows))
        retained.append((output, inputs.clone()))
        assert model._teacher_pre_norm is None
    del model
    gc.collect()
    for output, inputs in retained:
        _assert_output(output, inputs)


@pytest.mark.parametrize(
    "failure", ["before_norm", "inside_norm", "after_norm", "invalid_auxiliary"]
)
def test_failed_forward_clears_teacher_and_recovers(bridge_class, failure):
    model = _make_model(bridge_class, graph=True)
    inputs = torch.ones(2, 8, dtype=torch.bfloat16)
    previous = model(inputs, torch.arange(2))
    model.failure = failure
    model.model.norm.fail = failure == "inside_norm"
    with pytest.raises(RuntimeError):
        model(inputs * 2, torch.arange(2))
    assert model._teacher_pre_norm is None
    _assert_output(previous, inputs)
    model.failure = None
    model.model.norm.fail = False
    output = model(inputs * 3, torch.arange(2))
    _assert_output(output, inputs * 3)
    assert model._teacher_pre_norm is None


def test_missing_hook_never_reuses_teacher_from_previous_forward(bridge_class):
    model = _make_model(bridge_class, graph=True)
    inputs = torch.ones(2, 8, dtype=torch.bfloat16)
    previous = model(inputs, torch.arange(2))
    model.skip_hook = True
    with pytest.raises(RuntimeError, match="did not capture"):
        model(inputs * 5, torch.arange(2))
    assert model._teacher_pre_norm is None
    _assert_output(previous, inputs)
    model.skip_hook = False
    _assert_output(model(inputs * 7, torch.arange(2)), inputs * 7)


@pytest.mark.parametrize("rows", [1, 3])
def test_real_npu_graph_replay_updates_returned_teacher_without_python_hook(
    bridge_class, rows
):
    # API source: https://github.com/Ascend/pytorch/blob/master/torch_npu/npu/graphs.py
    # Missing hardware skips; a real capture/replay failure must fail, never skip.
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is required for real NPUGraph capture/replay")
    if not hasattr(torch.npu, "NPUGraph") or not hasattr(torch.npu, "graph"):
        pytest.skip("This torch_npu version does not expose NPUGraph/graph")

    model = _make_model(bridge_class, graph=True)
    static_inputs = torch.ones(rows, 8, dtype=torch.bfloat16, device="npu")
    positions = torch.arange(rows, device="npu")
    capture_stream = torch.npu.Stream()
    capture_stream.wait_stream(torch.npu.current_stream())
    with torch.inference_mode():
        with torch.npu.stream(capture_stream):
            for _ in range(3):
                model(static_inputs, positions)
        torch.npu.current_stream().wait_stream(capture_stream)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        # Capture the complete production bridge, including clone and final cleanup.
        with torch.npu.graph(graph, stream=capture_stream):
            output = model(static_inputs, positions)
        torch.npu.synchronize()
        forward_calls, norm_calls = model.forward_calls, model.model.norm.calls
        assert model._teacher_pre_norm is None
        teacher_address = output[1][-1].data_ptr()
        snapshots = []
        for value in (1, 7, -3):
            expected = torch.full((rows, 8), value, dtype=torch.bfloat16)
            with torch.npu.stream(capture_stream):
                static_inputs.copy_(expected)
                graph.replay()
            capture_stream.synchronize()
            cpu_output = (output[0].cpu(), [tensor.cpu() for tensor in output[1]])
            _assert_output(cpu_output, expected)
            snapshots.append(cpu_output[1][-1].clone())
            assert output[1][-1].data_ptr() == teacher_address
            assert model.forward_calls == forward_calls
            assert model.model.norm.calls == norm_calls
            assert model._teacher_pre_norm is None
        for snapshot, value in zip(snapshots, (1, 7, -3), strict=True):
            torch.testing.assert_close(snapshot, torch.full_like(snapshot, value * 2))
