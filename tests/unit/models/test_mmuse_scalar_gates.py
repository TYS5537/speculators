"""FSDP gate shapes, mixed-precision parity, and legacy checkpoint migration."""

import copy
from functools import partial

import pytest
import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_init import _get_managed_states
from torch.distributed.tensor import DTensor
from transformers import Qwen3Config

from speculators import SpeculatorModel
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.mmuse import MMuseDraftModel, MMuseSpeculatorConfig
from speculators.models.mmuse.checkpoint_compat import SCALAR_GATE_NAMES
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train import checkpointer as checkpoint_module
from speculators.train import distributed
from speculators.train.checkpointer import (
    DistributedCheckpointer,
    SingleGPUCheckpointer,
)


def _model(block_size=15):
    transformer = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 2,
    )
    transformer._attn_implementation = "sdpa"
    model = MMuseDraftModel(
        MMuseSpeculatorConfig(
            transformer_layer_config=transformer,
            draft_vocab_size=32,
            block_size=block_size,
            aux_hidden_state_layer_ids=[0, 1],
            mask_token_id=0,
            markov_rank=4,
            enable_correction_head=True,
            correction_output_mode="logits",
            correction_hidden_size=16,
            correction_num_heads=4,
            correction_rank=4,
            correction_hidden_aux_loss=True,
            correction_hidden_feedback=True,
            correction_project_corrected_hidden=True,
            correction_with_markov=True,
            selector_correction_feedback="corrected",
            dflash_gated_layer_fusion=True,
            dflash_context_residual=True,
            dflash_block_position_embedding=True,
            dflash2_dynamic_conv=True,
            dflash2_conv_group_size=4,
            dflash2_candidate_selector=True,
            dflash2_selector_rank=4,
            dflash2_selector_top_k=4,
            enable_confidence_head=True,
            confidence_head_with_markov=True,
            speculators_config=SpeculatorsConfig(
                algorithm="mmuse",
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=block_size)
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=None, architectures=["Qwen3ForCausalLM"]
                ),
            ),
        )
    )
    with torch.no_grad():
        for module in (model.embed_tokens, model.lm_head, model.verifier_lm_head):
            nn.init.normal_(module.weight, std=0.1)
        for name in SCALAR_GATE_NAMES:
            model.get_parameter(name).fill_(0.3)
        # Exercise feedback/collaboration at nonzero outputs, not only init gates.
        nn.init.normal_(model.correction_head.correction_up.weight, std=0.05)
        nn.init.normal_(model.correction_head.auxiliary_hidden_up.weight, std=0.05)
    return model


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _legacy(model):
    for name in SCALAR_GATE_NAMES:
        parameter = model.get_parameter(name)
        parameter.data = parameter.data.reshape(())
    return model


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_scalar_to_vector_preserves_forward_gradients_and_update(bf16, checkpointing):
    model = _model().train()
    reference = _legacy(copy.deepcopy(model))
    model.set_activation_checkpointing(checkpointing)
    reference.set_activation_checkpointing(checkpointing)
    inputs = {
        "hidden_states": torch.randn(1, 32, 32),
        "input_ids": torch.randint(1, 32, (1, 32)),
        "loss_mask": torch.ones(1, 32),
        "verifier_last_hidden_states": torch.randn(1, 32, 16),
        "document_ids": torch.zeros(1, 32, dtype=torch.long),
        "max_anchors": 2,
        "gamma": 15.0,
    }
    outputs = []
    for current in (reference, model):
        torch.manual_seed(73)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            _, loss, metrics = current(**inputs)
        loss.backward()
        outputs.append((loss.detach(), metrics))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        previous = reference.get_parameter(name)
        assert (parameter.grad is None) == (previous.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            if name in SCALAR_GATE_NAMES:
                assert parameter.grad.abs().sum() > 0, name
            torch.testing.assert_close(
                parameter.grad.reshape_as(previous.grad), previous.grad, rtol=0, atol=0
            )
    for current in (reference, model):
        torch.optim.AdamW(current.parameters(), lr=6e-5, weight_decay=0.01).step()
    for name, parameter in model.named_parameters():
        previous = reference.get_parameter(name)
        torch.testing.assert_close(
            parameter.reshape_as(previous), previous, rtol=0, atol=0
        )


def test_fsdp_parameter_validation_accepts_all_features_and_rejects_old_scalars():
    model = _model()
    _get_managed_states(list(model.modules()), set())
    _legacy(model)
    with pytest.raises(ValueError, match="doesn't support scalar parameters"):
        _get_managed_states(list(model.modules()), set())


def test_real_fsdp_initialization_accepts_every_mmuse_parameter(tmp_path, monkeypatch):
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is required for the CPU FSDP initialization check")
    try:
        torch.distributed.init_process_group(
            "gloo", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1
        )
    except RuntimeError as error:
        if "unsupported gloo device" in str(error):
            pytest.skip(f"This torch build cannot initialize Gloo: {error}")
        raise
    try:
        mesh = init_device_mesh("cpu", (1,))
        # Exercise real FSDP2 parameter validation/partitioning on this CPU host.
        monkeypatch.setattr(distributed, "fully_shard", partial(fully_shard, mesh=mesh))
        model = _model()
        distributed.apply_fully_sharded(model)
        for name in SCALAR_GATE_NAMES:
            parameter = model.get_parameter(name)
            assert isinstance(parameter, DTensor)
            assert parameter.shape == (1,)
            assert parameter.dtype == torch.float32
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("assign", [False, True])
def test_direct_load_migrates_legacy_gates_inside_wrapper(assign):
    old = nn.ModuleDict({"draft": _legacy(_model())})
    model = nn.ModuleDict({"draft": _model()})
    state = copy.deepcopy(old.state_dict())
    model.load_state_dict(state, strict=True, assign=assign)
    for name in SCALAR_GATE_NAMES:
        key = "draft." + name
        assert state[key].shape == ()
        assert model.get_parameter(key).shape == (1,)
        torch.testing.assert_close(model.get_parameter(key).squeeze(0), state[key])


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_hf_loading_keeps_legacy_and_new_gate_values(tmp_path, legacy, sharded):
    model = _legacy(_model()) if legacy else _model()
    model.save_pretrained(tmp_path, max_shard_size="8KB" if sharded else "1GB")
    loaded, info = SpeculatorModel.from_pretrained(
        tmp_path, local_files_only=True, output_loading_info=True
    )
    assert not info["mismatched_keys"]
    for name, parameter in loaded.named_parameters():
        expected = model.get_parameter(name)
        if expected.requires_grad:
            torch.testing.assert_close(
                parameter.reshape_as(expected), expected, msg=name
            )
    assert all(loaded.get_parameter(name).shape == (1,) for name in SCALAR_GATE_NAMES)


@pytest.mark.parametrize("name", ["layer_fusion_gate", "fc.weight"])
def test_hf_loading_still_rejects_unrelated_shape_mismatches(tmp_path, name):
    model = _model()
    state = model.state_dict()
    state[name] = torch.ones(2)
    model.save_pretrained(tmp_path, state_dict=state)
    with pytest.raises((ValueError, RuntimeError), match="[Mm]ismatch|size"):
        SpeculatorModel.from_pretrained(tmp_path, local_files_only=True)


def _optimizer_step(model, optimizer):
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, 0.01)
    optimizer.step()
    optimizer.zero_grad()


def test_old_adam_checkpoint_can_resume_and_update_vector_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_module, "get_current_device", lambda: "cpu")
    old = _legacy(_model())
    optimizer = torch.optim.AdamW(old.named_parameters(), lr=6e-5)
    _optimizer_step(old, optimizer)
    SingleGPUCheckpointer(tmp_path).save_checkpoint(
        old, optimizer, 0, float_dtype=torch.float32
    )
    restored = _model()
    restored_optimizer = torch.optim.AdamW(restored.named_parameters(), lr=6e-5)
    checkpointer = SingleGPUCheckpointer(tmp_path)
    checkpointer.load_model_state_dict(restored)
    checkpointer.load_optimizer_state_dict(restored, restored_optimizer)
    for name in SCALAR_GATE_NAMES:
        state = restored_optimizer.state[restored.get_parameter(name)]
        assert state["step"].ndim == 0
        assert state["exp_avg"].shape == state["exp_avg_sq"].shape == (1,)
    _optimizer_step(old, optimizer)
    _optimizer_step(restored, restored_optimizer)
    for name, parameter in restored.named_parameters():
        if parameter.requires_grad:
            expected = old.get_parameter(name)
            torch.testing.assert_close(
                parameter.reshape_as(expected), expected, rtol=0, atol=0
            )


@pytest.mark.parametrize("flattened", [False, True])
def test_distributed_optimizer_migration_preserves_scalar_step(flattened):
    model = _model()
    state = {
        "step": torch.tensor(7.0),
        "exp_avg": torch.tensor(0.25),
        "exp_avg_sq": torch.tensor(0.5),
    }
    payload = (
        {
            f"state.{name}.{key}": value
            for name in SCALAR_GATE_NAMES
            for key, value in state.items()
        }
        if flattened
        else {"state": dict.fromkeys(SCALAR_GATE_NAMES, state), "param_groups": []}
    )
    migrated = model._prepare_optimizer_state_dict(payload)
    for name in SCALAR_GATE_NAMES:
        values = (
            {key: migrated[f"state.{name}.{key}"] for key in state}
            if flattened
            else migrated["state"][name]
        )
        assert values["step"].ndim == 0
        for key in ("exp_avg", "exp_avg_sq"):
            assert values[key].shape == (1,)
            torch.testing.assert_close(values[key].squeeze(0), state[key])
            assert state[key].ndim == 0


def test_distributed_checkpoint_reshapes_before_state_distribution(
    tmp_path, monkeypatch
):
    old = _legacy(_model())
    optimizer = torch.optim.AdamW(old.named_parameters(), lr=6e-5)
    _optimizer_step(old, optimizer)
    old.save_pretrained(tmp_path / "0")
    torch.save(
        get_optimizer_state_dict(old, optimizer),
        tmp_path / "0" / "optimizer_state_dict.pt",
    )
    restored = _model()
    restored_optimizer = torch.optim.AdamW(restored.named_parameters(), lr=6e-5)

    def load_model(model, state, **_kwargs):
        assert all(state[name].shape == (1,) for name in SCALAR_GATE_NAMES)
        model.load_state_dict(state, strict=False)

    def load_optimizer(model, optimizers, state, **_kwargs):
        for name in SCALAR_GATE_NAMES:
            assert state["state"][name]["exp_avg"].shape == (1,)
            assert state["state"][name]["step"].ndim == 0
        # Exercise real name-to-parameter restoration, without a device collective.
        set_optimizer_state_dict(
            model, optimizers, state, options=StateDictOptions(full_state_dict=True)
        )

    monkeypatch.setattr(checkpoint_module, "set_model_state_dict", load_model)
    monkeypatch.setattr(checkpoint_module, "set_optimizer_state_dict", load_optimizer)
    monkeypatch.setattr(checkpoint_module.dist, "barrier", lambda: None)
    checkpointer = DistributedCheckpointer(tmp_path)
    checkpointer.load_model_state_dict(restored)
    checkpointer.load_optimizer_state_dict(restored, restored_optimizer)
    _optimizer_step(old, optimizer)
    _optimizer_step(restored, restored_optimizer)
    for name in SCALAR_GATE_NAMES:
        torch.testing.assert_close(
            restored.get_parameter(name).squeeze(0),
            old.get_parameter(name),
            rtol=0,
            atol=0,
        )
