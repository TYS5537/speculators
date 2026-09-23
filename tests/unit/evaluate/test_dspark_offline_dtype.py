"""Draft-loading precision at the HF target/draft hidden-state boundary."""

# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).parents[3]
EVALUATOR = ROOT / "scripts/evaluate/dspark_offline_eval.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluator():
    module = load_module("dspark_dtype_eval_test", EVALUATOR)
    module.torch = torch
    return module


def load_draft(evaluator, monkeypatch, *, target_dtype, checkpoint_dtype, backend="hf"):
    calls = []

    class TinyDraft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(8, 4, bias=False, dtype=checkpoint_dtype)
            self.lm_head = torch.nn.Linear(4, 5, bias=False, dtype=checkpoint_dtype)
            self.verifier_lm_head = torch.nn.Linear(
                4, 5, bias=False, dtype=checkpoint_dtype
            )

        @classmethod
        def from_pretrained(cls, path, *, config, d2t, t2d, **kwargs):
            calls.append(kwargs)
            # Match HF's explicit dtype conversion followed by the repository's
            # borrowed verifier-weight refresh via load_state_dict (not assign).
            model = cls().to(dtype=kwargs.get("torch_dtype", torch.float32))
            model.verifier_lm_head.load_state_dict(
                {"weight": torch.ones(5, 4, dtype=checkpoint_dtype)}
            )
            return model

        def _fuse_target_hidden(self, hidden_states):
            return self.fc(hidden_states)

    args = SimpleNamespace(
        draft_model="local-checkpoint",
        dtype="auto",
        device="cpu",
        sample_from_anchor=None,
        draft_attn_impl="auto",
        d2t_path=None,
        t2d_path=None,
    )
    config = SimpleNamespace(transformer_layer_config=SimpleNamespace())
    report = {"model_path": "local-target"} if backend == "dsv4-vllm" else None
    setup = evaluator._TargetSetup(backend, {}, report, None)
    monkeypatch.setattr(evaluator, "_load_draft_config", lambda _path: config)
    monkeypatch.setattr(
        evaluator, "_load_vocab_mapping_tensors", lambda **kwargs: (None, None)
    )
    checked_mapping = Mock()
    monkeypatch.setattr(evaluator, "_ensure_loaded_vocab_mappings", checked_mapping)
    contract = ModuleType("speculators_dsv4.eval_contract")
    contract.bind_draft_verifier = Mock()
    monkeypatch.setitem(sys.modules, contract.__name__, contract)
    draft, returned_config = evaluator._load_evaluation_draft(
        args,
        setup,
        None if backend == "dsv4-vllm" else SimpleNamespace(dtype=target_dtype),
        torch.device("cpu"),
        model_class=TinyDraft,
    )
    assert returned_config is config
    checked_mapping.assert_called_once_with(draft, args)
    if backend == "dsv4-vllm":
        contract.bind_draft_verifier.assert_called_once_with(config, report)
    else:
        contract.bind_draft_verifier.assert_not_called()
    return draft, calls


@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("checkpoint_dtype", [torch.bfloat16, torch.float32])
def test_hf_draft_matches_loaded_target_and_accepts_its_hidden_states(
    evaluator, monkeypatch, target_dtype, checkpoint_dtype
):
    draft, calls = load_draft(
        evaluator,
        monkeypatch,
        target_dtype=target_dtype,
        checkpoint_dtype=checkpoint_dtype,
    )
    assert calls == [{"torch_dtype": target_dtype}]
    assert not draft.training
    assert all(parameter.dtype == target_dtype for parameter in draft.parameters())
    assert torch.equal(
        draft.verifier_lm_head.weight, torch.ones_like(draft.verifier_lm_head.weight)
    )
    hidden = torch.randn(1, 3, 8, dtype=target_dtype)
    with torch.inference_mode():
        projected = evaluator._prepare_dflash_target_context(draft, hidden)
        logits = draft.lm_head(projected)
    assert projected.dtype == logits.dtype == target_dtype
    assert torch.isfinite(logits).all()


def test_dsv4_draft_stays_bf16_without_a_local_target(evaluator, monkeypatch):
    draft, calls = load_draft(
        evaluator,
        monkeypatch,
        target_dtype=None,
        checkpoint_dtype=torch.float32,
        backend="dsv4-vllm",
    )
    assert calls == [{"torch_dtype": torch.bfloat16}]
    assert all(parameter.dtype == torch.bfloat16 for parameter in draft.parameters())


@pytest.mark.parametrize("sparse", [False, True])
def test_bf16_correction_still_computes_previous_softmax_in_fp32(
    evaluator, monkeypatch, sparse
):
    draft, _ = load_draft(
        evaluator,
        monkeypatch,
        target_dtype=torch.bfloat16,
        checkpoint_dtype=torch.float32,
    )
    definitions = load_module(
        "dspark_dtype_head_test",
        ROOT / "src/speculators/models/mmuse/correction.py",
    )
    head = definitions.CausalCorrectionHead(
        input_hidden_size=4,
        token_embedding_size=4,
        block_size=3,
        correction_hidden_size=4,
        correction_rank=2,
        num_heads=1,
        output_mode="logits",
        draft_vocab_size=5,
    ).to(dtype=draft.fc.weight.dtype)
    softmax_dtypes = []
    original_softmax = torch.softmax

    def softmax(input_tensor, *args, **kwargs):
        softmax_dtypes.append(input_tensor.dtype)
        return original_softmax(input_tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", softmax)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.bfloat16)
    arguments = (
        {"candidate_ids": torch.tensor([[0, 1]]), "candidate_logits": logits[:, :2]}
        if sparse
        else {"previous_logits": logits}
    )
    result = head.encode_previous_distribution(torch.tensor([True]), **arguments)
    assert softmax_dtypes == [torch.float32]
    assert result.dtype == torch.bfloat16
    assert torch.isfinite(result).all()
