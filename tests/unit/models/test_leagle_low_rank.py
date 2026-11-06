"""Factorized main-head supervision, native gradients and checkpoint semantics."""

import copy

import pytest
import torch
from safetensors.torch import save_file

from speculators import SpeculatorModel
from speculators.config import SpeculatorModelConfig
from speculators.losses import resolve_loss_config
from speculators.models.eagle3.metrics import compute_metrics
from speculators.models.leagle.candidate_head import LowRankCandidateHead
from speculators.train.leagle_optimizations import resolve_optimization_groups

from .test_leagle import batch, make_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(73)
    torch.set_num_threads(1)


def low_rank_model(precision="fp32"):
    model = make_model(
        layers=1,
        loops=2,
        attention="sdpa",
        candidate_head=True,
        candidate_rerank=False,
    ).train()
    return model.to(torch.bfloat16 if precision == "bf16" else torch.float32)


def inputs_for(model, empty=False):
    inputs = batch()
    inputs["document_ids"] = torch.tensor([[0, 0, 0, 1, 1, -1]])
    inputs["loss_mask"] = torch.tensor([[1, 0, 1, 1, 1, 0]])
    if empty:
        inputs["loss_mask"].zero_()
    inputs["verifier_last_hidden_states"] = torch.randn(1, 6, 16)
    return {
        key: value.to(model.dtype) if value.is_floating_point() else value
        for key, value in inputs.items()
    }


def gradients(model):
    return {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


@pytest.mark.parametrize("precision", ["fp32", "bf16", "amp"])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("loss_fn", ["kl_div", '{"ce": 0.1, "tv": 0.9}'])
def test_main_loss_matches_native_reference_and_trains_backbone(
    precision, checkpointed, loss_fn, monkeypatch
):
    model = low_rank_model(precision)
    model.set_activation_checkpointing(checkpointed)
    reference = copy.deepcopy(model)
    inputs = inputs_for(model)
    loss_config = resolve_loss_config(loss_fn, implementation="eager")
    captured = []
    hook = reference.candidate_head.up_proj.register_forward_hook(
        lambda _m, _args, output: captured.append(output)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Low-rank main training must not use a dense draft or auxiliary KL")

    monkeypatch.setattr(LowRankCandidateHead, "loss_accumulator", forbidden)
    monkeypatch.setattr(model.lm_head, "forward", forbidden)
    monkeypatch.setattr(reference.lm_head, "forward", forbidden)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "amp"):
        _, loss, metrics = model(
            **inputs, ttt_steps=3, ttt_step_loss_decay=0.7, loss_config=loss_config
        )
        reference(
            **{k: v for k, v in inputs.items() if k != "verifier_last_hidden_states"},
            ttt_steps=3,
            teacher_forcing=True,
        )
        with torch.no_grad():
            targets = reference.verifier_lm_head(
                reference.verifier_norm(inputs["verifier_last_hidden_states"])
            )
        expected = targets.new_zeros(())
        valid = inputs["loss_mask"].bool() & (inputs["document_ids"] >= 0)
        for step, logits in enumerate(captured):
            mask = valid.clone()
            if step:
                docs = inputs["document_ids"]
                mask[:, step:] &= docs[:, step:] == docs[:, :-step]
            value, _ = compute_metrics(
                logits,
                targets,
                mask,
                None,
                step,
                0.7,
                loss_config=loss_config,
                collect_metrics=False,
            )
            expected = expected + value
    hook.remove()
    torch.testing.assert_close(loss, expected, atol=0, rtol=0)
    (loss * 0.37).backward()
    (expected * 0.37).backward()
    torch.testing.assert_close(gradients(model), gradients(reference), atol=0, rtol=0)
    for name, gradient in gradients(model).items():
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
    assert model.candidate_head.down_proj.weight.grad.count_nonzero()
    assert model.candidate_head.up_proj.weight.grad.count_nonzero()
    assert model.layers[0].mlp.up_proj.weight.grad.count_nonzero()
    assert model.lm_head.weight.grad is None
    assert not any(key.startswith("candidate_") for key in metrics)
    assert len(model.gradient_clip_groups()) == 1


@pytest.mark.parametrize("precision", ["fp32", "bf16", "amp"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("mode", ["chunked", "masked", "cached"])
def test_low_rank_chunked_and_masked_gradients_match_dense(precision, empty, mode):
    reference = low_rank_model(precision)
    model = copy.deepcopy(reference)
    inputs = inputs_for(model, empty)
    options = {"leagle_chunked_loss": True, "leagle_loss_chunk_size": 2}
    if mode != "chunked":
        options["leagle_masked_head"] = True
    if mode == "cached":
        options.update(
            leagle_teacher_cache=True,
            leagle_teacher_cache_mb=1,
            leagle_cache_frozen_head_weights=True,
            leagle_cache_teacher_indices=True,
            leagle_teacher_cache_supervised=True,
            leagle_shared_loss_normalization=True,
        )
    outputs = []
    for item, kwargs in ((reference, {}), (model, options)):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "amp"):
            result = item(
                **inputs,
                ttt_steps=7,
                ttt_step_loss_decay=0.7,
                loss_config=resolve_loss_config(
                    '{"ce": 0.1, "tv": 0.9}', implementation="eager"
                ),
                return_predictions=False,
                **kwargs,
            )
        (result[1] * 0.37).backward()
        outputs.append(result)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=2e-6, rtol=2e-5)
    tolerance = (
        {"atol": 2e-7, "rtol": 2e-4}
        if precision == "fp32"
        else {"atol": 2e-4, "rtol": 0.03}
    )
    torch.testing.assert_close(gradients(model), gradients(reference), **tolerance)
    for name, gradient in gradients(model).items():
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        if empty:
            assert not gradient.count_nonzero(), name
    assert not any(key.startswith("candidate_") for key in outputs[1][2])


def test_primary_training_ignores_inference_override_and_round_trips(tmp_path):
    model = low_rank_model()
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    model.config.transformer_layer_config.save_pretrained(verifier)
    save_file(
        {
            "model.embed_tokens.weight": model.embed_tokens.weight.detach(),
            "lm_head.weight": model.verifier_lm_head.weight.detach(),
            "model.norm.weight": model.verifier_norm.weight.detach(),
        },
        verifier / "model.safetensors",
    )
    model.config.speculators_config.verifier.name_or_path = str(verifier)
    inputs = inputs_for(model)
    inputs["loss_config"] = resolve_loss_config("kl_div", implementation="eager")
    _, loss, _ = model(**inputs, ttt_steps=2)
    loss.backward()
    torch.optim.AdamW(model.parameters(), lr=0.01).step()
    model.eval()
    baseline = model(**inputs, ttt_steps=2)[1]
    model.set_candidate_head(True, rerank=True)
    torch.testing.assert_close(
        model(**inputs, ttt_steps=2)[1], baseline, atol=0, rtol=0
    )
    assert model.config.candidate_rerank is False
    checkpoint = tmp_path / "draft"
    model.save_pretrained(checkpoint)
    config = SpeculatorModelConfig.from_pretrained(checkpoint)
    config.transformer_layer_config._attn_implementation = "sdpa"
    restored = SpeculatorModel.from_pretrained(checkpoint, config=config).eval()
    assert restored.low_rank_main
    assert not restored.candidate_rerank
    torch.testing.assert_close(restored(**inputs, ttt_steps=2)[1], baseline)
    _, logits = restored.propose(
        inputs["hidden_states"], inputs["input_ids"][:, :1], num_tokens=2, temperature=1
    )
    assert torch.isfinite(logits).all()


def test_low_rank_main_groups_disable_auxiliary_and_frozen_head_paths():
    options = resolve_optimization_groups(
        {
            "candidate_head": True,
            "candidate_rerank": False,
            "leagle_safe_optimizations": True,
            "leagle_numerical_optimizations": True,
            "activation_checkpointing": True,
        }
    )
    for flag in (
        "leagle_candidate_grad_addmm",
        "leagle_candidate_backward_recompute",
        "leagle_align_candidate_chunks",
        "leagle_stream_main_loss",
    ):
        assert options[flag] is False
    model = low_rank_model()
    model.set_activation_checkpointing(True)
    inputs = inputs_for(model)
    inputs["loss_config"] = resolve_loss_config("kl_div", implementation="eager")
    options.update(
        leagle_loss_chunk_size=2, leagle_attn_chunk_size=2, leagle_teacher_cache_mb=1
    )
    _, loss, _ = model(**inputs, ttt_steps=3, **options)
    loss.backward()
    assert all(gradient is not None for gradient in gradients(model).values())


@pytest.mark.parametrize(
    "flag",
    [
        "leagle_candidate_grad_addmm",
        "leagle_candidate_backward_recompute",
        "leagle_align_candidate_chunks",
        "leagle_stream_main_loss",
    ],
)
def test_low_rank_main_rejects_auxiliary_only_or_frozen_head_execution(flag):
    model = low_rank_model()
    with pytest.raises(
        ValueError, match="auxiliary distillation|frozen output matrices"
    ):
        model(
            **inputs_for(model),
            loss_config=resolve_loss_config("kl_div", implementation="eager"),
            **{flag: True},
        )
