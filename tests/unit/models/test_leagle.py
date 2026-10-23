"""Numerical contracts for shared depth, autoregression and prefix isolation."""

import argparse
import copy

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config

from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.leagle import LEagleDraftModel, LEagleSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.config import TrainConfig
from speculators.train.legacy_cli import parse_train_args
from speculators.train.model_config import plan_pretrained_config_overrides


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def make_model(layers=2, loops=2, attention="eager", window=None, reduced=False):
    decoder = Qwen3Config(
        vocab_size=24,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        attention_dropout=0.0,
        sliding_window=window,
        layer_types=["full_attention" if window is None else "sliding_attention"]
        * layers,
    )
    decoder._attn_implementation = attention
    config = LEagleSpeculatorConfig(
        transformer_layer_config=decoder,
        draft_vocab_size=12 if reduced else 24,
        aux_hidden_state_layer_ids=[0, 1],
        num_loops=loops,
        speculators_config=SpeculatorsConfig(
            algorithm="leagle",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=7)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None, architectures=["Qwen3ForCausalLM"]
            ),
        ),
    )
    model = LEagleDraftModel(config)
    with torch.no_grad():
        model.embed_tokens.weight.normal_(std=0.3)
        model.lm_head.weight.normal_(std=0.3)
        model.verifier_lm_head.weight.copy_(model.lm_head.weight)
    if reduced:
        selected = torch.arange(0, 24, 2)
        t2d = torch.zeros(24, dtype=torch.bool)
        t2d[selected] = True
        model.load_vocab_mappings(t2d, selected - torch.arange(12))
    return model


def batch(length=6):
    return {
        "hidden_states": torch.randn(1, length, 32),
        "input_ids": torch.randint(0, 24, (1, length)),
        "document_ids": torch.zeros(1, length, dtype=torch.long),
        "position_ids": torch.arange(1, length + 1).unsqueeze(0),
    }


def forward_logits(model, inputs, **kwargs):
    outputs = []
    hook = model.lm_head.register_forward_hook(lambda _m, _a, out: outputs.append(out))
    try:
        result = model(**inputs, **kwargs)
    finally:
        hook.remove()
    return torch.stack(outputs, dim=1), result


@pytest.mark.parametrize("loops", [1, 2, 4])
def test_loop_caches_only_grow_on_token_steps(loops):
    model = make_model(loops=loops).eval()
    inputs = batch()
    lengths = []
    hook = model.layers[0].self_attn.register_forward_hook(
        lambda _m, _a, out: lengths.append(out[1][0].shape[-2])
    )
    model(**inputs, ttt_steps=3)
    hook.remove()
    assert lengths == [6] * loops + [12] * loops + [18] * loops
    assert not any(
        "markov" in name or "confidence" in name for name, _ in model.named_parameters()
    )
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in make_model(loops=1).parameters()
    )


def test_shared_loop_matches_independent_layers_and_summed_gradients():
    shared = make_model(layers=2, loops=3)
    unrolled = make_model(layers=6, loops=1)
    source = shared.state_dict()
    expanded = {}
    for name in unrolled.state_dict():
        if name.startswith("layers."):
            _, index, tail = name.split(".", 2)
            expanded[name] = source[f"layers.{int(index) % 2}.{tail}"]
        else:
            expanded[name] = source[name]
    unrolled.load_state_dict(expanded)
    inputs = batch()
    inputs["verifier_last_hidden_states"] = torch.randn(1, 6, 16)
    loss_config = resolve_loss_config("kl_div", implementation="eager")
    logits_a, (_, loss_a, _) = forward_logits(
        shared, inputs, ttt_steps=3, loss_config=loss_config
    )
    logits_b, (_, loss_b, _) = forward_logits(
        unrolled, inputs, ttt_steps=3, loss_config=loss_config
    )
    torch.testing.assert_close(logits_a, logits_b)
    torch.testing.assert_close(loss_a, loss_b)
    loss_a.backward()
    loss_b.backward()
    independent = dict(unrolled.named_parameters())
    for name, parameter in shared.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, name
        if name.startswith("layers."):
            _, index, tail = name.split(".", 2)
            expected = sum(
                independent[f"layers.{int(index) + 2 * r}.{tail}"].grad
                for r in range(3)
            )
        else:
            expected = independent[name].grad
        torch.testing.assert_close(parameter.grad, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("window", [None, 3])
def test_parallel_training_trajectories_match_real_autoregressive_rollout(
    attention, window
):
    model = make_model(attention=attention, window=window, reduced=True).eval()
    inputs = batch()
    logits, predictions = forward_logits(model, inputs, ttt_steps=4)
    for anchor in [0, 2, 5]:
        tokens, single_logits = model.propose(
            inputs["hidden_states"][:, : anchor + 1],
            inputs["input_ids"][:, anchor : anchor + 1],
            num_tokens=4,
        )
        torch.testing.assert_close(
            logits[:, :, anchor], single_logits, atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            tokens, torch.stack(predictions, dim=1)[:, :, anchor]
        )
        assert torch.all(tokens % 2 == 0)  # verifier IDs, not draft vocabulary indices


def test_future_teacher_context_and_other_trajectories_are_invisible():
    model = make_model().eval()
    inputs = batch()
    original, _ = forward_logits(model, inputs, ttt_steps=4)
    changed = copy.deepcopy(inputs)
    changed["hidden_states"][:, 3:] += 100
    changed["input_ids"][:, 3:] = (changed["input_ids"][:, 3:] + 7) % 24
    perturbed, _ = forward_logits(model, changed, ttt_steps=4)
    torch.testing.assert_close(original[:, :, :3], perturbed[:, :, :3])


def test_packed_documents_and_padding_do_not_leak_or_receive_cross_document_loss():
    model = make_model().eval()
    inputs = batch()
    inputs["document_ids"] = torch.tensor([[0, 0, 0, 1, 1, -1]])
    inputs["position_ids"] = torch.tensor([[1, 2, 3, 1, 2, 0]])
    inputs["verifier_last_hidden_states"] = torch.randn(1, 6, 16)
    logits, (_, loss, metrics) = forward_logits(
        model,
        inputs,
        ttt_steps=3,
        teacher_forcing=False,
        loss_config=resolve_loss_config("kl_div", implementation="eager"),
    )
    assert torch.isfinite(loss)
    assert [metrics[f"position_{s}_acc_total"].item() for s in range(3)] == [5, 3, 1]
    tokens, separate_logits = model.propose(
        inputs["hidden_states"][:, 3:5],
        inputs["input_ids"][:, 4:5],
        num_tokens=3,
    )
    assert tokens.shape == (1, 3)
    torch.testing.assert_close(logits[:, :, 4], separate_logits, atol=2e-6, rtol=2e-5)


def test_teacher_forcing_is_explicit_and_disabled_for_proposals():
    model = make_model().eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()  # every greedy prediction is verifier token 0
    inputs = batch()
    inputs["input_ids"] = torch.tensor([[1, 2, 3, 4, 5, 6]])
    seen = []
    hook = model.embed_tokens.register_forward_pre_hook(
        lambda _m, args: seen.append(args[0].clone())
    )
    model(**inputs, ttt_steps=3, teacher_forcing=False)
    assert torch.equal(seen[1], torch.zeros_like(inputs["input_ids"]))
    seen.clear()
    model(**inputs, ttt_steps=3, teacher_forcing=True)
    assert torch.equal(seen[1], torch.tensor([[2, 3, 4, 5, 6, 0]]))
    hook.remove()


@pytest.mark.parametrize("selected", [None, [1]])
@pytest.mark.parametrize(
    ("loss_fn", "implementation", "attention"),
    [("kl_div", "eager", "eager"), ('{"ce": 0.1, "tv": 0.9}', "legacy", "sdpa")],
)
def test_checkpoint_recompute_preserves_cache_and_full_loop_gradients(
    selected, loss_fn, implementation, attention
):
    plain = make_model(attention=attention)
    recomputed = copy.deepcopy(plain)
    recomputed.set_activation_checkpointing(True, layer_ids=selected)
    inputs = batch()
    inputs["verifier_last_hidden_states"] = torch.randn(1, 6, 16)
    train_kwargs, _ = LEagleDraftModel.get_trainer_kwargs(
        loss_fn=loss_fn, loss_implementation=implementation, ttt_steps=3
    )
    losses = []
    for model in [plain, recomputed]:
        _, loss, _ = model(
            **inputs,
            **train_kwargs,
        )
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(*losses)
    for (name, a), (_, b) in zip(
        plain.named_parameters(), recomputed.named_parameters(), strict=True
    ):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4, msg=name)


def test_checkpoint_retains_architecture_and_single_copy_of_shared_weights(tmp_path):
    model = make_model(layers=2, loops=3)
    model.config.save_pretrained(tmp_path)
    restored = SpeculatorModelConfig.from_pretrained(tmp_path)
    assert isinstance(restored, LEagleSpeculatorConfig)
    assert restored.num_loops == 3
    assert restored.effective_depth == 6
    assert (
        SpeculatorModel.registered_model_class_from_config(restored) is LEagleDraftModel
    )
    assert not any(name.startswith("layers.2.") for name in model.state_dict())


def test_training_config_keeps_depth_loop_and_horizon_independent():
    config = TrainConfig(
        speculator_type="leagle",
        draft={"num_layers": 2, "num_loops": 3},
    )
    flat = config.flatten()
    assert (flat["num_layers"], flat["num_loops"], flat["ttt_steps"]) == (2, 3, 7)
    assert flat["draft_arch"] == "qwen3"
    assert flat["norm_output"] is True
    explicit = TrainConfig(speculator_type="leagle", loss={"ttt_steps": 4})
    assert explicit.flatten()["ttt_steps"] == 4
    with pytest.raises(ValueError, match="num-loops"):
        TrainConfig(speculator_type="dspark", draft={"num_loops": 2})


def test_both_cli_paths_parse_loop_count_and_protect_checkpoint_policy():
    argv = [
        "--verifier-name-or-path",
        "fixture",
        "--data-path",
        "fixture",
        "--speculator-type",
        "leagle",
        "--num-layers",
        "2",
        "--num-loops",
        "3",
    ]
    typed = argparse.Namespace(**TrainConfig.resolve(argv).flatten())
    legacy = parse_train_args(argv)
    for args in [typed, legacy]:
        assert (args.num_layers, args.num_loops, args.ttt_steps) == (2, 3, 7)
    config = make_model(loops=4).config
    with pytest.raises(ValueError, match="num-loops"):
        plan_pretrained_config_overrides(typed, config, {"num_loops"})
    inherited, _ = plan_pretrained_config_overrides(typed, config, set())
    assert inherited["num_loops"] == 4


def test_optimizer_step_and_real_checkpoint_reload_with_local_verifier(tmp_path):
    template = make_model(loops=2)
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    decoder = copy.deepcopy(template.config.transformer_layer_config)
    decoder.architectures = ["Qwen3ForCausalLM"]
    decoder.save_pretrained(verifier)
    save_file(
        {
            "model.embed_tokens.weight": template.embed_tokens.weight.detach(),
            "lm_head.weight": template.lm_head.weight.detach(),
            "model.norm.weight": template.verifier_norm.weight.detach(),
        },
        verifier / "model.safetensors",
    )
    model = LEagleDraftModel.from_training_args(
        decoder,
        verifier_name_or_path=str(verifier),
        draft_vocab_size=24,
        target_layer_ids=[0, 1],
        draft_attn_impl="eager",
        num_loops=2,
        ttt_steps=7,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    inputs = batch()
    inputs["verifier_last_hidden_states"] = torch.randn(1, 6, 16)
    initial = model.layers[0].self_attn.q_proj.weight.detach().clone()
    _, loss, _ = model(
        **inputs,
        ttt_steps=3,
        loss_config=resolve_loss_config("kl_div", implementation="eager"),
    )
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(initial, model.layers[0].self_attn.q_proj.weight)
    model.eval()
    expected = model.propose(
        inputs["hidden_states"], inputs["input_ids"][:, -1:], num_tokens=3
    )
    checkpoint_path = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint_path)
    config = SpeculatorModelConfig.from_pretrained(checkpoint_path)
    config.transformer_layer_config._attn_implementation = "eager"
    restored = SpeculatorModel.from_pretrained(
        checkpoint_path, config=config, local_files_only=True
    ).eval()
    actual = restored.propose(
        inputs["hidden_states"], inputs["input_ids"][:, -1:], num_tokens=3
    )
    for before, after in zip(expected, actual, strict=True):
        torch.testing.assert_close(before, after)
    assert restored.config.num_loops == 2
    assert not restored.embed_tokens.weight.requires_grad
    assert not restored.lm_head.weight.requires_grad
