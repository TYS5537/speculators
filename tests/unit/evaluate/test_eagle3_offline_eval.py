"""Draft proposal alignment, cached recurrence and checkpointed training parity."""

import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators_eval import leagle
from speculators_eval.eagle3_decoding import propose_eagle3
from speculators_eval.leagle_decoding import check_greedy_output, generate_greedy

from .test_leagle_offline_eval import make_real_models, save_fixture


def make_eagle(
    *, layers=2, arch="llama", reduced=True, norm_output=True, fc_norm=False
):
    target, _ = make_real_models(layers=layers, reduced=reduced)
    decoder = copy.deepcopy(target.config)
    decoder.num_hidden_layers = layers
    decoder.layer_types = ["full_attention"] * layers
    if arch == "llama":
        decoder = LlamaConfig(**decoder.to_dict())
    decoder._attn_implementation = "sdpa"
    draft = Eagle3DraftModel(
        Eagle3SpeculatorConfig(
            transformer_layer_config=decoder,
            draft_vocab_size=16 if reduced else 32,
            eagle_aux_hidden_state_layer_ids=[1, 2],
            norm_output=norm_output,
            norm_before_residual=True,
            norm_before_fc=not fc_norm,
            fc_norm=fc_norm,
            speculators_config=SpeculatorsConfig(
                algorithm="eagle3",
                proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=7)],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=None, architectures=["Qwen3ForCausalLM"]
                ),
            ),
        )
    ).eval()
    selected = torch.arange(0, 32, 2) if reduced else torch.arange(32)
    if reduced:
        mask = torch.zeros(32, dtype=torch.bool)
        mask[selected] = True
        draft.load_vocab_mappings(mask, selected - torch.arange(16))
    with torch.no_grad():
        draft.embed_tokens.weight.copy_(target.model.embed_tokens.weight)
        draft.lm_head.weight.copy_(target.lm_head.weight[selected])
        draft.verifier_lm_head.weight.copy_(target.lm_head.weight[selected])
    return target, draft


@pytest.mark.parametrize("arch", ["llama", "qwen3"])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("norm_output", [False, True])
def test_proposal_matches_training_rollout_and_target_greedy(
    arch, layers, reduced, norm_output
):
    torch.manual_seed(1)
    torch.set_num_threads(1)
    target, draft = make_eagle(
        arch=arch,
        layers=layers,
        reduced=reduced,
        norm_output=norm_output,
        fc_norm=not norm_output,
    )
    prefix = torch.tensor([[5, 9, 2]])
    hidden = torch.randn(1, 3, 32)
    with_anchor = torch.cat([prefix, torch.tensor([[4]])], dim=1)
    proposals, logits = propose_eagle3(
        draft,
        hidden_states=hidden,
        input_ids=with_anchor,
        position_ids=torch.arange(3).unsqueeze(0),
        num_tokens=3,
    )
    # Feed the proposed tokens as teacher-forcing inputs in the independent
    # training path. Position 2 follows exactly this autoregressive trajectory.
    rollout_hidden = torch.cat([hidden, torch.zeros(1, 2, 32)], dim=1)
    captured = []
    handle = draft.lm_head.register_forward_hook(
        lambda _m, _a, out: captured.append(out)
    )
    try:
        with torch.no_grad():
            draft(
                hidden_states=rollout_hidden,
                input_ids=torch.cat([with_anchor[:, 1:], proposals[:, :2]], dim=1),
                document_ids=torch.zeros(1, 5, dtype=torch.long),
                ttt_steps=3,
            )
    finally:
        handle.remove()
    for step, actual in enumerate(captured):
        torch.testing.assert_close(logits[:, step], actual[:, 2], atol=1e-6, rtol=1e-5)
    response = generate_greedy(
        target, draft, prefix, propose=propose_eagle3, max_new_tokens=20
    )
    check_greedy_output(target, prefix, response, 20, [])


@pytest.mark.parametrize("arch", ["llama", "qwen3"])
@pytest.mark.parametrize("selected", [None, [0], [1]])
def test_eagle_checkpointing_preserves_ttt_loss_and_all_gradients(arch, selected):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    _, plain = make_eagle(arch=arch)
    plain.train()
    recomputed = copy.deepcopy(plain)
    recomputed.set_activation_checkpointing(True, layer_ids=selected)
    inputs = {
        "hidden_states": torch.randn(1, 6, 32),
        "input_ids": torch.randint(0, 32, (1, 6)),
        "document_ids": torch.zeros(1, 6, dtype=torch.long),
        "verifier_last_hidden_states": torch.randn(1, 6, 16),
        "ttt_steps": 3,
        "loss_config": resolve_loss_config("kl_div", implementation="eager"),
    }
    losses = []
    for model in (plain, recomputed):
        _, loss, _ = model(**inputs)
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(*losses)
    for (name, left), (_, right) in zip(
        plain.named_parameters(), recomputed.named_parameters(), strict=True
    ):
        if left.requires_grad:
            assert left.grad is not None, name
            torch.testing.assert_close(left.grad, right.grad, msg=name)


def test_eagle_saved_checkpoint_uses_shared_eval_pipeline(tmp_path):
    args = save_fixture(tmp_path)
    target, draft = make_eagle()
    target.save_pretrained(args.verifier_model)
    draft.save_pretrained(args.draft_model)
    leagle.run(args, entrypoint=Path("scripts/evaluate/eagle3_offline_eval.py"))
    metadata = json.loads((args.output_dir / "eval_config.json").read_text())
    assert metadata["algorithm"] == "eagle3"
    assert metadata["num_loops"] == 1
    assert metadata["spec_tokens"] == 7
    assert (args.output_dir / "drafter_checkpoint_sha256.txt").is_file()
