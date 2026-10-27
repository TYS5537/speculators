"""Trainable retrieval, detached gradients, memory bounds and checkpoint integrity."""

import copy

import pytest
import torch
from safetensors.torch import load_file, save_file

from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.losses import resolve_loss_config
from speculators.models.leagle.candidate_head import LowRankCandidateHead
from speculators.proposals.sampling import logits_to_probs

from .test_leagle import batch, make_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def test_coarse_selection_then_exact_rescore_and_conditional_probabilities():
    head = LowRankCandidateHead(2, 4, 1)
    with torch.no_grad():
        head.down_proj.weight.copy_(torch.tensor([[1.0, 0.0]]))
        head.up_proj.weight.copy_(torch.tensor([[3.0], [4.0], [2.0], [1.0]]))
    weight = torch.tensor([[2.0, 0], [1.0, 0], [0, 9.0], [-1.0, 0]])
    hidden = torch.tensor([[[1.0, 0.0]]])
    assert head.coarse_logits(hidden).argmax(-1).item() == 1
    logits = head(hidden, weight, top_k=2)
    torch.testing.assert_close(
        logits, torch.tensor([[[2.0, 1.0, -torch.inf, -torch.inf]]])
    )
    assert logits.argmax(-1).item() == 0
    probs = logits_to_probs(logits, 1.0)
    torch.testing.assert_close(
        probs[..., :2], torch.tensor([2.0, 1.0]).softmax(-1).reshape(1, 1, 2)
    )
    assert probs[..., 2:].count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("empty", [False, True])
def test_chunked_distillation_matches_full_kl_and_gradients_without_backbone_gradient(
    dtype, empty
):
    chunked = LowRankCandidateHead(16, 24, 4).to(dtype)
    reference = copy.deepcopy(chunked)
    hidden = torch.randn(1, 7, 16, dtype=dtype, requires_grad=True)
    weight = torch.randn(24, 16, dtype=dtype, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 1, 0, 0, 1]], dtype=torch.bool)
    if empty:
        mask.zero_()
    count = mask.sum().clamp_min(1)
    loss_sum, recall, total = chunked.distillation_loss(
        hidden, weight, mask, top_k=3, chunk_size=2
    )
    with torch.no_grad():
        log_p = torch.nn.functional.linear(hidden, weight).float().log_softmax(-1)
    log_q = reference.coarse_logits(hidden.detach()).float().log_softmax(-1)
    expected = ((log_p.exp() * (log_p - log_q)).sum(-1) * mask).sum()
    torch.testing.assert_close(loss_sum, expected, atol=1e-5, rtol=1e-5)
    (loss_sum / count).backward()
    (expected / count).backward()
    assert hidden.grad is None
    assert weight.grad is None
    assert total.item() == mask.sum().item()
    assert 0 <= recall.item() <= total.item()
    for actual, wanted in zip(
        chunked.parameters(), reference.parameters(), strict=True
    ):
        assert actual.grad is not None
        assert torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, wanted.grad, atol=0.003, rtol=0.03)
        if empty:
            assert actual.grad.count_nonzero() == 0


def test_distillation_updates_both_factors_and_reduces_loss():
    head = LowRankCandidateHead(8, 12, 3)
    hidden = torch.randn(1, 32, 8)
    weight = (torch.randn(12, 3) @ torch.randn(3, 8)) * 0.25
    mask = torch.ones(1, 32, dtype=torch.bool)
    original = [p.clone() for p in head.parameters()]
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.05, weight_decay=0)
    losses = []
    for _ in range(45):
        loss, _, total = head.distillation_loss(
            hidden, weight, mask, top_k=3, chunk_size=8
        )
        loss = loss / total
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * 0.2
    assert all(
        not torch.equal(a, b) for a, b in zip(original, head.parameters(), strict=True)
    )


@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_propose_skips_dense_projection_and_returns_actual_sampling_support(
    reduced, dtype, monkeypatch
):
    model = make_model(loops=3, reduced=reduced, candidate_head=True).to(dtype).eval()
    seen = []
    hook = model.candidate_head.register_forward_pre_hook(
        lambda _m, args: seen.append(args[0].clone())
    )

    def forbidden(*args):
        pytest.fail("Candidate proposals must not execute the full dense head")

    monkeypatch.setattr(model.lm_head, "forward", forbidden)
    try:
        tokens, logits = model.propose(
            torch.randn(1, 5, 32).to(dtype),
            torch.tensor([[7]]),
            num_tokens=4,
            temperature=1.0,
        )
    finally:
        hook.remove()
    assert len(seen) == 4
    full = torch.nn.functional.linear(torch.cat(seen, dim=1), model.lm_head.weight)
    support = torch.isfinite(logits)
    assert torch.all(support.sum(-1) == 3)
    torch.testing.assert_close(logits[support], full[support], atol=0.03, rtol=0.02)
    probs = logits_to_probs(logits, 1.0)
    torch.testing.assert_close(probs.sum(-1), torch.ones(1, 4))
    assert not probs[~support].any()
    if reduced:
        assert torch.all(tokens % 2 == 0)
        tokens = tokens // 2
    assert torch.isfinite(logits.gather(-1, tokens.unsqueeze(-1))).all()


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("loss_fn", ["kl_div", '{"ce": 0.1, "tv": 0.9}'])
def test_joint_objective_preserves_dense_initialization_gradients_and_clipped_updates(
    training, checkpointed, loss_fn
):
    torch.manual_seed(71)
    dense = make_model().train(training)
    dense_rng = torch.get_rng_state()
    torch.manual_seed(71)
    joint = make_model(candidate_head=True).train(training)
    assert torch.equal(dense_rng, torch.get_rng_state())
    for name, weight in dense.state_dict().items():
        torch.testing.assert_close(weight, joint.state_dict()[name], atol=0, rtol=0)
    dense.set_activation_checkpointing(checkpointed)
    joint.set_activation_checkpointing(checkpointed)
    inputs = batch()
    inputs["verifier_last_hidden_states"] = torch.randn(1, 6, 16)
    results = []
    for model in (dense, joint):
        predictions, loss, metrics = model(
            **inputs,
            ttt_steps=3,
            loss_config=resolve_loss_config(loss_fn, implementation="eager"),
        )
        loss.backward()
        results.append((predictions, loss, metrics))
    torch.testing.assert_close(
        results[0][1], results[1][2]["draft_loss_sum"], atol=0, rtol=0
    )
    assert results[1][1] > results[0][1]
    for name, parameter in dense.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(
                parameter.grad,
                dict(joint.named_parameters())[name].grad,
                atol=0,
                rtol=0,
            )
    for model in (dense, joint):
        for group in model.gradient_clip_groups():
            torch.nn.utils.clip_grad_norm_(group, 0.1)
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    for name, parameter in dense.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(
                parameter, dict(joint.named_parameters())[name], atol=0, rtol=0
            )
    assert all(
        p.grad is not None and p.grad.count_nonzero()
        for p in joint.candidate_head.parameters()
    )


def test_packed_mask_and_ttt_decay_align_candidate_supervision(monkeypatch):
    model = make_model(candidate_head=True)
    inputs = batch()
    inputs["document_ids"] = torch.tensor([[0, 0, 0, 1, 1, -1]])
    inputs["loss_mask"] = torch.tensor([[1, 0, 1, 1, 1, 0]])
    inputs["verifier_last_hidden_states"] = torch.randn(1, 6, 16)
    calls = []
    original = model.candidate_head.distillation_loss

    def record(hidden, weight, mask, **kwargs):
        result = original(hidden, weight, mask, **kwargs)
        calls.append((mask.clone(), result))
        return result

    monkeypatch.setattr(model.candidate_head, "distillation_loss", record)
    _, loss, metrics = model(
        **inputs,
        ttt_steps=7,
        ttt_step_loss_decay=0.5,
        loss_config=resolve_loss_config("kl_div", implementation="eager"),
    )
    assert [mask.sum().item() for mask, _ in calls] == [4, 1, 1, 0, 0, 0]
    assert metrics["candidate_kl_total"].item() == 6
    expected = sum(
        0.5**i * values[0] / values[2].clamp_min(1)
        for i, (_, values) in enumerate(calls)
    )
    torch.testing.assert_close(loss, metrics["draft_loss_sum"] + expected)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.candidate_head.parameters())


def test_auxiliary_checkpoint_does_not_retain_sequence_vocabulary_activations():
    head = LowRankCandidateHead(8, 128, 4)
    hidden, weight = torch.randn(1, 17, 8), torch.randn(128, 8)
    saved_shapes = []

    def pack(value):
        saved_shapes.append(tuple(value.shape))
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        loss, _, _ = head.distillation_loss(
            hidden, weight, torch.ones(1, 17, dtype=torch.bool), top_k=3, chunk_size=2
        )
    assert (17, 128) not in saved_shapes
    assert (2, 128) not in saved_shapes
    loss.backward()


@pytest.mark.parametrize("reduced", [False, True])
def test_train_save_reload_and_runtime_switch(tmp_path, reduced):
    model = make_model(reduced=reduced, candidate_head=True)
    inputs = batch()
    inputs["verifier_last_hidden_states"] = torch.randn(1, 6, 16)
    _, loss, _ = model(
        **inputs,
        ttt_steps=2,
        loss_config=resolve_loss_config("kl_div", implementation="eager"),
    )
    loss.backward()
    torch.optim.AdamW(model.parameters(), lr=0.01).step()
    model.eval().save_pretrained(tmp_path)
    restored = SpeculatorModel.from_pretrained(
        tmp_path, attn_implementation="eager"
    ).eval()
    assert restored.config.candidate_head
    assert restored.candidate_head_enabled
    assert all(p.requires_grad for p in restored.candidate_head.parameters())
    for name, parameter in model.candidate_head.state_dict().items():
        torch.testing.assert_close(
            parameter, restored.candidate_head.state_dict()[name]
        )
    for enabled in (False, True):
        model.set_candidate_head(enabled)
        restored.set_candidate_head(enabled)
        a = model.propose(
            inputs["hidden_states"], inputs["input_ids"][:, -1:], num_tokens=3
        )
        b = restored.propose(
            inputs["hidden_states"], inputs["input_ids"][:, -1:], num_tokens=3
        )
        for actual, expected in zip(a, b, strict=True):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("corruption", ["missing", "nonfinite", "shape"])
def test_incomplete_trained_checkpoints_cannot_use_random_factors(tmp_path, corruption):
    model = make_model(candidate_head=True).eval()
    model.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    weights = load_file(path)
    key = "candidate_head.down_proj.weight"
    if corruption == "missing":
        del weights[key]
    elif corruption == "nonfinite":
        weights[key][0, 0] = torch.nan
    else:
        weights[key] = weights[key][:2]
    save_file(weights, path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="[Cc]andidate"):
        SpeculatorModel.from_pretrained(tmp_path, ignore_mismatched_sizes=True)


def test_missing_head_and_invalid_dimensions_rejected():
    model = make_model()
    assert model.candidate_head is None
    with pytest.raises(ValueError, match="Train LEAGLE"):
        model.set_candidate_head(True)
    for overrides in (
        {"candidate_head_rank": 17},
        {"candidate_head_top_k": 25},
        {"candidate_head_rank": 0},
        {"candidate_head_loss_weight": float("nan")},
        {"candidate_head_chunk_size": 0},
    ):
        with pytest.raises(ValueError):
            SpeculatorModelConfig.from_dict(
                {**model.config.to_dict(), "candidate_head": True, **overrides}
            )
