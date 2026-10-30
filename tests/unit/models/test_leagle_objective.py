"""Check the shared training objective against full-vocabulary PyTorch formulas."""

import itertools

import pytest
import torch
from torch.nn import functional

from speculators.models.metrics import resolve_training_loss

from .test_leagle import make_model
from .test_leagle_training_diagnostics import packed_batch


def explicit_objective(model, states, inputs, decay):
    """Independent hard CE + TV and auxiliary KL, with whole-step denominators."""
    with torch.no_grad():
        teacher = model.verifier_lm_head(
            model.verifier_norm(inputs["verifier_last_hidden_states"])
        )
    docs = inputs["document_ids"]
    valid = (docs >= 0) & inputs["loss_mask"].bool()
    main = teacher.new_zeros((), dtype=torch.float32)
    auxiliary = main.clone()
    for step, hidden in enumerate(states):
        width = hidden.shape[1] - step
        if width <= 0:
            continue
        hidden = hidden[:, :width]
        logits = functional.linear(hidden, model.lm_head.weight)
        targets = teacher[:, step:]
        mask = valid[:, step:] & (docs[:, :width] == docs[:, step:])
        ce = functional.cross_entropy(
            logits.flatten(0, 1), targets.argmax(-1).flatten(), reduction="none"
        ).reshape_as(mask)
        tv = 1 - torch.minimum(
            logits.float().softmax(-1), targets.float().softmax(-1)
        ).sum(-1)
        # Compute each weighted term separately, matching the specified objective.
        for vector, weight in ((ce, 0.1), (tv, 0.9)):
            main = main + weight * (vector * mask * decay**step).sum() / (
                mask.sum() + 1e-5
            )
        # Ordinary autograd through both candidate matrices is independent of the
        # streamed, manually accumulated gradients used by production training.
        coarse = model.candidate_head.up_proj(
            model.candidate_head.down_proj(hidden.detach())
        )
        target_logp = logits.detach().float().log_softmax(-1)
        kl = (target_logp.exp() * (target_logp - coarse.float().log_softmax(-1))).sum(
            -1
        )
        candidate_mask = valid[:, :width] & mask
        auxiliary = auxiliary + (
            kl * candidate_mask
        ).sum() * decay**step / candidate_mask.sum().clamp_min(1)
    return main + model.config.candidate_head_loss_weight * auxiliary


@pytest.mark.parametrize(
    ("reuse", "chunked", "sparse"), list(itertools.product([False, True], repeat=3))
)
def test_shared_objective_loss_and_gradients_against_explicit_formula(
    reuse, chunked, sparse, monkeypatch
):
    torch.manual_seed(738)
    torch.set_num_threads(1)
    model = make_model(layers=2, loops=2, attention="sdpa", candidate_head=True)
    model.set_activation_checkpointing(True)
    inputs = packed_batch()
    states = []
    original = model._step

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        states.append(result[3])
        return result

    monkeypatch.setattr(model, "_step", record)
    _, loss, _ = model(
        **inputs,
        ttt_steps=11,
        ttt_step_loss_decay=0.7,
        collect_metrics=False,
        loss_config=resolve_training_loss(
            '{"ce": 0.1, "tv": 0.9}', loss_implementation="legacy"
        ),
        leagle_reuse_training_kv=reuse,
        leagle_chunked_loss=chunked,
        leagle_sparse_attention=sparse,
        leagle_loss_chunk_size=4,
        leagle_attn_chunk_size=3,
    )
    expected = explicit_objective(model, states, inputs, decay=0.7)
    parameters = [p for p in model.parameters() if p.requires_grad]
    actual_grads = torch.autograd.grad(loss, parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, parameters)
    torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_grads, expected_grads, atol=2e-6, rtol=2e-5)
