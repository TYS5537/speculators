"""Teacher index plans preserve projection shapes and checkpointed gradients."""

import copy

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from speculators.models.leagle.teacher_cache import TeacherCache
from speculators.models.metrics import resolve_training_loss

from .test_leagle import make_model
from .test_leagle_training_diagnostics import packed_batch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(172)
    torch.set_num_threads(1)


class CountNonzero(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.nonzero.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("supervised", [False, True])
@pytest.mark.parametrize("rows", [0, 2, 9])
@pytest.mark.parametrize("compact", [False, True])
def test_planned_partial_cache_lookups_are_exact_without_dynamic_nonzero(
    supervised, rows, compact
):
    hidden, weight = torch.randn(1, 9, 16), torch.randn(24, 16)
    priority = torch.tensor([[1, 0, 1, 0, 1, 1, 0, 1, 1]], dtype=torch.bool)
    if not rows:
        # An empty supervised subset is valid; a nonempty cache needs one row.
        priority.zero_()
        supervised = True
    cache = TeacherCache(
        hidden,
        weight,
        budget_bytes=72 + max(rows, 1) * (24 * 4 + 8),
        chunk_size=3,
        priority_mask=priority if supervised else None,
        cache_indices=True,
    )
    for start in (0, 2, 8, 9):
        positions = torch.arange(start, 9)
        if compact:
            positions = positions[::2]
            if supervised and positions.numel():
                # Indexed caches also support arbitrary order and duplicates.
                positions = torch.cat((positions.flip(0), positions[:1]))
        indices = positions if compact else None
        plans = cache.plan_chunks(positions.numel(), 3, start=start, indices=indices)
        assert (plans is None) is (not supervised and not compact)
        for chunk, offset in enumerate(range(0, positions.numel(), 3)):
            selected = positions[offset : offset + 3]
            arguments = {
                "start": start + offset,
                "indices": selected if compact else None,
            }
            expected = cache.get(hidden[:, selected], weight, **arguments)
            counter = CountNonzero()
            with counter:
                actual = cache.get(
                    hidden[:, selected],
                    weight,
                    **arguments,
                    plan=plans[chunk] if plans is not None else None,
                )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert counter.count == 0


def test_plans_are_opt_in_and_check_width():
    hidden, weight = torch.randn(1, 4, 16), torch.randn(24, 16)
    cache = TeacherCache(hidden, weight, budget_bytes=1024, chunk_size=2)
    assert cache.plan_chunks(4, 2, indices=torch.arange(4)) is None
    cache.cache_indices = True
    plans = cache.plan_chunks(4, 2, indices=torch.arange(4))
    with pytest.raises(ValueError, match="query length"):
        cache.get(hidden, weight, plan=plans[0])
    with pytest.raises(ValueError, match="query length"):
        cache.plan_chunks(4, 2, indices=torch.arange(3))
    with pytest.raises(ValueError, match="valid length"):
        cache.plan_chunks(4, 0, indices=torch.arange(4))


@pytest.mark.parametrize(
    "mode", ["dense", "masked", "candidate", "empty", "ar", "stream"]
)
@pytest.mark.parametrize("supervised", [False, True])
@pytest.mark.parametrize(
    ("dtype", "amp"),
    [(torch.float32, False), (torch.float32, True), (torch.bfloat16, False)],
)
@pytest.mark.parametrize("loss_fn", ["kl_div", '{"ce":0.1,"tv":0.9}'])
def test_plans_preserve_loss_gradients_updates_rng_and_remove_backward_nonzero(
    mode, supervised, dtype, amp, loss_fn, monkeypatch
):
    original_init = TeacherCache.__init__

    def partial_cache(self, hidden, weight, **kwargs):
        kwargs["budget_bytes"] = hidden.shape[1] * 8 + 2 * (4 * weight.shape[0] + 8)
        original_init(self, hidden, weight, **kwargs)

    monkeypatch.setattr(TeacherCache, "__init__", partial_cache)
    base = make_model(
        layers=1, loops=2, attention="sdpa", candidate_head=mode == "candidate"
    ).to(dtype)
    base.set_activation_checkpointing(True)
    inputs = packed_batch()
    for name in ("hidden_states", "verifier_last_hidden_states"):
        inputs[name] = inputs[name].to(dtype)
    if mode == "empty":
        inputs["loss_mask"].zero_()
    outcomes, counts = [], []
    for planned in (False, True):
        model = copy.deepcopy(base).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        torch.manual_seed(118)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            result = model(
                **inputs,
                ttt_steps=4,
                ttt_step_loss_decay=0.7,
                teacher_forcing=mode != "ar",
                loss_config=resolve_training_loss(loss_fn, loss_implementation="eager"),
                leagle_chunked_loss=True,
                leagle_loss_chunk_size=3,
                leagle_cache_frozen_head_weights=True,
                leagle_teacher_cache=True,
                leagle_teacher_cache_supervised=supervised,
                leagle_cache_teacher_indices=planned,
                leagle_masked_head=mode != "dense",
                leagle_shared_loss_normalization=mode == "stream",
                leagle_stream_main_loss=mode == "stream",
                leagle_reuse_training_kv=True,
                leagle_sparse_attention=True,
                leagle_attn_chunk_size=2,
                leagle_selective_checkpoint=True,
                return_predictions=mode == "ar",
            )
        counter = CountNonzero()
        with counter:
            (result[1] * 0.37).backward()
        counts.append(counter.count)
        gradients = {name: p.grad for name, p in model.named_parameters()}
        optimizer.step()
        outcomes.append(
            (
                result,
                gradients,
                model.state_dict(),
                optimizer.state_dict(),
                torch.get_rng_state(),
            )
        )
    torch.testing.assert_close(outcomes[0], outcomes[1], rtol=0, atol=0)
    assert counts[1] == 0
    if supervised and mode not in ("empty", "stream"):
        assert counts[0] > counts[1]


def test_forward_rejects_index_cache_without_teacher_cache():
    with pytest.raises(ValueError, match="Teacher index plans"):
        make_model()(**packed_batch(), leagle_cache_teacher_indices=True)
