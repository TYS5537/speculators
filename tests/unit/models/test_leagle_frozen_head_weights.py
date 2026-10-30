"""Batch-local frozen head casts preserve AMP training while avoiding recasts."""

import copy

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from speculators.models.leagle.output_head import training_output_weights
from speculators.models.metrics import resolve_training_loss

from .test_leagle import make_model
from .test_leagle_training_diagnostics import packed_batch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(674)
    torch.set_num_threads(1)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_copies_are_local_deduplicated_and_refreshed(shared, dtype):
    model = make_model(layers=1, loops=1)
    if shared:
        model.share_output_head()
    original = (model.lm_head.weight, model.verifier_lm_head.weight)
    before = copy.deepcopy(model.state_dict())
    with torch.autocast("cpu", dtype=dtype):
        copies = training_output_weights(model, cache_autocast=True)
        disabled = training_output_weights(model)
    for weight, value, unchanged in zip(original, copies, disabled, strict=True):
        assert unchanged is weight
        assert value is not weight
        assert not value.requires_grad
        torch.testing.assert_close(value, weight.to(dtype), rtol=0, atol=0)
    assert (copies[0] is copies[1]) is shared
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    with torch.no_grad():
        model.lm_head.weight.add_(0.25)
    with torch.autocast("cpu", dtype=dtype):
        refreshed = training_output_weights(model, cache_autocast=True)
    assert refreshed[0] is not copies[0]
    torch.testing.assert_close(refreshed[0], original[0].to(dtype), rtol=0, atol=0)
    assert not torch.equal(refreshed[0], copies[0])
    assert model.lm_head.weight is original[0]
    assert model.verifier_lm_head.weight is original[1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("trainable", [False, True])
def test_only_frozen_fp32_autocast_weights_are_copied(dtype, amp, trainable):
    model = make_model(layers=1, loops=1).to(dtype)
    model.lm_head.requires_grad_(trainable)
    model.verifier_lm_head.requires_grad_(trainable)
    original = (model.lm_head.weight, model.verifier_lm_head.weight)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        weights = training_output_weights(model, cache_autocast=True)
    copied = dtype == torch.float32 and amp and not trainable
    for actual, parameter in zip(weights, original, strict=True):
        assert (actual is parameter) is not copied


class CountHeadCasts(TorchDispatchMode):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (
            func is torch.ops.aten._to_copy.default
            and args[0].shape == self.shape
            and args[0].dtype == torch.float32
            and kwargs.get("dtype") == torch.bfloat16
        ):
            self.count += 1
        return func(*args, **kwargs)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_full_forward_backward_casts_once_per_distinct_frozen_head(
    shared, cached, chunked
):
    model = make_model(layers=1, loops=1, attention="sdpa")
    if shared:
        model.share_output_head()
    inputs = packed_batch()
    counter = CountHeadCasts(model.lm_head.weight.shape)
    with counter:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss, _ = model(
                **inputs,
                ttt_steps=4,
                leagle_chunked_loss=chunked,
                leagle_loss_chunk_size=2,
                leagle_cache_frozen_head_weights=cached,
                loss_config=resolve_training_loss(
                    '{"ce":0.1,"tv":0.9}', loss_implementation="eager"
                ),
                return_predictions=True,
            )
        (loss * 0.37).backward()
    if cached and chunked:
        assert counter.count == (1 if shared else 2)
    else:
        assert counter.count > 2


@pytest.mark.parametrize(
    "mode",
    [
        "plain",
        "teacher",
        "masked",
        "candidate",
        "native_candidate",
        "trainable",
        "empty",
        "autoregressive",
    ],
)
@pytest.mark.parametrize(
    ("dtype", "amp"),
    [(torch.float32, True), (torch.float32, False), (torch.bfloat16, False)],
)
@pytest.mark.parametrize("loss_fn", ["kl_div", '{"ce":0.1,"tv":0.9}'])
def test_cache_preserves_losses_metrics_gradients_updates_and_rng(
    mode, dtype, amp, loss_fn
):
    candidate = mode in ("candidate", "native_candidate")
    base = make_model(layers=1, loops=2, attention="sdpa", candidate_head=candidate).to(
        dtype
    )
    base.set_activation_checkpointing(True)
    if mode == "trainable":
        base.lm_head.requires_grad_(True)
    elif mode != "plain":
        base.share_output_head()
    inputs = packed_batch()
    for name in ("hidden_states", "verifier_last_hidden_states"):
        inputs[name] = inputs[name].to(dtype)
    if mode == "empty":
        inputs["loss_mask"].zero_()
    outcomes = []
    for cached in (False, True):
        model = copy.deepcopy(base).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        torch.manual_seed(93)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            result = model(
                **inputs,
                ttt_steps=10 if mode == "autoregressive" else 4,
                ttt_step_loss_decay=0.7,
                teacher_forcing=mode != "autoregressive",
                loss_config=resolve_training_loss(loss_fn, loss_implementation="eager"),
                leagle_chunked_loss=True,
                leagle_loss_chunk_size=3,
                leagle_cache_frozen_head_weights=cached,
                leagle_teacher_cache=mode not in ("plain", "trainable"),
                leagle_teacher_cache_supervised=mode not in ("plain", "trainable"),
                leagle_masked_head=mode not in ("plain", "teacher", "trainable"),
                leagle_candidate_backward_recompute=mode == "native_candidate",
                leagle_reuse_training_kv=True,
                leagle_sparse_attention=True,
                leagle_attn_chunk_size=2,
                leagle_selective_checkpoint=True,
                leagle_cache_attention_masks=True,
                return_predictions=mode in ("plain", "autoregressive", "empty"),
            )
        (result[1] * 0.37).backward()
        gradients = {name: p.grad for name, p in model.named_parameters()}
        optimizer.step()
        outcomes.append((result, gradients, model.state_dict(), torch.get_rng_state()))
    torch.testing.assert_close(outcomes[0], outcomes[1], rtol=0, atol=0)
