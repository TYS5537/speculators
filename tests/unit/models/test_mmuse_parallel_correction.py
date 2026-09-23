"""Parallel Correction feedback, residual dtype and projection-boundary contracts."""

import pytest
import torch

from speculators.models.mmuse.parallel_correction import (
    add_parallel_base_residual,
    add_parallel_hidden_residual,
    add_parallel_projected_residual,
    build_parallel_logit_kwargs,
)
from tests.unit.models.test_mmuse_anchor_correction import _ParallelHarness


@pytest.mark.parametrize("start", [0, 1])
@pytest.mark.parametrize("block", [1, 2, 5])
@pytest.mark.parametrize("orphan_rank", [False, True])
def test_dense_feedback_alignment_and_gradient_are_unchanged(start, block, orphan_rank):
    targets = torch.randn(2, block, 6, dtype=torch.float64, requires_grad=True)
    positions = torch.arange(block).expand(2, -1)
    output = build_parallel_logit_kwargs(
        targets.view(1, 2 * block, 6),
        positions,
        positions[:, start:],
        num_blocks=2,
        block_size=block,
        start_position=start,
        selector_previous_rank_features=torch.empty(1) if orphan_rank else None,
        selector_previous_logits_mask=None,
    )
    assert list(output) == ["previous_logits", "previous_logits_mask"]
    expected = targets[:, :-1]
    if not start:
        expected = torch.cat([torch.zeros_like(targets[:, :1]), expected], dim=1)
    torch.testing.assert_close(output["previous_logits"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output["previous_logits_mask"], (positions > 0)[:, start:]
    )
    assert output["previous_logits"].dtype == targets.dtype
    output["previous_logits"].sum().backward()
    expected_grad = torch.ones_like(targets)
    expected_grad[:, -1] = 0
    torch.testing.assert_close(targets.grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("start", [0, 1])
@pytest.mark.parametrize("has_rank", [False, True])
def test_compact_feedback_ignores_dense_targets_and_preserves_views(start, has_rank):
    source = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
    rank = source.transpose(0, 1)
    # Existing compact masks are sliced as supplied, not recast to bool here.
    mask = torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]], dtype=torch.float64)
    positions = torch.arange(3).expand(2, -1)
    output = build_parallel_logit_kwargs(
        torch.empty(1),
        positions,
        positions[:, start:],
        num_blocks=2,
        block_size=3,
        start_position=start,
        selector_previous_rank_features=rank if has_rank else None,
        selector_previous_logits_mask=mask,
    )
    assert output["previous_logits"] is None
    assert output["previous_logits_mask"].dtype == mask.dtype
    torch.testing.assert_close(output["previous_logits_mask"], mask[:, start:])
    if has_rank:
        torch.testing.assert_close(output["previous_rank_features"], rank[:, start:])
        assert output["previous_rank_features"].stride() == rank[:, start:].stride()
        output["previous_rank_features"].sum().backward()
        expected_grad = torch.ones_like(source)
        expected_grad[:start] = 0
        torch.testing.assert_close(source.grad, expected_grad, rtol=0, atol=0)
    else:
        assert "previous_rank_features" not in output


@pytest.mark.parametrize("start", [0, 1])
@pytest.mark.parametrize(
    ("output_dtype", "residual_dtype"),
    [
        (torch.float64, torch.float32),
        (torch.float32, torch.float64),
        (torch.bfloat16, torch.float32),
    ],
)
def test_residual_paths_preserve_casts_reserved_slots_and_gradients(
    start, output_dtype, residual_dtype
):
    hidden = torch.randn(2, 3, 4, dtype=output_dtype, requires_grad=True)
    delta_hidden = torch.randn(
        2, 3 - start, 4, dtype=residual_dtype, requires_grad=True
    )
    base = torch.randn(2, 3, 6, dtype=output_dtype, requires_grad=True)
    residual = torch.randn(2, 3 - start, 6, dtype=residual_dtype, requires_grad=True)
    corrected = add_parallel_hidden_residual(hidden, delta_hidden, start_position=start)
    base_result = add_parallel_base_residual(
        base,
        residual,
        start_position=start,
        mask_tokens_size=6,
    ).view(2, 3, 6)
    projected_result = add_parallel_projected_residual(
        base.view(1, 6, 6),
        residual,
        num_blocks=2,
        start_position=start,
        mask_tokens_size=6,
    ).view(2, 3, 6)
    assert (
        corrected.dtype == base_result.dtype == projected_result.dtype == output_dtype
    )
    for position in range(3):
        expected_hidden, expected_logits = hidden[:, position], base[:, position]
        if position >= start:
            expected_hidden = expected_hidden + delta_hidden[:, position - start].to(
                output_dtype
            )
            expected_logits = expected_logits + residual[:, position - start].to(
                output_dtype
            )
        torch.testing.assert_close(
            corrected[:, position], expected_hidden, rtol=0, atol=0
        )
        torch.testing.assert_close(
            base_result[:, position], expected_logits, rtol=0, atol=0
        )
        torch.testing.assert_close(
            projected_result[:, position], expected_logits, rtol=0, atol=0
        )
    (corrected.sum() + base_result.sum() + projected_result.sum()).backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden), rtol=0, atol=0)
    torch.testing.assert_close(
        delta_hidden.grad, torch.ones_like(delta_hidden), rtol=0, atol=0
    )
    torch.testing.assert_close(base.grad, torch.full_like(base, 2), rtol=0, atol=0)
    torch.testing.assert_close(
        residual.grad, torch.full_like(residual, 2), rtol=0, atol=0
    )


def test_hidden_residual_does_not_reuse_the_correction_input_view():
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    active_hidden = hidden[:, 1:]
    delta_hidden = active_hidden * 0.3
    output = add_parallel_hidden_residual(hidden, delta_hidden, start_position=1)
    # Cat(anchor, Add(fresh_slice, delta)); merging the two views changes backward
    # accumulation order. Identity here complements numerical regression tests.
    addition = output.grad_fn.next_functions[1][0]
    residual_view = addition.next_functions[0][0]
    assert residual_view is not active_hidden.grad_fn
    assert addition.next_functions[1][0] is delta_hidden.grad_fn
    assert add_parallel_hidden_residual(hidden, None, start_position=1) is None


@pytest.mark.parametrize("mode", ["hidden", "logits", "logits_aux", "dual", "dual_aux"])
@pytest.mark.parametrize("missing_head", [False, True])
def test_head_and_base_failures_keep_embedding_and_projection_order(mode, missing_head):
    model = _ParallelHarness(False, mode)
    encoder = model.correction_head
    events = []
    if missing_head:
        model.correction_head = None
    handles = [
        model.embed_tokens.register_forward_hook(
            lambda *_args: events.append(("embedding", torch.is_grad_enabled()))
        ),
        model.lm_head.register_forward_hook(
            lambda *_args: events.append(("projection", True))
        ),
    ]
    kwargs = {
        "targets": torch.zeros(1, 6, 6),
        "base_logits_blocks": None,
    }
    try:
        if missing_head or mode.startswith("logits"):
            message = (
                "Parallel Correction requires an enabled head"
                if missing_head
                else "Logit-residual Correction requires base logits"
            )
            with pytest.raises(RuntimeError) as error:
                model._teacher_forced_parallel_correction(
                    torch.zeros(2, 3, 4),
                    torch.zeros(2, 3, dtype=torch.long),
                    torch.arange(3).expand(2, -1),
                    **kwargs,
                )
            assert str(error.value) == message
            assert encoder.calls == []
            assert encoder.auxiliary_calls == 0
        else:
            logits, _, _ = model._teacher_forced_parallel_correction(
                torch.zeros(2, 3, 4),
                torch.zeros(2, 3, dtype=torch.long),
                torch.arange(3).expand(2, -1),
                **kwargs,
            )
            assert (logits is None) == (mode == "hidden")
    finally:
        for handle in handles:
            handle.remove()
    expected_events = [] if missing_head else [("embedding", False)]
    if not missing_head and mode.startswith("dual"):
        expected_events.append(("projection", True))
    assert events == expected_events


@pytest.mark.parametrize("start", [0, 1])
def test_parallel_helpers_compile_without_graph_breaks(start):
    positions = torch.arange(3).expand(2, -1)

    def function(hidden, delta_hidden, residual, base, targets):
        kwargs = build_parallel_logit_kwargs(
            targets,
            positions,
            positions[:, start:],
            num_blocks=2,
            block_size=3,
            start_position=start,
            selector_previous_rank_features=None,
            selector_previous_logits_mask=None,
        )
        return (
            add_parallel_hidden_residual(hidden, delta_hidden, start_position=start),
            add_parallel_base_residual(
                base, residual, start_position=start, mask_tokens_size=6
            ),
            add_parallel_projected_residual(
                base.view(1, 6, 6),
                residual,
                num_blocks=2,
                start_position=start,
                mask_tokens_size=6,
            ),
            kwargs["previous_logits"],
        )

    shapes = ((2, 3, 4), (2, 3 - start, 4), (2, 3 - start, 6), (2, 3, 6), (1, 6, 6))
    inputs = tuple(
        torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in shapes
    )
    reference = tuple(value.detach().clone().requires_grad_() for value in inputs)
    compiled = torch.compile(function, backend="eager", fullgraph=True)
    actual, expected = compiled(*inputs), function(*reference)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    actual_gradients = torch.autograd.grad(
        sum(value.square().sum() for value in actual), inputs
    )
    expected_gradients = torch.autograd.grad(
        sum(value.square().sum() for value in expected), reference
    )
    for left, right in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
