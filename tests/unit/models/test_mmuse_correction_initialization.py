"""Seeded weights, module ordering and validation contracts for Correction setup."""

import io

import pytest
import torch
from torch import nn

from speculators.models.mmuse.correction import CausalCorrectionHead, _TinyCausalLayer


def _options(mode="hidden", *, auxiliary=False, feedback=False, gate_bias=0.0):
    return {
        "input_hidden_size": 8,
        "token_embedding_size": 6,
        "block_size": 4,
        "correction_hidden_size": 12,
        "correction_rank": 4,
        "num_layers": 2,
        "num_heads": 3,
        "output_mode": mode,
        "draft_vocab_size": 11,
        "enable_hidden_auxiliary": auxiliary,
        "enable_hidden_feedback": feedback,
        "gate_bias": gate_bias,
    }


def _reference_parameters(options):
    """Replay the established allocation order independently of head setup helpers.

    The causal layers are unchanged by this refactor. Everything surrounding them
    is constructed explicitly so changes to projection order or early special
    initialization alter this reference's weights/RNG comparison.
    """
    width = options["correction_hidden_size"]
    rank = options["correction_rank"]
    hidden = options["input_hidden_size"]
    vocab = options["draft_vocab_size"]
    logits = options["output_mode"] == "logits"
    modules = nn.ModuleDict(
        {
            "hidden_proj": nn.Linear(hidden, width, bias=False),
            "token_proj": nn.Linear(options["token_embedding_size"], width, bias=False),
            "position_embedding": nn.Embedding(options["block_size"], width),
            "layers": nn.ModuleList(
                _TinyCausalLayer(width, options["num_heads"])
                for _ in range(options["num_layers"])
            ),
            "output_norm": nn.RMSNorm(width),
            "correction_down": nn.Linear(width, rank, bias=False),
            "correction_up": nn.Linear(rank, vocab if logits else hidden, bias=False),
        }
    )
    if logits:
        modules["previous_logits_down"] = nn.Linear(vocab, rank, bias=False)
        modules["previous_logits_proj"] = nn.Linear(rank, width, bias=False)
    if options["enable_hidden_feedback"]:
        modules["hidden_feedback_proj"] = nn.Linear(hidden, width, bias=False)
    if logits and (
        options["enable_hidden_auxiliary"] or options["enable_hidden_feedback"]
    ):
        modules["auxiliary_hidden_up"] = nn.Linear(rank, hidden, bias=False)
        modules["auxiliary_hidden_gate"] = nn.Linear(width, 1)
    modules["residual_gate"] = nn.Linear(width, 1)
    nn.init.normal_(modules["position_embedding"].weight, mean=0.0, std=0.02)
    nn.init.zeros_(modules["correction_up"].weight)
    nn.init.zeros_(modules["residual_gate"].weight)
    nn.init.constant_(modules["residual_gate"].bias, options["gate_bias"])
    if "auxiliary_hidden_up" in modules:
        nn.init.zeros_(modules["auxiliary_hidden_up"].weight)
        nn.init.zeros_(modules["auxiliary_hidden_gate"].weight)
        nn.init.constant_(modules["auxiliary_hidden_gate"].bias, options["gate_bias"])
    return modules


@pytest.fixture(params=[torch.float32, torch.float64, torch.bfloat16])
def default_dtype(request):
    original = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize("mode", ["hidden", "logits"])
@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("feedback", [False, True])
@pytest.mark.parametrize("gate_bias", [-2.0, 0.75])
@pytest.mark.parametrize("seed", [0, 29])
def test_seeded_initialization_keeps_weights_rng_and_registration_order(
    mode, auxiliary, feedback, gate_bias, seed, default_dtype
):
    options = _options(
        mode, auxiliary=auxiliary, feedback=feedback, gate_bias=gate_bias
    )
    options["num_layers"] = 1 + seed % 3
    torch.manual_seed(seed)
    expected = _reference_parameters(options)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    head = CausalCorrectionHead(**options)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert tuple(head._modules) == tuple(expected)
    assert tuple(head.state_dict()) == tuple(expected.state_dict())
    assert tuple(dict(head.named_parameters())) == tuple(
        dict(expected.named_parameters())
    )
    torch.testing.assert_close(head.state_dict(), expected.state_dict(), rtol=0, atol=0)
    assert all(value.dtype == default_dtype for value in head.parameters())
    assert all(value.requires_grad for value in head.parameters())
    assert tuple(head.named_buffers()) == ()


@pytest.mark.parametrize("mode", ["hidden", "logits"])
@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("feedback", [False, True])
def test_optional_attributes_and_initial_residuals_keep_their_original_roles(
    mode, auxiliary, feedback
):
    head = CausalCorrectionHead(
        **_options(mode, auxiliary=auxiliary, feedback=feedback, gate_bias=-0.5)
    )
    enabled = {
        "previous_logits_down": mode == "logits",
        "previous_logits_proj": mode == "logits",
        "hidden_feedback_proj": feedback,
        "auxiliary_hidden_up": mode == "logits" and (auxiliary or feedback),
        "auxiliary_hidden_gate": mode == "logits" and (auxiliary or feedback),
    }
    for name, present in enabled.items():
        if present:
            assert isinstance(getattr(head, name), nn.Linear)
            assert name in head._modules
        else:
            assert getattr(head, name) is None
            assert name not in head._modules
            assert name in vars(head)
    states = torch.randn(2, 3, 12, requires_grad=True)
    assert torch.count_nonzero(head._residual_from_causal_states(states)) == 0
    torch.testing.assert_close(head.residual_gate(states), torch.full((2, 3, 1), -0.5))
    if mode == "hidden" or enabled["auxiliary_hidden_up"]:
        assert torch.count_nonzero(head.auxiliary_hidden_residual(states)) == 0
    else:
        with pytest.raises(
            RuntimeError, match="Auxiliary hidden residual was not enabled"
        ):
            head.auxiliary_hidden_residual(states)
    assert head._lm_fusion_signature is None
    assert head._lm_fusion_weight is None


@pytest.mark.parametrize("first_failure", range(6))
@pytest.mark.parametrize("invalid_dimension", [0, -1])
def test_constructor_validation_precedes_allocations_in_the_existing_order(
    first_failure, invalid_dimension
):
    options = _options("logits")
    faults = [
        (
            "correction_hidden_size",
            invalid_dimension,
            "correction_hidden_size must be positive",
        ),
        ("correction_rank", invalid_dimension, "correction_rank must be positive"),
        ("block_size", invalid_dimension, "block_size must be positive"),
        ("num_layers", invalid_dimension, "num_layers must be positive"),
        ("output_mode", "unknown", "Unsupported correction output mode"),
        ("draft_vocab_size", None, "draft_vocab_size must be positive"),
    ]
    for name, value, _ in faults[first_failure:]:
        options[name] = value
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match=faults[first_failure][2]):
        CausalCorrectionHead(**options)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize(
    ("heads", "error", "message"),
    [(0, ZeroDivisionError, "modulo by zero"), (5, ValueError, "must be divisible")],
)
def test_attention_dimension_failure_still_follows_base_input_projections(
    heads, error, message
):
    torch.manual_seed(73)
    nn.Linear(8, 12, bias=False)
    nn.Linear(6, 12, bias=False)
    nn.Embedding(4, 12)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(73)
    options = _options()
    options["num_heads"] = heads
    with pytest.raises(error, match=message):
        CausalCorrectionHead(**options)
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("mode", ["hidden", "logits"])
@pytest.mark.parametrize("feedback", [False, True])
def test_checkpoint_loading_stays_strict_and_fusion_cache_stays_nonpersistent(
    mode, feedback
):
    options = _options(mode, auxiliary=True, feedback=feedback)
    expected = _reference_parameters(options)
    with torch.no_grad():
        for value in expected.parameters():
            value.uniform_(-0.2, 0.2)
    archive = io.BytesIO()
    torch.save(expected.state_dict(), archive)
    archive.seek(0)
    head = CausalCorrectionHead(**options)
    loaded = head.load_state_dict(torch.load(archive, weights_only=True), strict=True)
    assert loaded.missing_keys == loaded.unexpected_keys == []
    assert tuple(head.state_dict()) == tuple(expected.state_dict())
    torch.testing.assert_close(head.state_dict(), expected.state_dict(), rtol=0, atol=0)
    with torch.no_grad():
        head.fused_lm_head_residual(torch.randn(2, 3, 12), torch.randn(11, 8))
    assert head._lm_fusion_signature is not None
    assert head._lm_fusion_weight is not None
    assert tuple(head.named_buffers()) == ()
    assert tuple(head.state_dict()) == tuple(expected.state_dict())
    torch.testing.assert_close(head.state_dict(), expected.state_dict(), rtol=0, atol=0)
