"""MMuse options agree across CLI, model construction and checkpoint overlays."""

import copy
from unittest.mock import Mock

import pytest
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from scripts.train import (
    PRETRAINED_MODEL_CONFIG_FLAGS,
    _reconcile_pretrained_config_args,
    build_draft_model,
    parse_args,
)
from speculators import SpeculatorModelConfig
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dspark import DSparkSpeculatorConfig
from speculators.models.mmuse import MMuseDraftModel, MMuseSpeculatorConfig
from speculators.models.mmuse.config import (
    MMUSE_OPTION_FIELDS,
    mmuse_option_defaults,
    validate_mmuse_options,
)
from speculators.models.mtp import MTPDraftModel, MTPSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

_SMALL_OPTIONS = {
    "markov_rank": 4,
    "correction_hidden_size": 16,
    "correction_rank": 8,
    "correction_num_heads": 4,
    "dflash2_selector_rank": 4,
    "dflash2_selector_top_k": 4,
}


def _config(config_class=MMuseSpeculatorConfig, **options):
    transformer = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        layer_types=["full_attention"],
    )
    transformer._attn_implementation = "eager"
    return config_class(
        transformer_layer_config=transformer,
        draft_vocab_size=32,
        block_size=3,
        aux_hidden_state_layer_ids=[0],
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm=config_class.model_fields["speculators_model_type"].default,
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=[]),
        ),
        **options,
    )


def _flags(options):
    flags = []
    for name, value in options.items():
        if name == "dflash2_selector_search_mode":
            flags.append(f"--dflash2-selector-{value}")
        elif isinstance(value, bool):
            flags.append(f"--{'' if value else 'no-'}{name.replace('_', '-')}")
        else:
            flags.extend([f"--{name.replace('_', '-')}", str(value)])
    return flags


def _parse(monkeypatch, options=None, *, checkpoint=None, algorithm="mmuse"):
    argv = ["train.py", "--verifier-name-or-path", "local-unused-verifier"]
    if algorithm is not None:
        argv.extend(["--speculator-type", algorithm])
    if checkpoint is not None:
        argv.extend(["--from-pretrained", str(checkpoint)])
    monkeypatch.setattr("sys.argv", argv + _flags(options or {}))
    return parse_args()


def _checkpoint_overlay(monkeypatch, tmp_path, config, options=None, **kwargs):
    config.save_pretrained(tmp_path)
    args = _parse(monkeypatch, options, checkpoint=tmp_path, **kwargs)
    restored = SpeculatorModelConfig.from_pretrained(tmp_path, local_files_only=True)
    return args, restored


def _factory(monkeypatch, **options):
    config = _config()
    monkeypatch.setattr(
        VerifierConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: config.speculators_config.verifier,
    )
    monkeypatch.setattr(MMuseDraftModel, "load_verifier_weights", lambda _self: None)
    return MMuseDraftModel.from_training_args(
        config.transformer_layer_config,
        verifier_name_or_path="local-unused-verifier",
        draft_vocab_size=32,
        mask_token_id=0,
        target_layer_ids=[0],
        draft_attn_impl="eager",
        **options,
    )


def test_defaults_match_cli_config_and_training_factory(monkeypatch):
    defaults = mmuse_option_defaults()
    assert set(defaults) == MMUSE_OPTION_FIELDS
    assert MMUSE_OPTION_FIELDS.issubset(PRETRAINED_MODEL_CONFIG_FLAGS)
    # Golden values protect the existing recipe, not just cross-path agreement.
    for name, expected in {
        "enable_correction_head": False,
        "correction_hidden_size": 512,
        "correction_rank": 256,
        "correction_num_heads": 8,
        "correction_output_mode": "hidden",
        "correction_markov_gate_bias": -2.0,
        "selector_correction_feedback": "static",
        "dflash2_dynamic_conv": False,
        "dflash2_conv_kernel_size": 2,
        "dflash2_conv_group_size": 16,
        "dflash2_selector_rank": 256,
        "dflash2_selector_top_k": 16,
        "dflash2_selector_search_mode": "greedy",
        "dflash2_selector_loss_weight": 1.0,
    }.items():
        assert defaults[name] == expected
    args = _parse(monkeypatch)
    config = _config()
    model = _factory(monkeypatch)
    for name, expected in defaults.items():
        assert getattr(args, name) == expected, name
        assert getattr(config, name) == expected, name
        assert getattr(model.config, name) == expected, name
    defaults["enable_correction_head"] = True
    assert mmuse_option_defaults()["enable_correction_head"] is False


def test_factory_retains_only_documented_none_as_default_options(monkeypatch):
    model = _factory(
        monkeypatch,
        sample_from_anchor=None,
        enable_confidence_head=None,
        confidence_head_with_markov=None,
    )
    assert model.config.sample_from_anchor is True
    assert model.config.enable_confidence_head is True
    assert model.config.confidence_head_with_markov is True
    with pytest.raises(ValueError):
        _factory(monkeypatch, correction_rank=None)


@pytest.mark.parametrize(
    "options",
    [
        {"correction_output_mode": "logits"},
        {"correction_lm_head_fusion": True},
        {
            "enable_correction_head": True,
            "correction_output_mode": "logits",
            "correction_lm_head_fusion": True,
        },
        {"correction_hidden_feedback": True},
        {"enable_correction_head": True, "correction_project_corrected_hidden": True},
        {"enable_correction_head": True, "correction_hidden_size": 15},
        {"selector_correction_feedback": "corrected"},
        {
            "enable_correction_head": True,
            "dflash2_candidate_selector": True,
            "selector_correction_feedback": "corrected",
            "dflash2_selector_search_mode": "global",
        },
        {"dflash2_candidate_selector": True, "dflash2_selector_search_mode": "global"},
        {"correction_with_markov": True},
        {
            "enable_correction_head": True,
            "correction_with_markov": True,
            "markov_rank": 0,
        },
        {
            "enable_correction_head": True,
            "correction_with_markov": True,
            "markov_head_type": "rnn",
        },
        {"markov_rank": 0},
    ],
    ids=[
        "logits-without-correction",
        "fusion-without-correction",
        "logits-fusion-without-dual",
        "feedback-without-correction",
        "dual-in-hidden-mode",
        "correction-head-divisibility",
        "corrected-without-heads",
        "corrected-global-search",
        "global-standalone-markov",
        "collaboration-without-correction",
        "collaboration-without-markov",
        "collaboration-rnn",
        "confidence-without-sequential-state",
    ],
)
def test_invalid_combinations_agree_between_cli_and_model(monkeypatch, options):
    values = _SMALL_OPTIONS | options
    # Cross-field checks intentionally happen at model construction, not config IO.
    config = _config(**values)
    with pytest.raises(ValueError):
        validate_mmuse_options(vars(config))
    with pytest.raises(ValueError):
        MMuseDraftModel(config)
    with pytest.raises(SystemExit):
        _parse(monkeypatch, values)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"enable_correction_head": True, "correction_lm_head_fusion": True},
        {
            "enable_correction_head": True,
            "correction_output_mode": "logits",
            "correction_project_corrected_hidden": True,
            "correction_lm_head_fusion": True,
        },
        {
            "enable_correction_head": True,
            "dflash2_candidate_selector": True,
            "selector_correction_feedback": "corrected",
        },
        {
            "enable_correction_head": True,
            "dflash2_candidate_selector": True,
            "dflash2_selector_search_mode": "global",
        },
        {"enable_correction_head": True, "correction_with_markov": True},
        {"markov_rank": 0, "enable_confidence_head": False},
        {"markov_rank": -1, "enable_confidence_head": False},
        {"markov_rank": 0, "confidence_head_with_markov": False},
        {"dflash2_selector_search_mode": "global"},
        {"correction_rollout_metrics": True, "correction_base_diagnostics": True},
    ],
)
def test_valid_and_inactive_options_remain_legal(monkeypatch, options):
    values = _SMALL_OPTIONS | options
    config = _config(**values)
    validate_mmuse_options(vars(config))
    model = MMuseDraftModel(config)
    args = _parse(monkeypatch, values)
    for name, expected in values.items():
        assert getattr(args, name) == getattr(model.config, name) == expected


@pytest.mark.parametrize(
    "options",
    [
        {"correction_rank": 0},
        {"correction_num_layers": 0},
        {"correction_hidden_aux_weight": -0.1},
        {"dflash2_conv_kernel_size": 0},
        {"dflash2_selector_top_k": 0},
        {"dflash2_selector_loss_weight": -0.1},
    ],
)
def test_partial_checkpoint_options_still_validate_scalar_ranges(monkeypatch, options):
    with pytest.raises(ValueError):
        validate_mmuse_options(options, partial=True)
    with pytest.raises(SystemExit):
        _parse(monkeypatch, options, checkpoint="not-loaded-by-parser")


def test_checkpoint_unspecified_options_inherit_saved_values(monkeypatch, tmp_path):
    args, config = _checkpoint_overlay(
        monkeypatch,
        tmp_path,
        _config(**(_SMALL_OPTIONS | {"correction_rollout_metrics": True})),
    )
    assert args.correction_rollout_metrics is False
    _reconcile_pretrained_config_args(args, config)
    assert args.correction_rollout_metrics is config.correction_rollout_metrics is True
    assert args.correction_hidden_size == config.correction_hidden_size == 16


def test_checkpoint_explicit_default_false_is_not_treated_as_omission(
    monkeypatch, tmp_path
):
    args, config = _checkpoint_overlay(
        monkeypatch,
        tmp_path,
        _config(correction_rollout_metrics=True),
        {"correction_rollout_metrics": False},
        algorithm=None,
    )
    assert "correction_rollout_metrics" in args._provided_model_config_dests
    _reconcile_pretrained_config_args(args, config)
    assert args.correction_rollout_metrics is config.correction_rollout_metrics is False


def test_checkpoint_matching_structure_is_allowed(monkeypatch, tmp_path):
    args, config = _checkpoint_overlay(
        monkeypatch,
        tmp_path,
        _config(dflash2_dynamic_conv=True),
        {"dflash2_dynamic_conv": True},
        algorithm=None,
    )
    _reconcile_pretrained_config_args(args, config)
    assert args.dflash2_dynamic_conv is config.dflash2_dynamic_conv is True


@pytest.mark.parametrize(
    ("saved", "overlay"),
    [
        ({}, {"dflash2_dynamic_conv": True, "correction_rollout_metrics": True}),
        (
            {
                "enable_correction_head": True,
                "dflash2_candidate_selector": True,
                "selector_correction_feedback": "corrected",
            },
            {"dflash2_selector_search_mode": "global"},
        ),
        (
            {"enable_correction_head": True, "correction_output_mode": "logits"},
            {"correction_lm_head_fusion": True},
        ),
    ],
    ids=["changed-structure", "invalid-selector-overlay", "invalid-fusion-overlay"],
)
def test_checkpoint_rejected_overlay_does_not_mutate_args_or_config(
    monkeypatch, tmp_path, saved, overlay
):
    args, config = _checkpoint_overlay(
        monkeypatch, tmp_path, _config(**(_SMALL_OPTIONS | saved)), overlay
    )
    args_before, config_before = copy.deepcopy(vars(args)), config.to_dict()
    with pytest.raises(ValueError):
        _reconcile_pretrained_config_args(args, config)
    assert vars(args) == args_before
    assert config.to_dict() == config_before


def test_checkpoint_valid_combination_is_checked_after_saved_values_are_merged(
    monkeypatch, tmp_path
):
    saved = _SMALL_OPTIONS | {
        "enable_correction_head": True,
        "correction_output_mode": "logits",
        "correction_project_corrected_hidden": True,
    }
    args, config = _checkpoint_overlay(
        monkeypatch,
        tmp_path,
        _config(**saved),
        {"correction_lm_head_fusion": True, "correction_output_mode": "logits"},
        algorithm=None,
    )
    assert args.enable_correction_head is False
    _reconcile_pretrained_config_args(args, config)
    assert args.enable_correction_head
    assert args.correction_project_corrected_hidden
    assert args.correction_output_mode == "logits"
    assert config.correction_lm_head_fusion is True
    MMuseDraftModel(config)


def test_checkpoint_partial_geometry_uses_saved_width_not_cli_default(
    monkeypatch, tmp_path
):
    saved = _SMALL_OPTIONS | {
        "enable_correction_head": True,
        "correction_hidden_size": 24,
        "correction_num_heads": 6,
    }
    args, config = _checkpoint_overlay(
        monkeypatch,
        tmp_path,
        _config(**saved),
        {"enable_correction_head": True, "correction_num_heads": 6},
    )
    assert args.correction_hidden_size == 512
    _reconcile_pretrained_config_args(args, config)
    assert args.correction_hidden_size == config.correction_hidden_size == 24
    assert args.correction_num_heads == config.correction_num_heads == 6
    MMuseDraftModel(config)


@pytest.mark.parametrize("enabled", [True, False])
def test_checkpoint_actual_baseline_type_rejects_mmuse_override(
    monkeypatch, tmp_path, enabled
):
    args, config = _checkpoint_overlay(
        monkeypatch,
        tmp_path,
        _config(DSparkSpeculatorConfig),
        {"correction_rollout_metrics": enabled},
    )
    before = config.to_dict()
    with pytest.raises(ValueError):
        _reconcile_pretrained_config_args(args, config)
    assert config.to_dict() == before


def test_mtp_weight_checkpoint_cannot_bypass_mmuse_option_guard(monkeypatch, tmp_path):
    template = _config()
    MTPSpeculatorConfig(
        transformer_layer_config=template.transformer_layer_config,
        speculators_config=template.speculators_config.model_copy(
            update={"algorithm": "mtp"}
        ),
    ).save_pretrained(tmp_path)
    # Select the weights path; the loader must not read this sentinel file.
    (tmp_path / "model.safetensors").touch()
    args = _parse(
        monkeypatch,
        {"correction_rollout_metrics": False},
        checkpoint=tmp_path,
        algorithm="mtp",
    )
    loader = Mock(side_effect=AssertionError("weights must not load before validation"))
    monkeypatch.setattr(MTPDraftModel, "from_pretrained", loader)
    with pytest.raises(ValueError):
        build_draft_model(args, MTPDraftModel, None, None, 32)
    loader.assert_not_called()
