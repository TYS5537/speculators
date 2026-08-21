"""Tests for CLI arguments."""

import pytest

from scripts.train import parse_args
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.metrics import ce_loss, kl_div_loss, tv_loss_fused_or_eager
from speculators.models.peagle.core import PEagleDraftModel


def _parse(monkeypatch, extra: list[str]):
    monkeypatch.setattr(
        "sys.argv", ["train.py", "--verifier-name-or-path", "dummy"] + extra
    )
    return parse_args()


# ---------------------------------------------------------------------------
# Ensure CLI args flow correctly through vars(args) into get_trainer_kwargs
# ---------------------------------------------------------------------------


def test_dflash_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]
    assert train_kw["gamma"] == 4.0
    assert val_kw["gamma"] == 4.0


def test_dflash_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert "ce" in val_kw["loss_config"]
    assert train_kw["gamma"] == 4.0
    assert val_kw["gamma"] == 4.0


def test_dflash_explicit_decay_gamma(monkeypatch):
    args = _parse(monkeypatch, ["--dflash-decay-gamma", "7.0"])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["gamma"] == 7.0
    assert val_kw["gamma"] == 7.0


def test_dflash_decay_gamma_falls_back_when_omitted():
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(loss_fn="kl_div")
    assert train_kw["gamma"] == 4.0
    assert val_kw["gamma"] == 4.0


def test_dflash_compound_loss(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", '{"ce": 0.1, "tv": 0.9}'])
    train_kw, val_kw = DFlashDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert "tv" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][1] == 0.1
    assert train_kw["loss_config"]["tv"][1] == 0.9
    assert "ce" in val_kw["loss_config"]
    assert "tv" in val_kw["loss_config"]


def test_eagle3_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = Eagle3DraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]


def test_eagle3_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = Eagle3DraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert "ce" in val_kw["loss_config"]


def test_peagle_default_uses_kl(monkeypatch):
    args = _parse(monkeypatch, [])
    train_kw, val_kw = PEagleDraftModel.get_trainer_kwargs(**vars(args))
    assert "kl_div" in train_kw["loss_config"]
    assert train_kw["loss_config"]["kl_div"][0] is kl_div_loss
    assert "kl_div" in val_kw["loss_config"]


def test_peagle_explicit_ce(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", "ce"])
    train_kw, val_kw = PEagleDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert "ce" in val_kw["loss_config"]


def test_dspark_defaults_match_paper_weighting(monkeypatch):
    args = _parse(monkeypatch, ["--speculator-type", "dspark"])
    assert args.block_size == 7
    assert args.dflash_decay_gamma == 7.0
    assert args.num_layers == 5
    assert args.epochs == 10
    assert args.enable_correction_head is False
    assert args.correction_lm_head_fusion is False
    assert args.correction_rollout_metrics is False
    assert args.dflash_context_residual is False
    assert args.dflash_block_position_embedding is False
    assert args.dflash_gated_layer_fusion is False
    assert args.dflash2_dynamic_conv is False
    assert args.dflash2_candidate_selector is False
    assert args.dflash2_conv_kernel_size == 2
    assert args.dflash2_conv_group_size == 16
    assert args.dflash2_selector_rank == 256
    assert args.dflash2_selector_top_k == 16
    assert args.dflash2_selector_search_mode == "greedy"
    assert args.selector_correction_feedback == "static"
    assert args.dflash2_selector_loss_weight == 1.0
    assert args.enable_confidence_head is True
    assert args.confidence_head_with_markov is True
    assert args.confidence_detach_features is False
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["loss_config"]["ce"] == (ce_loss, 0.1)
    assert train_kw["loss_config"]["tv"] == (tv_loss_fused_or_eager, 0.9)
    assert val_kw["loss_config"]["ce"] == (ce_loss, 0.1)
    assert val_kw["loss_config"]["tv"] == (tv_loss_fused_or_eager, 0.9)
    assert train_kw["gamma"] == 7.0
    assert val_kw["gamma"] == 7.0
    assert train_kw["confidence_head_alpha"] == 1.0
    assert val_kw["confidence_head_alpha"] == 1.0
    assert train_kw["confidence_loss_weighting"] == "match-draft"
    assert val_kw["confidence_loss_weighting"] == "match-draft"


def test_dspark_explicit_recipe_overrides_win(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--speculator-type",
            "dspark",
            "--block-size",
            "8",
            "--dflash-decay-gamma",
            "4",
            "--num-layers",
            "3",
            "--epochs",
            "2",
            "--loss-fn",
            "kl_div",
            "--confidence-loss-weighting",
            "uniform",
        ],
    )
    assert args.block_size == 8
    assert args.dflash_decay_gamma == 4.0
    assert args.num_layers == 3
    assert args.epochs == 2
    assert args.loss_fn == "kl_div"
    assert args.confidence_loss_weighting == "uniform"


def test_dspark_compound_loss(monkeypatch):
    args = _parse(monkeypatch, ["--loss-fn", '{"ce": 0.1, "tv": 0.9}'])
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert "ce" in train_kw["loss_config"]
    assert train_kw["loss_config"]["ce"][0] is ce_loss
    assert train_kw["loss_config"]["ce"][1] == 0.1
    assert "tv" in train_kw["loss_config"]
    assert train_kw["loss_config"]["tv"][0] is tv_loss_fused_or_eager
    assert train_kw["loss_config"]["tv"][1] == 0.9
    assert "ce" in val_kw["loss_config"]
    assert "tv" in val_kw["loss_config"]


def test_dspark_confidence_head_alpha(monkeypatch):
    args = _parse(monkeypatch, ["--confidence-head-alpha", "0.5"])
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["confidence_head_alpha"] == 0.5
    assert val_kw["confidence_head_alpha"] == 0.5


def test_dspark_adaptive_and_confidence_cli(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--adaptive-loss",
            "ssal",
            "--ssal-curriculum",
            "--confidence-length-alpha",
            "0.1",
            "--confidence-loss-weighting",
            "match-draft",
            "--first-error-focal-alpha",
            "0.3",
        ],
    )
    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw["adaptive_loss"] == "ssal"
    assert train_kw["ssal_curriculum"] is True
    assert train_kw["confidence_length_alpha"] == 0.1
    assert train_kw["confidence_loss_weighting"] == "match-draft"
    assert train_kw["first_error_focal_alpha"] == 0.3
    assert "ssal_curriculum" not in val_kw
    assert val_kw["ssal_decay_weight"] == 0.0


def test_dspark_preprojection_correction_head_cli(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--speculator-type",
            "dspark",
            "--enable-correction-head",
            "--correction-hidden-size",
            "96",
            "--correction-rank",
            "48",
            "--correction-num-layers",
            "2",
            "--correction-num-heads",
            "6",
            "--correction-gate-bias",
            "-1.0",
            "--correction-lm-head-fusion",
            "--no-correction-rollout-metrics",
            "--correction-base-diagnostics",
            "--confidence-detach-features",
        ],
    )
    assert args.enable_correction_head is True
    assert args.correction_output_mode == "hidden"
    assert args.correction_hidden_size == 96
    assert args.correction_rank == 48
    assert args.correction_num_layers == 2
    assert args.correction_num_heads == 6
    assert args.correction_gate_bias == -1.0
    assert args.correction_lm_head_fusion is True
    assert args.correction_rollout_metrics is False
    assert args.correction_base_diagnostics is True
    assert args.confidence_detach_features is True

    train_kw, val_kw = DSparkDraftModel.get_trainer_kwargs(**vars(args))
    assert train_kw == val_kw


def test_dspark_logit_residual_correction_head_cli(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--speculator-type",
            "dspark",
            "--enable-correction-head",
            "--correction-output-mode",
            "logits",
        ],
    )
    assert args.enable_correction_head is True
    assert args.correction_output_mode == "logits"


def test_dspark_lm_head_fusion_rejects_logit_mode_without_dual_projection(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(
            monkeypatch,
            [
                "--speculator-type",
                "dspark",
                "--enable-correction-head",
                "--correction-output-mode",
                "logits",
                "--correction-lm-head-fusion",
            ],
        )


def test_dspark_lm_head_fusion_accepts_logit_mode_with_dual_projection(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--speculator-type",
            "dspark",
            "--enable-correction-head",
            "--correction-output-mode",
            "logits",
            "--correction-project-corrected-hidden",
            "--correction-lm-head-fusion",
        ],
    )

    assert args.correction_project_corrected_hidden is True
    assert args.correction_lm_head_fusion is True


def test_dspark_correction_hidden_auxiliary_features_cli(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--speculator-type",
            "dspark",
            "--enable-correction-head",
            "--correction-hidden-aux-loss",
            "--correction-hidden-aux-weight",
            "0.2",
            "--correction-hidden-feedback",
            "--correction-project-corrected-hidden",
            "--correction-output-mode",
            "logits",
        ],
    )
    assert args.correction_hidden_aux_loss is True
    assert args.correction_hidden_aux_weight == 0.2
    assert args.correction_hidden_feedback is True
    assert args.correction_project_corrected_hidden is True


def test_corrected_selector_feedback_cli(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--speculator-type",
            "dspark",
            "--enable-correction-head",
            "--dflash2-candidate-selector",
            "--selector-correction-feedback",
            "corrected",
        ],
    )
    assert args.selector_correction_feedback == "corrected"


def test_corrected_selector_feedback_rejects_global_search(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(
            monkeypatch,
            [
                "--speculator-type",
                "dspark",
                "--enable-correction-head",
                "--dflash2-candidate-selector",
                "--dflash2-selector-global",
                "--selector-correction-feedback",
                "corrected",
            ],
        )


def test_retained_dflash_and_collaboration_features_default_off(monkeypatch):
    args = _parse(monkeypatch, ["--speculator-type", "dspark"])
    assert args.correction_with_markov is False
    assert args.dflash_context_residual is False
    assert args.dflash_block_position_embedding is False
    assert args.dflash_gated_layer_fusion is False


def test_dspark_collaboration_and_dflash_feature_cli(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--speculator-type",
            "dspark",
            "--enable-correction-head",
            "--correction-with-markov",
            "--correction-markov-gate-bias",
            "-1.5",
            "--dflash-context-residual",
            "--dflash-block-position-embedding",
            "--dflash-gated-layer-fusion",
            "--dflash2-dynamic-conv",
            "--dflash2-conv-kernel-size",
            "3",
            "--dflash2-conv-group-size",
            "8",
            "--dflash2-candidate-selector",
            "--dflash2-selector-rank",
            "64",
            "--dflash2-selector-top-k",
            "4",
            "--dflash2-selector-global",
            "--dflash2-selector-loss-weight",
            "0.5",
        ],
    )
    assert args.correction_with_markov is True
    assert args.correction_markov_gate_bias == -1.5
    assert args.dflash_context_residual is True
    assert args.dflash_block_position_embedding is True
    assert args.dflash_gated_layer_fusion is True
    assert args.dflash2_dynamic_conv is True
    assert args.dflash2_conv_kernel_size == 3
    assert args.dflash2_conv_group_size == 8
    assert args.dflash2_candidate_selector is True
    assert args.dflash2_selector_rank == 64
    assert args.dflash2_selector_top_k == 4
    assert args.dflash2_selector_search_mode == "global"
    assert args.dflash2_selector_loss_weight == 0.5


# ---------------------------------------------------------------------------
# Per-speculator-type defaults for draft_arch, norm_before_fc, norm_output
# ---------------------------------------------------------------------------


def test_eagle3_defaults_to_llama_arch(monkeypatch):
    args = _parse(monkeypatch, [])
    assert args.draft_arch == "llama"


def test_eagle3_defaults_norm_before_fc_true(monkeypatch):
    args = _parse(monkeypatch, [])
    assert args.norm_before_fc is True


def test_eagle3_defaults_norm_output_true(monkeypatch):
    args = _parse(monkeypatch, [])
    assert args.norm_output is True


def test_dflash_defaults_to_qwen3_arch(monkeypatch):
    args = _parse(monkeypatch, ["--speculator-type", "dflash"])
    assert args.draft_arch == "qwen3"


def test_dflash_defaults_norm_before_fc_false(monkeypatch):
    args = _parse(monkeypatch, ["--speculator-type", "dflash"])
    assert args.norm_before_fc is False


def test_dflash_defaults_norm_output_false(monkeypatch):
    args = _parse(monkeypatch, ["--speculator-type", "dflash"])
    assert args.norm_output is False


def test_no_norm_before_fc_flag(monkeypatch):
    args = _parse(monkeypatch, ["--no-norm-before-fc"])
    assert args.norm_before_fc is False


def test_no_norm_output_flag(monkeypatch):
    args = _parse(monkeypatch, ["--no-norm-output"])
    assert args.norm_output is False
