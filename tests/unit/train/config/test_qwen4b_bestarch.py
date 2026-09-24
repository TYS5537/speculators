"""Compare real resolved 4B recipes with an independent historical argv fixture."""

import json
import shlex
import warnings

import pytest
import torch
from torch import nn

from speculators.models.dspark.model_definitions import MarkovHead
from speculators.models.mmuse.core import MMuseDraftModel
from speculators.train import legacy_cli
from speculators.train.config import TrainConfig
from speculators.train.optimizers import build_optimizers
from speculators.train.trainer import TrainerConfig
from tests.standalone.test_qwen4b_bestarch_scripts import (
    BASH,
    TRAINERS,
    training_arguments,
)

pytestmark = pytest.mark.skipif(
    not BASH, reason="Bash is required to resolve launchers"
)

# The pasted enhanced DSpark is now named MMuse. Otherwise these are the original
# training arguments, independently transcribed (not extracted from the recipes).
HISTORICAL_ARGS = shlex.split("""
--verifier-name-or-path ../../Qwen3-4B
--data-path ../../datasets/open_perfectblend_qwen3_4b_700k
--vllm-endpoint http://localhost:8001/v1
--save-path ./output/dspark_qwen3_4b_perfectblend_ascend_bestarch/checkpoints
--epochs 10 --lr 6e-5 --logger tensorboard --total-seq-len 3072
--speculator-type mmuse --block-size 7 --max-anchors 512 --num-layers 5
--draft-attn-impl sdpa --target-layer-ids 1 9 17 25 33
--markov-rank 256 --markov-head-type vanilla
--enable-correction-head --correction-output-mode logits
--correction-hidden-size 768 --correction-rank 256 --correction-lm-head-fusion
--correction-num-layers 1 --correction-num-heads 8 --correction-gate-bias 0
--correction-hidden-aux-loss --correction-hidden-aux-weight 0.1
--correction-hidden-feedback --selector-correction-feedback corrected
--correction-project-corrected-hidden --correction-with-markov
--correction-markov-gate-bias -2.0 --no-correction-rollout-metrics
--no-correction-base-diagnostics --dflash-context-residual
--dflash-block-position-embedding --dflash-gated-layer-fusion
--dflash2-dynamic-conv --dflash2-conv-kernel-size 2 --dflash2-conv-group-size 16
--dflash2-candidate-selector --dflash2-selector-rank 256 --dflash2-selector-top-k 16
--dflash2-selector-greedy --dflash2-selector-loss-weight 0.1
--enable-confidence-head --confidence-head-with-markov
--loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0
--confidence-length-alpha 0.0 --confidence-loss-weighting match-draft
--no-confidence-detach-features --first-error-focal-alpha 0.0 --adaptive-loss none
--no-ssal-curriculum --ssal-curriculum-start 0.1 --ssal-curriculum-end 0.6
--on-missing generate --on-generate delete
""")


def resolve(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return TrainConfig.resolve(training_arguments(name))


def test_legacy_bestarch_matches_historical_settings_except_full_attention():
    old = legacy_cli.parse_train_args(HISTORICAL_ARGS)
    expected = TrainConfig.from_flat(
        {**vars(old), "training_recipe": "legacy"}
    ).flatten()
    actual = resolve(TRAINERS[1]).flatten()
    excluded = {
        "save_path",
        "log_dir",
        "run_name",
        "no_resume_from_checkpoint",
        "vllm_endpoint",
        "muon_parameter_policy",
        "muon_lr",
        "full_attention_indices",
    } | {key for key in expected if key.startswith("mooncake_")}
    assert {key: value for key, value in actual.items() if key not in excluded} == {
        key: value for key, value in expected.items() if key not in excluded
    }
    assert actual["muon_lr"] == pytest.approx(expected["muon_lr"], rel=1e-15)
    assert actual["draft_vocab_size"] is None
    assert expected["full_attention_indices"] == []
    assert actual["full_attention_indices"] == list(range(actual["num_layers"]))


def test_mmuse_optimizer_comparison_keeps_model_init_and_losses_identical():
    old, new = (resolve(name).flatten() for name in TRAINERS[1:])
    assert old.keys() == new.keys()
    assert {key for key in old if old[key] != new[key]} == {
        "muon_parameter_policy",
        "save_path",
        "log_dir",
        "run_name",
    }
    assert old["training_recipe"] == new["training_recipe"] == "legacy"
    assert (
        old["full_attention_indices"]
        == new["full_attention_indices"]
        == [0, 1, 2, 3, 4]
    )
    assert MMuseDraftModel.get_trainer_kwargs(
        **old
    ) == MMuseDraftModel.get_trainer_kwargs(**new)


def test_baseline_uses_paper_adamw_without_changing_local_data_and_losses():
    original = TrainConfig.resolve(
        shlex.split("""
--training-recipe legacy --loss-implementation legacy --optimizer muon
--muon-parameter-policy legacy --no-resume-from-checkpoint
--verifier-name-or-path ../../Qwen3-4B
--data-path ../../datasets/open_perfectblend_qwen3_4b_700k
--vllm-endpoint http://127.0.0.1:8001/v1
--save-path ./original/checkpoints --log-dir ./original/logs --run-name original
--lr 6e-5 --muon-lr 6e-4 --weight-decay 0.01 --muon-weight-decay 0.1
--seed 42 --epochs 10 --logger tensorboard --total-seq-len 3072
--speculator-type dspark --block-size 7 --max-anchors 512 --num-layers 5
--draft-attn-impl sdpa --target-layer-ids 1 9 17 25 33
--markov-rank 256 --markov-head-type vanilla
--enable-confidence-head --confidence-head-with-markov
--loss-fn '{"ce": 0.1, "tv": 0.9}'
--confidence-head-alpha 1 --confidence-length-alpha 0
--confidence-loss-weighting match-draft --no-confidence-detach-features
--first-error-focal-alpha 0 --adaptive-loss none
--no-ssal-curriculum --ssal-curriculum-start 0.1 --ssal-curriculum-end 0.6
--on-missing generate --on-generate delete
""")
    ).flatten()
    baseline = resolve(TRAINERS[0]).flatten()
    assert original.keys() == baseline.keys()
    metadata = {"save_path", "log_dir", "run_name"}
    # AdamW never consumes Muon options; no longer pin those inert CLI fields.
    inactive = {key for key in baseline if key.startswith("muon_")}
    assert {
        key
        for key in baseline
        if baseline[key] != original[key] and key not in metadata | inactive
    } == {
        "optimizer",
        "lr",
        "weight_decay",
        "scheduler_type",
        "scheduler_warmup_ratio",
        "full_attention_indices",
    }
    assert baseline["lr"] == 6e-4
    assert baseline["weight_decay"] == 0
    assert baseline["scheduler_type"] == "cosine"
    assert baseline["scheduler_warmup_ratio"] == 0.04
    assert baseline["full_attention_indices"] == [0, 1, 2, 3, 4]
    assert baseline["training_recipe"] == "legacy"
    assert baseline["optimizer"] == "adamw"
    assert baseline["muon_parameter_policy"] is None
    assert baseline["total_seq_len"] == 3072
    assert json.loads(baseline["loss_fn"]) == {"ce": 0.1, "tv": 0.9}
    assert baseline["loss_implementation"] == "legacy"
    assert baseline["dflash_decay_gamma"] == 7
    assert baseline["noise_std"] == 0.05
    assert baseline["train_data_ratio"] == 0.9
    assert baseline["draft_vocab_size"] is None
    assert baseline["target_layer_ids"] == [1, 9, 17, 25, 33]


def test_dspark_resolved_recipe_builds_one_adamw_including_markov():
    cfg = resolve(TRAINERS[0])
    model = nn.ModuleDict(
        {
            "backbone": nn.Linear(4, 4),
            "markov_head": MarkovHead(
                verifier_vocab_size=16,
                draft_vocab_size=16,
                markov_rank=4,
                hidden_size=4,
            ),
        }
    )
    trainer_cfg = TrainerConfig(
        lr=cfg.optimizer.lr,
        num_epochs=cfg.trainer.epochs,
        save_path="unused",
        optimizer=cfg.optimizer.optimizer,
        weight_decay=cfg.optimizer.weight_decay,
        training_recipe=cfg.training_recipe,
        muon_parameter_policy=cfg.optimizer.muon_parameter_policy,
    )
    optimizers = build_optimizers(model, trainer_cfg)
    assert len(optimizers) == 1
    optimizer = optimizers[0]
    assert type(optimizer) is torch.optim.AdamW
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.parameters()
    }
    assert all(group["lr"] == 6e-4 for group in optimizer.param_groups)
    assert all(group["weight_decay"] == 0 for group in optimizer.param_groups)
    before = [p.detach().clone() for p in model.parameters()]
    sum(p.sum() for p in model.parameters()).backward()
    optimizer.step()
    for previous, parameter in zip(before, model.parameters(), strict=True):
        assert torch.isfinite(parameter).all()
        assert not torch.equal(previous, parameter)


def test_policy_and_recipe_survive_saved_config_roundtrip(tmp_path):
    cfg = resolve(TRAINERS[2])
    path = tmp_path / "run.yaml"
    path.write_text(cfg.dump_yaml(), encoding="utf-8")
    restored = TrainConfig.resolve(["--config", str(path)])
    assert restored.flatten() == cfg.flatten()
    assert restored.optimizer.muon_parameter_policy == "upstream"
    assert restored.training_recipe == "legacy"
