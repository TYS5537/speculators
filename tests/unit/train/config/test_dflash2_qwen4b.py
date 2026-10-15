"""Resolve the actual Qwen3-4B DFlash2 launcher against the typed config."""

import pytest

from speculators.train.config import TrainConfig
from tests.standalone.test_qwen4b_bestarch_scripts import (
    BASH,
    DFLASH2_TRAINER,
    training_arguments,
)

pytestmark = pytest.mark.skipif(
    not BASH, reason="Bash is required to resolve launchers"
)


def test_dflash2_recipe_matches_shared_data_and_seven_token_budget():
    draft = TrainConfig.resolve(training_arguments(DFLASH2_TRAINER)).flatten()
    baseline = TrainConfig.resolve(training_arguments("train_dspark.sh")).flatten()
    for field in (
        "verifier_name_or_path",
        "data_path",
        "vllm_endpoint",
        "training_recipe",
        "loss_implementation",
        "optimizer",
        "lr",
        "weight_decay",
        "scheduler_type",
        "scheduler_warmup_ratio",
        "total_seq_len",
        "epochs",
        "seed",
        "num_layers",
        "max_anchors",
        "target_layer_ids",
        "full_attention_indices",
        "draft_attn_impl",
        "loss_fn",
        "noise_std",
        "dflash_decay_gamma",
    ):
        assert draft[field] == baseline[field], field
    assert draft["speculator_type"] == "dflash2"
    assert draft["block_size"] == baseline["block_size"] + 1 == 8
    assert not draft["sample_from_anchor"]
    assert draft["selector_loss_alpha"] == 0.1
    assert (draft["conv_kernel_size"], draft["conv_group_size"]) == (2, 16)
    assert (draft["selector_rank"], draft["selector_top_k"]) == (256, 16)
    assert draft["draft_vocab_size"] is None
    assert draft["save_path"] != baseline["save_path"]
    assert draft["no_resume_from_checkpoint"]
    assert draft["save_best"]
