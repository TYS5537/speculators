"""DSpark-specific tests for the native GLM draft backbone."""

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash.config import GlmMoeDsaDraftConfig
from speculators.models.dflash.model_definitions import GlmDSparkDecoderLayer
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _tiny_glm_config():
    return GlmMoeDsaDraftConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=6,
        qk_rope_head_dim=2,
        v_head_dim=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=16,
        first_k_dense_replace=1,
        index_topk=8,
        index_topk_freq=1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        _attn_implementation="eager",
    )


def test_dspark_glm_model_uses_dspark_backbone_and_heads():
    block_size = 4
    config = DSparkSpeculatorConfig(
        transformer_layer_config=_tiny_glm_config(),
        draft_vocab_size=64,
        block_size=block_size,
        max_anchors=2,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=0,
        markov_rank=8,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        speculators_config=SpeculatorsConfig(
            algorithm="dspark",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=block_size - 1)
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["GlmMoeDsaForCausalLM"],
            ),
        ),
    )

    model = DSparkDraftModel(config)

    assert isinstance(model.layers[0], GlmDSparkDecoderLayer)
    assert model.markov_head is not None
    assert model.confidence_head is not None
    assert model.config.speculators_model_type == "dspark"
    assert model.config.transformer_layer_config.model_type == "glm_moe_dsa"
