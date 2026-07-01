"""Unit tests for DFlash model-definition dispatch and GLM draft layers."""

import torch

from speculators.models.dflash.config import GlmMoeDsaDraftConfig
from speculators.models.dflash.model_definitions import (
    GlmDFlashDecoderLayer,
    GlmDFlashMLAAttention,
    GlmDFlashMoE,
    GlmDFlashRotaryEmbedding,
    GlmDSparkDecoderLayer,
    dflash_model_classes,
    resolve_dflash_model_components,
)


def _tiny_glm_config(**overrides):
    kwargs = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 128,
        "q_lora_rank": 16,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 6,
        "qk_rope_head_dim": 2,
        "v_head_dim": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_intermediate_size": 16,
        "first_k_dense_replace": 1,
        "index_topk": 8,
        "index_topk_freq": 1,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "_attn_implementation": "eager",
    }
    kwargs.update(overrides)
    return GlmMoeDsaDraftConfig(**kwargs)


def test_dflash_model_registry_has_glm_components():
    components = dflash_model_classes["glm_moe_dsa"]
    assert components.decoder_layer_class is GlmDFlashDecoderLayer
    assert components.no_split_module == "GlmDFlashDecoderLayer"


def test_dspark_resolves_dspark_named_glm_components():
    components = resolve_dflash_model_components("glm_moe_dsa", algorithm="dspark")
    assert components.decoder_layer_class is GlmDSparkDecoderLayer
    assert components.no_split_module == "GlmDSparkDecoderLayer"


def test_glm_mla_attention_injects_context_and_preserves_shape():
    config = _tiny_glm_config()
    attention = GlmDFlashMLAAttention(config, layer_idx=0)
    rotary = GlmDFlashRotaryEmbedding(config)
    hidden_states = torch.randn(1, 3, config.hidden_size)
    target_hidden = torch.randn(1, 5, config.hidden_size)
    position_ids = torch.arange(8).unsqueeze(0)
    position_embeddings = rotary(hidden_states, position_ids)

    output, attn_weights = attention(
        hidden_states=hidden_states,
        target_hidden=target_hidden,
        attention_mask=None,
        position_embeddings=position_embeddings,
    )

    assert output.shape == hidden_states.shape
    assert attn_weights is None
    assert attention.indexer.enabled
    assert attention.indexer.last_topk_indices is not None
    assert attention.indexer.last_topk_indices.shape == (1, 3, 8)


def test_glm_moe_preserves_shape_and_is_finite():
    config = _tiny_glm_config()
    moe = GlmDFlashMoE(config)
    hidden_states = torch.randn(2, 3, config.hidden_size)

    output = moe(hidden_states)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()


def test_glm_decoder_layer_forward_shape():
    config = _tiny_glm_config()
    layer = GlmDFlashDecoderLayer(config, layer_idx=1)
    rotary = GlmDFlashRotaryEmbedding(config)
    hidden_states = torch.randn(1, 3, config.hidden_size)
    target_hidden = torch.randn(1, 5, config.hidden_size)
    position_ids = torch.arange(8).unsqueeze(0)
    position_embeddings = rotary(hidden_states, position_ids)

    output = layer(
        hidden_states=hidden_states,
        target_hidden=target_hidden,
        attention_mask=None,
        position_embeddings=position_embeddings,
    )

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
