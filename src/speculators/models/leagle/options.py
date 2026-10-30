"""Shared defaults for LEAGLE's opt-in training execution paths."""

from types import MappingProxyType

TRAINING_DEFAULTS = MappingProxyType(
    {
        "leagle_reuse_training_kv": False,
        "leagle_chunked_loss": False,
        "leagle_loss_chunk_size": 128,
        "leagle_sparse_attention": False,
        "leagle_attn_chunk_size": 128,
        "leagle_candidate_grad_addmm": False,
        "leagle_candidate_backward_recompute": False,
        "leagle_teacher_cache": False,
        "leagle_teacher_cache_mb": 1024,
        "leagle_masked_head": False,
        "leagle_teacher_cache_supervised": False,
        "leagle_shared_loss_normalization": False,
        "leagle_stream_main_loss": False,
        "leagle_align_candidate_chunks": False,
        "leagle_selective_checkpoint": False,
        "leagle_cache_attention_masks": False,
        "leagle_attention_mask_cache_mb": 64,
    }
)

RUNTIME_DEFAULTS = MappingProxyType(
    {
        "leagle_fsdp_keep_forward_params": False,
        "leagle_share_output_head": False,
    }
)
