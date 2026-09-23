import logging
from typing import ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.attention import create_float_mask
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.model_definitions import (
    Qwen3DFlashDecoderLayer,
)
from speculators.models.dflash.target_distribution import project_target_distribution
from speculators.models.dflash.utils import (
    build_anchored_loss_mask,
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.metrics import (
    LossConfig,
    resolve_training_loss,
)
from speculators.models.utils import (
    conditional_torch_compile,
    flatten_rope_parameters,
    resolve_target_layer_ids,
    resolve_verifier_norm_class,
)

logger = logging.getLogger(__name__)

# Compile so the mask builds block-sparse instead of materializing DFlash's huge
# dense [Q, KV] grid every step. (No benefit for EAGLE3's small autoregressive mask.)
_compiled_create_block_mask = torch.compile(create_block_mask)


def _reject_enhanced_baseline_features(
    algorithm: str, options: dict | DFlashSpeculatorConfig
) -> None:
    """Reject MMuse-only options before a baseline factory can discard them."""
    if algorithm not in {"dflash", "dspark"}:
        return
    get_option = (
        options.get
        if isinstance(options, dict)
        else lambda name, default: getattr(options, name, default)
    )
    enhanced_features = (
        "dflash_context_residual",
        "dflash_block_position_embedding",
        "dflash_gated_layer_fusion",
        "dflash2_dynamic_conv",
        "dflash2_candidate_selector",
        "enable_correction_head",
        "correction_hidden_aux_loss",
        "correction_lm_head_fusion",
        "correction_hidden_feedback",
        "correction_project_corrected_hidden",
        "correction_with_markov",
        "correction_rollout_metrics",
        "correction_base_diagnostics",
    )
    enabled_features = [name for name in enhanced_features if get_option(name, False)]
    for name, default in (
        ("selector_correction_feedback", "static"),
        ("correction_output_mode", "hidden"),
    ):
        if get_option(name, default) != default:
            enabled_features.append(name)
    if enabled_features:
        raise ValueError(
            "DFlash and DSpark are baseline architectures; these features "
            f"require the mmuse architecture: {', '.join(enabled_features)}"
        )


@SpeculatorModel.register("dflash")
class DFlashDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    _needs_full_verifier_distribution: ClassVar[bool] = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        # verifier_lm_head is reloaded from the verifier (see load_verifier_weights)
        # and excluded on save, so it is expected to be absent from checkpoints.
        "verifier_lm_head.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(
        self,
        config: DFlashSpeculatorConfig,
    ) -> None:
        _reject_enhanced_baseline_features(config.speculators_model_type, config)
        # Forcibly override config settings
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                "simple_flex_attention"
            )
        self._attn_impl = config.transformer_layer_config._attn_implementation  # noqa: SLF001
        self._create_mask_fn = (
            _compiled_create_block_mask
            if self._attn_impl == "simple_flex_attention"
            else create_float_mask
            if self._attn_impl == "eager"
            else create_mask
        )
        super().__init__(config=config)
        # Runtime-only training policy: never changes saved architecture or weights.
        self.activation_checkpointing = False
        self._init_vocab(config)

        tl_config = config.transformer_layer_config

        # Number of draft layers is encoded in transformer_layer_config
        num_draft_layers = tl_config.num_hidden_layers
        hidden_size = tl_config.hidden_size
        num_target_layers = len(self.target_layer_ids)
        self.block_size = config.block_size
        self.layers = nn.ModuleList(
            [
                self._make_decoder_layer(config, layer_idx)
                for layer_idx in range(num_draft_layers)
            ]
        )
        self.sliding_window = getattr(tl_config, "sliding_window", None)
        self.sliding_window_indices = [
            i
            for i, layer_type in enumerate(
                getattr(tl_config, "layer_types", None) or []
            )
            if layer_type == "sliding_attention"
        ]
        self.uses_sliding_window_attn = bool(self.sliding_window_indices)
        self.uses_full_attn = bool(num_draft_layers - len(self.sliding_window_indices))
        self.sliding_window_non_causal = config.sliding_window_non_causal

        self.norm = Qwen3RMSNorm(
            hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.rotary_emb = Qwen3RotaryEmbedding(
            flatten_rope_parameters(config.transformer_layer_config)
        )  # type: ignore[arg-type]

        self.fc = nn.Linear(
            num_target_layers * hidden_size,
            hidden_size,
            bias=False,
        )
        self._init_layer_fusion(config, hidden_size, num_target_layers)

        self.hidden_norm = Qwen3RMSNorm(
            hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm = resolve_verifier_norm_class(config)(
            hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm.weight.requires_grad = False

        self._init_query_features(config, hidden_size)

        # Warn if using DFlash with sample_from_anchor=True (may not be supported)
        if type(self).__name__ == "DFlashDraftModel" and config.sample_from_anchor:
            logger.warning(
                "DFlash with sample_from_anchor=True may not be supported in "
                "all inference engines (e.g., vLLM). Verify compatibility with your "
                "deployment target."
            )

        self.post_init()
        self._reset_backbone_extensions(config)
        # Verifier-owned weights are reconstructed on load. Keep a reduced-vocab
        # lm_head serialized because current runtimes cannot derive it from the
        # full verifier head.
        #
        # Shadow the ClassVar lists with per-instance copies so full- and
        # reduced-vocabulary siblings cannot mutate each other's save rules.
        keys_to_ignore_on_save = list(type(self)._keys_to_ignore_on_save)  # noqa: SLF001
        keys_to_ignore_on_load_missing = list(
            type(self)._keys_to_ignore_on_load_missing  # noqa: SLF001
        )
        keys_to_ignore_on_save.append("embed_tokens.weight")
        if not self.use_draft_vocab:
            keys_to_ignore_on_save.append("lm_head.weight")
            keys_to_ignore_on_load_missing.append("lm_head.weight")
        self.__dict__["_keys_to_ignore_on_save"] = keys_to_ignore_on_save
        self.__dict__["_keys_to_ignore_on_load_missing"] = (
            keys_to_ignore_on_load_missing
        )

    def _make_decoder_layer(
        self, config: DFlashSpeculatorConfig, layer_idx: int
    ) -> Qwen3DFlashDecoderLayer:
        return Qwen3DFlashDecoderLayer(config.transformer_layer_config, layer_idx)

    def _init_layer_fusion(
        self, config: DFlashSpeculatorConfig, hidden_size: int, num_target_layers: int
    ) -> None:
        """Extension point after the shared verifier-state projection."""

    def _init_query_features(
        self, config: DFlashSpeculatorConfig, hidden_size: int
    ) -> None:
        """Extension point for block-query conditioning before initialization."""

    def _reset_backbone_extensions(self, config: DFlashSpeculatorConfig) -> None:
        """Extension point for identity-initialized optional modules."""

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.aux_hidden_state_layer_ids

    def load_verifier_weights(self):
        """Reconstruct weights intentionally omitted from DFlash checkpoints."""
        self._load_verifier_weights(
            overwrite_embed_tokens=True,
            overwrite_lm_head=not self.use_draft_vocab,
        )

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlashDraftModel":
        """Create DFlash model from training arguments.

        Args:
            verifier_config: Verifier model configuration. This should be a config
                with num_hidden_layers set to the number of DRAFT layers (created
                by create_transformer_layer_config in train.py).
            t2d: Target-to-draft vocabulary mapping tensor (optional)
            d2t: Draft-to-target vocabulary mapping tensor (optional)
            **kwargs: Training arguments with DFlash-specific params
                - draft_vocab_size: Size of draft vocabulary
                - block_size: Block size for draft predictions (default: 8)
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized DFlashDraftModel

        Note:
            The number of draft layers is encoded in verifier_config.num_hidden_layers,
            following the same pattern as EAGLE3.
        """
        config = DFlashSpeculatorConfig(
            **cls._build_base_config_kwargs("dflash", verifier_config, **kwargs)
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def _build_base_config_kwargs(
        algorithm: str,
        verifier_config: "PretrainedConfig",
        **kwargs,
    ) -> dict:
        """Shared DFlash-family config kwargs for ``from_training_args``.

        DSpark reuses this and appends its Markov/confidence/loss fields.
        """
        _reject_enhanced_baseline_features(algorithm, kwargs)

        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"), kwargs["verifier_name_or_path"]
        )
        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )
        block_size = kwargs.get("block_size", 8)

        default_sample_from_anchor = algorithm == "dspark"
        sample_from_anchor_arg = kwargs.get("sample_from_anchor")
        sample_from_anchor = (
            default_sample_from_anchor
            if sample_from_anchor_arg is None
            else sample_from_anchor_arg
        )

        # Calculate speculative tokens based on sample_from_anchor
        # False: anchor is bonus token (block_size - 1 tokens)
        # True: sample from anchor too (block_size tokens)
        speculative_tokens = block_size if sample_from_anchor else block_size - 1

        return {
            "transformer_layer_config": verifier_config,
            "draft_vocab_size": kwargs["draft_vocab_size"],
            "block_size": block_size,
            "aux_hidden_state_layer_ids": target_layer_ids,
            "target_hidden_state_format": kwargs.get(
                "target_hidden_state_format", "standard"
            ),
            "target_training_contract": kwargs.get("target_training_contract"),
            "mask_token_id": kwargs.get("mask_token_id"),
            "sliding_window_non_causal": kwargs.get("sliding_window_non_causal", False),
            "sample_from_anchor": sample_from_anchor,
            "speculators_config": SpeculatorsConfig(
                algorithm=algorithm,
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=speculative_tokens)
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(
                    kwargs["verifier_name_or_path"]
                ),
            ),
        }

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for DFlash.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        loss_config = resolve_training_loss(**kwargs)
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
        }
        return dict(shared), dict(shared)

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError(
                "mask_token_id is not set on the config. "
                "Pass --mask-token-id during training or ensure the config "
                "was saved with mask_token_id set."
            )
        return self.config.mask_token_id

    @torch.compiler.disable
    def _create_attention_mask(
        self,
        document_ids: torch.Tensor,
        total_seq_len: int,
        anchor_positions: torch.Tensor,
        device: torch.device,
        sliding_window: int | None = None,
        sliding_window_non_causal: bool = False,
    ):
        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            document_ids=document_ids.squeeze(0).to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            sliding_window=sliding_window,
            sliding_window_non_causal=sliding_window_non_causal,
        )
        return self._create_mask_fn(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    @torch.compiler.disable
    def _build_attention_mask(self, loss_mask, max_anchors, document_ids, device):
        total_seq_len = loss_mask.shape[1]

        anchor_positions, anchor_valid = select_anchors(
            loss_mask, max_anchors, self.block_size, document_ids=document_ids
        )

        full_attn_mask = None
        if self.uses_full_attn:
            full_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=None,
            )

        sliding_window_attn_mask = None
        if self.uses_sliding_window_attn:
            sliding_window_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=self.sliding_window,
                sliding_window_non_causal=self.sliding_window_non_causal,
            )

        return full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid

    def _prepare_target_hidden(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Project the concatenated verifier states into the draft hidden space."""
        return self.fc(hidden_states), None

    def _fuse_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the existing shared FC plus token-adaptive target context."""
        shared_projection, _ = self._prepare_target_hidden(hidden_states)
        return self.hidden_norm(shared_projection)

    def _condition_noise_embedding(
        self,
        noise_embedding: torch.Tensor,
        fused_context: torch.Tensor,  # noqa: ARG002
        anchor_positions: torch.Tensor,  # noqa: ARG002
        document_ids: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        """Extension point for block-query conditioning; baseline is unchanged."""
        return noise_embedding

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Recompute decoder layers only; leave sampling, heads and loss unchanged."""
        self.activation_checkpointing = enabled

    def _backbone_forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        *,
        project_logits: bool = True,
        **kwargs,
    ):
        """Run the anchored-block draft transformer and optionally project logits.

        Returns ``(hidden, logits, targets, aligned_loss_mask,
        anchored_block_indices, target_log_normalizer, target_argmax_ids)``.
        The last two entries are only needed with a pruned vocabulary; otherwise
        they are ``None``. ``logits`` is ``None`` when
        ``project_logits=False`` so DSpark can correct hidden states before the
        single draft-vocabulary projection.
        """
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        num_anchors = kwargs.pop("max_anchors", 3072)

        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, num_anchors, document_ids, device)
        )

        mask_tokens_size = num_anchors * self.block_size

        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )  # shape: [1, num_anchors*block_size]
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)
        # shape: [1, num_anchors*block_size, hidden_size]

        fc_output = self._fuse_target_hidden(hidden_states)
        noise_embedding = self._condition_noise_embedding(
            noise_embedding,
            fc_output,
            anchor_positions,
            document_ids,
        )
        # shape: [1, total_seq_len, hidden_size]

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        # shape: [1, total_seq_len + num_anchors*block_size]

        # the hidden_states shape doesn't match position_ids but doesn't need
        # to, as hidden_states is only used to set dtype and device in rotary_emb
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )  # shape: [num_anchors*block_size]

        target_log_normalizer = None
        target_argmax_ids = None
        with torch.no_grad():
            target_positions = anchored_block_indices
            if not self.config.sample_from_anchor:
                target_positions = (target_positions - 1) % total_seq_len
            verifier_pre_lm_hidden = self.verifier_norm(
                verifier_last_hidden_states[:, target_positions].to(
                    self.verifier_norm.weight.dtype
                )
            )
            if self.use_draft_vocab:
                if self.d2t is None:
                    raise RuntimeError(
                        "Pruned target statistics require vocabulary mappings"
                    )
                draft_token_ids = self.d2t + torch.arange(
                    self.draft_vocab_size, device=self.d2t.device
                )
                verifier_logits, target_log_normalizer, target_argmax_ids = (
                    project_target_distribution(
                        verifier_pre_lm_hidden, self.verifier_lm_head, draft_token_ids
                    )
                )
            else:
                verifier_logits = self.verifier_lm_head(verifier_pre_lm_hidden)
            targets = verifier_logits

        for layer_idx, layer in enumerate(self.layers):
            layer_kwargs = dict(
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if (
                self.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                # Bind the module and this call's inputs directly (no loop closure).
                # Positional tensors let checkpoint detect NPU RNG/autocast state.
                # Call the module, not forward(), to retain FSDP's hooks.
                noise_embedding = checkpoint(
                    layer,
                    fc_output,
                    noise_embedding,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    **layer_kwargs,
                )
            else:
                noise_embedding = layer(
                    target_hidden=fc_output,
                    hidden_states=noise_embedding,
                    **layer_kwargs,
                )

        hidden = self.norm(noise_embedding)
        logits = self.lm_head(hidden) if project_logits else None
        # shape when projected: [1, num_anchors*block_size, vocab_size]

        aligned_loss_mask = build_anchored_loss_mask(
            loss_mask,
            document_ids,
            anchor_positions,
            anchor_valid,
            self.block_size,
            sample_from_anchor=self.config.sample_from_anchor,
        )

        return (
            hidden,
            logits,
            targets,
            aligned_loss_mask,
            anchored_block_indices,
            target_log_normalizer,
            target_argmax_ids,
        )

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1,total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        document_ids: torch.Tensor,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 3072,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        (
            hidden,
            logits,
            targets,
            aligned_loss_mask,
            anchored_block_indices,
            _target_log_normalizer,
            target_argmax_ids,
        ) = self._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            **kwargs,
        )
        if logits is None:
            raise RuntimeError("DFlash forward requires projected draft logits")
        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
            target_argmax_ids=target_argmax_ids,
        )
        return None, loss, metrics
