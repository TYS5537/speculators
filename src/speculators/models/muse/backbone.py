"""Muse backbone extensions composed with the baseline DSpark/DFlash model."""

from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from speculators.models.muse.config import MuseOptions
from speculators.models.muse.decoder import Qwen3MuseDecoderLayer
from speculators.models.muse.dynamic_conv import DFlash2GroupedConv
from speculators.models.muse.selector import DFlash2CandidateSelector
from speculators.models.muse.selector_runtime import MuseSelectorMixin

_MISSING_KEY_PREVIEW = 8


def _reject_missing_optional_weights(
    *,
    enabled: bool,
    missing_keys: tuple[str, ...],
    fragments: tuple[str, ...],
    feature_name: str,
) -> None:
    """Reject a config-edited checkpoint that lacks trained optional weights."""
    if not enabled:
        return
    missing = [
        key for key in missing_keys if any(fragment in key for fragment in fragments)
    ]
    if not missing:
        return
    preview = ", ".join(missing[:_MISSING_KEY_PREVIEW])
    suffix = " ..." if len(missing) > _MISSING_KEY_PREVIEW else ""
    raise RuntimeError(
        f"The checkpoint enables {feature_name} but does not contain its "
        f"trained weights: {preview}{suffix}. Do not enable this feature by "
        "editing an older checkpoint config."
    )


class MuseBackboneMixin(MuseSelectorMixin):
    """Add Muse modules without making either baseline depend on Muse."""

    _no_split_modules: ClassVar[list[str]] = [
        "Qwen3DFlashDecoderLayer",
        "Qwen3MuseDecoderLayer",
    ]

    def _make_decoder_layer(self, config, layer_idx: int) -> Qwen3MuseDecoderLayer:
        return Qwen3MuseDecoderLayer(
            config.transformer_layer_config,
            layer_idx,
            dflash2_dynamic_conv=config.dflash2_dynamic_conv,
            dflash2_conv_kernel_size=config.dflash2_conv_kernel_size,
            dflash2_conv_group_size=config.dflash2_conv_group_size,
            block_size=config.block_size,
        )

    def _init_layer_fusion(
        self, config, hidden_size: int, num_target_layers: int
    ) -> None:
        self.dflash_gated_layer_fusion = config.dflash_gated_layer_fusion
        self.layer_fusion_norms: nn.ModuleList | None = None
        self.layer_fusion_score: nn.Linear | None = None
        self.layer_fusion_proj: nn.Linear | None = None
        self.layer_fusion_gate: nn.Parameter | None = None
        if self.dflash_gated_layer_fusion:
            self.layer_fusion_norms = nn.ModuleList(
                [
                    Qwen3RMSNorm(
                        hidden_size,
                        eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
                    )
                    for _ in range(num_target_layers)
                ]
            )
            self.layer_fusion_score = nn.Linear(hidden_size, 1, bias=False)
            self.layer_fusion_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.layer_fusion_gate = nn.Parameter(torch.zeros(()))

    def _init_query_features(self, config, hidden_size: int) -> None:
        self.context_hidden_proj: nn.Linear | None = None
        self.context_hidden_gate: nn.Parameter | None = None
        if config.dflash_context_residual:
            self.context_hidden_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.context_hidden_gate = nn.Parameter(torch.zeros(()))

        self.block_position_embedding: nn.Embedding | None = None
        if config.dflash_block_position_embedding:
            self.block_position_embedding = nn.Embedding(self.block_size, hidden_size)

        self.candidate_selector: DFlash2CandidateSelector | None = None
        if config.dflash2_candidate_selector:
            self.candidate_selector = DFlash2CandidateSelector(
                hidden_size=hidden_size,
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                rank=config.dflash2_selector_rank,
                top_k=config.dflash2_selector_top_k,
            )

    def _reset_backbone_extensions(self, config) -> None:
        tl_config = config.transformer_layer_config
        if self.layer_fusion_score is not None:
            nn.init.zeros_(self.layer_fusion_score.weight)
        if self.block_position_embedding is not None:
            nn.init.zeros_(self.block_position_embedding.weight)
        for module in self.modules():
            if isinstance(module, DFlash2GroupedConv):
                module.reset_identity()
        if self.candidate_selector is not None:
            self.candidate_selector.reset_unary(tl_config.initializer_range)

    @classmethod
    def _build_base_config_kwargs(
        cls, algorithm: str, verifier_config: PretrainedConfig, **kwargs
    ) -> dict:
        if algorithm == "muse" and kwargs.get("sample_from_anchor") is None:
            kwargs["sample_from_anchor"] = cls.config_class.model_fields[
                "sample_from_anchor"
            ].get_default()
        config_kwargs = super()._build_base_config_kwargs(
            algorithm, verifier_config, **kwargs
        )
        config_kwargs.update(
            {
                name: kwargs.get(name, field.default)
                for name, field in MuseOptions.model_fields.items()
            }
        )
        return config_kwargs

    def _prepare_target_hidden(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build the shared target projection and expose raw per-layer states."""
        baseline_projection = self.fc(hidden_states)
        if not self.dflash_gated_layer_fusion:
            return baseline_projection, None

        if (
            self.layer_fusion_norms is None
            or self.layer_fusion_score is None
            or self.layer_fusion_proj is None
            or self.layer_fusion_gate is None
        ):
            raise RuntimeError("Gated layer fusion modules were not initialized")
        num_layers = len(self.layer_fusion_norms)
        hidden_size = self.config.transformer_layer_config.hidden_size
        expected_size = num_layers * hidden_size
        if hidden_states.shape[-1] != expected_size:
            raise ValueError(
                "Expected concatenated verifier hidden size "
                f"{expected_size}, got {hidden_states.shape[-1]}"
            )

        layer_states = hidden_states.reshape(
            *hidden_states.shape[:-1], num_layers, hidden_size
        )
        normalized = torch.stack(
            [
                norm(layer_states[..., layer_idx, :])
                for layer_idx, norm in enumerate(self.layer_fusion_norms)
            ],
            dim=-2,
        )
        scores = self.layer_fusion_score(normalized).squeeze(-1)
        weights = torch.softmax(scores.float(), dim=-1).to(normalized.dtype)
        fused = (normalized * weights.unsqueeze(-1)).sum(dim=-2)
        fusion_residual = self.layer_fusion_proj(fused)
        projected = baseline_projection + (
            torch.tanh(self.layer_fusion_gate) * fusion_residual
        )
        return projected, layer_states

    def _prepare_missing_checkpoint_weights(self, loading_info: dict) -> None:
        """Safely initialize optional DFlash modules absent from a checkpoint."""
        missing_keys = tuple(loading_info.get("missing_keys", ()))
        _reject_missing_optional_weights(
            enabled=self.config.dflash2_dynamic_conv,
            missing_keys=missing_keys,
            fragments=(".attention_conv.", ".mlp_conv."),
            feature_name="DFlash2 dynamic convolution",
        )
        _reject_missing_optional_weights(
            enabled=self.config.dflash2_candidate_selector,
            missing_keys=missing_keys,
            fragments=("candidate_selector.",),
            feature_name="the DFlash2 candidate selector",
        )

    def _draft_ids_to_verifier(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Map draft-vocabulary IDs to the verifier vocabulary."""
        if self.d2t is None:
            return draft_ids.long()
        draft_ids = draft_ids.long()
        return draft_ids + self.d2t[draft_ids]

    def _condition_noise_embedding(
        self,
        noise_embedding: torch.Tensor,
        fused_context: torch.Tensor,
        anchor_positions: torch.Tensor,
        document_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply opt-in, inference-safe DFlash block conditioning."""
        if self.block_position_embedding is not None:
            slot_ids = torch.arange(
                self.block_size,
                dtype=torch.long,
                device=noise_embedding.device,
            ).repeat(anchor_positions.numel())
            noise_embedding = noise_embedding + self.block_position_embedding(
                slot_ids
            ).unsqueeze(0).to(noise_embedding.dtype)

        if self.context_hidden_proj is not None:
            if self.context_hidden_gate is None:
                raise RuntimeError("Context residual gate was not initialized")
            context_positions = (anchor_positions - 1).clamp_min(0)
            last_context = fused_context[:, context_positions, :]
            residual = self.context_hidden_proj(last_context)
            residual = residual.repeat_interleave(self.block_size, dim=1)

            anchor_docs = document_ids[:, anchor_positions]
            context_docs = document_ids[:, context_positions]
            valid_context = (
                (anchor_positions.unsqueeze(0) > 0)
                & (anchor_docs == context_docs)
                & (anchor_docs != -1)
            )
            valid_context = valid_context.repeat_interleave(
                self.block_size, dim=1
            ).unsqueeze(-1)
            residual = residual * valid_context.to(residual.dtype)
            noise_embedding = noise_embedding + (
                torch.tanh(self.context_hidden_gate)
                * residual.to(noise_embedding.dtype)
            )

        return noise_embedding
