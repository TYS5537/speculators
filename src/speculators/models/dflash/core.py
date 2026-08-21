import logging
from typing import ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
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
    DFlash2CandidateSelector,
    DFlash2GroupedConv,
    Qwen3DFlashDecoderLayer,
)
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.metrics import LossConfig, resolve_loss_config
from speculators.models.utils import conditional_torch_compile, resolve_target_layer_ids

logger = logging.getLogger(__name__)

# Compile so the mask builds block-sparse instead of materializing DFlash's huge
# dense [Q, KV] grid every step. (No benefit for EAGLE3's small autoregressive mask.)
_compiled_create_block_mask = torch.compile(create_block_mask)
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


@SpeculatorModel.register("dflash")
class DFlashDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
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
        self._init_vocab(config)

        tl_config = config.transformer_layer_config

        # Number of draft layers is encoded in transformer_layer_config
        num_draft_layers = tl_config.num_hidden_layers
        hidden_size = tl_config.hidden_size
        num_target_layers = len(self.target_layer_ids)
        self.block_size = config.block_size
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(
                    config.transformer_layer_config,  # type: ignore[arg-type]
                    layer_idx,
                    dflash2_dynamic_conv=config.dflash2_dynamic_conv,
                    dflash2_conv_kernel_size=config.dflash2_conv_kernel_size,
                    dflash2_conv_group_size=config.dflash2_conv_group_size,
                    block_size=config.block_size,
                )
                for layer_idx in range(num_draft_layers)
            ]
        )
        self.sliding_window = tl_config.sliding_window
        self.sliding_window_indices = [
            i
            for i, layer_type in enumerate(tl_config.layer_types)
            if layer_type == "sliding_attention"
        ]
        self.uses_sliding_window_attn = bool(self.sliding_window_indices)
        self.uses_full_attn = bool(num_draft_layers - len(self.sliding_window_indices))
        self.sliding_window_non_causal = config.sliding_window_non_causal

        self.norm = Qwen3RMSNorm(
            hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config.transformer_layer_config)  # type: ignore[arg-type]

        self.dflash_gated_layer_fusion = config.dflash_gated_layer_fusion
        self.fc = nn.Linear(
            num_target_layers * hidden_size,
            hidden_size,
            bias=False,
        )
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

        self.hidden_norm = Qwen3RMSNorm(
            hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm = Qwen3RMSNorm(
            hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm.weight.requires_grad = False

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

        # Warn if using DFlash with sample_from_anchor=True (may not be supported)
        if type(self).__name__ == "DFlashDraftModel" and config.sample_from_anchor:
            logger.warning(
                "DFlash with sample_from_anchor=True may not be supported in "
                "all inference engines (e.g., vLLM). Verify compatibility with your "
                "deployment target."
            )

        self.post_init()
        if self.layer_fusion_score is not None:
            nn.init.zeros_(self.layer_fusion_score.weight)
        if self.block_position_embedding is not None:
            nn.init.zeros_(self.block_position_embedding.weight)
        for module in self.modules():
            if isinstance(module, DFlash2GroupedConv):
                module.reset_identity()
        if self.candidate_selector is not None:
            self.candidate_selector.reset_unary(tl_config.initializer_range)

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.aux_hidden_state_layer_ids

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
            "mask_token_id": kwargs.get("mask_token_id"),
            "sliding_window_non_causal": kwargs.get("sliding_window_non_causal", False),
            "dflash_context_residual": kwargs.get("dflash_context_residual", False),
            "dflash_block_position_embedding": kwargs.get(
                "dflash_block_position_embedding", False
            ),
            "dflash_gated_layer_fusion": kwargs.get("dflash_gated_layer_fusion", False),
            "dflash2_dynamic_conv": kwargs.get("dflash2_dynamic_conv", False),
            "dflash2_conv_kernel_size": kwargs.get("dflash2_conv_kernel_size", 2),
            "dflash2_conv_group_size": kwargs.get("dflash2_conv_group_size", 16),
            "dflash2_candidate_selector": kwargs.get(
                "dflash2_candidate_selector", False
            ),
            "dflash2_selector_rank": kwargs.get("dflash2_selector_rank", 256),
            "dflash2_selector_top_k": kwargs.get("dflash2_selector_top_k", 16),
            "dflash2_selector_search_mode": kwargs.get(
                "dflash2_selector_search_mode", "greedy"
            ),
            "dflash2_selector_loss_weight": kwargs.get(
                "dflash2_selector_loss_weight", 1.0
            ),
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
        loss_config = resolve_loss_config(kwargs["loss_fn"])
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
            loss_mask, max_anchors, self.block_size
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

    def _fuse_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the existing shared FC plus token-adaptive target context."""
        shared_projection, _ = self._prepare_target_hidden(hidden_states)
        return self.hidden_norm(shared_projection)

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

    def dflash2_select_candidates(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score one position's Top-K candidates for DFlash2 rollout.

        ``previous_token_ids`` are verifier-vocabulary IDs. Returned candidate
        IDs remain in the draft vocabulary, matching ``logits``.
        """
        if self.candidate_selector is None:
            raise RuntimeError("DFlash2 candidate selector is not enabled")
        if logits.shape[:-1] != hidden_states.shape[:-1]:
            raise ValueError("DFlash2 logits and hidden states must align")
        if previous_token_ids.shape != hidden_states.shape[:-1]:
            raise ValueError("DFlash2 previous-token IDs and hidden states must align")
        unary_logits, candidate_ids = torch.topk(
            logits,
            k=self.candidate_selector.top_k,
            dim=-1,
        )
        candidate_logits = self.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            previous_token_ids,
        )
        return candidate_ids, candidate_logits

    def dflash2_sparse_logits(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the selector's realized sparse logits and Top-K draft IDs."""
        candidate_ids, candidate_logits = self.dflash2_select_candidates(
            logits,
            hidden_states,
            previous_token_ids,
        )
        sparse_logits = torch.full_like(logits, -torch.inf)
        sparse_logits.scatter_(-1, candidate_ids, candidate_logits)
        return sparse_logits, candidate_ids

    def _dflash2_select_topk_path(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_blocks: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select one block path from fixed DFlash Top-K candidates.

        Greedy mode reproduces the public DFlash2 walk. Global mode streams the
        block-local K-by-K edge lattice one position at a time and uses Viterbi
        backtracking to maximize the locally normalized joint path probability.
        Path selection is intentionally discrete; selector parameters are trained
        by the teacher-row loss in :meth:`_dflash2_block_outputs`.
        """
        if self.candidate_selector is None:
            raise RuntimeError("DFlash2 candidate selector is not enabled")
        if candidate_ids.shape != unary_logits.shape:
            raise ValueError("DFlash2 candidate IDs and unary logits must align")
        if candidate_ids.shape[:-1] != hidden_blocks.shape[:-1]:
            raise ValueError("DFlash2 candidates and hidden blocks must align")
        if anchor_token_ids.shape != (candidate_ids.shape[0],):
            raise ValueError("DFlash2 path selection requires one anchor per block")

        start_position = 0 if self.config.sample_from_anchor else 1
        selected_ids = candidate_ids[..., 0].detach().clone()
        realized_logits = unary_logits.detach().clone()
        active_candidates = candidate_ids[:, start_position:]
        if active_candidates.shape[1] == 0:
            return selected_ids, realized_logits

        active_unary = unary_logits[:, start_position:]
        active_hidden = hidden_blocks[:, start_position:]
        search_mode = self.config.dflash2_selector_search_mode
        with torch.no_grad():
            if search_mode == "greedy":
                previous_ids = anchor_token_ids.long()
                active_selected: list[torch.Tensor] = []
                active_rows: list[torch.Tensor] = []
                for position in range(active_candidates.shape[1]):
                    row = self.candidate_selector(
                        active_candidates[:, position],
                        active_unary[:, position],
                        active_hidden[:, position],
                        previous_ids,
                    )
                    selected_indices = row.argmax(dim=-1, keepdim=True)
                    selected_draft_ids = (
                        active_candidates[:, position]
                        .gather(-1, selected_indices)
                        .squeeze(-1)
                    )
                    active_rows.append(row)
                    active_selected.append(selected_draft_ids)
                    previous_ids = self._draft_ids_to_verifier(selected_draft_ids)
                selected_active = torch.stack(active_selected, dim=1)
                realized_active = torch.stack(active_rows, dim=1)
            elif search_mode == "global":
                # The first predecessor is the anchor. Later predecessor rows are
                # the verifier IDs corresponding to every prior Top-K candidate.
                first_row = self.candidate_selector(
                    active_candidates[:, 0],
                    active_unary[:, 0],
                    active_hidden[:, 0],
                    anchor_token_ids.long(),
                )
                # Selector rows define locally normalized conditional
                # distributions.  Viterbi must therefore add row log-probabilities,
                # not raw energies whose partition function varies by predecessor.
                best_scores = torch.log_softmax(first_row.float(), dim=-1)
                backpointers: list[torch.Tensor] = []
                for position in range(1, active_candidates.shape[1]):
                    predecessor_ids = self._draft_ids_to_verifier(
                        active_candidates[:, position - 1]
                    )
                    lattice = self.candidate_selector.score_lattice(
                        active_candidates[:, position : position + 1],
                        active_unary[:, position : position + 1],
                        active_hidden[:, position : position + 1],
                        predecessor_ids.unsqueeze(1),
                    )[:, 0]
                    edge_log_probs = torch.log_softmax(lattice.float(), dim=-1)
                    path_scores = best_scores.unsqueeze(-1) + edge_log_probs
                    best_scores, previous_indices = path_scores.max(dim=-2)
                    backpointers.append(previous_indices)

                active_indices: list[torch.Tensor] = [best_scores.argmax(dim=-1)]
                for previous_indices in reversed(backpointers):
                    active_indices.append(
                        previous_indices.gather(
                            -1, active_indices[-1].unsqueeze(-1)
                        ).squeeze(-1)
                    )
                active_indices.reverse()
                selected_indices = torch.stack(active_indices, dim=1)
                selected_active = active_candidates.gather(
                    -1, selected_indices.unsqueeze(-1)
                ).squeeze(-1)

                active_rows = []
                previous_ids = anchor_token_ids.long()
                for position in range(active_candidates.shape[1]):
                    active_rows.append(
                        self.candidate_selector(
                            active_candidates[:, position],
                            active_unary[:, position],
                            active_hidden[:, position],
                            previous_ids,
                        )
                    )
                    previous_ids = self._draft_ids_to_verifier(
                        selected_active[:, position]
                    )
                realized_active = torch.stack(active_rows, dim=1)
            else:
                raise ValueError(
                    f"Unsupported DFlash2 selector search mode: {search_mode!r}"
                )

        selected_ids[:, start_position:] = selected_active
        realized_logits[:, start_position:] = realized_active
        return selected_ids, realized_logits

    def dflash2_select_path(
        self,
        logits: torch.Tensor,
        hidden_blocks: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Top-K IDs, their realized path rows, and selected draft IDs."""
        if self.candidate_selector is None:
            raise RuntimeError("DFlash2 candidate selector is not enabled")
        if logits.shape[:-1] != hidden_blocks.shape[:-1]:
            raise ValueError("DFlash2 logits and hidden blocks must align")
        unary_logits, candidate_ids = torch.topk(
            logits,
            k=self.candidate_selector.top_k,
            dim=-1,
        )
        selected_ids, realized_logits = self._dflash2_select_topk_path(
            candidate_ids,
            unary_logits,
            hidden_blocks,
            anchor_token_ids,
        )
        return candidate_ids, realized_logits, selected_ids

    def _dflash2_proposal_logits(
        self,
        candidate_ids: torch.Tensor,
        realized_logits: torch.Tensor,
        selected_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return candidate logits that reproduce the configured path proposal.

        The public greedy walk selects the local argmax of every realized row, so
        its original logits already describe the proposal. A Viterbi path can
        deliberately take a locally suboptimal edge to improve later positions;
        expose that deterministic whole-path decision as a one-hot distribution.
        """
        search_mode = getattr(self.config, "dflash2_selector_search_mode", "greedy")
        if search_mode == "greedy":
            return realized_logits
        if search_mode != "global":
            raise ValueError(
                f"Unsupported DFlash2 selector search mode: {search_mode!r}"
            )
        selected_mask = candidate_ids == selected_ids.unsqueeze(-1)
        if not selected_mask.any(dim=-1).all():
            raise RuntimeError("DFlash2 global path selected an ID outside its Top-K")
        return torch.where(
            selected_mask,
            torch.zeros_like(realized_logits),
            torch.full_like(realized_logits, -torch.inf),
        )

    def _dflash2_block_outputs(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        hidden_blocks: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        teacher_previous_token_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build selector rollout scores and its restricted-Top-K train loss.

        The public DFlash2 inference implementation exposes a K-by-K transition
        lattice but not its training objective. This repository directly scores
        the teacher/realized predecessor row and distils the verifier distribution
        restricted to the current Top-K. This avoids materializing an
        ``anchors x positions x K x K`` autograd graph. The existing
        full-vocabulary draft loss is untouched.
        """
        if self.candidate_selector is None:
            raise RuntimeError("DFlash2 candidate selector is not enabled")
        num_blocks, block_size, _ = hidden_blocks.shape
        if block_size != self.block_size:
            raise ValueError("DFlash2 hidden blocks use the wrong block size")
        expected_logits_shape = (1, num_blocks * block_size, self.draft_vocab_size)
        if logits.shape != expected_logits_shape or targets.shape != logits.shape:
            raise ValueError("DFlash2 logits/targets have an unexpected shape")
        if loss_mask.shape != logits.shape[:-1]:
            raise ValueError("DFlash2 loss mask must align with logits")
        if anchor_token_ids.shape != (num_blocks,):
            raise ValueError("DFlash2 requires one verifier anchor ID per block")
        if teacher_previous_token_ids.shape != (num_blocks, block_size):
            raise ValueError(
                "DFlash2 teacher predecessor IDs must align with hidden blocks"
            )
        logits_blocks = logits.view(num_blocks, block_size, -1)
        target_blocks = targets.view_as(logits_blocks)
        mask_blocks = loss_mask.view(num_blocks, block_size)
        unary_logits, candidate_ids = torch.topk(
            logits_blocks,
            k=self.candidate_selector.top_k,
            dim=-1,
        )
        start_position = 0 if self.config.sample_from_anchor else 1
        active_candidates = candidate_ids[:, start_position:]
        active_unary = unary_logits[:, start_position:]
        active_hidden = hidden_blocks[:, start_position:]
        active_targets = target_blocks[:, start_position:]
        active_mask = mask_blocks[:, start_position:]
        teacher_rows = self.candidate_selector(
            active_candidates,
            active_unary,
            active_hidden,
            teacher_previous_token_ids[:, start_position:],
        )
        selected_ids, realized_logits = self._dflash2_select_topk_path(
            candidate_ids,
            unary_logits,
            hidden_blocks,
            anchor_token_ids,
        )

        target_topk_logits = active_targets.gather(-1, active_candidates)
        target_topk_probs = torch.softmax(target_topk_logits.float(), dim=-1).detach()
        selector_loss_per_position = -(
            target_topk_probs * torch.log_softmax(teacher_rows.float(), dim=-1)
        ).sum(dim=-1)
        selector_mask = active_mask.to(selector_loss_per_position.dtype)
        selector_loss = (selector_loss_per_position * selector_mask).sum() / (
            selector_mask.sum().clamp_min(1.0)
        )
        full_teacher_rows = torch.cat(
            [unary_logits[:, :start_position], teacher_rows],
            dim=1,
        )
        return (
            candidate_ids,
            realized_logits,
            selector_loss,
            selected_ids,
            full_teacher_rows,
        )

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
        anchored_block_indices)``. ``logits`` is ``None`` when
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

        with torch.no_grad():
            verifier_pre_lm_hidden = self.verifier_norm(
                verifier_last_hidden_states.to(self.verifier_norm.weight.dtype)
            )

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

        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(verifier_pre_lm_hidden)
            if not self.config.sample_from_anchor:
                # False: shift right by 1 so slot j predicts token at position j
                verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            # else: True, slot k predicts token at position k+1 (next), no shift
            targets = verifier_logits[:, anchored_block_indices]
            # shape: [1, num_anchors*block_size, draft_vocab_size]

        for layer_idx, layer in enumerate(self.layers):
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden = self.norm(noise_embedding)
        logits = self.lm_head(hidden) if project_logits else None
        # shape when projected: [1, num_anchors*block_size, vocab_size]

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        # shape: [1, num_anchors*block_size]

        # zero out any padded anchor blocks
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )  # shape: [1, num_anchors*block_size]

        # For sample_from_anchor=False, mask slot 0 (anchor) since it's not trained
        if not self.config.sample_from_anchor:
            aligned_loss_mask[:, :: self.block_size] = 0

        return hidden, logits, targets, aligned_loss_mask, anchored_block_indices

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
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **kwargs,
            )
        )
        if logits is None:
            raise RuntimeError("DFlash forward requires projected draft logits")
        proposal_candidate_ids = None
        proposal_candidate_logits = None
        selector_loss = None
        if self.candidate_selector is not None:
            num_blocks = hidden.shape[1] // self.block_size
            hidden_blocks = hidden.view(num_blocks, self.block_size, -1)
            block_tokens = input_ids[0, anchored_block_indices].view(
                num_blocks, self.block_size
            )
            candidate_ids, candidate_logits, selector_loss, selected_ids, _ = (
                self._dflash2_block_outputs(
                    logits,
                    targets,
                    hidden_blocks,
                    block_tokens[:, 0],
                    aligned_loss_mask,
                    teacher_previous_token_ids=(
                        block_tokens
                        if self.config.sample_from_anchor
                        else torch.cat(
                            [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
                        )
                    ),
                )
            )
            candidate_logits = self._dflash2_proposal_logits(
                candidate_ids,
                candidate_logits,
                selected_ids,
            )
            proposal_candidate_ids = candidate_ids.view(
                1, num_blocks * self.block_size, -1
            )
            proposal_candidate_logits = candidate_logits.view_as(proposal_candidate_ids)
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
            proposal_candidate_ids=proposal_candidate_ids,
            proposal_candidate_logits=proposal_candidate_logits,
        )
        if selector_loss is not None:
            loss = loss + self.config.dflash2_selector_loss_weight * selector_loss
            metrics["loss_sum"] = loss.detach().clone()
            metrics["dflash2_selector_loss_sum"] = selector_loss.detach().clone()
            metrics["dflash2_selector_loss_total"] = torch.ones((), device=loss.device)
        return None, loss, metrics
