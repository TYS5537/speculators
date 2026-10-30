"""EAGLE-style token/state recurrence with DFlash-style target KV injection."""

import logging
from typing import ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.attention import create_float_mask
from speculators.models.leagle.attention import rollout_mask, trajectory_mask
from speculators.models.leagle.cache import ReusableDraftKV, StaticDraftKV
from speculators.models.leagle.candidate_head import LowRankCandidateHead
from speculators.models.leagle.config import LEagleSpeculatorConfig
from speculators.models.leagle.model_definitions import DraftKV, LEagleDecoderLayer
from speculators.models.leagle.objective import TrainingObjective
from speculators.models.leagle.options import TRAINING_DEFAULTS
from speculators.models.leagle.output_head import (
    share_output_head,
    validate_shared_output_head,
)
from speculators.models.leagle.sparse_attention import (
    TrajectoryAttention,
    TrajectoryMaskCache,
)
from speculators.models.leagle.sparse_candidates import SparseCandidateLogits
from speculators.models.metrics import resolve_training_loss
from speculators.models.utils import (
    flatten_rope_parameters,
    resolve_target_layer_ids,
    validate_kv_reuse_rope,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.proposals.sampling import sample_from_logits, validate_temperature
from speculators.train.leagle_optimizations import resolve_optimization_groups

logger = logging.getLogger(__name__)


@SpeculatorModel.register("leagle")
class LEagleDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[LEagleSpeculatorConfig]] = LEagleSpeculatorConfig
    _no_split_modules = ["LEagleDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [
        "verifier_norm.weight",
        "verifier_lm_head.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [
        "verifier_norm.weight",
        "verifier_lm_head.weight",
    ]

    def __init__(self, config: LEagleSpeculatorConfig):
        decoder = config.transformer_layer_config
        if decoder._attn_implementation is None:  # noqa: SLF001
            decoder._attn_implementation = "simple_flex_attention"  # noqa: SLF001
        self._attn_impl = decoder._attn_implementation  # noqa: SLF001
        if self._attn_impl not in {"eager", "sdpa", "simple_flex_attention"}:
            raise ValueError("LEAGLE supports eager, sdpa or simple_flex_attention")
        super().__init__(config)
        self._init_vocab(config)
        if not config.aux_hidden_state_layer_ids:
            raise ValueError("LEAGLE requires explicit auxiliary target layer IDs")
        width = self.hidden_size
        eps = decoder.rms_norm_eps
        # Same persistent target memory as the baseline DFlash/DSpark backbone.
        self.fc = nn.Linear(len(self.target_layer_ids) * width, width, bias=False)
        self.hidden_norm = Qwen3RMSNorm(width, eps=eps)
        # Fuse the known token and the previous token's draft state once per
        # token, outside the shared core. No new token is consumed by a loop.
        self.embed_norm = Qwen3RMSNorm(width, eps=eps)
        self.state_norm = Qwen3RMSNorm(width, eps=eps)
        self.input_fc = nn.Linear(2 * width, width, bias=False)
        self.layers = nn.ModuleList(
            [LEagleDecoderLayer(decoder, i) for i in range(decoder.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(width, eps=eps)
        self.verifier_norm = Qwen3RMSNorm(width, eps=eps)
        self.verifier_norm.requires_grad_(False)
        self.rotary_emb = Qwen3RotaryEmbedding(flatten_rope_parameters(decoder))
        self.activation_checkpointing = False
        self.activation_checkpoint_layers: frozenset[int] | None = None
        self.post_init()
        self.candidate_head = None
        if config.candidate_head:
            # Build the head after backbone initialization and isolate its RNG:
            # adding the auxiliary objective must not change the seeded backbone.
            with torch.random.fork_rng(devices=[]):
                self.candidate_head = LowRankCandidateHead(
                    width,
                    self.draft_vocab_size,
                    config.candidate_head_rank,
                    initializer_range=decoder.initializer_range,
                )
        self.candidate_head_enabled = config.candidate_head
        self.candidate_top_k = config.candidate_head_top_k
        self.sparse_candidates = False
        self.static_kv = False
        self.reuse_rollout_buffers = False
        self.static_target_kv = False
        self.selective_target_hidden_states = False
        self._inference_shared_output = False

    def set_inference_optimizations(
        self,
        *,
        sparse_candidates=False,
        static_kv=False,
        reuse_rollout_buffers=False,
        static_target_kv=False,
        selective_target_hidden_states=False,
        share_output_head=False,
    ):
        if sparse_candidates and not self.candidate_head_enabled:
            raise ValueError("Sparse candidates require an enabled candidate head")
        if static_kv:
            validate_kv_reuse_rope(self.rotary_emb)
        if reuse_rollout_buffers and not static_kv:
            raise ValueError("Rollout buffer reuse requires static KV")
        self.sparse_candidates = sparse_candidates
        self.static_kv = static_kv
        self.reuse_rollout_buffers = reuse_rollout_buffers
        self.static_target_kv = static_target_kv
        self.selective_target_hidden_states = selective_target_hidden_states
        if share_output_head:
            self.share_output_head()
        elif self._inference_shared_output:
            self.verifier_lm_head.weight = nn.Parameter(
                self.lm_head.weight.detach().clone(), requires_grad=False
            )
        self._inference_shared_output = share_output_head

    @property
    def target_layer_ids(self) -> list[int]:
        return self.config.aux_hidden_state_layer_ids

    def share_output_head(self):
        return share_output_head(self)

    def validate_shared_output_head(self, reference):
        validate_shared_output_head(self, reference)

    def set_candidate_head(self, enabled: bool, *, top_k: int | None = None):
        """Override proposal retrieval; training uses the checkpoint's architecture."""
        if enabled:
            if self.candidate_head is None:
                raise ValueError(
                    "Checkpoint has no candidate head. "
                    "Train LEAGLE with --candidate-head."
                )
            selected = self.config.candidate_head_top_k if top_k is None else top_k
            if not 1 <= selected <= self.draft_vocab_size:
                raise ValueError("Candidate top_k must be in [1, draft_vocab_size]")
            self.candidate_head.validate_weights()
            self.candidate_top_k = selected
        elif top_k is not None:
            raise ValueError("Candidate top_k requires an enabled candidate head")
        self.candidate_head_enabled = enabled

    def _prepare_missing_checkpoint_weights(self, loading_info):
        if self.candidate_head is None:
            if any(
                key.startswith("candidate_head.")
                for key in loading_info.get("unexpected_keys", [])
            ):
                raise ValueError(
                    "Candidate head weights require candidate_head=true in config"
                )
            return
        missing = loading_info.get("missing_keys", [])
        mismatched = loading_info.get("mismatched_keys", [])
        invalid = [
            *missing,
            *(key[0] if isinstance(key, tuple) else key for key in mismatched),
        ]
        if any(key.startswith("candidate_head.") for key in invalid):
            raise ValueError("Checkpoint is missing compatible candidate head weights")
        self.candidate_head.validate_weights()
        self.candidate_head.requires_grad_(True)

    def gradient_clip_groups(self):
        """Clip the auxiliary head separately so it cannot rescale backbone updates."""
        if self.candidate_head is None:
            return [list(self.parameters())]
        backbone = [
            p
            for name, p in self.named_parameters()
            if not name.startswith("candidate_head.")
        ]
        return [backbone, list(self.candidate_head.parameters())]

    def set_activation_checkpointing(self, enabled: bool, *, layer_ids=None):
        selected = None if layer_ids is None else frozenset(layer_ids)
        if selected is not None and any(
            i < 0 or i >= len(self.layers) for i in selected
        ):
            raise ValueError("Checkpoint layer IDs must index physical decoder layers")
        self.activation_checkpointing = enabled
        self.activation_checkpoint_layers = selected

    def _create_mask(self, mask_mod, query_length, kv_length, device):
        factory = (
            create_block_mask
            if self._attn_impl == "simple_flex_attention"
            else create_float_mask
            if self._attn_impl == "eager"
            else create_mask
        )
        return factory(
            mask_mod,
            B=1,
            H=None,
            Q_LEN=query_length,
            KV_LEN=kv_length,
            device=device,
        )

    def _step(
        self,
        token_ids,
        state,
        context,
        context_positions,
        query_positions,
        masks,
        caches,
        context_kv=None,
        use_candidate_head=False,
        training_context_cache=None,
        compute_logits=True,
        selective_checkpoint=False,
    ):
        embeddings = self.embed_tokens(token_ids)
        hidden = self.input_fc(
            torch.cat(
                [
                    self.embed_norm(embeddings),
                    self.state_norm(state),
                ],
                dim=-1,
            )
        )
        positions = (
            torch.cat([context_positions, query_positions], dim=-1)
            if context_kv is None
            else query_positions
        )
        position_embeddings = self.rotary_emb(hidden, positions)
        next_caches: list[DraftKV | StaticDraftKV] = []
        for loop in range(self.config.num_loops):
            for index, layer in enumerate(self.layers):
                logical_depth = loop * len(self.layers) + index
                mask = masks[layer.self_attn.sliding_window]
                layer_args = (
                    context,
                    hidden,
                    position_embeddings,
                    mask,
                    caches[logical_depth],
                )
                if context_kv is not None:
                    # Target context and projection weights are shared by loops;
                    # only the synthetic draft KV remains per logical depth.
                    layer_args = (*layer_args, context_kv[index])
                elif training_context_cache is not None:
                    layer_args = (*layer_args, training_context_cache[index], True)
                recompute = (
                    self.training
                    and self.activation_checkpointing
                    and (
                        self.activation_checkpoint_layers is None
                        or index in self.activation_checkpoint_layers
                    )
                )
                if recompute and selective_checkpoint:
                    # Attention already checkpoints each compact query chunk.
                    # Keep its Q/K/V graph and recompute only the MLP here.
                    result = layer(*layer_args, checkpoint_mlp=True)
                elif recompute:
                    result = checkpoint(
                        layer,
                        *layer_args,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                else:
                    result = layer(*layer_args)
                hidden, present = result[:2]
                if training_context_cache is not None:
                    training_context_cache[index] = result[2]
                next_caches.append(present)
        normalized = self.norm(hidden)
        logits = (
            self._output_logits(normalized, use_candidate_head)
            if compute_logits
            else None
        )
        state = normalized if self.config.norm_output else hidden
        return logits, state, next_caches, normalized

    def _output_logits(self, normalized, use_candidate_head):
        if use_candidate_head:
            return self.candidate_head(
                normalized,
                self.lm_head.weight,
                self.candidate_top_k,
                sparse=self.sparse_candidates,
            )
        return self.lm_head(normalized)

    def _target_token_ids(self, draft_ids):
        # d2t stores OFFSETS, not absolute verifier IDs.
        return draft_ids + self.d2t[draft_ids] if self.use_draft_vocab else draft_ids

    def _training_positions(self, hidden_states, input_ids, document_ids, position_ids):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:  # noqa: PLR2004
            raise ValueError("LEAGLE forward expects one nonempty packed sequence")
        length = input_ids.shape[1]
        if hidden_states.shape != (1, length, self.fc.in_features):
            raise ValueError(
                "Target features must align with the shifted token sequence"
            )
        if document_ids.shape != input_ids.shape:
            raise ValueError("document_ids must match input_ids")
        if position_ids is None:
            indices = torch.arange(length, device=input_ids.device)
            docs = document_ids[0]
            starts = torch.where(docs != docs.roll(1), indices, 0)
            position_ids = (indices - starts.cummax(0).values + 1).unsqueeze(0)
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must match input_ids")
        return position_ids

    def _training_masks(
        self,
        document_ids,
        positions,
        step,
        windows,
        chunk_size,
        mask_cache=None,
    ):
        length = document_ids.shape[1]
        return {
            window: TrajectoryAttention(
                document_ids[0],
                positions[0],
                step,
                window,
                chunk_size,
                mask_cache,
            )
            if chunk_size
            else self._create_mask(
                trajectory_mask(document_ids[0], positions[0], step, window),
                length,
                length * (step + 2),
                document_ids.device,
            )
            for window in windows
        }

    def _validate_execution(
        self,
        reuse_kv,
        sparse_attention,
        candidate_addmm,
        cache_mb,
        *,
        candidate_recompute=False,
        cache_indices=False,
    ):
        if candidate_addmm and self.candidate_head is None:
            raise ValueError(
                "Candidate gradient addmm requires a trained candidate head"
            )
        if candidate_recompute and self.candidate_head is None:
            raise ValueError(
                "Candidate backward recomputation requires a candidate head"
            )
        if cache_mb is not None and cache_mb < 1:
            raise ValueError("Teacher cache budget must be positive")
        if cache_indices and cache_mb is None:
            raise ValueError("Teacher index plans require --leagle-teacher-cache")
        if reuse_kv:
            validate_kv_reuse_rope(self.rotary_emb)
        if sparse_attention and self._attn_impl != "sdpa":
            raise ValueError("LEAGLE sparse training attention requires SDPA")

    def _resolve_memory_options(
        self,
        chunk_size,
        align_candidates,
        selective_checkpoint,
        cache_masks,
        sparse_attention,
    ):
        if align_candidates:
            if self.candidate_head is None:
                raise ValueError("Candidate chunk alignment requires a candidate head")
            if chunk_size < 1:
                raise ValueError("LEAGLE optimization chunk sizes must be positive")
            chunk_size = max(chunk_size, self.config.candidate_head_chunk_size)
        if (selective_checkpoint or cache_masks) and not sparse_attention:
            raise ValueError(
                "Selective checkpoint and mask cache require sparse attention"
            )
        if selective_checkpoint and not self.activation_checkpointing:
            raise ValueError("Selective checkpoint requires activation checkpointing")
        return chunk_size

    def forward(
        self,
        hidden_states,
        input_ids,
        document_ids,
        loss_mask=None,
        position_ids=None,
        verifier_last_hidden_states=None,
        ttt_steps=7,
        ttt_step_loss_decay=1.0,
        loss_config=None,
        teacher_forcing=None,
        collect_metrics=True,
        leagle_reuse_training_kv=TRAINING_DEFAULTS["leagle_reuse_training_kv"],
        leagle_chunked_loss=TRAINING_DEFAULTS["leagle_chunked_loss"],
        leagle_loss_chunk_size=TRAINING_DEFAULTS["leagle_loss_chunk_size"],
        leagle_sparse_attention=TRAINING_DEFAULTS["leagle_sparse_attention"],
        leagle_attn_chunk_size=TRAINING_DEFAULTS["leagle_attn_chunk_size"],
        leagle_candidate_grad_addmm=TRAINING_DEFAULTS["leagle_candidate_grad_addmm"],
        leagle_candidate_backward_recompute=TRAINING_DEFAULTS[
            "leagle_candidate_backward_recompute"
        ],
        leagle_teacher_cache=TRAINING_DEFAULTS["leagle_teacher_cache"],
        leagle_cache_frozen_head_weights=TRAINING_DEFAULTS[
            "leagle_cache_frozen_head_weights"
        ],
        leagle_cache_teacher_indices=TRAINING_DEFAULTS["leagle_cache_teacher_indices"],
        leagle_teacher_cache_mb=TRAINING_DEFAULTS["leagle_teacher_cache_mb"],
        leagle_masked_head=TRAINING_DEFAULTS["leagle_masked_head"],
        leagle_teacher_cache_supervised=TRAINING_DEFAULTS[
            "leagle_teacher_cache_supervised"
        ],
        leagle_shared_loss_normalization=TRAINING_DEFAULTS[
            "leagle_shared_loss_normalization"
        ],
        leagle_stream_main_loss=TRAINING_DEFAULTS["leagle_stream_main_loss"],
        leagle_align_candidate_chunks=TRAINING_DEFAULTS[
            "leagle_align_candidate_chunks"
        ],
        leagle_selective_checkpoint=TRAINING_DEFAULTS["leagle_selective_checkpoint"],
        leagle_cache_attention_masks=TRAINING_DEFAULTS["leagle_cache_attention_masks"],
        leagle_attention_mask_cache_mb=TRAINING_DEFAULTS[
            "leagle_attention_mask_cache_mb"
        ],
        return_predictions=True,
        **_kwargs,
    ):
        """Run one shared TTT recurrence with optional training execution paths.

        With teacher labels, default to ground-truth-token teacher forcing.
        Without labels, return greedy autoregressive predictions only.
        collect_metrics=False skips diagnostics without changing the objective.
        """
        if ttt_steps < 1:
            raise ValueError("ttt_steps must be positive")
        return_loss = verifier_last_hidden_states is not None
        chunked_head = (
            leagle_chunked_loss
            or leagle_teacher_cache
            or leagle_masked_head
            or leagle_shared_loss_normalization
            or leagle_stream_main_loss
            or leagle_align_candidate_chunks
        )
        leagle_loss_chunk_size = self._resolve_memory_options(
            leagle_loss_chunk_size,
            leagle_align_candidate_chunks,
            leagle_selective_checkpoint,
            leagle_cache_attention_masks,
            leagle_sparse_attention,
        )
        if leagle_teacher_cache_supervised and not leagle_teacher_cache:
            raise ValueError("Supervised teacher cache requires --leagle-teacher-cache")
        if (
            leagle_reuse_training_kv
            or chunked_head
            or leagle_sparse_attention
            or leagle_candidate_grad_addmm
            or leagle_candidate_backward_recompute
        ):
            if min(leagle_loss_chunk_size, leagle_attn_chunk_size) < 1:
                raise ValueError("LEAGLE optimization chunk sizes must be positive")
            if not return_loss:
                raise ValueError(
                    "LEAGLE training optimizations require teacher hidden states"
                )
        self._validate_execution(
            leagle_reuse_training_kv,
            leagle_sparse_attention,
            leagle_candidate_grad_addmm,
            leagle_teacher_cache_mb if leagle_teacher_cache else None,
            candidate_recompute=leagle_candidate_backward_recompute,
            cache_indices=leagle_cache_teacher_indices,
        )
        positions = self._training_positions(
            hidden_states, input_ids, document_ids, position_ids
        )
        teacher_forcing = return_loss if teacher_forcing is None else teacher_forcing
        context = self.hidden_norm(self.fc(hidden_states))
        mask_cache = (
            TrajectoryMaskCache(leagle_attention_mask_cache_mb * 1024 * 1024)
            if leagle_cache_attention_masks
            else None
        )
        objective = (
            TrainingObjective(
                self,
                verifier_last_hidden_states,
                document_ids,
                loss_mask,
                ttt_steps,
                ttt_step_loss_decay,
                loss_config,
                collect_metrics,
                leagle_loss_chunk_size if chunked_head else 0,
                context.new_zeros(()),
                grad_addmm=leagle_candidate_grad_addmm,
                candidate_backward_recompute=leagle_candidate_backward_recompute,
                teacher_cache_mb=leagle_teacher_cache_mb if leagle_teacher_cache else 0,
                cache_frozen_head_weights=leagle_cache_frozen_head_weights,
                cache_teacher_indices=leagle_cache_teacher_indices,
                masked_head=leagle_masked_head,
                need_predictions=return_predictions or not teacher_forcing,
                cache_supervised=leagle_teacher_cache_supervised,
                shared_normalization=leagle_shared_loss_normalization,
                stream_main_loss=leagle_stream_main_loss,
            )
            if return_loss
            else None
        )
        state, tokens = context, input_ids
        predictions = []
        caches = [None] * self.config.effective_depth
        context_cache = [None] * len(self.layers) if leagle_reuse_training_kv else None
        windows = {layer.self_attn.sliding_window for layer in self.layers}
        for step in range(ttt_steps):
            masks = self._training_masks(
                document_ids,
                positions,
                step,
                windows,
                leagle_attn_chunk_size if leagle_sparse_attention else 0,
                mask_cache,
            )
            logits, state, caches, normalized = self._step(
                tokens,
                state,
                context,
                positions - 1,
                positions + step,
                masks,
                caches,
                training_context_cache=context_cache,
                compute_logits=not chunked_head,
                selective_checkpoint=leagle_selective_checkpoint,
            )
            draft_ids = (
                objective.step(logits, normalized, step)
                if objective is not None
                else logits.argmax(-1)
            )
            if draft_ids is not None:
                tokens = self._target_token_ids(draft_ids)
                if return_predictions:
                    predictions.append(tokens.detach())
            if teacher_forcing:
                offset = min(step + 1, input_ids.shape[1])
                tokens = torch.cat(
                    [input_ids[:, offset:], input_ids.new_zeros(1, offset)], dim=-1
                )
        return objective.finish(predictions) if objective is not None else predictions

    def _proposal_caches(self, length, num_tokens, prepared_context, rollout_buffers):
        if self.reuse_rollout_buffers and rollout_buffers is None:
            raise ValueError(
                "Rollout buffer reuse requires a request-local cached proposer"
            )
        if rollout_buffers is not None:
            if (
                not self.reuse_rollout_buffers
                or prepared_context is None
                or len(rollout_buffers) != self.config.effective_depth
                or any(
                    not isinstance(cache, ReusableDraftKV) or cache.length != length
                    for cache in rollout_buffers
                )
            ):
                raise ValueError(
                    "Rollout buffers must match the verified prefix and logical depth"
                )
            return rollout_buffers
        return (
            [
                StaticDraftKV(length + num_tokens)
                for _ in range(self.config.effective_depth)
            ]
            if self.static_kv
            else [None] * self.config.effective_depth
        )

    @staticmethod
    def _proposal_positions(hidden_states, anchor_token, position_ids, num_tokens):
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive")
        if (
            hidden_states.ndim != 3  # noqa: PLR2004
            or hidden_states.shape[0] != 1
            or hidden_states.shape[1] < 1
        ):
            raise ValueError("propose requires one nonempty target prefix")
        if anchor_token.shape != (1, 1):
            raise ValueError("anchor_token must have shape [1, 1]")
        length = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden_states.device).unsqueeze(
                0
            )
        if position_ids.shape != (1, length):
            raise ValueError("position_ids must match the target prefix")
        return position_ids

    @torch.no_grad()
    def propose(
        self,
        hidden_states,
        anchor_token,
        *,
        position_ids=None,
        num_tokens=7,
        temperature=0.0,
        prepared_context=None,
        rollout_buffers=None,
        return_logits=True,
    ):
        """Greedy or sampled proposals for one prefix, without teacher forcing.

        hidden_states: [1, context_length, auxiliary_count * hidden_size], up to
        the last verified token. anchor_token: [1, 1], the next known/bonus token.
        Returns verifier-vocabulary IDs [1, num_tokens] and draft-vocabulary
        logits [1, num_tokens, draft_vocab_size]. Draft trajectory caches are
        local to this round. An inference caller may supply prepared_context as
        (last fused target state, per-physical-layer target KV) to reuse confirmed
        context across rounds. It must cover exactly the supplied verified prefix.
        """
        validate_temperature(temperature)
        if not return_logits and temperature > 0:
            raise ValueError("Sampled proposals must retain logits for verification")
        if self.training:
            raise ValueError("Call model.eval() before proposing tokens")
        position_ids = self._proposal_positions(
            hidden_states, anchor_token, position_ids, num_tokens
        )
        length = hidden_states.shape[1]
        if prepared_context is None:
            context = self.hidden_norm(self.fc(hidden_states))
            context_kv = None
        else:
            context, context_kv = prepared_context
        state = context[:, -1:]
        caches = self._proposal_caches(
            length, num_tokens, prepared_context, rollout_buffers
        )
        windows = {layer.self_attn.sliding_window for layer in self.layers}
        tokens = anchor_token
        proposals, all_logits = [], []
        skip_full_mask = self.static_kv and bool(
            (position_ids <= position_ids[:, -1:] + 1).all()
        )
        for step in range(num_tokens):
            query_positions = position_ids[:, -1:] + 1 + step
            masks = {
                window: None
                if skip_full_mask and window is None
                else self._create_mask(
                    rollout_mask(position_ids[0], query_positions[0, 0], step, window),
                    1,
                    length + step + 1,
                    hidden_states.device,
                )
                for window in windows
            }
            logits, state, caches, _ = self._step(
                tokens,
                state,
                context,
                position_ids,
                query_positions,
                masks,
                caches,
                context_kv=context_kv,
                use_candidate_head=self.candidate_head_enabled,
            )
            tokens, logits = self._sample_proposal(logits, temperature)
            proposals.append(tokens)
            if return_logits:
                all_logits.append(logits)
        scores = (
            None
            if not return_logits
            else SparseCandidateLogits.concatenate(all_logits)
            if isinstance(all_logits[0], SparseCandidateLogits)
            else torch.cat(all_logits, dim=1)
        )
        return torch.cat(proposals, dim=1), scores

    def _sample_proposal(self, logits, temperature):
        if isinstance(logits, SparseCandidateLogits):
            tokens = self._target_token_ids(logits.sample(temperature))
            logits = SparseCandidateLogits(
                self._target_token_ids(logits.indices),
                logits.values,
                self.config.target_vocab_size,
            )
            return tokens, logits
        return self._target_token_ids(sample_from_logits(logits, temperature)), logits

    @classmethod
    def from_training_args(cls, verifier_config, t2d=None, d2t=None, **kwargs):
        target_layers = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"),
            kwargs["verifier_name_or_path"],
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )
        config = LEagleSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            aux_hidden_state_layer_ids=target_layers,
            num_loops=kwargs.get("num_loops", 1),
            norm_output=kwargs.get("norm_output", True),
            candidate_head=kwargs.get("candidate_head", False),
            candidate_head_rank=kwargs.get("candidate_head_rank", 256),
            candidate_head_top_k=kwargs.get("candidate_head_top_k", 128),
            candidate_head_loss_weight=kwargs.get("candidate_head_loss_weight", 1.0),
            candidate_head_chunk_size=kwargs.get("candidate_head_chunk_size", 32),
            speculators_config=SpeculatorsConfig(
                algorithm="leagle",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs.get("ttt_steps", 7),
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(
                    kwargs["verifier_name_or_path"]
                ),
            ),
        )
        model = cls(config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs):
        kwargs.update(resolve_optimization_groups(kwargs))
        if (
            str(kwargs.get("hidden_states_dtype", "bfloat16")).removeprefix("torch.")
            != "float32"
        ):
            for flag, label in (
                ("leagle_stream_main_loss", "streaming main loss"),
                ("leagle_shared_loss_normalization", "shared loss normalization"),
            ):
                if kwargs.get(flag, False):
                    logger.warning(
                        "LEAGLE %s is FP32-only. Low-precision heads use the "
                        "reference checkpointed chunks to preserve loss scaling "
                        "and native loss precision.",
                        label,
                    )
        call_kwargs = {
            "ttt_steps": kwargs.get("ttt_steps", 7),
            "ttt_step_loss_decay": kwargs.get("ttt_step_loss_decay", 1.0),
            "loss_config": resolve_training_loss(**kwargs),
            "teacher_forcing": True,
            "collect_metrics": True,
            # Predictions are unused by the trainer, even on metric steps.
            "return_predictions": False,
            **{
                name: kwargs.get(name, default)
                for name, default in TRAINING_DEFAULTS.items()
            },
        }
        if call_kwargs["leagle_candidate_backward_recompute"]:
            logger.info(
                "Candidate backward recomputation enabled: extra teacher/candidate "
                "projections, native loss scaling and FP32 step accumulation. "
                "Candidate addmm and masked-row compaction are overridden."
            )
        if (
            kwargs.get("candidate_head", False)
            and not call_kwargs["leagle_candidate_backward_recompute"]
            and not call_kwargs["leagle_align_candidate_chunks"]
            and any(
                call_kwargs[name]
                for name in (
                    "leagle_chunked_loss",
                    "leagle_teacher_cache",
                    "leagle_masked_head",
                    "leagle_shared_loss_normalization",
                    "leagle_stream_main_loss",
                )
            )
            and kwargs.get("candidate_head_chunk_size", 32)
            > call_kwargs["leagle_loss_chunk_size"]
        ):
            logger.warning(
                "Candidate chunks are capped by leagle_loss_chunk_size=%s. "
                "Use --leagle-align-candidate-chunks to enlarge the main chunks; "
                "this increases temporary device memory.",
                call_kwargs["leagle_loss_chunk_size"],
            )
        return call_kwargs, dict(call_kwargs)
