"""EAGLE-style token/state recurrence with DFlash-style target KV injection."""

from typing import ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.attention import create_float_mask
from speculators.models.eagle3.metrics import EagleTrainingMetrics, compute_metrics
from speculators.models.leagle.attention import rollout_mask, trajectory_mask
from speculators.models.leagle.config import LEagleSpeculatorConfig
from speculators.models.leagle.model_definitions import DraftKV, LEagleDecoderLayer
from speculators.models.metrics import resolve_training_loss
from speculators.models.utils import flatten_rope_parameters, resolve_target_layer_ids
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.proposals.sampling import sample_from_logits, validate_temperature


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

    @property
    def target_layer_ids(self) -> list[int]:
        return self.config.aux_hidden_state_layer_ids

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
        positions = torch.cat([context_positions, query_positions], dim=-1)
        position_embeddings = self.rotary_emb(hidden, positions)
        next_caches: list[DraftKV] = []
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
                recompute = (
                    self.training
                    and self.activation_checkpointing
                    and (
                        self.activation_checkpoint_layers is None
                        or index in self.activation_checkpoint_layers
                    )
                )
                if recompute:
                    hidden, present = checkpoint(
                        layer,
                        *layer_args,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                else:
                    hidden, present = layer(*layer_args)
                next_caches.append(present)
        normalized = self.norm(hidden)
        logits = self.lm_head(normalized)
        state = normalized if self.config.norm_output else hidden
        return logits, state, next_caches

    def _target_token_ids(self, logits):
        draft_ids = logits.argmax(dim=-1)
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
        **_kwargs,
    ):
        """Train with shifted EAGLE batches; optionally roll out sampled tokens.

        With teacher labels, default to EAGLE's ground-truth-token TTT training.
        Without labels, default to genuine greedy autoregressive rollout.
        Returned tokens always use the verifier vocabulary.
        """
        if ttt_steps < 1:
            raise ValueError("ttt_steps must be positive")
        position_ids = self._training_positions(
            hidden_states,
            input_ids,
            document_ids,
            position_ids,
        )
        length = input_ids.shape[1]
        return_loss = verifier_last_hidden_states is not None
        if teacher_forcing is None:
            teacher_forcing = return_loss
        context = self.hidden_norm(self.fc(hidden_states))
        state = context
        caches: list[DraftKV | None] = [None] * self.config.effective_depth
        windows = {layer.self_attn.sliding_window for layer in self.layers}
        tokens = input_ids
        predictions = []
        loss = context.new_zeros(())
        if return_loss:
            with torch.no_grad():
                targets = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )
            valid_mask = document_ids >= 0
            if loss_mask is not None:
                valid_mask = valid_mask & loss_mask.bool()
            prev_correct = valid_mask.clone()
            metrics = EagleTrainingMetrics(document_ids, loss_mask, ttt_steps)
        for step in range(ttt_steps):
            masks = {
                window: self._create_mask(
                    trajectory_mask(document_ids[0], position_ids[0], step, window),
                    length,
                    length * (step + 2),
                    input_ids.device,
                )
                for window in windows
            }
            logits, state, caches = self._step(
                tokens,
                state,
                context,
                position_ids - 1,
                position_ids + step,
                masks,
                caches,
            )
            predicted = self._target_token_ids(logits)
            predictions.append(predicted.detach())
            if return_loss and step < length:
                metrics.update(logits, targets, step)
                step_mask = valid_mask.clone()
                if step:
                    # No supervision across packed-document boundaries. The
                    # teacher at q+step is valid only within q's original document.
                    step_mask[:, step:] &= (
                        document_ids[:, step:] == document_ids[:, :-step]
                    )
                prev_correct[:, : length - step] &= step_mask[:, step:]
                step_loss, _ = compute_metrics(
                    logits,
                    targets,
                    step_mask,
                    prev_correct,
                    step,
                    ttt_step_loss_decay,
                    loss_config=loss_config,
                )
                loss = loss + step_loss
            if teacher_forcing:
                offset = min(step + 1, length)
                tokens = torch.cat(
                    [
                        input_ids[:, offset:],
                        input_ids.new_zeros(1, offset),
                    ],
                    dim=-1,
                )
            else:
                tokens = predicted
        if return_loss:
            return predictions, loss, metrics.finish(loss)
        return predictions

    @torch.no_grad()
    def propose(
        self,
        hidden_states,
        anchor_token,
        *,
        position_ids=None,
        num_tokens=7,
        temperature=0.0,
    ):
        """Greedy or sampled proposals for one prefix, without teacher forcing.

        hidden_states: [1, context_length, auxiliary_count * hidden_size], up to
        the last verified token. anchor_token: [1, 1], the next known/bonus token.
        Returns verifier-vocabulary IDs [1, num_tokens] and draft-vocabulary
        logits [1, num_tokens, draft_vocab_size]. Caches are local to this round;
        call again with the newly verified prefix after acceptance/rejection.
        """
        validate_temperature(temperature)
        if self.training:
            raise ValueError("Call model.eval() before proposing tokens")
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
        context = self.hidden_norm(self.fc(hidden_states))
        state = context[:, -1:]
        caches: list[DraftKV | None] = [None] * self.config.effective_depth
        windows = {layer.self_attn.sliding_window for layer in self.layers}
        tokens = anchor_token
        proposals, all_logits = [], []
        for step in range(num_tokens):
            query_positions = position_ids[:, -1:] + 1 + step
            masks = {
                window: self._create_mask(
                    rollout_mask(position_ids[0], query_positions[0, 0], step, window),
                    1,
                    length + step + 1,
                    hidden_states.device,
                )
                for window in windows
            }
            logits, state, caches = self._step(
                tokens,
                state,
                context,
                position_ids,
                query_positions,
                masks,
                caches,
            )
            tokens = sample_from_logits(logits, temperature)
            if self.use_draft_vocab:
                tokens = tokens + self.d2t[tokens]
            proposals.append(tokens)
            all_logits.append(logits)
        return torch.cat(proposals, dim=1), torch.cat(all_logits, dim=1)

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
        call_kwargs = {
            "ttt_steps": kwargs.get("ttt_steps", 7),
            "ttt_step_loss_decay": kwargs.get("ttt_step_loss_decay", 1.0),
            "loss_config": resolve_training_loss(**kwargs),
            "teacher_forcing": True,
        }
        return call_kwargs, dict(call_kwargs)
