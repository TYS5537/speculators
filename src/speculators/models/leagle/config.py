"""LEAGLE keeps physical depth, recurrence, and proposal length independent."""

from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator, model_validator
from transformers import PretrainedConfig, Qwen3Config

from speculators.config import SpeculatorModelConfig


@SpeculatorModelConfig.register("leagle")
class LEagleSpeculatorConfig(SpeculatorModelConfig):
    speculators_model_type: Literal["leagle"] = "leagle"
    architectures: list[str] = Field(default_factory=lambda: ["LEagleDraftModel"])
    transformer_layer_config: PretrainedConfig = Field(default_factory=Qwen3Config)
    draft_vocab_size: int = Field(default=32000, gt=0)
    aux_hidden_state_layer_ids: list[int] | None = None
    num_loops: int = Field(
        default=1,
        ge=1,
        description="Visits to the same decoder stack per token; weights are shared.",
    )
    norm_output: bool = Field(
        default=True,
        description="Feed normalized hidden states into the next token step.",
    )

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        if isinstance(value, dict):
            if value.get("model_type", "qwen3") != "qwen3":
                raise ValueError("LEAGLE currently requires a Qwen3 draft decoder")
            return Qwen3Config(**value)
        return value

    @model_validator(mode="after")
    def validate_architecture(self) -> "LEagleSpeculatorConfig":
        decoder = self.transformer_layer_config
        if decoder.model_type != "qwen3":
            raise ValueError("LEAGLE currently requires a Qwen3 draft decoder")
        if decoder.num_hidden_layers < 1:
            raise ValueError("LEAGLE needs at least one physical decoder layer")
        if self.draft_vocab_size > decoder.vocab_size:
            raise ValueError("Draft vocabulary cannot exceed the verifier vocabulary")
        layers = self.aux_hidden_state_layer_ids
        if layers is not None and (
            not layers or min(layers) < 0 or len(set(layers)) != len(layers)
        ):
            raise ValueError("Auxiliary layer IDs must be nonempty, distinct and >= 0")
        return self

    @property
    def target_vocab_size(self) -> int:
        return self.transformer_layer_config.vocab_size

    @property
    def effective_depth(self) -> int:
        return self.transformer_layer_config.num_hidden_layers * self.num_loops
