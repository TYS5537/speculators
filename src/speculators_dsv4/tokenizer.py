"""Use the running target's DeepSeek V4 renderer, not checkpoint Python code."""

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any


class DSV4ServerTokenizer:
    """Small tokenizer adapter for the offline evaluator's single-prompt path.

    ``client`` must target the server root, not its ``/v1`` API prefix: vLLM
    0.26.0 exposes ``/tokenize`` at the root. The target must have been started
    with ``--tokenizer-mode deepseek_v4``. Rendered prompts deliberately remain
    token IDs even when the caller requests ``tokenize=False``; decoding and
    re-encoding them would risk changing special tokens or duplicating BOS.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        eos_token_id: int | list[int] | None,
        *,
        vocab_size: int | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("DSV4 tokenizer requires the served model name")
        if vocab_size is not None and (type(vocab_size) is not int or vocab_size <= 0):
            raise ValueError("vocab_size must be a positive integer")
        self.client = client
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        if eos_token_id is not None:
            self._validate_ids(
                eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
            )

    def _validate_ids(self, value: Any) -> list[int]:
        if not isinstance(value, list) or not value:
            raise ValueError("DSV4 tokenization requires a non-empty token ID list")
        if any(
            type(token_id) is not int
            or token_id < 0
            or token_id > 2**63 - 1
            or (self.vocab_size is not None and token_id >= self.vocab_size)
            for token_id in value
        ):
            raise ValueError("DSV4 tokenization returned an invalid token ID")
        return list(value)

    def _tokenize(self, body: dict[str, Any]) -> list[int]:
        response = self.client.post("/tokenize", cast_to=dict, body=body)
        if not isinstance(response, Mapping):
            raise ValueError("DSV4 /tokenize did not return a JSON object")
        token_ids = self._validate_ids(response.get("tokens"))
        if "count" in response and (
            type(response["count"]) is not int or response["count"] != len(token_ids)
        ):
            raise ValueError("DSV4 /tokenize count does not match its token IDs")
        return token_ids

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> list[int]:
        """Render on the target and preserve its exact token IDs for evaluation."""
        if type(tokenize) is not bool or type(add_generation_prompt) is not bool:
            raise ValueError("tokenize and add_generation_prompt must be booleans")
        if not isinstance(messages, list) or not messages:
            raise ValueError("DSV4 chat rendering requires a non-empty message list")
        if "chat_template" in kwargs:
            raise ValueError("DSV4 evaluation must use the target's own renderer")
        body = {
            "model": self.model_name,
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "add_special_tokens": False,
        }
        for key in ("tools", "continue_final_message"):
            if key in kwargs:
                body[key] = kwargs.pop(key)
        if body.get("continue_final_message") and add_generation_prompt:
            raise ValueError("Cannot continue a message and add a generation prompt")
        template_kwargs = kwargs.pop("chat_template_kwargs", {})
        if not isinstance(template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be a dictionary")
        # vLLM's V4 tokenizer maps enable_thinking/thinking to thinking_mode.
        # Forward these unchanged so the server owns that versioned behavior.
        body["chat_template_kwargs"] = {**template_kwargs, **kwargs}
        return self._tokenize(body)

    def __call__(
        self,
        prompt: str | list[int],
        return_tensors: str = "pt",
    ) -> SimpleNamespace:
        if return_tensors != "pt":
            raise ValueError("DSV4 offline evaluation requires return_tensors='pt'")
        if isinstance(prompt, str):
            token_ids = self._tokenize(
                {
                    "model": self.model_name,
                    "prompt": prompt,
                    "add_special_tokens": False,
                }
            )
        else:
            token_ids = self._validate_ids(prompt)

        import torch  # noqa: PLC0415 -- Keep server-only rendering torch-independent.

        return SimpleNamespace(input_ids=torch.tensor([token_ids], dtype=torch.long))
