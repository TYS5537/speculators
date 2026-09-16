"""Experimental, eager-only HS exporter for vLLM Ascend 0.26.0rc1.

Uses a separate architecture so Qwen and the native V4 serving path are untouched.
"""

import logging
from importlib.metadata import version

import torch
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM

from speculators_dsv4.block_protocol import BLOCK_CONNECTOR
from speculators_dsv4.contract import (
    replace_teacher_hidden,
    validate_config,
    validate_layers,
)

logger = logging.getLogger(__name__)


class SpeculatorsDeepseekV4ForCausalLM(AscendDeepseekV4ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):  # noqa: C901
        for package, expected in (("vllm", "0.26.0"), ("vllm-ascend", "0.26.0rc1")):
            actual = version(package).split("+")[0]
            if actual != expected:
                raise RuntimeError(
                    f"DSV4 HS bridge requires {package}=={expected}, got {actual}"
                )
        model = vllm_config.model_config
        validate_config(model.hf_config.to_dict())
        if not model.enforce_eager or model.dtype != torch.bfloat16:
            raise ValueError(
                "DSV4 HS export requires --enforce-eager --dtype bfloat16."
            )
        # Quantization is resolved by the native target backend. BF16 describes
        # the exported activations, not the storage format of target weights.
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Disable prefix caching so every input token exports HS.")
        if vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError("Disable chunked prefill for the initial DSV4 HS bridge.")
        speculation = vllm_config.speculative_config
        if speculation is None or speculation.method != "extract_hidden_states":
            raise ValueError(
                "This architecture is for extract_hidden_states only, "
                "not DSpark serving."
            )
        parallel = vllm_config.parallel_config
        for name in (
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(parallel, name, 1) != 1:
                raise ValueError(
                    f"Initial DSV4 HS bridge requires {name}=1; TP/EP are allowed."
                )
        ascend = get_ascend_config()
        if getattr(ascend, "enable_flashcomm1", False) or getattr(
            ascend, "enable_dsa_cp", False
        ):
            raise ValueError("Disable FlashComm1 and DSA-CP for DSV4 HS export.")
        if vllm_config.compilation_config.pass_config.enable_sp:
            raise ValueError("Disable sequence parallelism for DSV4 HS export.")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._block_verify = (
            getattr(
                getattr(vllm_config, "kv_transfer_config", None), "kv_connector", None
            )
            == BLOCK_CONNECTOR
        )
        self._teacher_pre_norm = None
        self._export_count = None
        self.model.norm.register_forward_pre_hook(self._capture_teacher)
        logger.warning(
            "Experimental DSV4 HS bridge active: "
            "auxiliary mean, teacher hc_head -> pre-norm."
        )

    def _capture_teacher(self, _module, inputs):
        # Clone before RMSNorm so any backend in-place implementation is harmless.
        self._teacher_pre_norm = inputs[0].detach().clone()

    def set_aux_hidden_state_layers(self, layers):
        if not layers or layers[-1] != self.config.num_hidden_layers:
            raise ValueError(
                "Append HS slot 43 for the teacher, after the auxiliary IDs."
            )
        validate_layers(list(layers[:-1]))
        self._export_count = len(layers)
        super().set_aux_hidden_state_layers(layers)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        if self._export_count is None:
            raise RuntimeError("DSV4 auxiliary HS IDs were not configured.")
        self._teacher_pre_norm = None
        try:
            output = super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )
            output = replace_teacher_hidden(
                output, self._teacher_pre_norm, self._export_count
            )
            if self._block_verify:
                from speculators_dsv4.block_connector import (  # noqa: PLC0415
                    export_block,
                )

                export_block(self, input_ids, positions, output)
            return output
        finally:
            self._teacher_pre_norm = None
