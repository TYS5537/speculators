"""Compare the trainer's frozen teacher head with native DSV4 probabilities.

This is a numerical integration check, not an acceptance evaluator. It loads only
the audited final RMSNorm and full-vocabulary head, never remote Python or a
target/draft model. Reference mode needs a training HS server started with the
evaluation/raw-full-logprob options; block mode needs the block export service.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from speculators_dsv4.contract import inspect_checkpoint

_MATRIX_DIMS = 2
_LOGPROB_POSITIVE_TOLERANCE = 1e-6
_TEACHER_HS_ID = 43


@dataclass(frozen=True)
class ParityThresholds:
    """Conservative smoke-test defaults, NOT universal BF16 error guarantees."""

    max_tv: float = 0.02
    max_logprob_error: float = 0.5
    min_argmax_agreement: float = 1.0

    def validate(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_tv > 1 or self.min_argmax_agreement > 1:
            raise ValueError("TV and argmax agreement thresholds must be in [0, 1]")


@dataclass
class FrozenTeacherHead:
    """Match Qwen3RMSNorm + verifier_lm_head used by the current DSpark trainer."""

    norm_weight: object
    head_weight: object
    epsilon: float

    def logprobs(self, hidden):
        import torch  # noqa: PLC0415

        if (
            hidden.ndim != _MATRIX_DIMS
            or hidden.shape[1] != self.norm_weight.numel()
            or hidden.dtype != torch.bfloat16
            or not torch.isfinite(hidden).all()
        ):
            raise ValueError("Teacher HS must be finite BF16 [positions, hidden_size]")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Target rms_norm_eps must be finite and positive")
        with torch.inference_mode():
            # Qwen3RMSNorm casts back BEFORE multiplying its weight. Keeping this
            # BF16 rounding point is important; F.rms_norm need not match it.
            values = hidden.to(
                device=self.norm_weight.device, dtype=self.norm_weight.dtype
            )
            values_float = values.float()
            variance = values_float.square().mean(-1, keepdim=True)
            normalized = (values_float * torch.rsqrt(variance + self.epsilon)).to(
                values.dtype
            )
            normalized = self.norm_weight * normalized
            logits = torch.nn.functional.linear(
                normalized.to(self.head_weight.dtype), self.head_weight
            )
            if not torch.isfinite(logits).all():
                raise ValueError("Reconstructed teacher logits are not finite")
            return torch.log_softmax(logits.float(), dim=-1).cpu()


def load_teacher_head(report, *, device="cpu", dtype="bfloat16", norm_dtype="float32"):
    """Read exactly the two inspected floating IO keys using safetensors only."""
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    if dtype not in {"bfloat16", "float32"}:
        raise ValueError("Teacher head dtype must be bfloat16 or float32")
    if norm_dtype not in {"bfloat16", "float32"}:
        raise ValueError("Teacher norm dtype must be bfloat16 or float32")
    root = Path(report["model_path"]).resolve(strict=True)
    index_path = root / "model.safetensors.index.json"
    weight_map = (
        json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        if index_path.exists()
        else None
    )
    tensors = {}
    for logical in ("model.norm.weight", "lm_head.weight"):
        actual = report["io_keys"][logical]
        shard = root / (weight_map[actual] if weight_map else "model.safetensors")
        shard = shard.resolve(strict=True)
        if not shard.is_relative_to(root):
            raise ValueError("Teacher IO shard escapes the inspected checkpoint")
        with safe_open(str(shard), framework="pt", device="cpu") as source:
            tensor = source.get_tensor(actual)
        expected = (
            (report["config"]["hidden_size"],)
            if logical == "model.norm.weight"
            else (report["config"]["vocab_size"], report["config"]["hidden_size"])
        )
        if (
            tensor.dtype not in {torch.bfloat16, torch.float16, torch.float32}
            or tuple(tensor.shape) != expected
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"Invalid/non-floating/non-finite teacher IO: {actual}")
        # train.py's fresh/config-only initialization keeps parameters FP32;
        # its Qwen3RMSNorm remains FP32 under BF16 autocast. Only the following
        # Linear casts its input/weight to BF16. A BF16 saved-model initialization
        # can select norm_dtype explicitly instead of silently changing semantics.
        target_dtype = norm_dtype if logical == "model.norm.weight" else dtype
        tensors[logical] = tensor.to(device=device, dtype=getattr(torch, target_dtype))
    # Reject a checkpoint changed between the header audit and tensor loading.
    checked_again = inspect_checkpoint(root)
    if checked_again["checkpoint_signature"] != report["checkpoint_signature"]:
        raise ValueError("Target checkpoint changed while loading teacher IO")
    epsilon = report["config"].get("rms_norm_eps")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("Target config must declare a finite positive rms_norm_eps")
    return FrozenTeacherHead(
        norm_weight=tensors["model.norm.weight"],
        head_weight=tensors["lm_head.weight"],
        epsilon=float(epsilon),
    )


def select_positions(token_count, *, positions=None, tail_positions=4):
    """Positions are zero-based HS rows: row p predicts token p+1 (no shift)."""
    if type(token_count) is not int or token_count <= 0:
        raise ValueError("A nonempty token prefix is required")
    if positions is None:
        if type(tail_positions) is not int or tail_positions <= 0:
            raise ValueError("tail_positions must be a positive integer")
        return list(range(max(0, token_count - tail_positions), token_count))
    if (
        not positions
        or any(type(p) is not int or not 0 <= p < token_count for p in positions)
        or list(positions) != sorted(set(positions))
    ):
        raise ValueError("positions must be nonempty, unique, ascending valid HS rows")
    return list(positions)


def position_chunks(positions, chunk_size):
    """Only contiguous rows share a block, bounding every response to chunk_size."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("position_chunk_size must be a positive integer")
    chunk = []
    for position in positions:
        if chunk and (position != chunk[-1] + 1 or len(chunk) == chunk_size):
            yield chunk
            chunk = []
        chunk.append(position)
    if chunk:
        yield chunk


def compare_rows(native, reconstructed, positions, thresholds):
    """Compare distributions, so a harmless constant logit offset is irrelevant."""
    import torch  # noqa: PLC0415

    thresholds.validate()
    if (
        native.ndim != _MATRIX_DIMS
        or tuple(native.shape) != tuple(reconstructed.shape)
        or native.shape[0] != len(positions)
        or native.shape[1] == 0
        or not torch.isfinite(native).all()
        or not torch.isfinite(reconstructed).all()
    ):
        raise ValueError("Probability comparison requires matching finite logprob rows")
    native = native.double().cpu()
    reconstructed = reconstructed.double().cpu()
    for values in (native, reconstructed):
        if (values > _LOGPROB_POSITIVE_TOLERANCE).any() or not torch.isclose(
            values.exp().sum(-1),
            torch.ones(len(positions), dtype=torch.float64),
            rtol=1e-3,
            atol=1e-5,
        ).all():
            raise ValueError("Comparison requires normalized full-vocabulary logprobs")
    differences = (native - reconstructed).abs()
    native_prob, reconstructed_prob = native.exp(), reconstructed.exp()
    prob_differences = (native_prob - reconstructed_prob).abs()
    rows = []
    for index, position in enumerate(positions):
        row = {
            "position": position,
            "predicts_position": position + 1,
            "max_abs_logprob_error": differences[index].max().item(),
            "mean_abs_logprob_error": differences[index].mean().item(),
            "max_abs_probability_error": prob_differences[index].max().item(),
            "tv": (0.5 * prob_differences[index].sum()).item(),
            "kl_native_to_teacher": max(
                0.0,
                (native_prob[index] * (native[index] - reconstructed[index]))
                .sum()
                .item(),
            ),
            "native_argmax": native[index].argmax().item(),
            "teacher_argmax": reconstructed[index].argmax().item(),
        }
        row["argmax_match"] = row["native_argmax"] == row["teacher_argmax"]
        row["threshold_violations"] = [
            name
            for name, exceeded in (
                ("max_tv", row["tv"] > thresholds.max_tv),
                (
                    "max_logprob_error",
                    row["max_abs_logprob_error"] > thresholds.max_logprob_error,
                ),
            )
            if exceeded
        ]
        rows.append(row)
    return rows


def check_teacher_parity(
    target,
    teacher,
    input_ids,
    *,
    positions=None,
    tail_positions=4,
    position_chunk_size=4,
    thresholds=None,
):
    """Use the evaluator's strict transport validation, without any drafter."""
    import torch  # noqa: PLC0415

    thresholds = thresholds or ParityThresholds()
    thresholds.validate()
    positions = select_positions(
        len(input_ids), positions=positions, tail_positions=tail_positions
    )
    if any(
        type(token) is not int or not 0 <= token < target.vocab_size
        for token in input_ids
    ):
        raise ValueError(
            "Input token IDs must be integers inside the target vocabulary"
        )
    if len(input_ids) + 1 > target.max_model_len:
        raise ValueError(
            "Input prefix plus the API output token exceeds the context limit"
        )
    if target.packet_layer_ids[-1] != _TEACHER_HS_ID:
        raise ValueError(
            "DSV4 teacher must be the final post-hc_head/pre-norm HS slot 43"
        )
    if target.verification_mode not in {"reference", "block"}:
        raise ValueError("Teacher check requires explicit reference or block mode")
    teacher_slot = len(target.packet_layer_ids) - 1
    rows = []
    for chunk in position_chunks(positions, position_chunk_size):
        if target.verification_mode == "block":
            native, hidden = target._request_block(  # noqa: SLF001
                input_ids[: chunk[-1] + 1], logits_start=chunk[0], hidden_start=chunk[0]
            )
            rows.extend(
                compare_rows(
                    native, teacher.logprobs(hidden[:, teacher_slot]), chunk, thresholds
                )
            )
        else:
            for position in chunk:
                native, hidden = target._request(  # noqa: SLF001
                    input_ids[: position + 1], need_hidden=True
                )
                rows.extend(
                    compare_rows(
                        torch.tensor([native], dtype=torch.float32),
                        teacher.logprobs(hidden[-1:, teacher_slot]),
                        [position],
                        thresholds,
                    )
                )
    agreement = sum(row["argmax_match"] for row in rows) / len(rows)
    violations = [
        {"position": row["position"], "metrics": row["threshold_violations"]}
        for row in rows
        if row["threshold_violations"]
    ]
    return {
        "passed": not violations and agreement >= thresholds.min_argmax_agreement,
        "verification_mode": target.verification_mode,
        "input_ids": list(input_ids),
        "positions": positions,
        "position_definition": "zero-based pre-norm HS row p predicts token p+1",
        "teacher_hs_id": _TEACHER_HS_ID,
        "teacher_norm": "Qwen3RMSNorm (FP32 variance, cast before weight multiply)",
        "rms_norm_eps": teacher.epsilon,
        "norm_dtype": str(teacher.norm_weight.dtype),
        "head_dtype": str(teacher.head_weight.dtype),
        "thresholds": asdict(thresholds),
        "threshold_note": (
            "Smoke-test thresholds only; not universal BF16 accuracy limits."
        ),
        "max_abs_logprob_error": max(row["max_abs_logprob_error"] for row in rows),
        "mean_abs_logprob_error": sum(row["mean_abs_logprob_error"] for row in rows)
        / len(rows),
        "max_tv": max(row["tv"] for row in rows),
        "mean_tv": sum(row["tv"] for row in rows) / len(rows),
        "max_kl_native_to_teacher": max(row["kl_native_to_teacher"] for row in rows),
        "argmax_agreement": agreement,
        "argmax_mismatch_positions": [
            row["position"] for row in rows if not row["argmax_match"]
        ],
        "threshold_violations": violations,
        "rows": rows,
    }
