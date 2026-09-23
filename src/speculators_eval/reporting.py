"""Offline evaluation counters, weighted aggregation and result artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from speculators_eval.data import load_jsonl

if TYPE_CHECKING:
    from pathlib import Path
    from types import SimpleNamespace

RESULT_COLUMNS = [
    "dataset",
    "num_requests",
    "elapsed_s",
    "requests_per_second",
    "output_tokens_per_second",
    "total_output_tokens",
    "base_elapsed_s",
    "base_output_tokens_per_second",
    "base_total_output_tokens",
    "speedup_vs_base",
    "num_proposals",
    "num_proposed_draft_tokens",
    "num_accepted_draft_tokens",
    "draft_length",
    "acceptance_length",
    "accepted_draft_length",
    "position_accept_rates",
    "position_accept_prob_means",
    "position_support_accept_rate_means",
    "position_accept_prob_sums",
    "position_support_accept_rate_sums",
    "position_accepted_counts",
    "position_proposed_counts",
]


@dataclass
class EvalStats:
    elapsed_s: float = 0.0
    total_output_tokens: int = 0
    num_proposals: int = 0
    num_proposed_draft_tokens: int = 0
    num_accepted_draft_tokens: int = 0
    position_proposed_counts: list[int] = field(default_factory=list)
    position_accepted_counts: list[int] = field(default_factory=list)
    position_accept_prob_sums: list[float] = field(default_factory=list)
    position_support_accept_rate_sums: list[float] = field(default_factory=list)

    @property
    def acceptance_length(self) -> float:
        if self.num_proposals == 0:
            return 1.0
        return 1.0 + self.num_accepted_draft_tokens / self.num_proposals

    @property
    def draft_length(self) -> float:
        if self.num_proposals == 0:
            return 0.0
        return self.num_proposed_draft_tokens / self.num_proposals

    @property
    def accepted_draft_length(self) -> float:
        if self.num_proposals == 0:
            return 0.0
        return self.num_accepted_draft_tokens / self.num_proposals

    @property
    def position_accept_rates(self) -> list[float]:
        return [
            accepted / proposed if proposed else 0.0
            for accepted, proposed in zip(
                self.position_accepted_counts,
                self.position_proposed_counts,
                strict=True,
            )
        ]

    @property
    def position_accept_prob_means(self) -> list[float]:
        return [
            value / proposed if proposed else 0.0
            for value, proposed in zip(
                self.position_accept_prob_sums,
                self.position_proposed_counts,
                strict=True,
            )
        ]

    @property
    def position_support_accept_rate_means(self) -> list[float]:
        return [
            value / proposed if proposed else 0.0
            for value, proposed in zip(
                self.position_support_accept_rate_sums,
                self.position_proposed_counts,
                strict=True,
            )
        ]

    def add_response(self, response: SimpleNamespace) -> None:
        self.total_output_tokens += int(response.num_output_tokens)
        proposal_lengths = getattr(response, "proposal_lengths", [])
        accepted_lengths = getattr(response, "accepted_draft_lengths", [])
        accept_prob_lists = getattr(response, "accept_prob_lists", [])
        support_accept_rate_lists = getattr(response, "support_accept_rate_lists", [])
        self.num_proposals += len(proposal_lengths)
        self.num_proposed_draft_tokens += sum(int(x) for x in proposal_lengths)
        self.num_accepted_draft_tokens += sum(int(x) for x in accepted_lengths)
        for proposal_len, accepted_len in zip(
            proposal_lengths,
            accepted_lengths,
            strict=True,
        ):
            self.add_proposal_positions(int(proposal_len), int(accepted_len))
        for proposal_len, accept_probs, support_accept_rates in zip(
            proposal_lengths,
            accept_prob_lists,
            support_accept_rate_lists,
            strict=True,
        ):
            self.add_proposal_probability_stats(
                int(proposal_len),
                accept_probs,
                support_accept_rates,
            )

    def add_proposal_positions(self, proposal_len: int, accepted_len: int) -> None:
        if accepted_len > proposal_len:
            raise ValueError(
                f"accepted_len must not exceed proposal_len: {accepted_len}"
            )
        missing = proposal_len - len(self.position_proposed_counts)
        if missing > 0:
            self.position_proposed_counts.extend([0] * missing)
            self.position_accepted_counts.extend([0] * missing)
        for pos in range(proposal_len):
            self.position_proposed_counts[pos] += 1
            if pos < accepted_len:
                self.position_accepted_counts[pos] += 1

    def add_proposal_probability_stats(
        self,
        proposal_len: int,
        accept_probs: list[float],
        support_accept_rates: list[float] | None,
    ) -> None:
        if len(accept_probs) != proposal_len:
            raise ValueError("accept_probs length does not match proposal_len")
        if (
            support_accept_rates is not None
            and len(support_accept_rates) != proposal_len
        ):
            raise ValueError("support_accept_rates length does not match proposal_len")

        missing = proposal_len - len(self.position_accept_prob_sums)
        if missing > 0:
            self.position_accept_prob_sums.extend([0.0] * missing)
            self.position_support_accept_rate_sums.extend([0.0] * missing)
        for pos in range(proposal_len):
            self.position_accept_prob_sums[pos] += float(accept_probs[pos])
            if support_accept_rates is not None:
                self.position_support_accept_rate_sums[pos] += float(
                    support_accept_rates[pos]
                )


def _parse_count_list(value: Any) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value) if value else []
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _parse_float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value) if value else []
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]


def dataset_output_path(directory: Path, dataset: str, suffix: str = "") -> Path:
    """Map a root-relative identity to an output without escaping its directory."""
    parts = dataset.split("/")
    if any(part in {"", ".", ".."} or "\\" in part or ":" in part for part in parts):
        raise ValueError(f"Invalid root-relative dataset identity: {dataset!r}")
    path = directory.joinpath(*parts[:-1], parts[-1] + suffix)
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Dataset output escapes its directory: {dataset!r}")
    return path


def aggregate_rows(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    position_proposed_counts: list[int] = []
    position_accepted_counts: list[int] = []
    position_accept_prob_sums: list[float] = []
    position_support_accept_rate_sums: list[float] = []
    for row in rows:
        proposed = _parse_count_list(row.get("position_proposed_counts", []))
        accepted = _parse_count_list(row.get("position_accepted_counts", []))
        accept_prob_sums = _parse_float_list(row.get("position_accept_prob_sums", []))
        support_sums = _parse_float_list(
            row.get("position_support_accept_rate_sums", [])
        )
        size = max(len(position_proposed_counts), len(proposed))
        if len(position_proposed_counts) < size:
            position_proposed_counts.extend(
                [0] * (size - len(position_proposed_counts))
            )
            position_accepted_counts.extend(
                [0] * (size - len(position_accepted_counts))
            )
            position_accept_prob_sums.extend(
                [0.0] * (size - len(position_accept_prob_sums))
            )
            position_support_accept_rate_sums.extend(
                [0.0] * (size - len(position_support_accept_rate_sums))
            )
        for idx, count in enumerate(proposed):
            position_proposed_counts[idx] += count
        for idx, count in enumerate(accepted):
            position_accepted_counts[idx] += count
        for idx, value in enumerate(accept_prob_sums):
            position_accept_prob_sums[idx] += value
        for idx, value in enumerate(support_sums):
            position_support_accept_rate_sums[idx] += value

    stats = EvalStats(
        elapsed_s=max((float(row["elapsed_s"]) for row in rows), default=0.0),
        total_output_tokens=sum(int(row["total_output_tokens"]) for row in rows),
        num_proposals=sum(int(row["num_proposals"]) for row in rows),
        num_proposed_draft_tokens=sum(
            int(row["num_proposed_draft_tokens"]) for row in rows
        ),
        num_accepted_draft_tokens=sum(
            int(row["num_accepted_draft_tokens"]) for row in rows
        ),
        position_proposed_counts=position_proposed_counts,
        position_accepted_counts=position_accepted_counts,
        position_accept_prob_sums=position_accept_prob_sums,
        position_support_accept_rate_sums=position_support_accept_rate_sums,
    )
    num_requests = sum(int(row["num_requests"]) for row in rows)
    summary = summary_row(dataset, num_requests, stats)
    base_elapsed_s = max(
        (float(row.get("base_elapsed_s", 0.0)) for row in rows),
        default=0.0,
    )
    if base_elapsed_s:
        base_total_output_tokens = sum(
            int(row.get("base_total_output_tokens", 0)) for row in rows
        )
        base_tps = base_total_output_tokens / base_elapsed_s
        summary.update(
            {
                "base_elapsed_s": base_elapsed_s,
                "base_output_tokens_per_second": base_tps,
                "base_total_output_tokens": base_total_output_tokens,
                "speedup_vs_base": (
                    summary["output_tokens_per_second"] / base_tps if base_tps else 0.0
                ),
            }
        )
    return summary


def summary_row(dataset: str, num_requests: int, stats: EvalStats) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "num_requests": num_requests,
        "elapsed_s": stats.elapsed_s,
        "requests_per_second": num_requests / stats.elapsed_s if stats.elapsed_s else 0,
        "output_tokens_per_second": (
            stats.total_output_tokens / stats.elapsed_s if stats.elapsed_s else 0
        ),
        "total_output_tokens": stats.total_output_tokens,
        "base_elapsed_s": 0.0,
        "base_output_tokens_per_second": 0.0,
        "base_total_output_tokens": 0,
        "speedup_vs_base": 0.0,
        "num_proposals": stats.num_proposals,
        "num_proposed_draft_tokens": stats.num_proposed_draft_tokens,
        "num_accepted_draft_tokens": stats.num_accepted_draft_tokens,
        "draft_length": stats.draft_length,
        "acceptance_length": stats.acceptance_length,
        "accepted_draft_length": stats.accepted_draft_length,
        "position_accept_rates": json.dumps(stats.position_accept_rates),
        "position_accept_prob_means": json.dumps(stats.position_accept_prob_means),
        "position_support_accept_rate_means": json.dumps(
            stats.position_support_accept_rate_means
        ),
        "position_accept_prob_sums": json.dumps(stats.position_accept_prob_sums),
        "position_support_accept_rate_sums": json.dumps(
            stats.position_support_accept_rate_sums
        ),
        "position_accepted_counts": json.dumps(stats.position_accepted_counts),
        "position_proposed_counts": json.dumps(stats.position_proposed_counts),
    }


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    artifacts_by_dataset: dict[str, list[dict[str, Any]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    if not artifacts_by_dataset:
        return
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for dataset, artifacts in artifacts_by_dataset.items():
        artifact_path = dataset_output_path(artifacts_dir, dataset, ".jsonl")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("w", encoding="utf-8") as f:
            for artifact in artifacts:
                f.write(json.dumps(artifact) + "\n")


def read_worker_row(output_dir: Path) -> dict[str, Any]:
    with (output_dir / "summary.json").open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"{output_dir}/summary.json must contain one result row")
    return rows[0]


def read_worker_artifacts(output_dir: Path, dataset: str) -> list[dict[str, Any]]:
    path = dataset_output_path(output_dir / "artifacts", dataset, ".jsonl")
    if not path.exists():
        return []
    return load_jsonl(path)
