from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    example_id: str
    task_family: str
    task_name: str
    suite_names: list[str]
    lineage_key: str
    source_kind: str
    source_id: str
    messages: list[dict[str, str]]
    input_context: dict[str, Any]
    gold_target: dict[str, Any]
    scoring_target: dict[str, Any]
    provenance: dict[str, Any]
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalFileFingerprint:
    path: str
    sha256: str
    size_bytes: int
    records: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalItemResult:
    item_id: str
    example_id: str
    task_family: str
    suite_names: list[str]
    score: float
    decision_correct: bool
    overclaim: bool
    false_refusal: bool
    parse_error: bool
    error_bucket: str
    field_scores: dict[str, float]
    gold_summary: dict[str, Any]
    predicted_summary: dict[str, Any]
    prediction_text: str
    parsed_prediction: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalAggregate:
    count: int
    mean_score: float
    decision_accuracy: float
    overclaim_rate: float
    false_refusal_rate: float
    parse_error_rate: float
    error_buckets: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

