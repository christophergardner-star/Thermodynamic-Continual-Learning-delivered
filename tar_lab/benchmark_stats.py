from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

from tar_lab.schemas import (
    BenchmarkExecutionStatisticalSummary,
    BenchmarkMetricStatistic,
    BenchmarkSpec,
    BenchmarkStatisticalSummary,
    ProblemExperimentResult,
)


PRIMARY_METRIC_PRIORITY = (
    "accuracy",
    "top1_accuracy",
    "node_accuracy",
    "episodic_return",
    "f1",
    "auroc",
    "shot_noise_robustness",
    "trainability_gap",
    "oversmoothing_gap",
    "effective_dimensionality",
    "representation_dimensionality",
    "sample_efficiency",
    "policy_entropy",
    "corruption_robustness",
)


def default_recommended_seed_runs(tier: str) -> int:
    if tier == "canonical":
        return 5
    if tier == "validation":
        return 3
    return 1


def default_statistical_validation_required(tier: str) -> bool:
    return tier in {"validation", "canonical"}


def build_benchmark_statistical_summary(
    spec: BenchmarkSpec,
    metrics: Dict[str, float],
    *,
    statistical_input: Optional[Dict[str, Any]] = None,
) -> BenchmarkStatisticalSummary:
    statistical_input = statistical_input or {}
    sample_count = int(statistical_input.get("sample_count") or metrics.get("seed_count") or 1)
    sample_count = max(1, sample_count)
    recommended_seed_runs = max(1, int(spec.recommended_seed_runs))
    primary_metric = (
        statistical_input.get("primary_metric")
        or _choose_primary_metric(metrics)
    )
    std_dev = _coerce_optional_float(statistical_input.get("std_dev"))
    if std_dev is None and "seed_variance" in metrics:
        std_dev = _coerce_optional_float(metrics.get("seed_variance"))

    metric_summaries: list[BenchmarkMetricStatistic] = []
    notes: list[str] = []

    if primary_metric and primary_metric in metrics:
        mean = float(metrics[primary_metric])
        ci95_low = None
        ci95_high = None
        if std_dev is not None and sample_count > 1:
            margin = 1.96 * (std_dev / math.sqrt(sample_count))
            ci95_low = mean - margin
            ci95_high = mean + margin
        metric_summaries.append(
            BenchmarkMetricStatistic(
                metric_name=primary_metric,
                mean=mean,
                std_dev=std_dev,
                ci95_low=ci95_low,
                ci95_high=ci95_high,
                sample_count=sample_count,
            )
        )
    else:
        notes.append("primary benchmark metric was not available for statistical summarization")

    statistically_ready = False
    if not spec.statistical_validation_required:
        statistically_ready = True
    elif sample_count < recommended_seed_runs:
        notes.append(
            f"benchmark statistical readiness requires {recommended_seed_runs} seed runs, but only {sample_count} were recorded"
        )
    elif std_dev is None:
        notes.append("benchmark run did not expose seed-dispersion evidence for the primary metric")
    else:
        statistically_ready = True

    return BenchmarkStatisticalSummary(
        statistically_ready=statistically_ready,
        sample_count=sample_count,
        recommended_seed_runs=recommended_seed_runs,
        primary_metric=primary_metric,
        metrics=metric_summaries,
        notes=notes,
    )


def build_execution_statistical_summary(
    experiments: Iterable[ProblemExperimentResult],
) -> BenchmarkExecutionStatisticalSummary:
    items = list(experiments)
    completed = [item for item in items if item.status == "completed"]
    ready = [
        item
        for item in completed
        if item.statistical_summary is not None and item.statistical_summary.statistically_ready
    ]
    canonical_completed = [item for item in completed if item.canonical_comparable]
    notes: list[str] = []
    if completed and len(ready) < len(completed):
        notes.append("one or more completed benchmark experiments remain statistically under-powered")
    if canonical_completed and len(ready) < len(canonical_completed):
        notes.append("canonical-comparable benchmark executions still need stronger seed evidence")

    return BenchmarkExecutionStatisticalSummary(
        experiment_count=len(items),
        completed_experiment_count=len(completed),
        statistically_ready_experiment_count=len(ready),
        canonical_ready_completed_count=len(canonical_completed),
        statistically_ready=bool(completed) and len(ready) == len(completed),
        notes=notes,
    )


def _choose_primary_metric(metrics: Dict[str, float]) -> Optional[str]:
    for key in PRIMARY_METRIC_PRIORITY:
        if key in metrics:
            return key
    for key in metrics:
        if key not in {"seed_variance", "seed_count"}:
            return key
    return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
