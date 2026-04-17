from __future__ import annotations

from statistics import pstdev

import pytest
from pydantic import ValidationError

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import (
    AblationResult,
    AnomalyElevationRecord,
    BreakthroughReport,
    CalibrationReport,
    MemorySearchHit,
    SeedVarianceReport,
    VerificationReport,
)


def _verification_report() -> VerificationReport:
    return VerificationReport(
        trial_id="trial-1",
        control_score=1.0,
        seed_variance=SeedVarianceReport(
            num_runs=3,
            loss_mean=0.5,
            loss_std=0.02,
            dimensionality_mean=5.0,
            dimensionality_std=0.1,
            calibration_ece_mean=0.05,
            stable=True,
        ),
        calibration=CalibrationReport(
            ece=0.05,
            accuracy=0.9,
            mean_confidence=0.88,
        ),
        ablations=[
            AblationResult(
                name="ablation",
                training_loss=0.6,
                effective_dimensionality=4.8,
                equilibrium_fraction=0.7,
                calibration_ece=0.06,
                score=0.9,
                delta_vs_control=-0.1,
            )
        ],
        verdict="verified",
    )


class _VaultStub:
    def __init__(self, hits):
        self._hits = list(hits)

    def search(self, query: str, n_results: int = 3, kind: str | None = None):
        return list(self._hits)[:n_results]


def _score_hit(score: float) -> MemorySearchHit:
    return MemorySearchHit(
        document_id=f"breakthrough:{score}",
        score=0.0,
        document="breakthrough",
        metadata={"score": score},
    )


def test_breakthrough_report_has_surprise_fields():
    report = BreakthroughReport(
        trial_id="trial-1",
        status="candidate",
        summary="Candidate result",
        novelty_score=0.5,
        stability_score=0.6,
        calibration_score=0.7,
        verification=_verification_report(),
    )
    assert report.surprise_score == 0.0
    assert report.prior_contradiction_score == 0.0


def test_anomaly_elevation_record_schema_valid():
    record = AnomalyElevationRecord(
        elevation_id="abc123",
        timestamp="2026-04-17T12:00:00",
        breakthrough_id="trial-1",
        project_id="project-1",
        surprise_score=0.4,
        prior_contradiction_score=0.0,
        vault_score_mean=0.5,
        vault_score_std=0.1,
        elevation_reason="Deviates from prior distribution",
    )
    assert record.replication_priority == "normal"
    assert record.replicated is False


def test_anomaly_elevation_record_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        AnomalyElevationRecord(
            elevation_id="abc123",
            timestamp="2026-04-17T12:00:00",
            breakthrough_id="trial-1",
            project_id="project-1",
            surprise_score=0.4,
            prior_contradiction_score=0.0,
            vault_score_mean=0.5,
            vault_score_std=0.1,
            elevation_reason="Deviates from prior distribution",
            unknown="x",
        )


def test_compute_surprise_score_insufficient_history(tmp_path):
    orch = TAROrchestrator(str(tmp_path))
    orch.vault = _VaultStub([_score_hit(0.4), _score_hit(0.5), _score_hit(0.6), _score_hit(0.5)])
    assert orch._compute_surprise_score(0.9) == (0.0, 0.0, 0.0)


def test_compute_surprise_score_normal_result(tmp_path):
    orch = TAROrchestrator(str(tmp_path))
    orch.vault = _VaultStub([_score_hit(0.5) for _ in range(10)])
    surprise, mean, std = orch._compute_surprise_score(0.5)
    assert surprise == 0.0
    assert mean == 0.5
    assert std == 0.0


def test_compute_surprise_score_high_deviation(tmp_path):
    orch = TAROrchestrator(str(tmp_path))
    scores = [0.4, 0.5, 0.6, 0.5, 0.4, 0.6, 0.5, 0.5, 0.4, 0.6]
    orch.vault = _VaultStub([_score_hit(score) for score in scores])
    mean = sum(scores) / len(scores)
    std = pstdev(scores)
    surprise, computed_mean, computed_std = orch._compute_surprise_score(mean + (2 * std))
    assert computed_mean == pytest.approx(mean)
    assert computed_std == pytest.approx(std)
    assert 0.0 < surprise <= 1.0


def test_elevate_anomalies_returns_empty_on_no_reports(tmp_path):
    orch = TAROrchestrator(str(tmp_path))
    orch.vault = _VaultStub([])
    assert orch.elevate_anomalies() == []
