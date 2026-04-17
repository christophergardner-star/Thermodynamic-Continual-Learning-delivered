from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import ContributionPositioningReport, MemorySearchHit, SoTAComparison


class _VaultStub:
    def __init__(self, hits):
        self._hits = list(hits)
        self.records: list[tuple[str, str, dict[str, object]]] = []

    def search(self, query: str, n_results: int = 3, kind: str | None = None):
        return list(self._hits)[:n_results]

    def _upsert(self, document_id: str, document: str, metadata: dict[str, object]) -> None:
        self.records.append((document_id, document, metadata))


def _hit(similarity: float, *, document_id: str = "paper:1") -> MemorySearchHit:
    return MemorySearchHit(
        document_id=document_id,
        score=similarity,
        document="similar paper",
        metadata={
            "paper_id": document_id,
            "paper_title": f"title-{document_id}",
            "domain": "cs.AI",
        },
    )


def test_sota_comparison_schema_valid():
    comparison = SoTAComparison(
        comparison_id="cmp-1",
        timestamp="2026-04-17T20:00:00",
        paper_id="paper-1",
        paper_title="Paper Title",
        domain="cs.AI",
        similarity_score=0.75,
        outperforms=False,
        delta_description="TAR differs in calibration handling",
    )
    assert isinstance(comparison.outperforms, bool)
    assert 0.0 <= comparison.similarity_score <= 1.0


def test_sota_comparison_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        SoTAComparison(
            comparison_id="cmp-1",
            timestamp="2026-04-17T20:00:00",
            paper_id="paper-1",
            paper_title="Paper Title",
            domain="cs.AI",
            similarity_score=0.75,
            outperforms=False,
            delta_description="delta",
            unknown="x",
        )


def test_contribution_positioning_report_schema_valid():
    report = ContributionPositioningReport(
        report_id="rep-1",
        timestamp="2026-04-17T20:00:00",
        project_id="proj-1",
        trial_id="trial-1",
        novelty_vs_literature=0.8,
        positioning_summary="Summary",
    )
    assert report.sota_comparisons == []
    assert report.competing_theories_open == 0
    assert report.vault_indexed is False


def test_contribution_positioning_report_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ContributionPositioningReport(
            report_id="rep-1",
            timestamp="2026-04-17T20:00:00",
            project_id="proj-1",
            trial_id="trial-1",
            novelty_vs_literature=0.8,
            positioning_summary="Summary",
            unknown="x",
        )


def test_position_contribution_empty_vault(tmp_path):
    orch = TAROrchestrator(str(tmp_path))
    orch.vault = _VaultStub([])
    result = orch.position_contribution("proj_abc", "trial_xyz", "test result")
    assert result.novelty_vs_literature == 1.0
    assert result.sota_comparisons == []
    assert isinstance(result.report_id, str) and result.report_id


def test_position_contribution_vault_hits_reduce_novelty(tmp_path):
    orch = TAROrchestrator(str(tmp_path))
    orch.vault = _VaultStub([
        _hit(0.8, document_id="paper:1"),
        _hit(0.6, document_id="paper:2"),
    ])
    result = orch.position_contribution("proj_abc", "trial_xyz", "test result")
    assert abs(result.novelty_vs_literature - 0.3) < 0.01
    assert len(result.sota_comparisons) == 2


def test_position_contribution_report_persisted(tmp_path):
    orch = TAROrchestrator(str(tmp_path))
    orch.vault = _VaultStub([])
    result = orch.position_contribution("proj_abc", "trial_xyz", "test result")
    report_path = Path(tmp_path) / "tar_state" / "positioning" / f"{result.report_id}.json"
    assert report_path.exists()


def test_publish_handoff_package_includes_positioning(tmp_path, monkeypatch):
    orch = TAROrchestrator(str(tmp_path))
    expected = ContributionPositioningReport(
        report_id="rep-1",
        timestamp="2026-04-17T20:00:00",
        project_id="proj",
        trial_id="trial",
        novelty_vs_literature=1.0,
        positioning_summary="Summary",
    )

    monkeypatch.setattr(orch, "position_contribution", lambda project_id, trial_id, description: expected)
    result = orch.publish_handoff_package("proj", "trial", "desc")
    assert "positioning_report_id" in result
