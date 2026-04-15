import tempfile
from pathlib import Path

from tar_lab.hierarchy import build_evidence_bundle, build_hypotheses
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import MemorySearchHit


REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_science_profiles(tmp: str) -> None:
    source_dir = REPO_ROOT / "science_profiles"
    target_dir = Path(tmp) / "science_profiles"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.json"):
        (target_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _hit(document_id: str, score: float, claim_id: str, contradiction: bool = False) -> MemorySearchHit:
    metadata = {
        "kind": "paper_claim",
        "paper_id": f"paper:{document_id}",
        "paper_title": f"Paper {document_id}",
        "claim_id": claim_id,
        "page_number": 3,
        "source_excerpt": "Optimization stability depends on calibrated representation geometry.",
    }
    if contradiction:
        metadata["contradictory_claims"] = [
            {
                "left_claim_id": claim_id,
                "right_claim_id": "claim:other",
                "reason": "opposite empirical conclusion",
            }
        ]
    return MemorySearchHit(document_id=document_id, score=score, document="Evidence text", metadata=metadata)


def test_evidence_bundle_surfaces_contradictions():
    bundle = build_evidence_bundle(
        "Investigate optimization stability in deep learning",
        [_hit("doc-1", 0.9, "claim:1", contradiction=True), _hit("doc-2", 0.8, "claim:2")],
    )
    assert bundle.contradiction_review is not None
    assert bundle.contradiction_review.contradiction_count >= 1
    hypotheses = build_hypotheses("Investigate optimization stability in deep learning", bundle, benchmark_ids=["cifar10_c"])
    assert hypotheses
    assert hypotheses[0].contradiction_review_id == bundle.contradiction_review.review_id


class _ConflictVault:
    def search(self, query: str, n_results: int = 5, kind=None, require_research_grade: bool = False):
        return [
            MemorySearchHit(
                document_id="claim_conflict:1",
                score=0.86,
                document="Claim conflict showing opposite outcomes on optimization stability.",
                metadata={
                    "kind": "claim_conflict",
                    "left_claim_id": "claim:left",
                    "right_claim_id": "claim:right",
                    "contradiction_count": 1,
                    "source_excerpt": "Paper A reports improved stability while Paper B reports collapse.",
                },
                contradiction_surfaced=True,
            ),
            _hit("doc-1", 0.81, "claim:1", contradiction=True),
        ]

    def index_problem_study(self, report) -> None:
        return None

    def close(self) -> None:
        return None


def test_retrieval_conflicts_raise_project_falsification_pressure():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            if orchestrator.vault is not None:
                orchestrator.vault.close()
            orchestrator.vault = _ConflictVault()  # type: ignore[assignment]
            orchestrator.memory_indexer = None

            project = orchestrator.create_project("Investigate conflicting stability evidence")
            report = orchestrator.study_problem(
                "Investigate conflicting stability evidence",
                project_id=project.project_id,
                build_env=False,
                max_results=0,
            )

            debt = orchestrator.store.latest_evidence_debt_record(project.project_id)

            assert report.retrieval_conflict_count >= 1
            assert debt is not None
            assert debt.falsification_gap >= 0.1
            assert any("contradiction-bearing evidence" in item for item in debt.rationale)
        finally:
            orchestrator.shutdown()
