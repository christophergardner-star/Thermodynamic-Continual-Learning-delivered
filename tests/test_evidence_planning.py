import tempfile

from tar_lab.hierarchy import build_evidence_bundle, build_hypotheses
from tar_lab.schemas import MemorySearchHit


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
