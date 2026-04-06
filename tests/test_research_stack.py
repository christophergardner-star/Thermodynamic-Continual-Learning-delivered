import json
import tempfile
from pathlib import Path

from build_researcher_dataset import main as build_dataset_main
from research_database import ResearchDatabase, ResearchEntry
from researcher_agent import classify_claims, render_claim_report


def test_claim_classification_covers_categories():
    claims = classify_claims(
        "Water freezes near 0C. We measured 91% accuracy on the benchmark. "
        "Therefore the method may generalize."
    )
    labels = [claim.label for claim in claims]
    assert "fact" in labels
    assert "measured_result" in labels
    assert "inference" in labels or "hypothesis" in labels


def test_render_claim_report_has_header():
    report = render_claim_report(classify_claims("This may work."))
    assert report.startswith("Claim Classification:")


def test_research_database_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        db = ResearchDatabase(str(Path(tmp) / "db.sqlite"))
        db.add_entry(
            ResearchEntry(
                session="s1",
                task_type="research",
                prompt="Prompt",
                response="Response",
                score=0.9,
                success=True,
                metadata={"confidence": 0.9},
            )
        )
        entries = list(db.iter_entries())
        db.close()
        assert len(entries) == 1
        assert entries[0].metadata["confidence"] == 0.9


def test_build_researcher_dataset_from_db_and_notes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db_path = root / "db.sqlite"
        notes_path = root / "notes.jsonl"
        out_path = root / "corpus.jsonl"

        db = ResearchDatabase(str(db_path))
        db.add_entry(
            ResearchEntry(
                session="s1",
                task_type="research",
                prompt="Prompt",
                response="Response",
                score=0.9,
                success=True,
                metadata={},
            )
        )
        db.close()
        notes_path.write_text(json.dumps({"text": "External note"}) + "\n", encoding="utf-8")

        import sys

        argv = sys.argv[:]
        try:
            sys.argv = [
                "build_researcher_dataset.py",
                "--db",
                str(db_path),
                "--notes_jsonl",
                str(notes_path),
                "--output",
                str(out_path),
            ]
            assert build_dataset_main() == 0
        finally:
            sys.argv = argv

        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
