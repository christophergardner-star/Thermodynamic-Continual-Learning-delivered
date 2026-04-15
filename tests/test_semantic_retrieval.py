import tempfile
from pathlib import Path

import pytest

from tar_lab.literature_engine import LiteratureEngine, ParsedPage
from tar_lab.errors import ScientificValidityError
from tar_lab.memory.vault import VectorVault
from tar_lab.schemas import BibliographyEntry, PaperArtifact, PaperSection, ResearchClaim


class FakeSentenceTransformer:
    def __init__(self, model_name: str, local_files_only: bool = True, device: str = "cpu"):
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.device = device

    def encode(self, text: str, normalize_embeddings: bool = True):
        lowered = text.lower()
        vector = [
            float("entropy" in lowered),
            float("production" in lowered),
            float("drift" in lowered),
            float("irreversible" in lowered),
            float("stable" in lowered or "stability" in lowered),
            float("not" in lowered or "worse" in lowered),
        ]
        norm = sum(item * item for item in vector) ** 0.5
        if norm and normalize_embeddings:
            return [item / norm for item in vector]
        return vector


def test_vault_retrieves_physics_claim_with_semantic_embedder(monkeypatch):
    import tar_lab.memory.vault as vault_module

    monkeypatch.setattr(vault_module, "SentenceTransformer", FakeSentenceTransformer)
    with tempfile.TemporaryDirectory() as tmp:
        vault = VectorVault(tmp)
        try:
            claim_a = ResearchClaim(
                claim_id="claim-a",
                paper_id="paper-a",
                section_id="section:0",
                label="fact",
                text="Entropy production increases under irreversible drift.",
                polarity="positive",
                page_number=1,
            )
            claim_b = ResearchClaim(
                claim_id="claim-b",
                paper_id="paper-b",
                section_id="section:0",
                label="fact",
                text="Entropy production does not increase under irreversible drift.",
                polarity="negative",
                page_number=2,
            )
            artifact_a = PaperArtifact(
                paper_id="paper-a",
                source_path="paper-a.pdf",
                title="Physics Paper A",
                sections=[PaperSection(section_id="section:0", heading="Results", text=claim_a.text, page_start=1, page_end=1)],
                claims=[claim_a],
            )
            artifact_b = PaperArtifact(
                paper_id="paper-b",
                source_path="paper-b.pdf",
                title="Physics Paper B",
                sections=[PaperSection(section_id="section:0", heading="Results", text=claim_b.text, page_start=2, page_end=2)],
                claims=[claim_b],
            )
            vault.index_paper_artifact(artifact_a)
            vault.index_paper_artifact(artifact_b)
            hits = vault.search(
                "Does entropy production increase under irreversible drift?",
                n_results=2,
                kind="paper_claim",
                require_research_grade=True,
            )
            assert hits
            assert vault.stats()["embedder"] == "BAAI/bge-small-en-v1.5"
            assert vault.stats()["semantic_research_ready"] is True
            assert hits[0].metadata["claim_id"] in {"claim-a", "claim-b"}
            assert "contradictory_claims" in hits[0].metadata
            assert "evidence_trace" in hits[0].metadata
            traces = vault.build_evidence_traces(hits)
            assert traces[0].claim_id in {"claim-a", "claim-b"}
        finally:
            vault.close()


def test_vault_indexes_claim_graph_and_surfaces_conflict_hit(monkeypatch):
    import tar_lab.memory.vault as vault_module

    monkeypatch.setattr(vault_module, "SentenceTransformer", FakeSentenceTransformer)
    with tempfile.TemporaryDirectory() as tmp:
        vault = VectorVault(tmp)
        try:
            claim_a = ResearchClaim(
                claim_id="claim-a",
                paper_id="paper-a",
                section_id="section:0",
                label="fact",
                text="Entropy production increases under irreversible drift.",
                polarity="positive",
                page_number=1,
            )
            claim_b = ResearchClaim(
                claim_id="claim-b",
                paper_id="paper-b",
                section_id="section:0",
                label="fact",
                text="Entropy production does not increase under irreversible drift.",
                polarity="negative",
                page_number=2,
            )
            artifact_a = PaperArtifact(
                paper_id="paper-a",
                source_path="paper-a.pdf",
                title="Physics Paper A",
                sections=[PaperSection(section_id="section:0", heading="Results", text=claim_a.text, page_start=1, page_end=1)],
                claims=[claim_a],
            )
            artifact_b = PaperArtifact(
                paper_id="paper-b",
                source_path="paper-b.pdf",
                title="Physics Paper B",
                sections=[PaperSection(section_id="section:0", heading="Results", text=claim_b.text, page_start=2, page_end=2)],
                claims=[claim_b],
            )
            vault.index_paper_artifact(artifact_a)
            vault.index_paper_artifact(artifact_b)

            stats = vault.stats()
            assert stats["claim_clusters"] >= 1
            assert stats["claim_conflicts"] >= 1

            hits = vault.search(
                "Find contradiction or conflict about entropy production under irreversible drift.",
                n_results=3,
                require_research_grade=True,
            )

            assert hits
            conflict_hit = next((hit for hit in hits if hit.metadata.get("kind") == "claim_conflict"), None)
            assert conflict_hit is not None
            assert float(conflict_hit.metadata["source_confidence"]) >= 0.5
            assert conflict_hit.metadata["evidence_trace"]["contradiction_count"] >= 1
        finally:
            vault.close()


def test_literature_engine_extracts_pdf_string_with_page_attribution(monkeypatch):
    def fake_read_pdf_pages(self, path: Path):
        return (
            [
                ParsedPage(
                    page_number=2,
                    text=(
                        "Mock PDF Title\n\n"
                        "Abstract\n"
                        "Quantum thermodynamic equilibrium remains stable.\n\n"
                        "1 Introduction\n"
                        "Quantum thermodynamic equilibrium remains stable under bounded drift [1].\n\n"
                        "References\n"
                        "[1] Gardner et al. 2026. Thermodynamic Equilibrium in Neural Systems."
                    ),
                    source="mock_pdf",
                    used_ocr=False,
                )
            ],
            ["mock_pdf"],
            False,
        )

    monkeypatch.setattr(LiteratureEngine, "_read_pdf_pages", fake_read_pdf_pages)
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "mock.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%mock\n")
        engine = LiteratureEngine(tmp)
        report = engine.ingest_paths([str(pdf_path)])
        assert report.ingested == 1
        artifact = report.artifacts[0]
        claim = next(item for item in artifact.claims if "bounded drift" in item.text)
        assert claim.page_number == 2
        assert claim.span_start is not None
        assert claim.span_end is not None
        assert claim.citation_entry_ids
        assert any(section.page_start == 2 for section in artifact.sections)
        assert artifact.bibliography
        assert artifact.citations[0].bibliography_entry_id == artifact.bibliography[0].entry_id
        assert report.capability_report is not None


def test_vault_requires_semantic_model_for_research_queries(monkeypatch):
    import tar_lab.memory.vault as vault_module

    monkeypatch.setattr(vault_module, "SentenceTransformer", None)
    with tempfile.TemporaryDirectory() as tmp:
        vault = VectorVault(tmp)
        try:
            with pytest.raises(ScientificValidityError, match="Research-grade literature retrieval requires a semantic embedding model"):
                vault.ensure_research_ready()
        finally:
            vault.close()
