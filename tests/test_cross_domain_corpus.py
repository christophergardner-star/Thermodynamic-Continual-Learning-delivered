from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from tar_lab.research_ingest import ResearchIngestor, _SourceSpec
from tar_lab.schemas import CrossDomainBridgeRecord, ResearchDocument


def _doc(*, document_id: str, domain: str, title: str, summary: str) -> ResearchDocument:
    return ResearchDocument(
        document_id=document_id,
        source_kind="manual",
        source_name="unit",
        domain=domain,
        title=title,
        summary=summary,
        url=f"https://example.test/{document_id}",
    )


def _sample_arxiv_feed() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <published>2026-04-23T00:00:00Z</published>
    <title> Thermodynamic regularization for continual learning </title>
    <summary> We study calibration robustness under distribution shift. </summary>
    <author><name>Alice Example</name></author>
    <category term="cs.LG" />
    <link href="http://arxiv.org/abs/1234.5678v1" rel="alternate" type="text/html" />
    <link title="pdf" href="http://arxiv.org/pdf/1234.5678v1" rel="related" type="application/pdf" />
  </entry>
</feed>
"""


def test_default_sources_contains_seven_domains(tmp_path):
    ingestor = ResearchIngestor(str(tmp_path))
    sources = ingestor._default_sources(topic="thermodynamics", max_results=2)
    arxiv_domains = {
        source.name.replace("arXiv ", "", 1)
        for source in sources
        if source.kind == "arxiv"
    }
    expected = {
        "cs.AI",
        "cs.LG",
        "cond-mat.stat-mech",
        "quant-ph",
        "math.OC",
        "math.ST",
        "q-bio.NC",
    }
    assert expected.issubset(arxiv_domains)


def test_parse_arxiv_extracts_pdf_url(tmp_path):
    ingestor = ResearchIngestor(str(tmp_path))
    source = _SourceSpec(kind="arxiv", name="arXiv cs.LG", url="https://export.arxiv.org/api/query")

    docs = ingestor._parse_arxiv(_sample_arxiv_feed(), source)

    assert len(docs) == 1
    assert docs[0].pdf_url == "https://arxiv.org/pdf/1234.5678v1.pdf"


def test_ingest_downloads_pdfs_and_reports_failures(tmp_path, monkeypatch):
    ingestor = ResearchIngestor(str(tmp_path))
    arxiv_source = _SourceSpec(kind="arxiv", name="arXiv cs.LG", url="https://export.arxiv.org/api/query")
    rss_source = _SourceSpec(kind="rss", name="RSS 1", url="https://example.test/rss")

    monkeypatch.setattr(
        ingestor,
        "_default_sources",
        lambda topic, max_results: [arxiv_source, rss_source],
    )

    def fake_fetch_text(url: str) -> str:
        if url == arxiv_source.url:
            return _sample_arxiv_feed()
        raise RuntimeError("rss unavailable")

    monkeypatch.setattr(ingestor, "_fetch_text", staticmethod(fake_fetch_text))
    monkeypatch.setattr(ingestor, "_fetch_bytes", staticmethod(lambda url: b"%PDF-1.4 fake pdf"))

    report = ingestor.ingest("thermodynamics", max_results=1)

    assert report.indexed == 1
    assert report.downloaded_pdfs == 1
    assert len(report.downloaded_paths) == 1
    assert Path(report.downloaded_paths[0]).exists()
    assert report.documents[0].local_pdf_path == report.downloaded_paths[0]
    assert report.documents[0].pdf_url == "https://arxiv.org/pdf/1234.5678v1.pdf"
    assert any(item["stage"] == "fetch_source" for item in report.failures)


def test_cross_domain_bridge_detected_shared_formalism(tmp_path):
    ingestor = ResearchIngestor(str(tmp_path))
    papers = [
        _doc(
            document_id="paper-a",
            domain="cs.AI",
            title="Entropy gradient loss regularization for continual learning",
            summary="We analyze convergence under distribution shift.",
        ),
        _doc(
            document_id="paper-b",
            domain="quant-ph",
            title="Entropy gradient loss in Hamiltonian diffusion systems",
            summary="Regularization supports equilibrium and convergence.",
        ),
    ]
    bridges = ingestor.detect_cross_domain_bridges(papers)
    assert len(bridges) >= 1
    assert bridges[0].bridge_type == "shared_formalism"
    assert bridges[0].confidence > 0.0


def test_same_domain_papers_produce_no_bridge(tmp_path):
    ingestor = ResearchIngestor(str(tmp_path))
    papers = [
        _doc(
            document_id="paper-a",
            domain="cs.AI",
            title="Entropy gradient loss regularization",
            summary="Convergence under distribution shift.",
        ),
        _doc(
            document_id="paper-b",
            domain="cs.AI",
            title="Entropy gradient loss in robust agents",
            summary="Regularization improves convergence.",
        ),
    ]
    assert ingestor.detect_cross_domain_bridges(papers) == []


def test_insufficient_shared_terms_produces_no_bridge(tmp_path):
    ingestor = ResearchIngestor(str(tmp_path))
    papers = [
        _doc(
            document_id="paper-a",
            domain="cs.AI",
            title="Entropy methods for agents",
            summary="Curriculum shaping for safe adaptation.",
        ),
        _doc(
            document_id="paper-b",
            domain="quant-ph",
            title="Hamiltonian evolution in quantum systems",
            summary="Spectral measurements for lattice transport.",
        ),
    ]
    assert ingestor.detect_cross_domain_bridges(papers) == []


def test_cross_domain_bridge_record_fields_valid(tmp_path):
    ingestor = ResearchIngestor(str(tmp_path))
    bridges = ingestor.detect_cross_domain_bridges(
        [
            _doc(
                document_id="paper-a",
                domain="cs.AI",
                title="Entropy gradient loss regularization",
                summary="Convergence in distribution-aware learners.",
            ),
            _doc(
                document_id="paper-b",
                domain="quant-ph",
                title="Entropy gradient loss in diffusion systems",
                summary="Regularization and convergence in equilibrium models.",
            ),
        ]
    )
    bridge = bridges[0]
    assert isinstance(bridge.bridge_id, str) and bridge.bridge_id
    assert re.fullmatch(r"[0-9a-f]{32}", bridge.bridge_id)
    assert bridge.source_domain != bridge.target_domain
    assert 0.0 < bridge.confidence <= 1.0
    assert bridge.vault_indexed is False


def test_cross_domain_bridge_schema_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        CrossDomainBridgeRecord(
            bridge_id="x",
            timestamp="2026-04-17T00:00:00",
            source_domain="cs.AI",
            target_domain="quant-ph",
            source_paper_id="a",
            target_paper_id="b",
            bridge_type="shared_formalism",
            confidence=0.5,
            summary="Shared formalism between cs.AI and quant-ph: entropy, gradient, loss",
            unknown_field="y",
        )
