from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from tar_lab.research_ingest import ResearchIngestor
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
