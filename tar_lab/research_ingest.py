from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime
from hashlib import blake2b
from itertools import combinations
from pathlib import Path
from typing import Iterable, List
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import uuid
from xml.etree import ElementTree as ET

from tar_lab.schemas import CrossDomainBridgeRecord, ResearchDocument, ResearchIngestReport


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_CROSS_DOMAIN_VOCAB = {
    "entropy",
    "gradient",
    "loss",
    "regularization",
    "convergence",
    "distribution",
    "manifold",
    "topology",
    "equilibrium",
    "partition",
    "diffusion",
    "kernel",
    "eigenvector",
    "Hamiltonian",
    "Lagrangian",
}


@dataclass(slots=True)
class _SourceSpec:
    kind: str
    name: str
    url: str


class ResearchIngestor:
    def __init__(self, workspace: str = "."):
        self.workspace = workspace

    def ingest(self, topic: str, max_results: int = 6) -> ResearchIngestReport:
        topic = topic.strip() or "frontier ai"
        docs: list[ResearchDocument] = []
        sources: list[str] = []
        failures: list[dict[str, str]] = []
        for source in self._default_sources(topic=topic, max_results=max_results):
            sources.append(source.name)
            try:
                raw = self._fetch_text(source.url)
                if source.kind == "arxiv":
                    docs.extend(self._parse_arxiv(raw, source))
                elif source.kind == "rss":
                    docs.extend(self._parse_rss(raw, source))
            except Exception as exc:
                failures.append(
                    {
                        "stage": "fetch_source",
                        "source": source.name,
                        "url": source.url,
                        "error": str(exc),
                    }
                )
                continue

        deduped: dict[str, ResearchDocument] = {}
        for doc in docs:
            existing = deduped.get(doc.document_id)
            deduped[doc.document_id] = self._merge_documents(existing, doc)

        documents = list(deduped.values())[: max_results * max(len(sources), 1)]
        enriched_documents: list[ResearchDocument] = []
        downloaded_paths: list[str] = []
        for document in documents:
            enriched = self._maybe_download_pdf(document, failures)
            if enriched.local_pdf_path:
                downloaded_paths.append(enriched.local_pdf_path)
            enriched_documents.append(enriched)
        return ResearchIngestReport(
            topic=topic,
            fetched=len(docs),
            indexed=len(enriched_documents),
            sources=sources,
            documents=enriched_documents,
            failures=failures,
            downloaded_pdfs=len(downloaded_paths),
            downloaded_paths=downloaded_paths,
        )

    def _default_sources(self, topic: str, max_results: int) -> list[_SourceSpec]:
        topic_q = quote_plus(topic)
        per_source = max(1, max_results)
        domains = [
            "cs.AI",
            "cs.LG",
            "cond-mat.stat-mech",
            "quant-ph",
            "math.OC",
            "math.ST",
            "q-bio.NC",
        ]
        sources = [
            _SourceSpec(
                kind="arxiv",
                name=f"arXiv {domain}",
                url=(
                    "https://export.arxiv.org/api/query?"
                    f"search_query=cat:{domain}+AND+all:{topic_q}&start=0&max_results={per_source}"
                    "&sortBy=submittedDate&sortOrder=descending"
                ),
            )
            for domain in domains
        ]
        for idx, rss_url in enumerate(self._rss_urls(), start=1):
            sources.append(_SourceSpec(kind="rss", name=f"RSS {idx}", url=rss_url))
        return sources

    @staticmethod
    def _rss_urls() -> Iterable[str]:
        raw = os.environ.get("TAR_RESEARCH_RSS_URLS", "")
        for item in raw.split(","):
            url = item.strip()
            if url:
                yield url

    @staticmethod
    def _fetch_text(url: str) -> str:
        req = Request(url, headers={"User-Agent": "TARResearchIngestor/1.0"})
        with urlopen(req, timeout=20) as response:  # nosec - controlled URLs/config
            return response.read().decode("utf-8", errors="replace")

    @staticmethod
    def _fetch_bytes(url: str) -> bytes:
        req = Request(url, headers={"User-Agent": "TARResearchIngestor/1.0"})
        with urlopen(req, timeout=30) as response:  # nosec - controlled URLs/config
            return response.read()

    def _parse_arxiv(self, raw: str, source: _SourceSpec) -> list[ResearchDocument]:
        root = ET.fromstring(raw)
        docs: list[ResearchDocument] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            doc_id = self._stable_id(
                source.kind,
                self._text(entry.findtext("atom:id", "", ATOM_NS)),
            )
            title = self._compact(self._text(entry.findtext("atom:title", "", ATOM_NS)))
            summary = self._compact(self._text(entry.findtext("atom:summary", "", ATOM_NS)))
            url = self._text(entry.findtext("atom:id", "", ATOM_NS))
            pdf_url = self._extract_arxiv_pdf_url(entry, url)
            authors = [
                self._compact(self._text(author.findtext("atom:name", "", ATOM_NS)))
                for author in entry.findall("atom:author", ATOM_NS)
            ]
            tags = [cat.attrib.get("term", "") for cat in entry.findall("atom:category", ATOM_NS)]
            primary_domain = next((tag for tag in tags if tag), source.name.replace("arXiv ", "", 1))
            docs.append(
                ResearchDocument(
                    document_id=doc_id,
                    source_kind="arxiv",
                    source_name=source.name,
                    domain=primary_domain,
                    title=title,
                    summary=summary,
                    url=url,
                    published_at=self._text(entry.findtext("atom:published", "", ATOM_NS)) or None,
                    authors=[author for author in authors if author],
                    tags=[tag for tag in tags if tag],
                    problem_statements=self._extract_problem_statements(title=title, summary=summary),
                    pdf_url=pdf_url,
                )
            )
        return docs

    def _parse_rss(self, raw: str, source: _SourceSpec) -> list[ResearchDocument]:
        root = ET.fromstring(raw)
        docs: list[ResearchDocument] = []
        for item in root.findall(".//item"):
            title = self._compact(self._text(item.findtext("title", "")))
            summary = self._compact(self._text(item.findtext("description", "")))
            url = self._text(item.findtext("link", ""))
            doc_id = self._stable_id(source.kind, url or title)
            docs.append(
                ResearchDocument(
                    document_id=doc_id,
                    source_kind="rss",
                    source_name=source.name,
                    domain="rss",
                    title=title,
                    summary=summary,
                    url=url,
                    published_at=self._text(item.findtext("pubDate", "")) or None,
                    problem_statements=self._extract_problem_statements(title=title, summary=summary),
                )
            )
        return docs

    def _maybe_download_pdf(
        self,
        document: ResearchDocument,
        failures: list[dict[str, str]],
    ) -> ResearchDocument:
        if document.source_kind != "arxiv":
            return document
        pdf_url = self._normalize_arxiv_pdf_url(document.pdf_url or "") or self._default_pdf_url(document.url)
        if not pdf_url:
            return document
        cache_path = self._pdf_cache_path(document)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return document.model_copy(update={"pdf_url": pdf_url, "local_pdf_path": str(cache_path)})
        try:
            payload = self._fetch_bytes(pdf_url)
            if not payload:
                raise ValueError("empty PDF response")
            self._write_bytes_atomic(cache_path, payload)
            return document.model_copy(update={"pdf_url": pdf_url, "local_pdf_path": str(cache_path)})
        except Exception as exc:
            failures.append(
                {
                    "stage": "download_pdf",
                    "source": document.source_name,
                    "document_id": document.document_id,
                    "url": pdf_url,
                    "error": str(exc),
                }
            )
            return document.model_copy(update={"pdf_url": pdf_url})

    def _pdf_cache_path(self, document: ResearchDocument) -> Path:
        cache_dir = Path(self.workspace) / "tar_state" / "literature" / "arxiv_pdfs"
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = document.document_id.replace(":", "__") + ".pdf"
        return cache_dir / filename

    @staticmethod
    def _write_bytes_atomic(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)

    def _extract_arxiv_pdf_url(self, entry: ET.Element, abs_url: str) -> str | None:
        for link in entry.findall("atom:link", ATOM_NS):
            href = self._text(link.attrib.get("href", "")).strip()
            link_type = self._text(link.attrib.get("type", "")).strip().lower()
            title = self._text(link.attrib.get("title", "")).strip().lower()
            if not href:
                continue
            if link_type == "application/pdf" or title == "pdf" or "/pdf/" in href or href.endswith(".pdf"):
                normalized = self._normalize_arxiv_pdf_url(href)
                if normalized:
                    return normalized
        return self._default_pdf_url(abs_url)

    @staticmethod
    def _normalize_arxiv_pdf_url(url: str) -> str | None:
        normalized = url.strip()
        if not normalized:
            return None
        normalized = normalized.replace("http://", "https://")
        normalized = normalized.replace("https://export.arxiv.org/", "https://arxiv.org/")
        if "/abs/" in normalized:
            normalized = normalized.replace("/abs/", "/pdf/", 1)
        if "/pdf/" in normalized and not normalized.endswith(".pdf"):
            normalized = normalized.rstrip("/") + ".pdf"
        return normalized

    def _default_pdf_url(self, abs_url: str) -> str | None:
        normalized = self._normalize_arxiv_pdf_url(abs_url)
        if normalized and "/pdf/" in normalized:
            return normalized
        return None

    @staticmethod
    def _merge_documents(existing: ResearchDocument | None, incoming: ResearchDocument) -> ResearchDocument:
        if existing is None:
            return incoming
        summary = incoming.summary if len(incoming.summary) > len(existing.summary) else existing.summary
        tags = sorted({*existing.tags, *incoming.tags})
        authors = existing.authors or incoming.authors
        if incoming.authors and len(incoming.authors) > len(existing.authors):
            authors = incoming.authors
        problem_statements = existing.problem_statements or incoming.problem_statements
        if incoming.problem_statements and len(incoming.problem_statements) > len(existing.problem_statements):
            problem_statements = incoming.problem_statements
        return existing.model_copy(
            update={
                "domain": existing.domain or incoming.domain,
                "title": incoming.title if len(incoming.title) > len(existing.title) else existing.title,
                "summary": summary,
                "url": existing.url or incoming.url,
                "published_at": existing.published_at or incoming.published_at,
                "authors": authors,
                "tags": tags,
                "problem_statements": problem_statements,
                "pdf_url": existing.pdf_url or incoming.pdf_url,
                "local_pdf_path": existing.local_pdf_path or incoming.local_pdf_path,
            }
        )

    @staticmethod
    def _stable_id(kind: str, value: str) -> str:
        digest = blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
        return f"{kind}:{digest}"

    @staticmethod
    def _text(value: str | None) -> str:
        return value or ""

    @staticmethod
    def _compact(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def _extract_problem_statements(self, title: str, summary: str) -> list[str]:
        text = f"{title}. {summary}".strip()
        if not text:
            return []
        sentences = [self._compact(part) for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        keywords = (
            "problem",
            "challenge",
            "limitation",
            "bottleneck",
            "failure",
            "risk",
            "alignment",
            "safety",
            "scalability",
            "calibration",
            "robustness",
        )
        matches = [sentence for sentence in sentences if any(word in sentence.lower() for word in keywords)]
        if matches:
            return matches[:3]
        return sentences[:2]

    def detect_cross_domain_bridges(
        self,
        papers: list[ResearchDocument],
    ) -> list[CrossDomainBridgeRecord]:
        if not papers:
            return []
        bridges: list[CrossDomainBridgeRecord] = []
        for paper_a, paper_b in combinations(papers, 2):
            if not paper_a.domain or not paper_b.domain or paper_a.domain == paper_b.domain:
                continue
            shared_terms = sorted(
                self._technical_terms(paper_a) & self._technical_terms(paper_b)
            )
            if len(shared_terms) < 3:
                continue
            bridges.append(
                CrossDomainBridgeRecord(
                    bridge_id=uuid.uuid4().hex,
                    timestamp=datetime.utcnow().isoformat(),
                    source_domain=paper_a.domain,
                    target_domain=paper_b.domain,
                    source_paper_id=self._paper_content_hash(paper_a),
                    target_paper_id=self._paper_content_hash(paper_b),
                    bridge_type="shared_formalism",
                    confidence=min(1.0, len(shared_terms) / 5.0),
                    summary=(
                        f"Shared formalism between {paper_a.domain} and {paper_b.domain}: "
                        f"{', '.join(shared_terms)}"
                    ),
                )
            )
        return bridges

    def _technical_terms(self, paper: ResearchDocument) -> set[str]:
        text = " ".join(
            filter(
                None,
                [paper.title, paper.summary, *paper.problem_statements],
            )
        ).lower()
        return {
            term for term in _CROSS_DOMAIN_VOCAB
            if re.search(rf"\b{re.escape(term.lower())}\b", text)
        }

    @staticmethod
    def _paper_content_hash(paper: ResearchDocument) -> str:
        payload = "\n".join(
            [
                paper.title.strip(),
                paper.summary.strip(),
                " ".join(item.strip() for item in paper.problem_statements if item.strip()),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
