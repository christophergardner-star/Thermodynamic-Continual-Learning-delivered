from __future__ import annotations

import os
import re
from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable, List
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from tar_lab.schemas import ResearchDocument, ResearchIngestReport


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


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
        for source in self._default_sources(topic=topic, max_results=max_results):
            sources.append(source.name)
            try:
                raw = self._fetch_text(source.url)
                if source.kind == "arxiv":
                    docs.extend(self._parse_arxiv(raw, source))
                elif source.kind == "rss":
                    docs.extend(self._parse_rss(raw, source))
            except Exception:
                continue

        deduped: dict[str, ResearchDocument] = {}
        for doc in docs:
            deduped.setdefault(doc.document_id, doc)

        documents = list(deduped.values())[: max_results * max(len(sources), 1)]
        return ResearchIngestReport(
            topic=topic,
            fetched=len(docs),
            indexed=len(documents),
            sources=sources,
            documents=documents,
        )

    def _default_sources(self, topic: str, max_results: int) -> list[_SourceSpec]:
        topic_q = quote_plus(topic)
        per_source = max(1, max_results)
        sources = [
            _SourceSpec(
                kind="arxiv",
                name="arXiv cs.AI",
                url=(
                    "https://export.arxiv.org/api/query?"
                    f"search_query=cat:cs.AI+AND+all:{topic_q}&start=0&max_results={per_source}"
                    "&sortBy=submittedDate&sortOrder=descending"
                ),
            ),
            _SourceSpec(
                kind="arxiv",
                name="arXiv cs.LG",
                url=(
                    "https://export.arxiv.org/api/query?"
                    f"search_query=cat:cs.LG+AND+all:{topic_q}&start=0&max_results={per_source}"
                    "&sortBy=submittedDate&sortOrder=descending"
                ),
            ),
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
            authors = [
                self._compact(self._text(author.findtext("atom:name", "", ATOM_NS)))
                for author in entry.findall("atom:author", ATOM_NS)
            ]
            tags = [cat.attrib.get("term", "") for cat in entry.findall("atom:category", ATOM_NS)]
            docs.append(
                ResearchDocument(
                    document_id=doc_id,
                    source_kind="arxiv",
                    source_name=source.name,
                    title=title,
                    summary=summary,
                    url=url,
                    published_at=self._text(entry.findtext("atom:published", "", ATOM_NS)) or None,
                    authors=[author for author in authors if author],
                    tags=[tag for tag in tags if tag],
                    problem_statements=self._extract_problem_statements(title=title, summary=summary),
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
                    title=title,
                    summary=summary,
                    url=url,
                    published_at=self._text(item.findtext("pubDate", "")) or None,
                    problem_statements=self._extract_problem_statements(title=title, summary=summary),
                )
            )
        return docs

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
