"""
OpenAlex API client.

OpenAlex provides broad cross-domain scholarly coverage with reliable metadata,
DOIs, venue information, and citation counts. TAR uses it as a resilient
secondary literature source when Semantic Scholar is thin or rate-limited.
"""
from __future__ import annotations

import json
import os
import re
import time
from hashlib import sha256
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from literature.schemas import Author, ExternalIDs, FetchResult, Paper
from literature.semantic_scholar import _classify_venue


_BASE_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted_index: Any) -> Optional[str]:
    if not isinstance(inverted_index, dict) or not inverted_index:
        return None
    positions: dict[int, str] = {}
    for token, idxs in inverted_index.items():
        if not isinstance(token, str) or not isinstance(idxs, list):
            continue
        for idx in idxs:
            try:
                positions[int(idx)] = token
            except Exception:
                continue
    if not positions:
        return None
    words = [positions[i] for i in sorted(positions)]
    text = " ".join(words).strip()
    return re.sub(r"\s+", " ", text) or None


def _author_id(name: str) -> str:
    return "openalex_author:" + sha256(name.encode("utf-8")).hexdigest()[:12]


def _parse_paper(raw: Dict[str, Any]) -> Optional[Paper]:
    title = str(raw.get("display_name", "") or "").strip()
    openalex_id = str(raw.get("id", "") or "").strip()
    if not title or not openalex_id:
        return None

    authors: List[Author] = []
    for authorship in raw.get("authorships") or []:
        author_obj = authorship.get("author") if isinstance(authorship, dict) else {}
        name = str((author_obj or {}).get("display_name", "") or "").strip()
        if not name:
            continue
        author_id = str((author_obj or {}).get("id", "") or "").strip() or _author_id(name)
        affiliations = []
        for inst in authorship.get("institutions") or []:
            inst_name = str((inst or {}).get("display_name", "") or "").strip()
            if inst_name:
                affiliations.append(inst_name)
        authors.append(Author(author_id=author_id, name=name, affiliations=affiliations))

    primary_location = raw.get("primary_location") if isinstance(raw.get("primary_location"), dict) else {}
    source_obj = primary_location.get("source") if isinstance(primary_location.get("source"), dict) else {}
    host_venue = raw.get("host_venue") if isinstance(raw.get("host_venue"), dict) else {}
    venue_name = str(source_obj.get("display_name", "") or host_venue.get("display_name", "") or "").strip() or None
    venue_type, venue_tier = _classify_venue(venue_name)

    fields: List[str] = []
    for concept in raw.get("concepts") or []:
        name = str((concept or {}).get("display_name", "") or "").strip()
        if name and name not in fields:
            fields.append(name)

    identifiers = raw.get("ids") if isinstance(raw.get("ids"), dict) else {}
    doi = str(identifiers.get("doi", "") or raw.get("doi", "") or "").strip()
    if doi.startswith("https://doi.org/"):
        doi = doi.removeprefix("https://doi.org/").strip()
    arxiv_id = None
    arxiv_url = str(identifiers.get("arxiv", "") or "").strip()
    if arxiv_url:
        arxiv_id = arxiv_url.rsplit("/", 1)[-1]

    return Paper(
        paper_id=f"openalex:{openalex_id.rsplit('/', 1)[-1]}",
        title=title,
        abstract=_reconstruct_abstract(raw.get("abstract_inverted_index")),
        year=raw.get("publication_year"),
        venue=venue_name,
        venue_type=venue_type,
        venue_tier=venue_tier,
        authors=authors,
        fields_of_study=fields,
        citation_count=int(raw.get("cited_by_count") or 0),
        influential_citation_count=int(raw.get("cited_by_count") or 0),
        reference_count=int(raw.get("referenced_works_count") or 0),
        external_ids=ExternalIDs(
            arxiv=arxiv_id,
            doi=doi or None,
            openalex=openalex_id,
        ),
        tldr=None,
        embedding=None,
        source="openalex",
    )


class OpenAlexClient:
    def __init__(self, *, timeout_s: float = 30.0, request_interval_s: float = 1.0) -> None:
        self._timeout = timeout_s
        self._interval = request_interval_s
        self._last_request = 0.0
        self._mailto = os.getenv("OPENALEX_MAILTO", "").strip()

    def search_topic(
        self,
        topic: str,
        *,
        min_year: int = 2021,
        max_results: int = 25,
    ) -> FetchResult:
        params: Dict[str, Any] = {
            "search": topic,
            "per-page": min(max_results, 50),
            "sort": "cited_by_count:desc",
            "filter": f"from_publication_date:{min_year}-01-01",
        }
        if self._mailto:
            params["mailto"] = self._mailto
        return self._fetch(params)

    def latest(
        self,
        *,
        topics: list[str],
        days_back: int = 7,
        max_results: int = 40,
    ) -> FetchResult:
        topic = " OR ".join([t for t in topics if t][:3]).strip() or "machine learning"
        params: Dict[str, Any] = {
            "search": topic,
            "per-page": min(max_results, 50),
            "sort": "publication_date:desc",
        }
        if self._mailto:
            params["mailto"] = self._mailto
        return self._fetch(params)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_request = time.monotonic()

    def _fetch(self, params: Dict[str, Any]) -> FetchResult:
        self._throttle()
        url = f"{_BASE_URL}?{urlencode(params)}"
        req = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "TAR-Literature-Harvester/1.0",
            },
        )
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(raw)
            results = payload.get("results", []) if isinstance(payload, dict) else []
            papers = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                paper = _parse_paper(item)
                if paper is not None:
                    papers.append(paper.model_dump(mode="json"))
            return FetchResult(ok=True, source="openalex", items=papers)
        except HTTPError as exc:
            rate_limited = exc.code == 429
            return FetchResult(ok=False, source="openalex", error=f"http_{exc.code}", rate_limited=rate_limited)
        except URLError as exc:
            return FetchResult(ok=False, source="openalex", error=str(exc.reason))
        except Exception as exc:
            return FetchResult(ok=False, source="openalex", error=str(exc))
