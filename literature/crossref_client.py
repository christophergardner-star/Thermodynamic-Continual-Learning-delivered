"""
Crossref API client.

Crossref is a reliable DOI-centric scholarly metadata source. TAR uses it as a
third-party fallback for verified metadata and source diversity when more
specialized APIs are degraded or rate-limited.
"""
from __future__ import annotations

import json
import os
import re
import time
from hashlib import sha256
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from literature.schemas import Author, ExternalIDs, FetchResult, Paper
from literature.semantic_scholar import _classify_venue


_BASE_URL = "https://api.crossref.org/works"


def _strip_jats(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").replace("\n", " ").strip()


def _author_id(name: str) -> str:
    return "crossref_author:" + sha256(name.encode("utf-8")).hexdigest()[:12]


def _parse_paper(raw: Dict[str, Any]) -> Paper | None:
    title_list = raw.get("title") or []
    title = str(title_list[0] if title_list else "").strip()
    doi = str(raw.get("DOI", "") or "").strip()
    if not title:
        return None

    authors: List[Author] = []
    for item in raw.get("author") or []:
        given = str((item or {}).get("given", "") or "").strip()
        family = str((item or {}).get("family", "") or "").strip()
        name = " ".join([part for part in [given, family] if part]).strip()
        if not name:
            name = str((item or {}).get("name", "") or "").strip()
        if not name:
            continue
        affiliations = [
            str((aff or {}).get("name", "") or "").strip()
            for aff in (item or {}).get("affiliation") or []
            if str((aff or {}).get("name", "") or "").strip()
        ]
        authors.append(Author(author_id=_author_id(name), name=name, affiliations=affiliations))

    container = raw.get("container-title") or []
    venue_name = str(container[0] if container else raw.get("publisher", "") or "").strip() or None
    venue_type, venue_tier = _classify_venue(venue_name)

    year = None
    issued = (raw.get("issued") or {}).get("date-parts") or []
    if issued and isinstance(issued[0], list) and issued[0]:
        try:
            year = int(issued[0][0])
        except Exception:
            year = None

    subjects = [str(item).strip() for item in raw.get("subject") or [] if str(item).strip()]
    abstract = _strip_jats(str(raw.get("abstract", "") or "")) or None
    paper_id = f"doi:{doi.lower()}" if doi else f"crossref:{sha256(title.encode('utf-8')).hexdigest()[:16]}"

    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        year=year,
        venue=venue_name,
        venue_type=venue_type,
        venue_tier=venue_tier,
        authors=authors,
        fields_of_study=subjects,
        citation_count=int(raw.get("is-referenced-by-count") or 0),
        influential_citation_count=int(raw.get("is-referenced-by-count") or 0),
        reference_count=int(raw.get("references-count") or 0),
        external_ids=ExternalIDs(doi=doi or None),
        source="crossref",
    )


class CrossrefClient:
    def __init__(self, *, timeout_s: float = 30.0, request_interval_s: float = 1.0) -> None:
        self._timeout = timeout_s
        self._interval = request_interval_s
        self._last_request = 0.0
        self._mailto = os.getenv("CROSSREF_MAILTO", os.getenv("OPENALEX_MAILTO", "")).strip()

    def search_topic(
        self,
        topic: str,
        *,
        min_year: int = 2021,
        max_results: int = 20,
    ) -> FetchResult:
        filters = [f"from-pub-date:{min_year}-01-01"]
        params: Dict[str, Any] = {
            "query.bibliographic": topic,
            "rows": min(max_results, 30),
            "sort": "is-referenced-by-count",
            "order": "desc",
            "filter": ",".join(filters),
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
            items = ((payload.get("message") or {}).get("items") or []) if isinstance(payload, dict) else []
            papers = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                paper = _parse_paper(item)
                if paper is not None:
                    papers.append(paper.model_dump(mode="json"))
            return FetchResult(ok=True, source="crossref", items=papers)
        except HTTPError as exc:
            rate_limited = exc.code == 429
            return FetchResult(ok=False, source="crossref", error=f"http_{exc.code}", rate_limited=rate_limited)
        except URLError as exc:
            return FetchResult(ok=False, source="crossref", error=str(exc.reason))
        except Exception as exc:
            return FetchResult(ok=False, source="crossref", error=str(exc))
