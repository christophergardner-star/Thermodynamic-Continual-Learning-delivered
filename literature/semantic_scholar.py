"""
Semantic Scholar API client.

Semantic Scholar indexes 200M+ academic papers with full citation graphs,
author profiles, and SPECTER2 paper embeddings. This is the primary source
for literature discovery across all AI/CS/ML domains.

API: https://api.semanticscholar.org/graph/v1/
Rate limits:
    - Without API key: ~1 req/s, 5000 results/search
    - With API key (free tier): 10 req/s, 10000 results/search
Set SS_API_KEY environment variable to increase throughput.

All methods return typed results and never raise on API errors —
failures are captured in FetchResult.error for the caller to handle.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from literature.schemas import Author, ExternalIDs, FetchResult, Paper, Venue


_BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Fields we request on every paper fetch — covers everything we need for the
# knowledge graph without pulling in large optional fields by default.
_PAPER_FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "year",
    "venue",
    "publicationVenue",
    "authors",
    "fieldsOfStudy",
    "s2FieldsOfStudy",
    "citationCount",
    "influentialCitationCount",
    "referenceCount",
    "externalIds",
    "publicationTypes",
    "tldr",
    "embedding",
])

# Minimal fields for citation/reference list traversal (saves quota).
_MINIMAL_FIELDS = "paperId,title,year,citationCount,influentialCitationCount"

# Venue tier classification based on known conference/journal names.
# Not exhaustive — unknown venues default to "unknown".
_TOP_VENUES = frozenset({
    "NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV",
    "ACL", "EMNLP", "NAACL", "AAAI", "IJCAI", "SIGKDD",
    "SIGIR", "WWW", "STOC", "FOCS", "SODA",
    "Nature", "Science", "Nature Machine Intelligence",
    "Journal of Machine Learning Research",
})
_STRONG_VENUES = frozenset({
    "AISTATS", "UAI", "COLING", "EACL", "CoNLL",
    "WACV", "BMVC", "INTERSPEECH", "ICASSP",
    "IEEE TPAMI", "IJCV", "IEEE TNNLS",
    "Machine Learning", "Artificial Intelligence",
    "ICRA", "IROS", "RSS",
})


def _classify_venue(name: Optional[str]) -> tuple[str, str]:
    """Return (venue_type, venue_tier) for a raw venue name string."""
    if not name:
        return "unknown", "unknown"
    n = name.strip()
    for known in _TOP_VENUES:
        if known.lower() in n.lower():
            return "conference", "top"
    for known in _STRONG_VENUES:
        if known.lower() in n.lower():
            return "conference", "strong"
    if any(kw in n.lower() for kw in ("arxiv", "preprint", "corr")):
        return "preprint", "unknown"
    if any(kw in n.lower() for kw in ("journal", "transactions", "letters")):
        return "journal", "standard"
    return "conference", "standard"


def _parse_paper(raw: Dict[str, Any]) -> Optional[Paper]:
    """Convert a raw Semantic Scholar API response dict into a Paper."""
    pid = raw.get("paperId")
    title = raw.get("title", "").strip()
    if not pid or not title:
        return None

    authors: List[Author] = []
    for a in raw.get("authors") or []:
        if a.get("authorId") and a.get("name"):
            authors.append(Author(author_id=a["authorId"], name=a["name"]))

    ext = raw.get("externalIds") or {}
    external_ids = ExternalIDs(
        arxiv=ext.get("ArXiv"),
        doi=ext.get("DOI"),
        semantic_scholar=pid,
        dblp=ext.get("DBLP"),
        acl=ext.get("ACL"),
        pubmed=ext.get("PubMed"),
    )

    venue_raw = raw.get("venue") or (raw.get("publicationVenue") or {}).get("name")
    venue_type, venue_tier = _classify_venue(venue_raw)

    fields: List[str] = []
    for f in raw.get("fieldsOfStudy") or []:
        if isinstance(f, str):
            fields.append(f)
    for f in raw.get("s2FieldsOfStudy") or []:
        cat = f.get("category") if isinstance(f, dict) else None
        if cat and cat not in fields:
            fields.append(cat)

    tldr_obj = raw.get("tldr")
    tldr = tldr_obj.get("text") if isinstance(tldr_obj, dict) else None

    embedding_obj = raw.get("embedding")
    embedding: Optional[List[float]] = None
    if isinstance(embedding_obj, dict) and "vector" in embedding_obj:
        embedding = embedding_obj["vector"]

    return Paper(
        paper_id=pid,
        title=title,
        abstract=raw.get("abstract"),
        year=raw.get("year"),
        venue=venue_raw,
        venue_type=venue_type,
        venue_tier=venue_tier,
        authors=authors,
        fields_of_study=fields,
        citation_count=raw.get("citationCount") or 0,
        influential_citation_count=raw.get("influentialCitationCount") or 0,
        reference_count=raw.get("referenceCount") or 0,
        external_ids=external_ids,
        tldr=tldr,
        embedding=embedding,
        source="semantic_scholar",
    )


class SemanticScholarClient:
    """
    Typed client for the Semantic Scholar Graph API.

    Thread-safe for read operations. Uses a simple token-bucket rate limiter
    to stay within API limits. Set SS_API_KEY environment variable for
    higher throughput (10 req/s vs 1 req/s).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.getenv("SS_API_KEY")
        self._timeout = timeout_s
        # Requests-per-second limit. With key: 10/s; without: 1/s.
        self._rps = 9.0 if self._api_key else 0.9
        self._last_request_time: float = 0.0

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def search_papers(
        self,
        query: str,
        *,
        fields_of_study: Optional[List[str]] = None,
        year_range: Optional[tuple[int, int]] = None,
        min_citations: int = 0,
        venue_tier: Optional[str] = None,
        max_results: int = 100,
        offset: int = 0,
    ) -> FetchResult:
        """
        Full-text search across all papers indexed by Semantic Scholar.

        fields_of_study: limit to specific S2 fields (e.g. ["Computer Science"])
        year_range: (start_year, end_year) inclusive
        min_citations: filter results with fewer citations
        max_results: capped at 100 per call (API limit); use offset to paginate
        """
        params: Dict[str, Any] = {
            "query": query,
            "fields": _PAPER_FIELDS,
            "limit": min(max_results, 100),
            "offset": offset,
        }
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        raw = self._get(f"{_BASE_URL}/paper/search", params)
        if not raw["ok"]:
            return FetchResult(ok=False, source="semantic_scholar", error=raw["error"])

        data = raw["data"]
        papers = []
        for item in data.get("data") or []:
            p = _parse_paper(item)
            if p is None:
                continue
            if p.citation_count < min_citations:
                continue
            if venue_tier and p.venue_tier != venue_tier:
                continue
            papers.append(p.model_dump(mode="json"))

        return FetchResult(ok=True, source="semantic_scholar", items=papers)

    def get_paper(self, paper_id: str) -> FetchResult:
        """
        Fetch full details for a single paper.

        paper_id accepts:
            Semantic Scholar ID: "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
            ArXiv ID:            "arXiv:1706.03762"
            DOI:                 "10.18653/v1/2020.acl-main.747"
        """
        encoded = quote(paper_id, safe=":/.")
        raw = self._get(f"{_BASE_URL}/paper/{encoded}", {"fields": _PAPER_FIELDS})
        if not raw["ok"]:
            return FetchResult(ok=False, source="semantic_scholar", error=raw["error"])

        p = _parse_paper(raw["data"])
        if p is None:
            return FetchResult(ok=False, source="semantic_scholar", error="parse_failed")
        return FetchResult(ok=True, source="semantic_scholar", items=[p.model_dump(mode="json")])

    def get_citations(
        self,
        paper_id: str,
        *,
        max_results: int = 500,
        min_year: Optional[int] = None,
    ) -> FetchResult:
        """Fetch papers that cite paper_id (forward citations)."""
        encoded = quote(paper_id, safe=":/.")
        params = {
            "fields": f"citingPaper.{_MINIMAL_FIELDS}",
            "limit": min(max_results, 1000),
        }
        raw = self._get(f"{_BASE_URL}/paper/{encoded}/citations", params)
        if not raw["ok"]:
            return FetchResult(ok=False, source="semantic_scholar", error=raw["error"])

        items = []
        for edge in raw["data"].get("data") or []:
            cp = edge.get("citingPaper") or {}
            if not cp.get("paperId"):
                continue
            year = cp.get("year")
            if min_year and year and year < min_year:
                continue
            items.append({
                "paper_id": cp["paperId"],
                "title": cp.get("title", ""),
                "year": year,
                "citation_count": cp.get("citationCount", 0),
            })

        return FetchResult(ok=True, source="semantic_scholar", items=items)

    def get_references(self, paper_id: str, *, max_results: int = 200) -> FetchResult:
        """Fetch papers referenced by paper_id (backward citations)."""
        encoded = quote(paper_id, safe=":/.")
        params = {
            "fields": f"citedPaper.{_MINIMAL_FIELDS}",
            "limit": min(max_results, 1000),
        }
        raw = self._get(f"{_BASE_URL}/paper/{encoded}/references", params)
        if not raw["ok"]:
            return FetchResult(ok=False, source="semantic_scholar", error=raw["error"])

        items = []
        for edge in raw["data"].get("data") or []:
            rp = edge.get("citedPaper") or {}
            if rp.get("paperId"):
                items.append({
                    "paper_id": rp["paperId"],
                    "title": rp.get("title", ""),
                    "year": rp.get("year"),
                    "citation_count": rp.get("citationCount", 0),
                })
        return FetchResult(ok=True, source="semantic_scholar", items=items)

    def batch_fetch(self, paper_ids: List[str]) -> FetchResult:
        """
        Fetch up to 500 papers in a single POST request.

        Significantly more efficient than individual get_paper calls when
        seeding the knowledge graph from a known paper list.
        """
        if not paper_ids:
            return FetchResult(ok=True, source="semantic_scholar", items=[])

        chunks = [paper_ids[i : i + 500] for i in range(0, len(paper_ids), 500)]
        all_items = []

        for chunk in chunks:
            raw = self._post(
                f"{_BASE_URL}/paper/batch",
                body={"ids": chunk},
                params={"fields": _PAPER_FIELDS},
            )
            if not raw["ok"]:
                return FetchResult(ok=False, source="semantic_scholar", error=raw["error"])
            for item in raw["data"] or []:
                if not item:
                    continue
                p = _parse_paper(item)
                if p:
                    all_items.append(p.model_dump(mode="json"))

        return FetchResult(ok=True, source="semantic_scholar", items=all_items)

    def get_author(self, author_id: str) -> FetchResult:
        """Fetch author profile including paper list."""
        encoded = quote(author_id, safe="")
        fields = "authorId,name,affiliations,hIndex,paperCount,citationCount,homepage"
        raw = self._get(f"{_BASE_URL}/author/{encoded}", {"fields": fields})
        if not raw["ok"]:
            return FetchResult(ok=False, source="semantic_scholar", error=raw["error"])

        d = raw["data"]
        author = Author(
            author_id=d.get("authorId", author_id),
            name=d.get("name", ""),
            affiliations=[a.get("name", "") for a in d.get("affiliations") or [] if a.get("name")],
            h_index=d.get("hIndex"),
            paper_count=d.get("paperCount"),
            citation_count=d.get("citationCount"),
            homepage=d.get("homepage"),
        )
        return FetchResult(ok=True, source="semantic_scholar", items=[author.model_dump(mode="json")])

    def search_by_topic(
        self,
        topic: str,
        *,
        min_year: int = 2019,
        min_citations: int = 5,
        max_results: int = 200,
    ) -> FetchResult:
        """
        High-level topic search across all AI/CS/ML literature.

        Wraps search_papers with sensible defaults for frontier research discovery:
        restricts to Computer Science field, post-2019, and filters very low-citation
        papers to reduce noise from preprint spam.
        """
        return self.search_papers(
            query=topic,
            fields_of_study=["Computer Science"],
            year_range=(min_year, 2030),
            min_citations=min_citations,
            max_results=max_results,
        )

    # -----------------------------------------------------------------------
    # HTTP internals
    # -----------------------------------------------------------------------

    def _throttle(self) -> None:
        """Simple token-bucket rate limiter."""
        min_interval = 1.0 / self._rps
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._api_key:
            h["x-api-key"] = self._api_key
        return h

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._throttle()
        if params:
            qs = urlencode({k: str(v) for k, v in params.items()})
            url = f"{url}?{qs}"
        req = Request(url, headers=self._headers())
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                return {"ok": True, "data": json.loads(resp.read().decode("utf-8"))}
        except HTTPError as exc:
            if exc.code == 429:
                return {"ok": False, "error": "rate_limited", "rate_limited": True}
            return {"ok": False, "error": f"http_{exc.code}: {exc.reason}"}
        except URLError as exc:
            return {"ok": False, "error": f"url_error: {exc.reason}"}
        except Exception as exc:
            return {"ok": False, "error": f"unexpected: {exc}"}

    def _post(
        self,
        url: str,
        body: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._throttle()
        if params:
            qs = urlencode({k: str(v) for k, v in params.items()})
            url = f"{url}?{qs}"
        data = json.dumps(body).encode("utf-8")
        req = Request(url, data=data, headers=self._headers(), method="POST")
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                return {"ok": True, "data": json.loads(resp.read().decode("utf-8"))}
        except HTTPError as exc:
            if exc.code == 429:
                return {"ok": False, "error": "rate_limited", "rate_limited": True}
            return {"ok": False, "error": f"http_{exc.code}: {exc.reason}"}
        except URLError as exc:
            return {"ok": False, "error": f"url_error: {exc.reason}"}
        except Exception as exc:
            return {"ok": False, "error": f"unexpected: {exc}"}
