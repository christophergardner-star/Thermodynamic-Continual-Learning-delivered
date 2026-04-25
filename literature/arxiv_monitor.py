"""
ArXiv monitor.

Covers the complete AI/CS/ML publication surface:
  cs.LG  — machine learning
  cs.AI  — artificial intelligence
  cs.CV  — computer vision
  cs.CL  — computation and language (NLP)
  cs.NE  — neural and evolutionary computing
  cs.RO  — robotics
  cs.IR  — information retrieval
  cs.HC  — human-computer interaction
  cs.DC  — distributed/parallel computing
  stat.ML — statistics / machine learning
  cond-mat.dis-nn — disordered systems (statistical physics ↔ ML bridge)

Two modes:
  search()  — one-shot query for seeding the knowledge graph
  monitor() — continuous polling loop for staying up-to-date on new work

ArXiv API: http://export.arxiv.org/api/query
Rate limit: ≤3 req/s (enforced here at 1 req/s to be conservative).
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from literature.schemas import ExternalIDs, FetchResult, Paper


_BASE_URL = "http://export.arxiv.org/api/query"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"

# Complete taxonomy of AI/CS/ML categories we care about.
# Grouped by research area for clarity.
AI_CATEGORIES: Dict[str, str] = {
    # Core ML & AI
    "cs.LG":           "Machine Learning",
    "cs.AI":           "Artificial Intelligence",
    "cs.NE":           "Neural and Evolutionary Computing",
    "stat.ML":         "Statistics - Machine Learning",
    # Perception
    "cs.CV":           "Computer Vision and Pattern Recognition",
    "cs.GR":           "Graphics",
    # Language & Knowledge
    "cs.CL":           "Computation and Language",
    "cs.IR":           "Information Retrieval",
    "cs.DB":           "Databases",
    # Systems & Efficiency
    "cs.DC":           "Distributed, Parallel, and Cluster Computing",
    "cs.AR":           "Hardware Architecture",
    "cs.PF":           "Performance",
    # Robotics & Control
    "cs.RO":           "Robotics",
    "cs.SY":           "Systems and Control",
    "eess.SY":         "Systems and Control (EESS)",
    # Human & Society
    "cs.HC":           "Human-Computer Interaction",
    "cs.CY":           "Computers and Society",
    # Theory & Foundations
    "cs.IT":           "Information Theory",
    "cs.LO":           "Logic in Computer Science",
    "math.ST":         "Statistics Theory",
    "math.OC":         "Optimization and Control",
    # Physics-ML bridge
    "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
    "quant-ph":        "Quantum Physics (for quantum ML)",
}

# Frontier topics for targeted searches — updated to reflect 2024-2026 landscape.
FRONTIER_TOPICS = [
    # Foundation models
    "large language model",
    "foundation model",
    "scaling law",
    "in-context learning",
    "chain of thought reasoning",
    # Efficiency
    "mixture of experts",
    "sparse model",
    "model compression",
    "quantization neural network",
    "efficient transformer",
    "state space model",
    # Learning paradigms
    "continual learning catastrophic forgetting",
    "meta-learning few-shot",
    "self-supervised representation learning",
    "reinforcement learning from human feedback",
    "reward model alignment",
    # Multimodal
    "vision language model",
    "multimodal learning",
    # Generative
    "diffusion model",
    "flow matching generative",
    # Mechanistic & interpretability
    "mechanistic interpretability",
    "neural circuit",
    "superposition polysemanticity",
    # Robustness & safety
    "adversarial robustness",
    "distribution shift",
    "out of distribution detection",
    # Graph & geometry
    "graph neural network",
    "geometric deep learning",
    # Biology-ML bridge
    "protein structure prediction",
    "drug discovery machine learning",
]


def _ns(tag: str, ns: str = _ATOM_NS) -> str:
    return f"{{{ns}}}{tag}"


def _text(elem: ET.Element, tag: str, ns: str = _ATOM_NS) -> Optional[str]:
    child = elem.find(_ns(tag, ns))
    return child.text.strip() if child is not None and child.text else None


def _parse_entry(entry: ET.Element) -> Optional[Paper]:
    """Convert one Atom <entry> element into a Paper."""
    raw_id = _text(entry, "id")
    if not raw_id:
        return None

    # Normalise: http://arxiv.org/abs/2401.12345v2 → 2401.12345
    arxiv_id = re.sub(r"v\d+$", "", raw_id.split("/abs/")[-1]).strip()
    if not arxiv_id:
        return None

    title_raw = _text(entry, "title") or ""
    title = " ".join(title_raw.split())  # collapse whitespace
    if not title:
        return None

    abstract_raw = _text(entry, "summary") or ""
    abstract = " ".join(abstract_raw.split())

    # Published date → year
    published = _text(entry, "published")
    year: Optional[int] = None
    if published:
        try:
            year = int(published[:4])
        except ValueError:
            pass

    authors: list[Any] = []
    from literature.schemas import Author
    for author_elem in entry.findall(_ns("author")):
        name_elem = author_elem.find(_ns("name"))
        if name_elem is not None and name_elem.text:
            aname = name_elem.text.strip()
            from hashlib import sha256
            aid = "arxiv_author:" + sha256(aname.encode()).hexdigest()[:12]
            authors.append(Author(author_id=aid, name=aname))

    # Primary category
    primary = entry.find(_ns("primary_category", _ARXIV_NS))
    category = primary.get("term") if primary is not None else None
    fields = [category] if category else []
    for cat_elem in entry.findall(_ns("category")):
        term = cat_elem.get("term")
        if term and term not in fields:
            fields.append(term)

    # Links
    pdf_url: Optional[str] = None
    for link in entry.findall(_ns("link")):
        if link.get("type") == "application/pdf":
            pdf_url = link.get("href")

    paper_id = f"arXiv:{arxiv_id}"

    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract or None,
        year=year,
        venue="arXiv preprint",
        venue_type="preprint",
        venue_tier="unknown",
        authors=authors,
        fields_of_study=fields,
        citation_count=0,
        external_ids=ExternalIDs(arxiv=arxiv_id, semantic_scholar=None),
        source="arxiv",
    )


class ArXivMonitor:
    """
    Queries and monitors arXiv across all AI/CS/ML categories.

    Usage:
        monitor = ArXivMonitor()
        result = monitor.search("continual learning", max_results=50)
        for paper_dict in result.items:
            process(Paper(**paper_dict))

    Continuous monitoring:
        for batch in monitor.monitor(categories=["cs.LG", "cs.AI"], poll_interval_s=3600):
            ingest(batch)   # called each hour with new papers
    """

    def __init__(self, request_interval_s: float = 1.5) -> None:
        self._interval = request_interval_s
        self._last_request: float = 0.0

    # -----------------------------------------------------------------------
    # Search interface
    # -----------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        categories: Optional[List[str]] = None,
        max_results: int = 100,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        start: int = 0,
    ) -> FetchResult:
        """
        Full-text search across arXiv.

        query: standard arXiv query syntax, e.g. "ti:continual AND ab:forgetting"
               or a plain natural-language query (arXiv handles both)
        categories: restrict to specific arXiv categories (AND-ed with query)
        sort_by: "relevance" | "lastUpdatedDate" | "submittedDate"
        """
        cat_filter = ""
        if categories:
            cat_parts = " OR ".join(f"cat:{c}" for c in categories)
            cat_filter = f" AND ({cat_parts})"

        full_query = query + cat_filter

        params = {
            "search_query": full_query,
            "start": start,
            "max_results": min(max_results, 2000),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        return self._fetch(params)

    def latest(
        self,
        categories: Optional[List[str]] = None,
        *,
        days_back: int = 1,
        max_results: int = 200,
    ) -> FetchResult:
        """
        Fetch papers submitted in the last N days.

        Used by the monitoring loop for daily/hourly updates.
        """
        cats = categories or list(AI_CATEGORIES.keys())
        cat_query = " OR ".join(f"cat:{c}" for c in cats)
        # submittedDate is YYYYMMDD format in arXiv query syntax
        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        date_str = since.strftime("%Y%m%d")
        query = f"({cat_query}) AND submittedDate:[{date_str}* TO *]"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": min(max_results, 2000),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        return self._fetch(params)

    def search_topic(
        self,
        topic: str,
        *,
        min_year: int = 2022,
        max_results: int = 100,
    ) -> FetchResult:
        """
        Convenience method for frontier topic search.

        Searches title and abstract, restricts to recent papers to surface
        cutting-edge work rather than historical surveys.
        """
        year_filter = f" AND submittedDate:[{min_year}0101* TO *]"
        query = f"(ti:{quote(topic)} OR abs:{quote(topic)})" + year_filter
        return self.search(
            query,
            categories=["cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.NE", "stat.ML"],
            max_results=max_results,
            sort_by="submittedDate",
        )

    def fetch_frontier(self, max_per_topic: int = 50) -> FetchResult:
        """
        Sweep all FRONTIER_TOPICS and return a deduplicated merged result.

        This is the primary seeding function for the knowledge graph — run
        it once on startup to populate the graph with cutting-edge work, then
        use monitor() to stay up to date.
        """
        seen: set[str] = set()
        all_items: list[Dict[str, Any]] = []

        for topic in FRONTIER_TOPICS:
            result = self.search_topic(topic, max_results=max_per_topic)
            if not result.ok:
                continue
            for item in result.items:
                pid = item.get("paper_id")
                if pid and pid not in seen:
                    seen.add(pid)
                    all_items.append(item)

        return FetchResult(ok=True, source="arxiv", items=all_items)

    # -----------------------------------------------------------------------
    # Monitoring loop
    # -----------------------------------------------------------------------

    def monitor(
        self,
        categories: Optional[List[str]] = None,
        *,
        poll_interval_s: float = 3600.0,
        on_new_papers: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Continuous monitoring loop. Yields batches of new papers each poll.

        Usage:
            for batch in monitor.monitor(poll_interval_s=3600):
                for paper_dict in batch:
                    knowledge_graph.upsert_paper(Paper(**paper_dict))

        Runs indefinitely — wrap in a daemon thread or async task.
        on_new_papers: optional callback fired for each batch (alternative to iterator).
        """
        cats = categories or ["cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.NE", "stat.ML"]
        seen: set[str] = set()

        while True:
            result = self.latest(categories=cats, days_back=2, max_results=500)
            if result.ok:
                new_batch = []
                for item in result.items:
                    pid = item.get("paper_id")
                    if pid and pid not in seen:
                        seen.add(pid)
                        new_batch.append(item)
                if new_batch:
                    if on_new_papers:
                        on_new_papers(new_batch)
                    yield new_batch

            time.sleep(poll_interval_s)

    # -----------------------------------------------------------------------
    # HTTP internals
    # -----------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_request = time.monotonic()

    def _fetch(self, params: Dict[str, Any]) -> FetchResult:
        self._throttle()
        qs = urlencode(params)
        url = f"{_BASE_URL}?{qs}"
        req = Request(url, headers={"Accept": "application/atom+xml"})
        try:
            with urlopen(req, timeout=30) as resp:
                xml_bytes = resp.read()
        except HTTPError as exc:
            return FetchResult(ok=False, source="arxiv", error=f"http_{exc.code}: {exc.reason}")
        except URLError as exc:
            return FetchResult(ok=False, source="arxiv", error=f"url_error: {exc.reason}")
        except Exception as exc:
            return FetchResult(ok=False, source="arxiv", error=f"unexpected: {exc}")

        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            return FetchResult(ok=False, source="arxiv", error=f"xml_parse: {exc}")

        papers = []
        for entry in root.findall(_ns("entry")):
            p = _parse_entry(entry)
            if p:
                papers.append(p.model_dump(mode="json"))

        return FetchResult(ok=True, source="arxiv", items=papers)
