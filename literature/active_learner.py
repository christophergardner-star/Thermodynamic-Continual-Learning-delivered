"""
Active Learner — autonomous continuous update loop for the Literature Brain.

The active learner runs as a background daemon and keeps the knowledge graph
current. It operates on a tiered schedule:

  FAST  (every 4 hours):   ArXiv latest submissions across all AI/CS/ML categories
  DAILY (every 24 hours):  Semantic Scholar enrichment for papers without embeddings
                            Gap detector re-run; top problems updated
  WEEKLY (every 7 days):   Full Papers With Code SoTA refresh for all benchmarks
                            Citation graph expansion for top-cited papers
                            Cross-domain bridge re-scan

The loop is entirely self-contained — it does not require human input.
Errors are logged and skipped; a single source failure never blocks the loop.

Design notes:
  - All operations are idempotent (upsert, not insert)
  - Each cycle logs a structured summary to the knowledge graph corpus status
  - The loop terminates cleanly on KeyboardInterrupt or stop() call
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional

from literature.arxiv_monitor import ArXivMonitor, AI_CATEGORIES, FRONTIER_TOPICS
from literature.gap_detector import GapDetector
from literature.knowledge_graph import LiteratureKnowledgeGraph
from literature.pwc_client import PapersWithCodeClient, BENCHMARK_REGISTRY
from literature.schemas import Benchmark, Paper, _stable_id, _utc_now
from literature.semantic_scholar import SemanticScholarClient


log = logging.getLogger(__name__)


# Cycle intervals in seconds
_FAST_INTERVAL   = 4 * 3600      # 4 hours  — ArXiv new submissions
_DAILY_INTERVAL  = 24 * 3600     # 24 hours — SS enrichment + gap refresh
_WEEKLY_INTERVAL = 7 * 24 * 3600 # 7 days   — full SoTA + citation refresh


@dataclass
class CycleStats:
    """Statistics for a single active learner cycle."""
    cycle_type: str
    started_at: str = field(default_factory=_utc_now)
    papers_added: int = 0
    papers_enriched: int = 0
    sota_entries_added: int = 0
    gaps_detected: int = 0
    errors: List[str] = field(default_factory=list)
    completed_at: Optional[str] = None

    def finish(self) -> None:
        self.completed_at = _utc_now()

    def log_summary(self) -> None:
        log.info(
            "[ActiveLearner] %s cycle complete: +%d papers, +%d enriched, "
            "+%d SoTA, %d gaps, %d errors",
            self.cycle_type,
            self.papers_added,
            self.papers_enriched,
            self.sota_entries_added,
            self.gaps_detected,
            len(self.errors),
        )


class ActiveLearner:
    """
    Autonomous background daemon for the Literature Brain.

    Usage (blocking):
        learner = ActiveLearner(graph)
        learner.run_forever()  # blocks

    Usage (background thread):
        learner = ActiveLearner(graph)
        learner.start()        # daemon thread
        # ... do other work ...
        learner.stop()

    Usage (manual single cycle):
        learner = ActiveLearner(graph)
        stats = learner.run_fast_cycle()
    """

    def __init__(
        self,
        graph: LiteratureKnowledgeGraph,
        ss_client: Optional[SemanticScholarClient] = None,
        arxiv_monitor: Optional[ArXivMonitor] = None,
        pwc_client: Optional[PapersWithCodeClient] = None,
        on_new_papers: Optional[Callable[[List[Paper]], None]] = None,
        on_gaps_updated: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._graph = graph
        self._ss = ss_client or SemanticScholarClient()
        self._arxiv = arxiv_monitor or ArXivMonitor()
        self._pwc = pwc_client or PapersWithCodeClient()
        self._gap_detector = GapDetector(graph)
        self._on_new_papers = on_new_papers
        self._on_gaps_updated = on_gaps_updated

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._last_fast   = 0.0
        self._last_daily  = 0.0
        self._last_weekly = 0.0

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start the active learner as a background daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run_forever,
            name="LiteratureActiveLearner",
            daemon=True,
        )
        self._thread.start()
        log.info("[ActiveLearner] Background thread started.")

    def stop(self) -> None:
        """Signal the active learner to stop after the current cycle."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=30)
        log.info("[ActiveLearner] Stopped.")

    def run_forever(self) -> None:
        """
        Main loop. Runs indefinitely until stop() is called.

        On first run, immediately executes all three cycle types to seed
        the knowledge graph before settling into scheduled operation.
        """
        log.info("[ActiveLearner] Starting. First run: executing all cycles.")
        self._run_weekly_cycle()
        self._run_daily_cycle()
        self._run_fast_cycle()

        while not self._stop_event.is_set():
            now = time.monotonic()

            if now - self._last_fast >= _FAST_INTERVAL:
                self._run_fast_cycle()

            if now - self._last_daily >= _DAILY_INTERVAL:
                self._run_daily_cycle()

            if now - self._last_weekly >= _WEEKLY_INTERVAL:
                self._run_weekly_cycle()

            # Sleep in short increments so stop() is responsive
            for _ in range(60):
                if self._stop_event.is_set():
                    break
                time.sleep(60)

    # -----------------------------------------------------------------------
    # Fast cycle — ArXiv new submissions (every 4 hours)
    # -----------------------------------------------------------------------

    def run_fast_cycle(self) -> CycleStats:
        """Public alias for external callers (e.g. tests, manual triggers)."""
        return self._run_fast_cycle()

    def _run_fast_cycle(self) -> CycleStats:
        stats = CycleStats(cycle_type="fast")
        log.info("[ActiveLearner] Fast cycle: fetching recent arXiv submissions.")

        try:
            result = self._arxiv.latest(
                categories=["cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.NE", "stat.ML"],
                days_back=1,
                max_results=500,
            )
            if result.ok:
                new_papers = self._ingest_paper_dicts(result.items)
                stats.papers_added = new_papers
                if self._on_new_papers and new_papers > 0:
                    pass  # callback fired inside _ingest_paper_dicts
            else:
                stats.errors.append(f"arxiv_latest: {result.error}")
        except Exception as exc:
            stats.errors.append(f"fast_cycle: {exc}")

        stats.finish()
        stats.log_summary()
        self._last_fast = time.monotonic()
        return stats

    # -----------------------------------------------------------------------
    # Daily cycle — SS enrichment + gaps (every 24 hours)
    # -----------------------------------------------------------------------

    def _run_daily_cycle(self) -> CycleStats:
        stats = CycleStats(cycle_type="daily")
        log.info("[ActiveLearner] Daily cycle: SS enrichment + gap refresh.")

        # 1. Enrich papers that have no embeddings yet
        try:
            ids_missing_embeddings = self._graph.papers_without_embeddings(limit=200)
            if ids_missing_embeddings:
                result = self._ss.batch_fetch(ids_missing_embeddings)
                if result.ok:
                    enriched = self._ingest_paper_dicts(result.items)
                    stats.papers_enriched = enriched
                else:
                    stats.errors.append(f"ss_batch_fetch: {result.error}")
        except Exception as exc:
            stats.errors.append(f"daily_enrichment: {exc}")

        # 2. Search for new work on frontier topics
        try:
            for topic in FRONTIER_TOPICS[:10]:  # rotate daily to limit API calls
                result = self._ss.search_by_topic(topic, max_results=20)
                if result.ok:
                    stats.papers_added += self._ingest_paper_dicts(result.items)
                time.sleep(1.2)  # respect SS rate limit
        except Exception as exc:
            stats.errors.append(f"daily_topic_search: {exc}")

        # 3. Re-run gap detector
        try:
            gaps = self._gap_detector.detect_all(top_n=200)
            for gap in gaps:
                self._graph.upsert_gap(gap)
            stats.gaps_detected = len(gaps)
            if self._on_gaps_updated:
                self._on_gaps_updated(len(gaps))
        except Exception as exc:
            stats.errors.append(f"daily_gap_detect: {exc}")

        stats.finish()
        stats.log_summary()
        self._last_daily = time.monotonic()
        return stats

    # -----------------------------------------------------------------------
    # Weekly cycle — full SoTA refresh + citation expansion (every 7 days)
    # -----------------------------------------------------------------------

    def _run_weekly_cycle(self) -> CycleStats:
        stats = CycleStats(cycle_type="weekly")
        log.info("[ActiveLearner] Weekly cycle: SoTA refresh + citation expansion.")

        # 1. Register benchmarks from PwC registry
        try:
            for slug, meta in BENCHMARK_REGISTRY.items():
                bmark = self._pwc.to_benchmark_schema(slug)
                self._graph.upsert_benchmark(bmark)
        except Exception as exc:
            stats.errors.append(f"benchmark_register: {exc}")

        # 2. Fetch SoTA tables for all registered benchmarks
        try:
            for slug in BENCHMARK_REGISTRY:
                table = self._pwc.get_sota_table(slug)
                if table:
                    added = self._graph.upsert_sota_table(table)
                    stats.sota_entries_added += added
                time.sleep(0.6)  # PwC rate limit
        except Exception as exc:
            stats.errors.append(f"weekly_sota_refresh: {exc}")

        # 3. Expand citation graph for high-influence papers
        try:
            high_influence = self._graph.conn.execute(
                "SELECT paper_id FROM papers "
                "WHERE influential_citation_count > 20 "
                "ORDER BY influential_citation_count DESC LIMIT 100"
            ).fetchall()
            for row in high_influence:
                pid = row[0]
                cit_result = self._ss.get_citations(pid, max_results=100, min_year=2022)
                if cit_result.ok:
                    for item in cit_result.items:
                        cited_id = item.get("paper_id")
                        if cited_id:
                            self._graph.add_citation(cited_id, pid)
                time.sleep(1.2)
        except Exception as exc:
            stats.errors.append(f"weekly_citations: {exc}")

        # 4. Seed with frontier topic papers from SS (deeper than daily)
        try:
            for topic in FRONTIER_TOPICS:
                result = self._ss.search_by_topic(topic, min_year=2023, max_results=50)
                if result.ok:
                    stats.papers_added += self._ingest_paper_dicts(result.items)
                time.sleep(1.2)
        except Exception as exc:
            stats.errors.append(f"weekly_frontier_seed: {exc}")

        stats.finish()
        stats.log_summary()
        self._last_weekly = time.monotonic()
        return stats

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _ingest_paper_dicts(self, items: list) -> int:
        """Deserialise raw dicts into Paper objects and upsert into the graph."""
        new_papers: List[Paper] = []
        for item in items:
            try:
                p = Paper(**item)
                new_papers.append(p)
            except Exception:
                continue

        for paper in new_papers:
            self._graph.upsert_paper(paper)

        if new_papers and self._on_new_papers:
            self._on_new_papers(new_papers)

        return len(new_papers)
