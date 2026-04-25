"""
Literature Brain — Layer 0 of the TAR research system.

Provides open-world grounding for scientific discovery:
  - live literature from Semantic Scholar (200M+ papers), arXiv, Papers With Code
  - structured SoTA tables across all AI/CS/ML benchmarks
  - knowledge graph with citation edges and method-benchmark coverage
  - gap detection: benchmark coverage, scale, replication, conflict, cross-domain
  - novelty gate: hard check before any result is called a breakthrough
  - active learner: autonomous background daemon keeping the graph current

Quick start:
    from literature import LiteratureBrain
    brain = LiteratureBrain()
    brain.start()   # launches background active learner

    # Check novelty of a result
    report = brain.novelty_gate.evaluate(
        method_name="MyMethod",
        method_description="...",
        benchmark_id=brain.graph.get_all_benchmarks(domain="continual_learning")[0].benchmark_id,
        metric_name="mean_forgetting",
        metric_value=0.10,
        higher_is_better=False,
    )
    print(report.verdict, report.contribution_statement)

    # Get top research problems
    problems = brain.top_problems(n=10)
    for p in problems:
        print(p.priority_score, p.title)
"""
from __future__ import annotations

from typing import List, Optional

from literature.active_learner import ActiveLearner
from literature.arxiv_monitor import ArXivMonitor
from literature.gap_detector import GapDetector
from literature.knowledge_graph import LiteratureKnowledgeGraph
from literature.novelty_gate import NoveltyGate
from literature.pwc_client import PapersWithCodeClient
from literature.schemas import (
    Benchmark,
    CorpusSummary,
    NoveltyReport,
    Paper,
    ProblemCandidate,
    ResearchGap,
    SoTATable,
)
from literature.semantic_scholar import SemanticScholarClient


class LiteratureBrain:
    """
    Unified facade over the Literature Brain.

    Instantiate once per process. The graph is persisted to SQLite at
    db_path. The active learner starts immediately on start() and runs
    as a daemon thread — it does not block the caller.
    """

    def __init__(
        self,
        db_path: str = "literature/literature_graph.db",
        load_embedding_model: bool = True,
    ) -> None:
        self.graph = LiteratureKnowledgeGraph(db_path=db_path)
        self.ss = SemanticScholarClient()
        self.arxiv = ArXivMonitor()
        self.pwc = PapersWithCodeClient()
        self.gap_detector = GapDetector(self.graph)
        self.novelty_gate = NoveltyGate(self.graph, load_embedding_model=load_embedding_model)
        self._learner: Optional[ActiveLearner] = None

    def start(self) -> None:
        """Launch the autonomous active learner in a background thread."""
        self._learner = ActiveLearner(
            graph=self.graph,
            ss_client=self.ss,
            arxiv_monitor=self.arxiv,
            pwc_client=self.pwc,
        )
        self._learner.start()

    def stop(self) -> None:
        if self._learner:
            self._learner.stop()

    # -----------------------------------------------------------------------
    # High-level convenience methods
    # -----------------------------------------------------------------------

    def corpus_summary(self) -> CorpusSummary:
        return self.graph.corpus_summary()

    def top_problems(self, n: int = 20, domain: Optional[str] = None) -> List[ProblemCandidate]:
        """Return top-N ranked research problems derived from gap detection."""
        gaps = self.gap_detector.detect_all(top_n=n * 3, domain=domain)
        return self.gap_detector.gaps_to_problems(gaps=gaps, top_n=n)

    def top_gaps(self, n: int = 20, domain: Optional[str] = None) -> List[ResearchGap]:
        return self.graph.get_top_gaps(n=n, domain=domain)

    def sota_for_benchmark(self, benchmark_id: str) -> SoTATable:
        return self.graph.get_sota_table(benchmark_id)

    def search_papers(self, query: str, max_results: int = 50) -> List[Paper]:
        result = self.ss.search_by_topic(query, max_results=max_results)
        if not result.ok:
            return []
        papers = []
        for item in result.items:
            try:
                p = Paper(**item)
                self.graph.upsert_paper(p)
                papers.append(p)
            except Exception:
                continue
        return papers


__all__ = [
    "LiteratureBrain",
    "LiteratureKnowledgeGraph",
    "SemanticScholarClient",
    "ArXivMonitor",
    "PapersWithCodeClient",
    "GapDetector",
    "NoveltyGate",
    "ActiveLearner",
    "Paper",
    "Benchmark",
    "SoTATable",
    "ResearchGap",
    "ProblemCandidate",
    "NoveltyReport",
    "CorpusSummary",
]
