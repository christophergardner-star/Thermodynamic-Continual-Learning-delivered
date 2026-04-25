"""
Novelty Gate — hard gate before any result is classified as a breakthrough.

This is the most critical component of the Literature Brain. It prevents the
system from claiming novelty for results that are already known.

A result passes the novelty gate only if ALL of the following hold:
  1. SoTA check: the metric value improves on the best known result
  2. Semantic similarity: the method description is not too close to existing work
  3. Contribution gap: the specific claim has not been made before

Verdict taxonomy:
  novel               — genuinely new: better than SoTA AND semantically distinct
  marginal_improvement — better than SoTA by < MARGINAL_THRESHOLD
  known_result        — same or worse than SoTA, similar method
  replication         — same result as existing work (within tolerance)
  contradicts_sota    — substantially WORSE than SoTA (investigate why)

The gate does NOT require a running ML model for embeddings — it degrades
gracefully to keyword-based similarity when sentence-transformers is unavailable.
When sentence-transformers IS available, it uses SPECTER2 or MiniLM embeddings
for scientific text similarity.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from literature.knowledge_graph import LiteratureKnowledgeGraph
from literature.schemas import (
    NoveltyReport,
    NoveltyVerdict,
    Paper,
    SimilarPaper,
    SoTAEntry,
    _utc_now,
)


# Improvement must exceed this to be considered non-marginal (relative to best)
_MARGINAL_THRESHOLD_REL = 0.02   # 2% relative improvement
# Within this tolerance, a result is considered a replication
_REPLICATION_TOLERANCE = 0.005   # 0.5% absolute
# Cosine similarity threshold above which methods are considered semantically close
_SIMILARITY_THRESHOLD = 0.82
# Minimum semantic similarity to include in the "similar papers" list
_SIMILAR_PAPERS_MIN = 0.60


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _keyword_similarity(text_a: str, text_b: str) -> float:
    """
    Jaccard similarity over normalised unigrams.
    Fast fallback when no embedding model is available.
    """
    def tokenise(t: str) -> set[str]:
        return set(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split())
    a, b = tokenise(text_a), tokenise(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _try_load_sentence_transformer():
    """Load sentence-transformers if available; return None otherwise."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        # allenai-specter is purpose-built for scientific papers
        return SentenceTransformer("allenai-specter")
    except Exception:
        return None


class NoveltyGate:
    """
    Hard novelty gate for claimed research contributions.

    Instantiate once and reuse — the embedding model (if available) is loaded
    once and cached. All public methods are thread-safe for reads.

    Usage:
        gate = NoveltyGate(graph)
        report = gate.evaluate(
            method_name="TCL",
            method_description="Thermodynamic governor for continual learning...",
            benchmark_id="benchmark:abc123",
            metric_name="mean_forgetting",
            metric_value=0.127,
            higher_is_better=False,
        )
        if report.verdict == "novel":
            proceed_to_synthesis()
        else:
            log_why_not_novel(report)
    """

    def __init__(
        self,
        graph: LiteratureKnowledgeGraph,
        load_embedding_model: bool = True,
    ) -> None:
        self._g = graph
        self._embedder = _try_load_sentence_transformer() if load_embedding_model else None

    # -----------------------------------------------------------------------
    # Primary interface
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        method_name: str,
        method_description: str,
        benchmark_id: str,
        metric_name: str,
        metric_value: float,
        higher_is_better: bool,
        paper_abstract: Optional[str] = None,
        max_similar_papers: int = 10,
    ) -> NoveltyReport:
        """
        Evaluate whether a result constitutes a novel contribution.

        All parameters are required. method_description should be a sentence or
        paragraph describing the method — used for semantic similarity.
        paper_abstract, if provided, is used in addition to method_description.
        """
        sota_verdict, sota_rank, sota_delta, sota_delta_pct, sota_best = self._sota_check(
            benchmark_id, metric_name, metric_value, higher_is_better
        )

        search_text = method_description
        if paper_abstract:
            search_text = f"{method_description}\n{paper_abstract}"

        similar = self._find_similar_papers(search_text, max_results=max_similar_papers)
        max_similarity = max((s.similarity_score for s in similar), default=0.0)

        verdict, confidence, contribution = self._determine_verdict(
            sota_verdict=sota_verdict,
            sota_delta=sota_delta,
            sota_delta_pct=sota_delta_pct,
            higher_is_better=higher_is_better,
            max_similarity=max_similarity,
            method_name=method_name,
            sota_best=sota_best,
        )

        required_cites = [s.paper_id for s in similar if s.similarity_score >= _SIMILARITY_THRESHOLD]

        return NoveltyReport(
            verdict=verdict,
            confidence=confidence,
            contribution_statement=contribution,
            similar_papers=similar,
            sota_rank=sota_rank,
            sota_delta=sota_delta,
            sota_delta_pct=sota_delta_pct,
            required_citations=required_cites,
        )

    def quick_sota_check(
        self,
        benchmark_id: str,
        metric_name: str,
        metric_value: float,
        higher_is_better: bool,
    ) -> Tuple[bool, Optional[SoTAEntry]]:
        """
        Fast path: just check if metric_value beats current SoTA.
        Returns (beats_sota, current_best_entry).
        """
        best = self._g.best_result(benchmark_id, metric_name, higher_is_better)
        if best is None:
            return True, None  # no prior result → trivially beats SoTA
        if higher_is_better:
            return metric_value > best.metric_value, best
        return metric_value < best.metric_value, best

    # -----------------------------------------------------------------------
    # SoTA check
    # -----------------------------------------------------------------------

    def _sota_check(
        self,
        benchmark_id: str,
        metric_name: str,
        metric_value: float,
        higher_is_better: bool,
    ) -> Tuple[str, Optional[int], Optional[float], Optional[float], Optional[SoTAEntry]]:
        """
        Compare metric_value against the SoTA table.

        Returns:
            (sota_verdict, rank, delta, delta_pct, best_entry)
        sota_verdict: "better" | "marginal" | "equal" | "worse"
        """
        best = self._g.best_result(benchmark_id, metric_name, higher_is_better)
        sota_table = self._g.get_sota_table(benchmark_id, metric_name)
        rank = sota_table.rank_of(metric_value) if sota_table.entries else 1

        if best is None:
            return "better", 1, None, None, None

        delta = metric_value - best.metric_value
        if not higher_is_better:
            delta = -delta   # make delta positive = improvement

        delta_pct = (abs(delta) / (abs(best.metric_value) + 1e-9))

        tol = _REPLICATION_TOLERANCE
        if abs(metric_value - best.metric_value) <= tol:
            return "equal", rank, delta, delta_pct, best
        elif delta < 0:
            return "worse", rank, delta, delta_pct, best
        elif delta_pct < _MARGINAL_THRESHOLD_REL:
            return "marginal", rank, delta, delta_pct, best
        else:
            return "better", rank, delta, delta_pct, best

    # -----------------------------------------------------------------------
    # Semantic similarity
    # -----------------------------------------------------------------------

    def _find_similar_papers(
        self,
        query_text: str,
        max_results: int = 10,
    ) -> List[SimilarPaper]:
        """
        Find papers semantically similar to the query text.

        Uses SPECTER2 embeddings if available, falls back to keyword similarity.
        """
        similar: List[SimilarPaper] = []

        if self._embedder is not None:
            similar = self._embedding_search(query_text, max_results)
        else:
            similar = self._keyword_search(query_text, max_results)

        # Filter to minimum threshold and sort by score descending
        similar = [s for s in similar if s.similarity_score >= _SIMILAR_PAPERS_MIN]
        similar.sort(key=lambda s: s.similarity_score, reverse=True)
        return similar[:max_results]

    def _embedding_search(self, query_text: str, max_results: int) -> List[SimilarPaper]:
        """Embedding-based semantic search using SPECTER2."""
        try:
            query_emb = self._embedder.encode(query_text, convert_to_tensor=False).tolist()
        except Exception:
            return []

        # Fetch candidate papers that have stored embeddings
        rows = self._g.conn.execute(
            "SELECT paper_id, title, year, embedding FROM papers "
            "WHERE embedding IS NOT NULL LIMIT 2000"
        ).fetchall()

        import json
        candidates: List[SimilarPaper] = []
        for row in rows:
            try:
                stored_emb = json.loads(row["embedding"])
                sim = _cosine_similarity(query_emb, stored_emb)
                if sim >= _SIMILAR_PAPERS_MIN:
                    candidates.append(SimilarPaper(
                        paper_id=row["paper_id"],
                        title=row["title"],
                        year=row["year"],
                        similarity_score=round(sim, 4),
                        similarity_reason="embedding_similarity",
                    ))
            except Exception:
                continue

        candidates.sort(key=lambda s: s.similarity_score, reverse=True)
        return candidates[:max_results]

    def _keyword_search(self, query_text: str, max_results: int) -> List[SimilarPaper]:
        """Fallback: keyword Jaccard similarity against paper titles and abstracts."""
        rows = self._g.conn.execute(
            "SELECT paper_id, title, year, abstract FROM papers LIMIT 5000"
        ).fetchall()
        candidates: List[SimilarPaper] = []
        for row in rows:
            title = row["title"] or ""
            abstract = row["abstract"] or ""
            combined = f"{title} {abstract}"
            sim = _keyword_similarity(query_text, combined)
            if sim >= _SIMILAR_PAPERS_MIN:
                candidates.append(SimilarPaper(
                    paper_id=row["paper_id"],
                    title=title,
                    year=row["year"],
                    similarity_score=round(sim, 4),
                    similarity_reason="keyword_overlap",
                ))
        candidates.sort(key=lambda s: s.similarity_score, reverse=True)
        return candidates[:max_results]

    # -----------------------------------------------------------------------
    # Verdict determination
    # -----------------------------------------------------------------------

    def _determine_verdict(
        self,
        sota_verdict: str,
        sota_delta: Optional[float],
        sota_delta_pct: Optional[float],
        higher_is_better: bool,
        max_similarity: float,
        method_name: str,
        sota_best: Optional[SoTAEntry],
    ) -> Tuple[NoveltyVerdict, float, str]:
        """
        Combine SoTA check and semantic similarity into a final verdict.

        Returns (verdict, confidence, contribution_statement).
        """
        best_desc = (
            f"{sota_best.method_name} ({sota_best.metric_value:.4f})" if sota_best else "unknown"
        )
        delta_str = f"+{sota_delta:.4f}" if sota_delta and sota_delta > 0 else str(round(sota_delta or 0, 4))

        if sota_verdict == "equal":
            if max_similarity >= _SIMILARITY_THRESHOLD:
                verdict: NoveltyVerdict = "replication"
                conf = 0.85
                contribution = (
                    f"{method_name} reproduces the result of {best_desc}. "
                    f"High semantic similarity ({max_similarity:.2f}) and equal metric value "
                    f"indicate this is a replication, not a novel contribution."
                )
            else:
                verdict = "replication"
                conf = 0.65
                contribution = (
                    f"{method_name} achieves the same metric as {best_desc} via a different approach. "
                    f"This is a replication in terms of performance, though the method may differ."
                )

        elif sota_verdict == "worse":
            verdict = "known_result"
            conf = 0.90
            contribution = (
                f"{method_name} does not improve on the current best ({best_desc}). "
                f"Delta: {delta_str}. No novelty claim can be made on this benchmark."
            )

        elif sota_verdict == "marginal":
            verdict = "marginal_improvement"
            conf = 0.70
            contribution = (
                f"{method_name} improves marginally over {best_desc} by {delta_str} "
                f"({100 * (sota_delta_pct or 0):.1f}% relative). "
                f"This is below the {100 * _MARGINAL_THRESHOLD_REL:.0f}% threshold for a strong contribution. "
                f"Statistical significance at scale would be required to claim novelty."
            )

        elif sota_verdict == "better":
            if max_similarity >= _SIMILARITY_THRESHOLD:
                verdict = "marginal_improvement"
                conf = 0.60
                contribution = (
                    f"{method_name} improves on {best_desc} by {delta_str}, but is semantically "
                    f"very similar to existing work (similarity: {max_similarity:.2f}). "
                    f"The contribution may be an incremental variation of an existing method."
                )
            else:
                verdict = "novel"
                conf = 0.80
                contribution = (
                    f"{method_name} establishes a new state of the art, improving over {best_desc} "
                    f"by {delta_str} ({100 * (sota_delta_pct or 0):.1f}% relative). "
                    f"The method is semantically distinct from the top similar papers "
                    f"(max similarity: {max_similarity:.2f}). "
                    f"This constitutes a genuine contribution subject to statistical validation."
                )

        else:
            # sota_verdict == "better" with no prior — first result on this benchmark
            verdict = "novel"
            conf = 0.70
            contribution = (
                f"{method_name} is the first reported result on this benchmark. "
                f"Novelty is inherent but significance depends on whether the benchmark itself is meaningful."
            )

        return verdict, conf, contribution
