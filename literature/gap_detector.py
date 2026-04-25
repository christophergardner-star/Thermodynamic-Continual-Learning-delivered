"""
Gap Detector.

Identifies the most promising research problems by analysing what the
literature does NOT know. Seven gap types are detected:

  benchmark_coverage  — method X never evaluated on benchmark Y
  scale               — method only tested on toy/small benchmarks
  replication         — influential result with no independent replication
  temporal            — SoTA is stale (>= STALE_YEARS old), prime for improvement
  metric_coverage     — benchmark evaluated on some metrics but missing others
  cross_domain        — technique from domain A not applied to domain B
  conflict            — two papers report contradictory results on same benchmark

Each gap is scored on three dimensions [0, 1]:
  impact_score      — how many researchers would care? (proxy: citation counts)
  tractability_score — how feasible is it to close? (compute, data, complexity)
  novelty_score     — how undiscovered is this gap?

Composite score: 0.40 × impact + 0.35 × novelty + 0.25 × tractability

The system is designed to surface problems that are hard enough to be
interesting, but tractable enough to make progress on.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from literature.knowledge_graph import LiteratureKnowledgeGraph
from literature.schemas import (
    Benchmark,
    ProblemCandidate,
    ResearchGap,
    SoTAEntry,
    _stable_id,
    _utc_now,
)


# A SoTA result older than this is considered potentially stale
_STALE_YEARS = 2

# Current year (will be refreshed when the module is first imported)
_CURRENT_YEAR = 2026

# Methods that are widely-known baselines — not interesting to re-test everywhere
_BASELINE_METHODS = frozenset({
    "SGD", "Adam", "Fine-tuning", "Finetuning", "Baseline",
    "Random", "Majority", "EWC", "EWC++", "SI", "LWF",
})

# Cross-domain opportunity pairs: (source_domain, target_domain, technique_hint)
_CROSS_DOMAIN_BRIDGES: List[Tuple[str, str, str]] = [
    ("language_modelling",     "continual_learning",      "in-context learning for task-incremental setup"),
    ("continual_learning",     "reinforcement_learning",   "catastrophic forgetting in policy networks"),
    ("image_classification",   "graph_neural_networks",    "vision backbone pre-training for graph tasks"),
    ("self_supervised",        "few_shot_learning",        "self-supervised pre-training for few-shot transfer"),
    ("generative_models",      "data_augmentation",        "diffusion-based augmentation for limited data"),
    ("knowledge_distillation", "continual_learning",       "distillation as anti-forgetting mechanism"),
    ("meta_learning",          "neural_architecture_search", "meta-learned architecture search"),
    ("reinforcement_learning", "reasoning",                "RL-based theorem proving / chain-of-thought"),
    ("efficient_inference",    "continual_learning",       "sparse/pruned networks for catastrophic forgetting"),
    ("safety_alignment",       "continual_learning",       "alignment preservation across sequential tasks"),
]

# Benchmarks that every serious method should be evaluated on, by domain
_EXPECTED_BENCHMARKS: Dict[str, List[str]] = {
    "continual_learning": [
        "continual-learning-on-split-cifar-10",
        "continual-learning-on-split-cifar-100",
        "continual-learning-on-permuted-mnist",
        "continual-learning-on-split-tiny-imagenet",
    ],
    "image_classification": [
        "image-classification-on-imagenet",
        "image-classification-on-cifar-10",
        "image-classification-on-cifar-100",
    ],
    "natural_language_processing": [
        "natural-language-inference-on-mnli",
        "question-answering-on-squad11",
        "question-answering-on-squad20",
    ],
    "graph_neural_networks": [
        "node-classification-on-ogbn-arxiv",
        "link-prediction-on-ogbl-collab",
    ],
}


def _sigmoid_normalise(value: float, scale: float = 1.0) -> float:
    """Map any positive value to [0, 1] via sigmoid with given scale."""
    return 1.0 / (1.0 + math.exp(-value / scale))


def _impact_from_citations(citation_count: int) -> float:
    """Normalise citation count → impact score in [0, 1]."""
    if citation_count <= 0:
        return 0.0
    # log scale: 10 citations → ~0.5, 1000 citations → ~0.87
    return min(1.0, math.log1p(citation_count) / math.log1p(1000))


def _tractability_from_scale(scale: str) -> float:
    """Benchmarks at smaller scale are more tractable to run."""
    return {"toy": 1.0, "small": 0.85, "medium": 0.65, "large": 0.40, "xlarge": 0.15}.get(scale, 0.5)


class GapDetector:
    """
    Analyses the knowledge graph and returns ranked research gaps.

    Usage:
        detector = GapDetector(graph)
        gaps = detector.detect_all(top_n=50)
        for gap in gaps:
            print(gap.composite_score, gap.title)
    """

    def __init__(self, graph: LiteratureKnowledgeGraph) -> None:
        self._g = graph

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def detect_all(
        self,
        top_n: int = 100,
        domain: Optional[str] = None,
    ) -> List[ResearchGap]:
        """
        Run all gap detectors and return a ranked, deduplicated list.

        top_n: maximum number of gaps to return
        domain: restrict to a specific research domain
        """
        all_gaps: List[ResearchGap] = []
        all_gaps.extend(self.detect_benchmark_coverage_gaps(domain=domain))
        all_gaps.extend(self.detect_scale_gaps(domain=domain))
        all_gaps.extend(self.detect_temporal_gaps(domain=domain))
        all_gaps.extend(self.detect_conflict_gaps(domain=domain))
        all_gaps.extend(self.detect_cross_domain_gaps())
        all_gaps.extend(self.detect_replication_gaps(domain=domain))

        # Deduplicate by gap_id
        seen: set[str] = set()
        unique: List[ResearchGap] = []
        for gap in all_gaps:
            gap.recompute_composite()
            if gap.gap_id not in seen:
                seen.add(gap.gap_id)
                unique.append(gap)

        # Sort by composite score descending
        unique.sort(key=lambda g: g.composite_score, reverse=True)
        return unique[:top_n]

    # -----------------------------------------------------------------------
    # Gap type: benchmark coverage
    # -----------------------------------------------------------------------

    def detect_benchmark_coverage_gaps(
        self, domain: Optional[str] = None
    ) -> List[ResearchGap]:
        """
        Find (method, benchmark) pairs where method has not been evaluated.

        Focus on methods that ARE evaluated on some benchmarks in the same domain
        but skipped others — indicating the evaluation is selective rather than
        the method being unknown.
        """
        gaps: List[ResearchGap] = []
        benchmarks = self._g.get_all_benchmarks(domain=domain)

        for bmark in benchmarks:
            expected = _EXPECTED_BENCHMARKS.get(bmark.domain, [])
            if not expected:
                continue

            existing_methods = set(self._g.methods_on_benchmark(bmark.benchmark_id))
            if not existing_methods:
                continue

            # Find methods in this domain's expected benchmarks that skip this one
            for other_slug in expected:
                from literature.schemas import _stable_id as sid
                other_bid = sid("benchmark", other_slug)
                if other_bid == bmark.benchmark_id:
                    continue
                other_methods = set(self._g.methods_on_benchmark(other_bid))
                uncovered = (other_methods - existing_methods) - _BASELINE_METHODS
                if not uncovered:
                    continue

                method_sample = sorted(uncovered)[:5]
                best = self._g.best_result(
                    bmark.benchmark_id,
                    metric_name="accuracy",
                    higher_is_better=True,
                )
                impact = _impact_from_citations(best.year and (2026 - best.year) * 20 or 0)
                tractability = _tractability_from_scale(bmark.scale)

                gap_id = _stable_id("gap_cov", f"{bmark.benchmark_id}:{','.join(method_sample)}")
                gap = ResearchGap(
                    gap_id=gap_id,
                    gap_type="benchmark_coverage",
                    title=f"{len(uncovered)} methods not evaluated on {bmark.name}",
                    description=(
                        f"Methods [{', '.join(method_sample[:3])}{'...' if len(uncovered) > 3 else ''}] "
                        f"are evaluated on related {bmark.domain} benchmarks but have no reported "
                        f"results on {bmark.name}. This leaves an important comparison gap."
                    ),
                    domain=bmark.domain,
                    benchmark_id=bmark.benchmark_id,
                    method_names=method_sample,
                    impact_score=round(impact, 3),
                    tractability_score=round(tractability, 3),
                    novelty_score=0.6,
                )
                gaps.append(gap)

        return gaps

    # -----------------------------------------------------------------------
    # Gap type: scale
    # -----------------------------------------------------------------------

    def detect_scale_gaps(self, domain: Optional[str] = None) -> List[ResearchGap]:
        """
        Find methods only evaluated on toy/small benchmarks within their domain.

        A method that only appears on Split-CIFAR-10 but not Split-CIFAR-100
        has a scale gap — we don't know if it scales.
        """
        gaps: List[ResearchGap] = []
        all_benchmarks = self._g.get_all_benchmarks(domain=domain)

        small_benchmarks = {b.benchmark_id for b in all_benchmarks if b.scale in ("toy", "small")}
        large_benchmarks = {b.benchmark_id for b in all_benchmarks if b.scale in ("large", "xlarge")}

        if not small_benchmarks or not large_benchmarks:
            return gaps

        # Methods evaluated on small but not large benchmarks in same domain
        small_methods: Dict[str, set] = {}
        for bid in small_benchmarks:
            for m in self._g.methods_on_benchmark(bid):
                small_methods.setdefault(m, set()).add(bid)

        large_methods: set = set()
        for bid in large_benchmarks:
            large_methods.update(self._g.methods_on_benchmark(bid))

        scale_gap_methods = {
            m for m in small_methods
            if m not in large_methods and m not in _BASELINE_METHODS
        }

        if scale_gap_methods:
            method_sample = sorted(scale_gap_methods)[:10]
            gap_id = _stable_id("gap_scale", f"{domain or 'all'}:{','.join(method_sample[:5])}")
            gaps.append(ResearchGap(
                gap_id=gap_id,
                gap_type="scale",
                title=f"{len(scale_gap_methods)} methods lack large-scale evaluation in {domain or 'this domain'}",
                description=(
                    f"Methods [{', '.join(method_sample[:5])}{'...' if len(scale_gap_methods) > 5 else ''}] "
                    f"have results on small benchmarks but no evaluation at scale. "
                    f"Scalability is unknown and may be the limiting factor."
                ),
                domain=domain or "other",
                method_names=method_sample,
                impact_score=0.75,
                tractability_score=0.35,  # large-scale runs are expensive
                novelty_score=0.70,
            ))

        return gaps

    # -----------------------------------------------------------------------
    # Gap type: temporal (stale SoTA)
    # -----------------------------------------------------------------------

    def detect_temporal_gaps(self, domain: Optional[str] = None) -> List[ResearchGap]:
        """
        Find benchmarks where the current SoTA result is >= STALE_YEARS old.

        Old SoTA = opportunity for improvement with modern techniques.
        """
        gaps: List[ResearchGap] = []
        benchmarks = self._g.get_all_benchmarks(domain=domain)

        for bmark in benchmarks:
            best = self._g.best_result(bmark.benchmark_id, metric_name="accuracy", higher_is_better=True)
            if best is None or best.year is None:
                continue
            age = _CURRENT_YEAR - best.year
            if age < _STALE_YEARS:
                continue

            # Impact is higher when the benchmark is widely cited (proxy: number of methods)
            n_methods = len(self._g.methods_on_benchmark(bmark.benchmark_id))
            impact = _impact_from_citations(n_methods * 50)
            tractability = _tractability_from_scale(bmark.scale)

            gap_id = _stable_id("gap_temporal", bmark.benchmark_id)
            gaps.append(ResearchGap(
                gap_id=gap_id,
                gap_type="temporal",
                title=f"SoTA on {bmark.name} is {age} years old ({best.method_name}, {best.year})",
                description=(
                    f"The current best result on {bmark.name} ({bmark.domain}) is "
                    f"{best.metric_value:.3f} {best.metric_name} by {best.method_name} in {best.year}. "
                    f"No improvement has been recorded for {age} years, suggesting this benchmark "
                    f"may be solvable with modern techniques."
                ),
                domain=bmark.domain,
                benchmark_id=bmark.benchmark_id,
                related_paper_ids=[best.paper_id] if best.paper_id else [],
                impact_score=round(impact, 3),
                tractability_score=round(tractability, 3),
                novelty_score=min(1.0, age / 4.0),
            ))

        return gaps

    # -----------------------------------------------------------------------
    # Gap type: conflict
    # -----------------------------------------------------------------------

    def detect_conflict_gaps(self, domain: Optional[str] = None) -> List[ResearchGap]:
        """
        Find benchmark/method pairs where different papers report contradictory results.

        A large spread in reported metric values for the same method on the same
        benchmark suggests either hyperparameter sensitivity or reproducibility issues.
        """
        gaps: List[ResearchGap] = []
        benchmarks = self._g.get_all_benchmarks(domain=domain)

        for bmark in benchmarks:
            sota_table = self._g.get_sota_table(bmark.benchmark_id)
            if len(sota_table.entries) < 2:
                continue

            # Group entries by method name
            by_method: Dict[str, List[float]] = {}
            for entry in sota_table.entries:
                by_method.setdefault(entry.method_name, []).append(entry.metric_value)

            for method, values in by_method.items():
                if len(values) < 2 or method in _BASELINE_METHODS:
                    continue
                spread = max(values) - min(values)
                # For accuracy, a spread > 5% is suspicious; for perplexity, > 2 points
                threshold = 0.05 if sota_table.higher_is_better else 2.0
                if spread <= threshold:
                    continue

                gap_id = _stable_id("gap_conflict", f"{bmark.benchmark_id}:{method}")
                gaps.append(ResearchGap(
                    gap_id=gap_id,
                    gap_type="conflict",
                    title=f"Conflicting results for {method} on {bmark.name} (spread: {spread:.3f})",
                    description=(
                        f"{method} reports values from {min(values):.3f} to {max(values):.3f} "
                        f"on {bmark.name} across {len(values)} entries. "
                        f"A spread of {spread:.3f} {bmark.domain} points suggests either "
                        f"hyperparameter sensitivity, implementation differences, or reporting inconsistency. "
                        f"A controlled replication study would resolve this."
                    ),
                    domain=bmark.domain,
                    benchmark_id=bmark.benchmark_id,
                    method_names=[method],
                    impact_score=0.70,
                    tractability_score=0.60,
                    novelty_score=0.65,
                ))

        return gaps

    # -----------------------------------------------------------------------
    # Gap type: cross-domain
    # -----------------------------------------------------------------------

    def detect_cross_domain_gaps(self) -> List[ResearchGap]:
        """
        Identify techniques from domain A that haven't been applied to domain B.

        Uses the pre-registered bridge list. For each bridge, checks whether
        the target domain has results that use the source technique.
        """
        gaps: List[ResearchGap] = []

        for source_domain, target_domain, technique in _CROSS_DOMAIN_BRIDGES:
            source_methods = set()
            for bmark in self._g.get_all_benchmarks(domain=source_domain):
                source_methods.update(self._g.methods_on_benchmark(bmark.benchmark_id))

            target_methods = set()
            for bmark in self._g.get_all_benchmarks(domain=target_domain):
                target_methods.update(self._g.methods_on_benchmark(bmark.benchmark_id))

            if not source_methods:
                continue

            # Proxy: check if any target method names contain source domain keywords
            # This is intentionally simple — the LLM hypothesis generator will refine it
            source_kw = source_domain.replace("_", " ").lower()
            target_has_transfer = any(source_kw in m.lower() for m in target_methods)

            if not target_has_transfer:
                gap_id = _stable_id("gap_xdomain", f"{source_domain}:{target_domain}")
                gaps.append(ResearchGap(
                    gap_id=gap_id,
                    gap_type="cross_domain",
                    title=f"Transfer opportunity: {source_domain} techniques in {target_domain}",
                    description=(
                        f"Specifically: {technique}. "
                        f"The {source_domain} literature has {len(source_methods)} known methods, "
                        f"but evidence of their application to {target_domain} is absent from the graph. "
                        f"Cross-domain transfer in this direction has not been systematically explored."
                    ),
                    domain=target_domain,
                    source_domain=source_domain,    # type: ignore[call-arg]
                    target_domain=target_domain,    # type: ignore[call-arg]
                    method_names=sorted(source_methods)[:5],
                    impact_score=0.72,
                    tractability_score=0.50,
                    novelty_score=0.85,
                ))

        return gaps

    # -----------------------------------------------------------------------
    # Gap type: replication
    # -----------------------------------------------------------------------

    def detect_replication_gaps(self, domain: Optional[str] = None) -> List[ResearchGap]:
        """
        Find influential results with no independent replication in the graph.

        A result is "influential" if the paper has >= MIN_CITATIONS citations.
        Replication is proxied by: another entry for the same method on the same
        benchmark, from a different paper.
        """
        MIN_CITATIONS = 50
        gaps: List[ResearchGap] = []
        benchmarks = self._g.get_all_benchmarks(domain=domain)

        for bmark in benchmarks:
            sota_table = self._g.get_sota_table(bmark.benchmark_id)
            if not sota_table.entries:
                continue

            # Group by method; look for methods with exactly 1 entry and high-citation paper
            by_method: Dict[str, List[SoTAEntry]] = {}
            for entry in sota_table.entries:
                by_method.setdefault(entry.method_name, []).append(entry)

            for method, entries in by_method.items():
                if len(entries) != 1 or method in _BASELINE_METHODS:
                    continue
                entry = entries[0]
                if not entry.paper_id:
                    continue
                paper = self._g.get_paper(entry.paper_id)
                if paper is None or paper.citation_count < MIN_CITATIONS:
                    continue

                gap_id = _stable_id("gap_repl", f"{bmark.benchmark_id}:{method}")
                impact = _impact_from_citations(paper.citation_count)
                gaps.append(ResearchGap(
                    gap_id=gap_id,
                    gap_type="replication",
                    title=f"Replication needed: {method} on {bmark.name} ({paper.citation_count} citations)",
                    description=(
                        f"'{paper.title}' (cited {paper.citation_count} times) reports {entry.metric_value:.3f} "
                        f"{entry.metric_name} for {method} on {bmark.name}. "
                        f"No independent replication appears in the graph. "
                        f"Given the paper's influence, an independent verification is scientifically warranted."
                    ),
                    domain=bmark.domain,
                    benchmark_id=bmark.benchmark_id,
                    method_names=[method],
                    related_paper_ids=[entry.paper_id],
                    impact_score=round(impact, 3),
                    tractability_score=_tractability_from_scale(bmark.scale),
                    novelty_score=0.45,  # replication is valuable but not "novel"
                ))

        return gaps

    # -----------------------------------------------------------------------
    # Problem candidate generation
    # -----------------------------------------------------------------------

    def gaps_to_problems(
        self,
        gaps: Optional[List[ResearchGap]] = None,
        top_n: int = 20,
    ) -> List[ProblemCandidate]:
        """
        Convert top-ranked gaps into structured ProblemCandidates.

        Each candidate includes a proposed experiment and falsification criterion.
        These are used by the hypothesis engine to generate pre-registered protocols.
        """
        if gaps is None:
            gaps = self.detect_all(top_n=top_n * 2)

        problems: List[ProblemCandidate] = []
        for gap in gaps[:top_n]:
            experiment, falsification, compute = self._gap_to_experiment(gap)
            pid = _stable_id("problem", gap.gap_id)
            problems.append(ProblemCandidate(
                problem_id=pid,
                title=gap.title,
                description=gap.description,
                domain=gap.domain,
                gap_ids=[gap.gap_id],
                proposed_experiment=experiment,
                falsification_criterion=falsification,
                compute_estimate=compute,
                priority_score=gap.composite_score,
                source_domain=getattr(gap, "source_domain", None),
                target_domain=getattr(gap, "target_domain", None),
            ))
        return problems

    def _gap_to_experiment(
        self, gap: ResearchGap
    ) -> Tuple[str, str, str]:
        """Return (proposed_experiment, falsification_criterion, compute_estimate)."""
        if gap.gap_type == "benchmark_coverage":
            return (
                f"Evaluate {', '.join(gap.method_names[:3])} on {gap.benchmark_id} "
                f"using the same hyperparameters reported in the original papers.",
                f"If any method achieves within 2% of its result on the related benchmark, "
                f"the gap is confirmed as an evaluation omission rather than a capability failure.",
                "single_gpu_days",
            )
        elif gap.gap_type == "scale":
            return (
                f"Scale the top-performing small-benchmark method(s) to a large benchmark "
                f"in the same domain, with a linear hyperparameter sweep.",
                f"If large-scale performance degrades by > 10% relative to small-scale, "
                f"the method does not scale and this constitutes a negative result.",
                "multi_gpu_days",
            )
        elif gap.gap_type == "temporal":
            return (
                f"Apply modern techniques (pre-training, architectural improvements, "
                f"improved regularisation) to the {gap.benchmark_id} benchmark.",
                f"If no new method improves on the {_STALE_YEARS}-year-old SoTA within "
                f"reasonable compute, the benchmark may be saturated.",
                "single_gpu_days",
            )
        elif gap.gap_type == "conflict":
            return (
                f"Controlled replication of {', '.join(gap.method_names)} on {gap.benchmark_id} "
                f"with fixed random seeds, reported hyperparameters, and independent implementation.",
                f"If the replication result falls within ±1% of the lower reported value, "
                f"the higher value is likely an outlier or reporting error.",
                "single_gpu_days",
            )
        elif gap.gap_type == "cross_domain":
            return (
                f"Adapt the core mechanism from {gap.source_domain} for use in {gap.target_domain}. "  # type: ignore
                f"Run a minimal proof-of-concept on a standard benchmark in the target domain.",
                f"If the adapted method does not outperform a domain-native baseline by a "
                f"statistically significant margin (p < 0.05, d > 0.5), the transfer does not hold.",
                "multi_gpu_days",
            )
        elif gap.gap_type == "replication":
            return (
                f"Independent replication of {', '.join(gap.method_names)} on {gap.benchmark_id} "
                f"from scratch, using only the information in the original paper.",
                f"If the replication result differs by > 5% from the published result, "
                f"a reproducibility failure is confirmed.",
                "single_gpu_days",
            )
        else:
            return (
                f"Investigate gap: {gap.title}",
                f"Gap is closed if a measurable improvement or explanation is produced.",
                "single_gpu_days",
            )
