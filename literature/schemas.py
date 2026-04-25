"""
Literature Brain schemas.

All data structures flowing through the Literature Brain are defined here.
Pydantic StrictModel throughout for fail-fast validation at API boundaries.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_id(namespace: str, key: str) -> str:
    digest = hashlib.sha256(f"{namespace}:{key}".encode()).hexdigest()[:16]
    return f"{namespace}:{digest}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------

class Author(StrictModel):
    author_id: str
    name: str
    affiliations: List[str] = Field(default_factory=list)
    h_index: Optional[int] = None
    paper_count: Optional[int] = None
    citation_count: Optional[int] = None
    homepage: Optional[str] = None


class Venue(StrictModel):
    name: str
    type: Literal["conference", "journal", "workshop", "preprint", "unknown"] = "unknown"
    # Known top venues for AI/CS/ML
    tier: Literal["top", "strong", "standard", "unknown"] = "unknown"
    abbreviation: Optional[str] = None


class ExternalIDs(StrictModel):
    arxiv: Optional[str] = None
    doi: Optional[str] = None
    semantic_scholar: Optional[str] = None
    pwc: Optional[str] = None
    dblp: Optional[str] = None
    acl: Optional[str] = None
    pubmed: Optional[str] = None


class Paper(StrictModel):
    paper_id: str                              # canonical ID (semantic_scholar preferred)
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    venue_type: Literal["conference", "journal", "workshop", "preprint", "unknown"] = "unknown"
    venue_tier: Literal["top", "strong", "standard", "unknown"] = "unknown"
    authors: List[Author] = Field(default_factory=list)
    fields_of_study: List[str] = Field(default_factory=list)
    citation_count: int = 0
    influential_citation_count: int = 0
    reference_count: int = 0
    external_ids: ExternalIDs = Field(default_factory=ExternalIDs)
    tldr: Optional[str] = None
    # SPECTER2 embedding stored as list[float]; None until fetched
    embedding: Optional[List[float]] = None
    source: Literal["semantic_scholar", "arxiv", "pwc", "manual"] = "semantic_scholar"
    fetched_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Paper title must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# SoTA tracking
# ---------------------------------------------------------------------------

AI_DOMAINS = Literal[
    "continual_learning",
    "image_classification",
    "object_detection",
    "semantic_segmentation",
    "natural_language_processing",
    "language_modelling",
    "question_answering",
    "machine_translation",
    "code_generation",
    "reinforcement_learning",
    "graph_neural_networks",
    "generative_models",
    "multimodal",
    "speech",
    "robotics",
    "meta_learning",
    "few_shot_learning",
    "self_supervised",
    "knowledge_distillation",
    "neural_architecture_search",
    "efficient_inference",
    "safety_alignment",
    "reasoning",
    "other",
]


class Benchmark(StrictModel):
    benchmark_id: str
    name: str
    task: str
    domain: str
    description: Optional[str] = None
    pwc_dataset_slug: Optional[str] = None
    pwc_task_slug: Optional[str] = None
    # Ordered list of metric names and their directionality
    metrics: List[str] = Field(default_factory=list)
    metrics_higher_better: Dict[str, bool] = Field(default_factory=dict)
    # Scale classification — used by gap detector
    scale: Literal["toy", "small", "medium", "large", "xlarge"] = "medium"


class SoTAEntry(StrictModel):
    entry_id: str
    benchmark_id: str
    method_name: str
    metric_name: str
    metric_value: float
    higher_is_better: bool
    paper_id: Optional[str] = None
    paper_title: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    venue_tier: Literal["top", "strong", "standard", "unknown"] = "unknown"
    # Additional metrics reported alongside the primary one
    extra_metrics: Dict[str, float] = Field(default_factory=dict)
    code_available: bool = False
    code_url: Optional[str] = None
    fetched_at: str = Field(default_factory=_utc_now)


class SoTATable(StrictModel):
    """All SoTA entries for a single benchmark, ordered by primary metric."""
    benchmark_id: str
    benchmark_name: str
    primary_metric: str
    higher_is_better: bool
    entries: List[SoTAEntry] = Field(default_factory=list)
    last_updated: str = Field(default_factory=_utc_now)

    def best(self) -> Optional[SoTAEntry]:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.metric_value) if self.higher_is_better \
            else min(self.entries, key=lambda e: e.metric_value)

    def rank_of(self, value: float) -> int:
        """1-indexed rank of a given metric value in this table."""
        if not self.entries:
            return 1
        if self.higher_is_better:
            return sum(1 for e in self.entries if e.metric_value > value) + 1
        return sum(1 for e in self.entries if e.metric_value < value) + 1


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

GapType = Literal[
    "benchmark_coverage",   # method not tested on benchmark
    "scale",                # only tested at small scale
    "replication",          # influential result with no independent replication
    "temporal",             # sota is old — ripe for improvement
    "metric_coverage",      # tested on benchmark but missing key metric
    "ablation",             # claim without ablation support
    "cross_domain",         # technique from domain A not tried in domain B
    "conflict",             # two papers report contradictory results
    "theoretical",          # empirical result without theoretical explanation
    "negative_result",      # important negative result needing broader validation
]


class ResearchGap(StrictModel):
    gap_id: str
    gap_type: GapType
    title: str
    description: str
    domain: str
    benchmark_id: Optional[str] = None
    method_names: List[str] = Field(default_factory=list)
    related_paper_ids: List[str] = Field(default_factory=list)
    # Scoring: each in [0, 1]
    impact_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tractability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Composite = 0.4*impact + 0.35*novelty + 0.25*tractability
    composite_score: float = Field(default=0.0, ge=0.0, le=1.0)
    status: Literal["open", "in_progress", "closed"] = "open"
    detected_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)

    def recompute_composite(self) -> None:
        self.composite_score = round(
            0.40 * self.impact_score
            + 0.35 * self.novelty_score
            + 0.25 * self.tractability_score,
            4,
        )


# ---------------------------------------------------------------------------
# Novelty verification
# ---------------------------------------------------------------------------

NoveltyVerdict = Literal[
    "novel",                # genuinely new contribution
    "marginal_improvement", # improves on existing by < threshold
    "known_result",         # same result already published
    "replication",          # reproduces existing result (valuable, not novel)
    "contradicts_sota",     # result contradicts established SoTA — high-value anomaly
]


class SimilarPaper(StrictModel):
    paper_id: str
    title: str
    year: Optional[int] = None
    similarity_score: float = Field(ge=0.0, le=1.0)
    similarity_reason: str


class NoveltyReport(StrictModel):
    verdict: NoveltyVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    # What specifically is new (empty if verdict != "novel")
    contribution_statement: str
    # Papers most similar to the proposed contribution
    similar_papers: List[SimilarPaper] = Field(default_factory=list)
    # How this ranks against SoTA on the relevant benchmark
    sota_rank: Optional[int] = None
    sota_delta: Optional[float] = None     # proposed_value - current_best
    sota_delta_pct: Optional[float] = None # as percentage of current_best
    # Papers that would need to be cited to contextualise the contribution
    required_citations: List[str] = Field(default_factory=list)
    evaluated_at: str = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# Problem candidates (output of gap detector + ranker)
# ---------------------------------------------------------------------------

class ProblemCandidate(StrictModel):
    problem_id: str
    title: str
    description: str
    domain: str
    gap_ids: List[str] = Field(default_factory=list)
    # Tractable hypothesis — what experiment would address this problem
    proposed_experiment: str
    # What result would confirm/refute the hypothesis
    falsification_criterion: str
    # Estimated compute scale
    compute_estimate: Literal["cpu_hours", "single_gpu_days", "multi_gpu_days", "cluster_weeks"] = "single_gpu_days"
    # Priority ranking
    priority_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Cross-domain opportunity: technique from a different domain
    source_domain: Optional[str] = None
    target_domain: Optional[str] = None
    status: Literal["proposed", "scheduled", "running", "complete", "abandoned"] = "proposed"
    created_at: str = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# Literature corpus summary
# ---------------------------------------------------------------------------

class CorpusSummary(StrictModel):
    total_papers: int = 0
    total_benchmarks: int = 0
    total_sota_entries: int = 0
    total_gaps: int = 0
    open_gaps: int = 0
    top_domains: List[str] = Field(default_factory=list)
    last_arxiv_check: Optional[str] = None
    last_pwc_check: Optional[str] = None
    last_ss_check: Optional[str] = None
    generated_at: str = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# API client responses (internal)
# ---------------------------------------------------------------------------

class FetchResult(StrictModel):
    """Typed wrapper around a raw API response — used internally by API clients."""
    ok: bool
    source: str
    items: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    rate_limited: bool = False
    fetched_at: str = Field(default_factory=_utc_now)
