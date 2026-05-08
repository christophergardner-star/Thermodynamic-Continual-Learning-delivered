"""
TAR External Evidence Ingestor
==============================

Harvests verified outside evidence from Semantic Scholar, arXiv, OpenAlex,
Crossref, and Papers With Code into TAR's persistent literature workspace.
The ingest loop is storage-aware and keeps all large artifacts under the
resolved TAR workspace (preferably E: or D: on this machine).

Outputs:
  {workspace}/tar_state/literature/literature_graph.db
  {workspace}/tar_state/literature/evidence_ingest_state.json
  {workspace}/tar_state/literature/learned_knowledge.json
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

from literature.arxiv_monitor import AI_CATEGORIES, ArXivMonitor
from literature.crossref_client import CrossrefClient
from literature.gap_detector import GapDetector
from literature.knowledge_graph import LiteratureKnowledgeGraph
from literature.openalex_client import OpenAlexClient
from literature.pwc_client import BENCHMARK_REGISTRY, PapersWithCodeClient
from literature.schemas import Paper
from literature.semantic_scholar import SemanticScholarClient
from tar_storage import ensure_workspace_layout, resolve_workspace


_REPO = Path(__file__).resolve().parent
_FAST_INTERVAL_S = 4 * 3600
_DAILY_INTERVAL_S = 24 * 3600
_WEEKLY_INTERVAL_S = 7 * 24 * 3600
_EARLY_FAST_INTERVAL_S = 2 * 3600
_EARLY_DAILY_INTERVAL_S = 12 * 3600
_EARLY_WEEKLY_INTERVAL_S = 3 * 24 * 3600
_EARLY_POLL_INTERVAL_S = 300.0

_DEFAULT_DOMAIN_INPUTS: list[dict[str, Any]] = [
    {
        "id": "general_ai",
        "label": "General AI Research",
        "keywords": ["foundation model", "machine learning", "reasoning", "artificial intelligence"],
        "seed_priority": 50,
    },
    {
        "id": "continual_learning",
        "label": "Continual Learning",
        "keywords": ["continual learning", "catastrophic forgetting", "class incremental learning", "lifelong learning"],
        "seed_priority": 10,
    },
    {
        "id": "thermodynamics_ml",
        "label": "Thermodynamics in ML",
        "keywords": ["thermodynamic machine learning", "activation entropy", "critical regime learning", "sigma star"],
        "seed_priority": 12,
    },
    {
        "id": "medical_ai",
        "label": "Medical AI",
        "keywords": ["medical ai", "clinical machine learning", "diagnostic model", "biomedical learning"],
        "seed_priority": 35,
    },
    {
        "id": "quantum_ml",
        "label": "Quantum / Quantum-Inspired ML",
        "keywords": ["quantum machine learning", "variational circuit", "quantum inspired optimization", "qubit learning"],
        "seed_priority": 38,
    },
    {
        "id": "quantitative_finance",
        "label": "Quantitative Finance",
        "keywords": ["quantitative finance machine learning", "portfolio optimization", "market prediction", "risk model"],
        "seed_priority": 40,
    },
    {
        "id": "multimodal_ai",
        "label": "Multimodal AI",
        "keywords": ["multimodal learning", "vision language model", "cross modal reasoning", "text image model"],
        "seed_priority": 45,
    },
]

_DOMAIN_ARXIV_CATEGORIES: dict[str, list[str]] = {
    "general_ai": ["cs.AI", "cs.LG", "stat.ML"],
    "continual_learning": ["cs.LG", "cs.AI", "stat.ML"],
    "thermodynamics_ml": ["cs.LG", "cond-mat.dis-nn", "stat.ML"],
    "medical_ai": ["cs.LG", "cs.AI", "stat.ML"],
    "quantum_ml": ["quant-ph", "cs.LG", "cs.AI"],
    "quantitative_finance": ["cs.LG", "stat.ML", "math.OC"],
    "multimodal_ai": ["cs.CV", "cs.CL", "cs.LG"],
}

_DOMAIN_BENCHMARK_MAP: dict[str, list[str]] = {
    "continual_learning": ["continual_learning"],
    "thermodynamics_ml": ["continual_learning"],
    "multimodal_ai": ["multimodal"],
    "general_ai": ["image_classification", "natural_language_processing", "reasoning"],
}

_DOMAIN_CONNECTED_TOPICS: dict[str, list[str]] = {
    "general_ai": [
        "statistical learning theory",
        "information theory in machine learning",
        "optimization for deep learning",
    ],
    "continual_learning": [
        "online learning",
        "memory consolidation in neural networks",
        "transfer learning under distribution shift",
    ],
    "thermodynamics_ml": [
        "statistical physics in machine learning",
        "thermodynamics of learning systems",
        "critical phenomena in neural networks",
        "entropy regularization",
    ],
    "medical_ai": [
        "clinical decision support",
        "biostatistics for machine learning",
        "medical imaging foundation models",
    ],
    "quantum_ml": [
        "quantum theory for machine learning",
        "variational quantum algorithms",
        "hamiltonian learning",
        "quantum optimization",
    ],
    "quantitative_finance": [
        "stochastic control in finance",
        "portfolio theory",
        "risk-sensitive optimization",
    ],
    "multimodal_ai": [
        "representation learning",
        "cross-modal alignment",
        "information fusion",
    ],
}

_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "using",
    "learning", "machine", "model", "models", "neural", "based", "toward",
    "under", "over", "study", "paper", "via", "their", "after", "before",
    "across", "task", "tasks", "data", "deep", "systems", "system",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jload(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _score_band(score: float) -> str:
    if score >= 85:
        return "high"
    if score >= 65:
        return "supported"
    if score >= 40:
        return "provisional"
    if score > 0:
        return "weak"
    return "seed"


def _tokenize_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", (text or "").lower())
    return [tok for tok in tokens if tok not in _STOPWORDS]


def _top_terms(texts: list[str], limit: int = 6) -> list[str]:
    counts: dict[str, int] = {}
    for text in texts:
        seen: set[str] = set()
        for tok in _tokenize_keywords(text):
            if tok in seen:
                continue
            seen.add(tok)
            counts[tok] = counts.get(tok, 0) + 1
    return [
        term for term, _ in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    ]


def _learning_maturity_label_for_band(confidence_band: str) -> str:
    if confidence_band == "high":
        return "well-supported working memory"
    if confidence_band in {"supported", "provisional"}:
        return "promising harvested memory"
    if confidence_band == "weak":
        return "tentative harvested memory"
    return "seeded memory"


def _mastery_status_for_metrics(
    *,
    learning_score: float,
    learning_confidence_score: float,
    source_diversity_count: int,
    verified_count: int,
    benchmark_count: int,
    degraded_sources: int,
    uncertainty_flags: list[str],
) -> tuple[str, str]:
    severe_flags = {
        "thin_source_diversity",
        "no_benchmark_grounding",
        "small_verified_corpus",
        "degraded_sources_present",
    }
    severe_count = sum(1 for flag in uncertainty_flags if flag in severe_flags)
    if (
        learning_confidence_score >= 82.0
        and learning_score >= 78.0
        and source_diversity_count >= 5
        and verified_count >= 40
        and benchmark_count >= 1
        and degraded_sources <= 0
        and severe_count == 0
    ):
        return (
            "mastered",
            "Evidence is broad, benchmark-grounded, and diverse enough to treat this domain as genuinely well learned.",
        )
    if learning_confidence_score >= 58.0 and verified_count >= 15 and source_diversity_count >= 3:
        return (
            "working_memory",
            "This domain is evidence-backed working knowledge, but it is not strong enough to claim mastery yet.",
        )
    return (
        "emerging",
        "This domain is still emerging knowledge and should be treated cautiously.",
    )


def _domain_learning_metrics(
    *,
    source_diversity_count: int,
    verified_count: int,
    strong_venue_count: int,
    benchmark_count: int,
    sota_count: int,
    trusted_term_count: int,
    connected_topic_count: int,
    recent_count: int,
    degraded_sources: int,
    rate_limited_sources: int,
) -> dict[str, Any]:
    source_diversity_score = round(_clamp(
        source_diversity_count * 18.0
        + min(10.0, verified_count * 0.18)
        - degraded_sources * 4.0
        - rate_limited_sources * 2.0,
        0.0,
        85.0,
    ), 1)
    verified_depth = min(28.0, math.log1p(verified_count) * 7.5)
    venue_depth = min(14.0, math.log1p(strong_venue_count) * 5.5)
    benchmark_depth = min(18.0, benchmark_count * 7.0 + min(6.0, sota_count * 0.5))
    topic_depth = min(8.0, trusted_term_count * 1.0 + connected_topic_count * 0.4)
    recency_depth = min(8.0, math.log1p(recent_count) * 2.5)

    uncertainty_penalty = 0.0
    uncertainty_flags: list[str] = []
    if source_diversity_count < 4:
        uncertainty_penalty += 12.0
        uncertainty_flags.append("thin_source_diversity")
    if benchmark_count == 0:
        uncertainty_penalty += 8.0
        uncertainty_flags.append("no_benchmark_grounding")
    if verified_count < 12:
        uncertainty_penalty += 6.0
        uncertainty_flags.append("small_verified_corpus")
    if degraded_sources:
        uncertainty_penalty += min(8.0, degraded_sources * 3.0)
        uncertainty_flags.append("degraded_sources_present")
    if rate_limited_sources:
        uncertainty_penalty += min(4.0, rate_limited_sources * 2.0)
        uncertainty_flags.append("rate_limited_sources_present")

    learning_score = round(_clamp(
        verified_depth
        + venue_depth
        + benchmark_depth
        + topic_depth
        + recency_depth
        + source_diversity_score * 0.28
        - uncertainty_penalty,
        0.0,
        92.0,
    ), 1)
    learning_confidence_score = round(_clamp(
        learning_score
        - max(0.0, 18.0 - verified_depth * 0.4)
        - uncertainty_penalty * 0.35,
        0.0,
        90.0,
    ), 1)
    learning_confidence_band = _score_band(learning_confidence_score)
    learning_maturity_label = _learning_maturity_label_for_band(learning_confidence_band)
    mastery_status, mastery_reason = _mastery_status_for_metrics(
        learning_score=learning_score,
        learning_confidence_score=learning_confidence_score,
        source_diversity_count=source_diversity_count,
        verified_count=verified_count,
        benchmark_count=benchmark_count,
        degraded_sources=degraded_sources,
        uncertainty_flags=uncertainty_flags,
    )
    return {
        "source_diversity_score": source_diversity_score,
        "learning_score": learning_score,
        "learning_confidence_score": learning_confidence_score,
        "learning_confidence_band": learning_confidence_band,
        "learning_maturity_label": learning_maturity_label,
        "mastery_status": mastery_status,
        "mastery_reason": mastery_reason,
        "uncertainty_flags": uncertainty_flags[:6],
    }


def _normalize_domain_profile_from_state(
    profile: dict[str, Any],
    *,
    source_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec = dict(profile or {})
    source_mix = rec.get("source_mix", {}) if isinstance(rec.get("source_mix", {}), dict) else {}
    connected_topics = [str(item) for item in rec.get("connected_topics", []) if str(item).strip()]
    trusted_terms = [str(item) for item in rec.get("trusted_terms", []) if str(item).strip()]
    degraded_sources = sum(
        1
        for entry in (source_health or {}).values()
        if isinstance(entry, dict) and not bool(entry.get("ok", True))
    )
    rate_limited_sources = sum(
        1
        for entry in (source_health or {}).values()
        if isinstance(entry, dict) and bool(entry.get("rate_limited", False))
    )
    matched_count = int(rec.get("matched_paper_count", 0) or 0)
    verified_count = int(rec.get("verified_paper_count", 0) or 0)
    recent_count = int(rec.get("recent_paper_count", 0) or 0)
    strong_venue_count = int(rec.get("strong_venue_paper_count", 0) or 0)
    benchmark_count = int(rec.get("benchmark_count", 0) or 0)
    sota_count = int(rec.get("sota_entry_count", 0) or 0)
    source_diversity_count = int(rec.get("source_diversity_count", 0) or len(source_mix))
    learning_metrics = _domain_learning_metrics(
        source_diversity_count=source_diversity_count,
        verified_count=verified_count,
        strong_venue_count=strong_venue_count,
        benchmark_count=benchmark_count,
        sota_count=sota_count,
        trusted_term_count=len(trusted_terms),
        connected_topic_count=len(connected_topics),
        recent_count=recent_count,
        degraded_sources=degraded_sources,
        rate_limited_sources=rate_limited_sources,
    )
    label = str(rec.get("label", "") or rec.get("domain_id", ""))
    rec.update({
        "source_diversity_count": source_diversity_count,
        **learning_metrics,
        "learned_summary": (
            f"{label} currently has {verified_count} verified papers and {matched_count} total matched papers. "
            f"Cross-source diversity={source_diversity_count}. "
            f"This should be treated as {learning_metrics['learning_maturity_label']}, not mastered knowledge. "
            f"Connected topics include {', '.join(connected_topics[:3]) or 'none yet'}."
        ),
    })
    return rec


def _build_learned_knowledge_payload(
    domain_profiles: list[dict[str, Any]],
    *,
    workspace: str = "",
    timestamp: str = "",
) -> dict[str, Any]:
    domains: list[dict[str, Any]] = []
    total_claims = 0
    connected_topic_count = 0
    unique_sources: set[str] = set()
    for profile in domain_profiles:
        if not isinstance(profile, dict):
            continue
        source_mix = profile.get("source_mix", {}) if isinstance(profile.get("source_mix", {}), dict) else {}
        unique_sources.update(str(src) for src in source_mix.keys() if str(src))
        claim_fragments = [str(item) for item in profile.get("claim_fragments", []) if str(item).strip()]
        connected_topics = [str(item) for item in profile.get("connected_topics", []) if str(item).strip()]
        total_claims += len(claim_fragments)
        connected_topic_count += len(connected_topics)
        domains.append({
            "domain_id": str(profile.get("domain_id", "") or ""),
            "label": str(profile.get("label", "") or profile.get("domain_id", "")),
            "learning_score": float(profile.get("learning_score", 0.0) or 0.0),
            "source_diversity_score": float(profile.get("source_diversity_score", 0.0) or 0.0),
            "learning_confidence_score": float(profile.get("learning_confidence_score", 0.0) or 0.0),
            "learning_confidence_band": str(profile.get("learning_confidence_band", "") or ""),
            "learning_maturity_label": str(profile.get("learning_maturity_label", "") or ""),
            "mastery_status": str(profile.get("mastery_status", "emerging") or "emerging"),
            "mastery_reason": str(profile.get("mastery_reason", "") or ""),
            "uncertainty_flags": [str(item) for item in profile.get("uncertainty_flags", []) if str(item).strip()][:6],
            "trusted_terms": [str(item) for item in profile.get("trusted_terms", []) if str(item).strip()][:8],
            "connected_topics": connected_topics[:6],
            "top_verified_titles": [str(item) for item in profile.get("top_verified_titles", []) if str(item).strip()][:4],
            "top_verified_paper_ids": [str(item) for item in profile.get("top_verified_paper_ids", []) if str(item).strip()][:4],
            "claim_fragments": claim_fragments[:5],
            "learned_summary": str(profile.get("learned_summary", "") or ""),
            "source_mix": source_mix,
        })
    domains.sort(
        key=lambda item: (
            -float(item.get("learning_score", 0.0) or 0.0),
            -float(item.get("learning_confidence_score", 0.0) or 0.0),
            -float(item.get("source_diversity_score", 0.0) or 0.0),
            item.get("label", ""),
        )
    )
    mastered_domains = [
        {
            "domain_id": str(item.get("domain_id", "") or ""),
            "label": str(item.get("label", "") or item.get("domain_id", "")),
            "learning_score": float(item.get("learning_score", 0.0) or 0.0),
            "learning_confidence_score": float(item.get("learning_confidence_score", 0.0) or 0.0),
            "mastery_reason": str(item.get("mastery_reason", "") or ""),
        }
        for item in domains
        if str(item.get("mastery_status", "") or "") == "mastered"
    ]
    return {
        "timestamp": timestamp or _now_iso(),
        "workspace": workspace,
        "summary": {
            "domain_count": len(domains),
            "claim_count": total_claims,
            "connected_topic_count": connected_topic_count,
            "source_count": len(unique_sources),
            "mastered_count": len(mastered_domains),
            "mastered_domain_ids": [str(item.get("domain_id", "") or "") for item in mastered_domains],
            "top_learning_domains": [str(item.get("domain_id", "")) for item in domains[:5]],
        },
        "domains": domains,
        "mastered_domains": mastered_domains,
    }


def normalize_literature_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload or {})
    source_health = data.get("source_health", {}) if isinstance(data.get("source_health", {}), dict) else {}
    raw_profiles = data.get("domain_profiles", []) if isinstance(data.get("domain_profiles", []), list) else []
    normalized_profiles = [
        _normalize_domain_profile_from_state(profile, source_health=source_health)
        for profile in raw_profiles
        if isinstance(profile, dict)
    ]
    if normalized_profiles:
        data["domain_profiles"] = normalized_profiles
        learned_timestamp = ""
        learned_workspace = str(data.get("workspace", "") or "")
        learned_existing = data.get("learned_knowledge", {})
        if isinstance(learned_existing, dict):
            learned_timestamp = str(learned_existing.get("timestamp", "") or "")
            learned_workspace = str(learned_existing.get("workspace", "") or learned_workspace)
        learned_payload = _build_learned_knowledge_payload(
            normalized_profiles,
            workspace=learned_workspace,
            timestamp=learned_timestamp or str(data.get("timestamp", "") or ""),
        )
        data["learned_knowledge"] = learned_payload
        summary = dict(data.get("summary", {})) if isinstance(data.get("summary", {}), dict) else {}
        summary.update({
            "learned_domain_count": int(learned_payload.get("summary", {}).get("domain_count", 0) or 0),
            "learned_claim_count": int(learned_payload.get("summary", {}).get("claim_count", 0) or 0),
            "connected_topic_count": int(learned_payload.get("summary", {}).get("connected_topic_count", 0) or 0),
            "source_diversity_count": int(learned_payload.get("summary", {}).get("source_count", 0) or 0),
            "mastered_domain_count": int(learned_payload.get("summary", {}).get("mastered_count", 0) or 0),
        })
        data["summary"] = summary
    return data


@dataclass
class SourceRun:
    source: str
    query: str
    ok: bool
    fetched_count: int = 0
    ingested_count: int = 0
    verified_count: int = 0
    weak_count: int = 0
    error: str = ""
    rate_limited: bool = False


@dataclass
class CycleResult:
    cycle: str
    started_at: str = field(default_factory=_now_iso)
    completed_at: str = ""
    ok: bool = True
    paper_count_before: int = 0
    paper_count_after: int = 0
    benchmark_count_before: int = 0
    benchmark_count_after: int = 0
    sota_count_before: int = 0
    sota_count_after: int = 0
    gap_count_before: int = 0
    gap_count_after: int = 0
    source_runs: list[SourceRun] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def finish(self) -> None:
        self.completed_at = _now_iso()
        self.ok = not self.errors


class ExternalEvidenceIngestor:
    def __init__(
        self,
        workspace: Path,
        *,
        poll_interval_s: float = 900.0,
        fast_interval_s: float = _FAST_INTERVAL_S,
        daily_interval_s: float = _DAILY_INTERVAL_S,
        weekly_interval_s: float = _WEEKLY_INTERVAL_S,
    ) -> None:
        self.workspace = ensure_workspace_layout(workspace, repo_root=_REPO)
        self.root = self.workspace / "tar_state" / "literature"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "evidence_ingest_state.json"
        self.db_path = self.root / "literature_graph.db"
        self.learned_path = self.root / "learned_knowledge.json"
        self.poll_interval_s = poll_interval_s
        self.fast_interval_s = fast_interval_s
        self.daily_interval_s = daily_interval_s
        self.weekly_interval_s = weekly_interval_s

        self.graph = LiteratureKnowledgeGraph(str(self.db_path))
        self.ss = SemanticScholarClient()
        self.arxiv = ArXivMonitor()
        self.openalex = OpenAlexClient()
        self.crossref = CrossrefClient()
        self.pwc = PapersWithCodeClient()
        self.gap_detector = GapDetector(self.graph)
        self._prior_state: dict[str, Any] = {}

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def read_state(self) -> dict[str, Any]:
        return _jload(self.state_path)

    def ensure_state(self) -> dict[str, Any]:
        state = self.read_state()
        if state:
            return state
        payload = self._build_state(cycle_result=None, prior_state={})
        self._write_state(payload)
        return payload

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run_forever,
            name="tar-evidence-ingestor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def run_forever(self) -> None:
        self.ensure_state()
        while not self._stop_event.is_set():
            try:
                self.run_once(force=False)
            except Exception:
                pass
            profile = self._cadence_profile(self.read_state())
            self._stop_event.wait(float(profile.get("poll_interval_s", self.poll_interval_s) or self.poll_interval_s))

    def run_once(self, *, force: bool = False, cycle: str = "auto") -> dict[str, Any]:
        prior_state = self.read_state()
        self._prior_state = prior_state if isinstance(prior_state, dict) else {}
        selected_cycle = cycle if cycle != "auto" else self._select_cycle(prior_state, force=force)
        if selected_cycle == "idle":
            payload = self._build_state(cycle_result=None, prior_state=prior_state)
            self._write_state(payload)
            return payload

        cycle_result = CycleResult(
            cycle=selected_cycle,
            paper_count_before=self.graph.paper_count(),
            benchmark_count_before=self.graph.benchmark_count(),
            sota_count_before=self.graph.sota_entry_count(),
            gap_count_before=self.graph.gap_count("open") + self.graph.gap_count("in_progress") + self.graph.gap_count("closed"),
        )
        try:
            queries = self._build_query_plan()
            if selected_cycle == "fast":
                self._run_fast_cycle(cycle_result, queries)
            elif selected_cycle == "daily":
                self._run_daily_cycle(cycle_result, queries)
            elif selected_cycle == "weekly":
                self._run_weekly_cycle(cycle_result, queries)
            elif selected_cycle == "bootstrap":
                self._run_fast_cycle(cycle_result, queries)
                self._run_daily_cycle(cycle_result, queries)
                self._run_weekly_cycle(cycle_result, queries)
            else:
                cycle_result.errors.append(f"unknown_cycle:{selected_cycle}")
        except Exception as exc:
            cycle_result.errors.append(str(exc))

        cycle_result.paper_count_after = self.graph.paper_count()
        cycle_result.benchmark_count_after = self.graph.benchmark_count()
        cycle_result.sota_count_after = self.graph.sota_entry_count()
        cycle_result.gap_count_after = self.graph.gap_count("open") + self.graph.gap_count("in_progress") + self.graph.gap_count("closed")
        cycle_result.finish()

        payload = self._build_state(cycle_result=cycle_result, prior_state=prior_state)
        self._write_state(payload)
        return payload

    def _select_cycle(self, prior_state: dict[str, Any], *, force: bool) -> str:
        if force:
            return "daily"
        cycles = prior_state.get("cycles", {}) if isinstance(prior_state, dict) else {}
        cadence = self._cadence_profile(prior_state)
        now = time.time()
        last_fast = float(cycles.get("fast", {}).get("completed_ts", 0.0) or 0.0)
        last_daily = float(cycles.get("daily", {}).get("completed_ts", 0.0) or 0.0)
        last_weekly = float(cycles.get("weekly", {}).get("completed_ts", 0.0) or 0.0)
        if not cycles:
            return "bootstrap"
        if now - last_weekly >= float(cadence.get("weekly_interval_s", self.weekly_interval_s) or self.weekly_interval_s):
            return "weekly"
        if now - last_daily >= float(cadence.get("daily_interval_s", self.daily_interval_s) or self.daily_interval_s):
            return "daily"
        if now - last_fast >= float(cadence.get("fast_interval_s", self.fast_interval_s) or self.fast_interval_s):
            return "fast"
        return "idle"

    def _cadence_profile(
        self,
        prior_state: dict[str, Any],
        *,
        total_papers: int | None = None,
        verified_total: int | None = None,
        domain_profile_count: int | None = None,
    ) -> dict[str, Any]:
        override = str(os.environ.get("TAR_EVIDENCE_CADENCE_MODE", "") or "").strip().lower()
        cycles = prior_state.get("cycles", {}) if isinstance(prior_state, dict) else {}
        summary = prior_state.get("summary", {}) if isinstance(prior_state, dict) else {}

        total_papers = int(
            total_papers
            if total_papers is not None
            else (summary.get("literature_total_papers", 0) or 0)
        )
        verified_total = int(
            verified_total
            if verified_total is not None
            else (summary.get("external_verified_sources", 0) or 0)
        )
        domain_profile_count = int(
            domain_profile_count
            if domain_profile_count is not None
            else (summary.get("domain_profile_count", 0) or 0)
        )

        completed_ts = [
            float((cycles.get(name, {}) or {}).get("completed_ts", 0.0) or 0.0)
            for name in ("fast", "daily", "weekly")
            if float((cycles.get(name, {}) or {}).get("completed_ts", 0.0) or 0.0) > 0.0
        ]
        first_completed_ts = min(completed_ts) if completed_ts else 0.0
        age_days = ((time.time() - first_completed_ts) / 86400.0) if first_completed_ts else 0.0

        mode = "standard"
        reason = "mature corpus"
        if override in {"accelerated", "early", "aggressive"}:
            mode = "accelerated"
            reason = "environment override"
        elif override in {"standard", "normal"}:
            mode = "standard"
            reason = "environment override"
        elif (
            total_papers < 250
            or verified_total < 30
            or domain_profile_count < 6
            or age_days < 21.0
        ):
            mode = "accelerated"
            reason = (
                f"early-stage evidence base "
                f"(papers={total_papers}, verified={verified_total}, domains={domain_profile_count}, age_days={age_days:.1f})"
            )

        if mode == "accelerated":
            return {
                "mode": mode,
                "reason": reason,
                "poll_interval_s": min(float(self.poll_interval_s), _EARLY_POLL_INTERVAL_S),
                "fast_interval_s": min(float(self.fast_interval_s), float(_EARLY_FAST_INTERVAL_S)),
                "daily_interval_s": min(float(self.daily_interval_s), float(_EARLY_DAILY_INTERVAL_S)),
                "weekly_interval_s": min(float(self.weekly_interval_s), float(_EARLY_WEEKLY_INTERVAL_S)),
            }
        return {
            "mode": mode,
            "reason": reason,
            "poll_interval_s": float(self.poll_interval_s),
            "fast_interval_s": float(self.fast_interval_s),
            "daily_interval_s": float(self.daily_interval_s),
            "weekly_interval_s": float(self.weekly_interval_s),
        }

    def _build_query_plan(self) -> dict[str, Any]:
        director_state = _jload(self.workspace / "tar_state" / "research_director_state.json")
        frontiers = director_state.get("frontier_directives", []) if isinstance(director_state.get("frontier_directives"), list) else []
        domain_inputs = director_state.get("knowledge_domains", []) if isinstance(director_state.get("knowledge_domains"), list) else []
        if not domain_inputs:
            domain_inputs = [dict(item) for item in _DEFAULT_DOMAIN_INPUTS]

        domain_inputs = [
            {
                "id": str(item.get("id", "")),
                "label": str(item.get("label", item.get("id", ""))),
                "keywords": list(item.get("keywords", [])) if isinstance(item.get("keywords"), list) else [],
                "seed_priority": int(item.get("seed_priority", 50) or 50),
                "expansion_status": str(item.get("expansion_status", "")),
            }
            for item in domain_inputs
            if str(item.get("id", ""))
        ]
        domain_inputs.sort(
            key=lambda item: (
                item.get("expansion_status") not in {"active_expansion", "stabilizing"},
                int(item.get("seed_priority", 50)),
                item.get("label", ""),
            )
        )
        top_domains = domain_inputs[:5]

        queries: list[str] = []
        connected_queries: list[str] = []
        for frontier in frontiers[:4]:
            title = str(frontier.get("title", "") or "").strip()
            if title:
                queries.append(title)
        for domain in top_domains:
            query = self._domain_query(domain)
            if query:
                queries.append(query)
            connected_queries.extend(_DOMAIN_CONNECTED_TOPICS.get(str(domain.get("id", "")), [])[:3])

        categories: set[str] = set()
        for domain in top_domains:
            categories.update(_DOMAIN_ARXIV_CATEGORIES.get(str(domain.get("id", "")), []))
        if not categories:
            categories.update(list(AI_CATEGORIES.keys())[:6])

        return {
            "queries": self._dedupe_keep_order(queries)[:8],
            "connected_queries": self._dedupe_keep_order(connected_queries)[:10],
            "domains": top_domains,
            "frontiers": frontiers[:6],
            "categories": sorted(categories),
        }

    def _domain_query(self, domain: dict[str, Any]) -> str:
        keywords = [str(item).strip() for item in domain.get("keywords", []) if str(item).strip()]
        phrase = next((item for item in keywords if " " in item), "")
        if phrase:
            return phrase
        label = str(domain.get("label", "") or "").strip()
        return label if label else "machine learning"

    @staticmethod
    def _dedupe_keep_order(values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out

    def _preferred_sources(self, *, connected: bool = False) -> list[str]:
        base = (
            ["openalex", "crossref", "semantic_scholar", "arxiv"]
            if connected else
            ["semantic_scholar", "openalex", "arxiv", "crossref"]
        )
        health = self._prior_state.get("source_health", {}) if isinstance(self._prior_state, dict) else {}

        def _rank(source_name: str) -> tuple[int, int, int]:
            entry = health.get(source_name, {}) if isinstance(health, dict) else {}
            last_error = str(entry.get("last_error", "") or "").lower()
            rate_limited = int("429" in last_error or "rate" in last_error)
            unhealthy = int(not bool(entry.get("ok", True)))
            return (rate_limited, unhealthy, base.index(source_name))

        return sorted(base, key=_rank)

    def _search_source(self, source: str, query: str, *, connected: bool = False) -> tuple[SourceRun, list[dict[str, Any]]]:
        if source == "semantic_scholar":
            result = self.ss.search_by_topic(
                query,
                min_year=2021 if connected else 2022,
                min_citations=3 if connected else 5,
                max_results=18 if connected else 25,
            )
        elif source == "openalex":
            result = self.openalex.search_topic(
                query,
                min_year=2021 if connected else 2022,
                max_results=20 if connected else 24,
            )
        elif source == "crossref":
            result = self.crossref.search_topic(
                query,
                min_year=2020 if connected else 2021,
                max_results=14 if connected else 16,
            )
        elif source == "arxiv":
            result = self.arxiv.search_topic(
                query,
                min_year=2022 if connected else 2023,
                max_results=12 if connected else 15,
            )
        else:
            run = SourceRun(source=source, query=query, ok=False, error="unknown_source")
            return run, []

        run = SourceRun(
            source=source,
            query=query,
            ok=result.ok,
            fetched_count=len(result.items),
            error=str(result.error or "") if not result.ok else "",
            rate_limited=bool(getattr(result, "rate_limited", False)),
        )
        return run, list(result.items)

    def _run_fast_cycle(self, cycle_result: CycleResult, queries: dict[str, Any]) -> None:
        result = self.arxiv.latest(
            categories=queries.get("categories") or ["cs.LG", "cs.AI", "stat.ML"],
            days_back=2,
            max_results=200,
        )
        run = SourceRun(
            source="arxiv",
            query="latest",
            ok=result.ok,
            fetched_count=len(result.items),
            rate_limited=bool(getattr(result, "rate_limited", False)),
        )
        if not result.ok:
            run.error = str(result.error or "fetch_failed")
            cycle_result.errors.append(f"arxiv_latest:{run.error}")
        else:
            ingested, verified, weak = self._ingest_papers(result.items)
            run.ingested_count = ingested
            run.verified_count = verified
            run.weak_count = weak
        cycle_result.source_runs.append(run)

        latest_topics = list(queries.get("queries", []))[:2] + list(queries.get("connected_queries", []))[:2]
        oa_result = self.openalex.latest(topics=latest_topics, days_back=5, max_results=30)
        oa_run = SourceRun(
            source="openalex",
            query="latest",
            ok=oa_result.ok,
            fetched_count=len(oa_result.items),
            rate_limited=bool(getattr(oa_result, "rate_limited", False)),
        )
        if not oa_result.ok:
            oa_run.error = str(oa_result.error or "fetch_failed")
            cycle_result.errors.append(f"openalex_latest:{oa_run.error}")
        else:
            ingested, verified, weak = self._ingest_papers(oa_result.items)
            oa_run.ingested_count = ingested
            oa_run.verified_count = verified
            oa_run.weak_count = weak
        cycle_result.source_runs.append(oa_run)

    def _run_daily_cycle(self, cycle_result: CycleResult, queries: dict[str, Any]) -> None:
        primary_queries = list(queries.get("queries", []))[:5]
        connected_queries = list(queries.get("connected_queries", []))[:6]

        for query in primary_queries:
            for source in self._preferred_sources(connected=False):
                run, items = self._search_source(source, query, connected=False)
                if not run.ok:
                    cycle_result.errors.append(f"{source}:{query}:{run.error or 'fetch_failed'}")
                else:
                    ingested, verified, weak = self._ingest_papers(items)
                    run.ingested_count = ingested
                    run.verified_count = verified
                    run.weak_count = weak
                cycle_result.source_runs.append(run)

        for query in connected_queries:
            for source in self._preferred_sources(connected=True)[:3]:
                run, items = self._search_source(source, query, connected=True)
                if not run.ok:
                    cycle_result.errors.append(f"{source}:{query}:{run.error or 'fetch_failed'}")
                else:
                    ingested, verified, weak = self._ingest_papers(items)
                    run.ingested_count = ingested
                    run.verified_count = verified
                    run.weak_count = weak
                cycle_result.source_runs.append(run)

        try:
            gaps = self.gap_detector.detect_all(top_n=120)
            for gap in gaps:
                self.graph.upsert_gap(gap)
        except Exception as exc:
            cycle_result.errors.append(f"gap_detect:{exc}")

    def _run_weekly_cycle(self, cycle_result: CycleResult, queries: dict[str, Any]) -> None:
        domain_ids = [str(domain.get("id", "")) for domain in queries.get("domains", [])]
        benchmark_domains: set[str] = set()
        for domain_id in domain_ids:
            benchmark_domains.update(_DOMAIN_BENCHMARK_MAP.get(domain_id, []))
        if not benchmark_domains:
            benchmark_domains.add("continual_learning")

        slugs = [
            slug for slug, meta in BENCHMARK_REGISTRY.items()
            if meta.get("domain") in benchmark_domains
        ][:18]
        for slug in slugs:
            try:
                benchmark = self.pwc.to_benchmark_schema(slug)
                self.graph.upsert_benchmark(benchmark)
                table = self.pwc.get_sota_table(slug)
                run = SourceRun(source="papers_with_code", query=slug, ok=table is not None, fetched_count=1 if table else 0)
                if table is None:
                    run.error = "no_sota_table"
                    cycle_result.errors.append(f"pwc:{slug}:no_sota_table")
                else:
                    added = self.graph.upsert_sota_table(table)
                    run.ingested_count = added
                    run.verified_count = added
                cycle_result.source_runs.append(run)
            except Exception as exc:
                cycle_result.errors.append(f"pwc:{slug}:{exc}")
                cycle_result.source_runs.append(SourceRun(source="papers_with_code", query=slug, ok=False, error=str(exc)))

        try:
            rows = self.graph.conn.execute(
                "SELECT paper_id FROM papers "
                "WHERE influential_citation_count >= 10 "
                "ORDER BY influential_citation_count DESC, citation_count DESC LIMIT 20"
            ).fetchall()
            for row in rows:
                paper_id = row["paper_id"]
                cit_result = self.ss.get_citations(paper_id, max_results=50, min_year=2022)
                run = SourceRun(source="semantic_scholar", query=f"citations:{paper_id}", ok=cit_result.ok, fetched_count=len(cit_result.items))
                if not cit_result.ok:
                    run.error = str(cit_result.error or "fetch_failed")
                    cycle_result.errors.append(f"citations:{paper_id}:{run.error}")
                else:
                    ingested, verified, weak = self._ingest_papers(cit_result.items)
                    run.ingested_count = ingested
                    run.verified_count = verified
                    run.weak_count = weak
                    for item in cit_result.items:
                        citing_id = str(item.get("paper_id", "") or "")
                        if citing_id:
                            self.graph.add_citation(citing_id, paper_id)
                cycle_result.source_runs.append(run)
        except Exception as exc:
            cycle_result.errors.append(f"citation_expand:{exc}")

    def _ingest_papers(self, items: list[dict[str, Any]]) -> tuple[int, int, int]:
        ingested = 0
        verified = 0
        weak = 0
        for item in items:
            try:
                paper = Paper(**item)
            except Exception:
                continue
            self.graph.upsert_paper(paper)
            ingested += 1
            if self._paper_is_verified(paper):
                verified += 1
            else:
                weak += 1
        return ingested, verified, weak

    def _paper_is_verified(self, paper: Paper) -> bool:
        if paper.source == "semantic_scholar":
            if paper.venue_tier in {"top", "strong"}:
                return True
            if paper.citation_count >= 20 or paper.influential_citation_count >= 5:
                return True
        if paper.source == "openalex":
            if paper.venue_tier in {"top", "strong"}:
                return True
            if paper.citation_count >= 25:
                return True
        if paper.source == "crossref":
            doi = getattr(paper.external_ids, "doi", None)
            if doi and paper.venue_tier in {"top", "strong"}:
                return True
            if doi and paper.citation_count >= 35:
                return True
        return False

    def _build_learned_knowledge(self, domain_profiles: list[dict[str, Any]]) -> dict[str, Any]:
        return _build_learned_knowledge_payload(
            domain_profiles,
            workspace=str(self.workspace),
            timestamp=_now_iso(),
        )

    def _build_state(self, *, cycle_result: CycleResult | None, prior_state: dict[str, Any]) -> dict[str, Any]:
        corpus = self.graph.corpus_summary()
        domain_inputs = self._build_query_plan().get("domains", [])
        domain_profiles = self._build_domain_profiles(domain_inputs)
        learned_knowledge = self._build_learned_knowledge(domain_profiles)
        cycles = dict(prior_state.get("cycles", {})) if isinstance(prior_state, dict) else {}
        last_errors = list(prior_state.get("last_errors", [])) if isinstance(prior_state, dict) else []

        if cycle_result is not None:
            cycles[cycle_result.cycle] = {
                "started_at": cycle_result.started_at,
                "completed_at": cycle_result.completed_at,
                "completed_ts": time.time(),
                "ok": cycle_result.ok,
                "papers_added": max(0, cycle_result.paper_count_after - cycle_result.paper_count_before),
                "benchmarks_added": max(0, cycle_result.benchmark_count_after - cycle_result.benchmark_count_before),
                "sota_added": max(0, cycle_result.sota_count_after - cycle_result.sota_count_before),
                "gaps_added": max(0, cycle_result.gap_count_after - cycle_result.gap_count_before),
                "errors": cycle_result.errors[:10],
            }
            last_errors = cycle_result.errors[:12]

        verified_total = sum(int(profile.get("verified_paper_count", 0) or 0) for profile in domain_profiles)
        weak_total = sum(int(profile.get("weak_paper_count", 0) or 0) for profile in domain_profiles)
        cadence = self._cadence_profile(
            {"cycles": cycles, "summary": dict(prior_state.get("summary", {})) if isinstance(prior_state, dict) else {}},
            total_papers=int(corpus.total_papers or 0),
            verified_total=verified_total,
            domain_profile_count=len(domain_profiles),
        )
        next_due: dict[str, dict[str, Any]] = {}
        for cycle_name in ("fast", "daily", "weekly"):
            interval_s = float(cadence.get(f"{cycle_name}_interval_s", 0.0) or 0.0)
            last_completed_ts = float((cycles.get(cycle_name, {}) or {}).get("completed_ts", 0.0) or 0.0)
            if last_completed_ts <= 0.0 or interval_s <= 0.0:
                next_due[cycle_name] = {"due_at": "", "due_in_s": 0.0}
                continue
            due_ts = last_completed_ts + interval_s
            next_due[cycle_name] = {
                "due_at": datetime.fromtimestamp(due_ts, timezone.utc).isoformat(),
                "due_in_s": round(max(0.0, due_ts - time.time()), 1),
            }
        if cycle_result is not None:
            source_health = self._source_health(cycle_result)
        else:
            source_health = self._source_health(None)
            prior_health = prior_state.get("source_health", {}) if isinstance(prior_state, dict) else {}
            if isinstance(prior_health, dict):
                for source_name, entry in prior_health.items():
                    if not isinstance(entry, dict):
                        continue
                    merged = dict(source_health.get(source_name, {}))
                    merged.update(entry)
                    source_health[source_name] = merged

        summary = {
            "literature_total_papers": corpus.total_papers,
            "literature_total_benchmarks": corpus.total_benchmarks,
            "literature_total_sota_entries": corpus.total_sota_entries,
            "literature_open_gaps": corpus.open_gaps,
            "external_verified_sources": verified_total,
            "external_weak_sources": weak_total,
            "domain_profile_count": len(domain_profiles),
            "last_literature_sync": cycle_result.completed_at if cycle_result else str(prior_state.get("summary", {}).get("last_literature_sync", "")),
            "last_cycle": cycle_result.cycle if cycle_result else str(prior_state.get("summary", {}).get("last_cycle", "idle")),
            "status": "ok" if not last_errors else "degraded",
            "cadence_mode": str(cadence.get("mode", "standard")),
            "cadence_reason": str(cadence.get("reason", "")),
            "poll_interval_s": float(cadence.get("poll_interval_s", self.poll_interval_s) or self.poll_interval_s),
            "fast_interval_s": float(cadence.get("fast_interval_s", self.fast_interval_s) or self.fast_interval_s),
            "daily_interval_s": float(cadence.get("daily_interval_s", self.daily_interval_s) or self.daily_interval_s),
            "weekly_interval_s": float(cadence.get("weekly_interval_s", self.weekly_interval_s) or self.weekly_interval_s),
            "learned_domain_count": int(learned_knowledge.get("summary", {}).get("domain_count", 0) or 0),
            "learned_claim_count": int(learned_knowledge.get("summary", {}).get("claim_count", 0) or 0),
            "connected_topic_count": int(learned_knowledge.get("summary", {}).get("connected_topic_count", 0) or 0),
            "source_diversity_count": int(learned_knowledge.get("summary", {}).get("source_count", 0) or 0),
        }

        payload = {
            "timestamp": _now_iso(),
            "workspace": str(self.workspace),
            "storage_drive": self.workspace.drive or self.workspace.anchor,
            "db_path": str(self.db_path),
            "state_path": str(self.state_path),
            "summary": summary,
            "cycles": cycles,
            "cadence": {
                **cadence,
                "next_due": next_due,
            },
            "source_health": source_health,
            "domain_profiles": domain_profiles,
            "learned_knowledge": learned_knowledge,
            "recent_queries": self._build_query_plan().get("queries", []),
            "recent_connected_queries": self._build_query_plan().get("connected_queries", []),
            "recent_harvests": [asdict(run) for run in (cycle_result.source_runs[:20] if cycle_result else [])],
            "last_errors": last_errors[:12],
        }
        return payload

    def _source_health(self, cycle_result: CycleResult | None) -> dict[str, dict[str, Any]]:
        health: dict[str, dict[str, Any]] = {
            "semantic_scholar": {"ok": True, "last_error": "", "last_sync": "", "ingested": 0, "rate_limited": False},
            "arxiv": {"ok": True, "last_error": "", "last_sync": "", "ingested": 0, "rate_limited": False},
            "openalex": {"ok": True, "last_error": "", "last_sync": "", "ingested": 0, "rate_limited": False},
            "crossref": {"ok": True, "last_error": "", "last_sync": "", "ingested": 0, "rate_limited": False},
            "papers_with_code": {"ok": True, "last_error": "", "last_sync": "", "ingested": 0, "rate_limited": False},
        }
        if cycle_result is None:
            return health
        for run in cycle_result.source_runs:
            entry = health.setdefault(run.source, {"ok": True, "last_error": "", "last_sync": "", "ingested": 0, "rate_limited": False})
            entry["ok"] = bool(entry["ok"]) and bool(run.ok)
            entry["last_sync"] = cycle_result.completed_at
            entry["ingested"] = int(entry.get("ingested", 0) or 0) + int(run.ingested_count or 0)
            entry["rate_limited"] = bool(entry.get("rate_limited", False)) or bool(run.rate_limited)
            if run.error:
                entry["last_error"] = run.error
        return health

    def _build_domain_profiles(self, domain_inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = self.graph.conn.execute(
            "SELECT paper_id, title, abstract, year, venue, venue_tier, source, "
            "citation_count, influential_citation_count, fields_of_study "
            "FROM papers ORDER BY COALESCE(year, 0) DESC, citation_count DESC LIMIT 5000"
        ).fetchall()
        profiles: list[dict[str, Any]] = []
        health = self._prior_state.get("source_health", {}) if isinstance(self._prior_state, dict) else {}
        degraded_sources = sum(
            1
            for entry in (health.values() if isinstance(health, dict) else [])
            if isinstance(entry, dict) and not bool(entry.get("ok", True))
        )
        rate_limited_sources = sum(
            1
            for entry in (health.values() if isinstance(health, dict) else [])
            if isinstance(entry, dict) and bool(entry.get("rate_limited", False))
        )

        for domain in domain_inputs:
            domain_id = str(domain.get("id", "") or "")
            label = str(domain.get("label", domain_id) or domain_id)
            keywords = [str(item).lower() for item in domain.get("keywords", []) if str(item).strip()]
            matched_rows = []
            for row in rows:
                text = " ".join([
                    str(row["title"] or ""),
                    str(row["abstract"] or ""),
                    str(row["fields_of_study"] or ""),
                ]).lower()
                if any(keyword in text for keyword in keywords):
                    matched_rows.append(row)

            verified_rows = [row for row in matched_rows if self._row_is_verified(row)]
            weak_rows = [row for row in matched_rows if not self._row_is_verified(row)]
            recent_rows = [row for row in matched_rows if int(row["year"] or 0) >= 2024]
            strong_venue_rows = [row for row in matched_rows if str(row["venue_tier"] or "") in {"top", "strong"}]
            benchmark_count, sota_count = self._domain_benchmark_counts(domain_id)
            citation_weight = sum(int(row["citation_count"] or 0) for row in verified_rows[:30])
            truth_delta = round(_clamp(len(verified_rows) * 3.8 + len(strong_venue_rows) * 2.4 + benchmark_count * 4.5 + min(14.0, citation_weight / 250.0), 0.0, 45.0), 1)
            proficiency_delta = round(_clamp(len(matched_rows) * 0.55 + len(verified_rows) * 1.8 + benchmark_count * 2.0 + sota_count * 0.18, 0.0, 35.0), 1)

            source_mix: dict[str, int] = {}
            for row in matched_rows:
                source = str(row["source"] or "unknown")
                source_mix[source] = source_mix.get(source, 0) + 1

            verified_texts = [
                " ".join([
                    str(row["title"] or ""),
                    str(row["abstract"] or ""),
                    str(row["fields_of_study"] or ""),
                ])
                for row in verified_rows[:40]
            ]
            trusted_terms = _top_terms(verified_texts, limit=8)
            connected_topics = _DOMAIN_CONNECTED_TOPICS.get(domain_id, [])[:5]
            source_diversity_count = len(source_mix)
            learning_metrics = _domain_learning_metrics(
                source_diversity_count=source_diversity_count,
                verified_count=len(verified_rows),
                strong_venue_count=len(strong_venue_rows),
                benchmark_count=benchmark_count,
                sota_count=sota_count,
                trusted_term_count=len(trusted_terms),
                connected_topic_count=len(connected_topics),
                recent_count=len(recent_rows),
                degraded_sources=degraded_sources,
                rate_limited_sources=rate_limited_sources,
            )

            top_titles = [
                str(row["title"] or "")
                for row in matched_rows[:4]
                if str(row["title"] or "")
            ]
            top_verified_titles = [
                str(row["title"] or "")
                for row in verified_rows[:3]
                if str(row["title"] or "")
            ]
            top_verified_ids = [
                str(row["paper_id"] or "")
                for row in verified_rows[:3]
                if str(row["paper_id"] or "")
            ]
            claim_fragments: list[str] = []
            if trusted_terms:
                claim_fragments.append(
                    f"Trusted literature themes concentrate around {', '.join(trusted_terms[:4])}."
                )
            if source_diversity_count > 1:
                claim_fragments.append(
                    f"Evidence is corroborated across {source_diversity_count} sources ({', '.join(sorted(source_mix)[:4])})."
                )
            if benchmark_count or sota_count:
                claim_fragments.append(
                    f"Structured benchmark evidence exists: {benchmark_count} benchmarks and {sota_count} SoTA entries."
                )
            if learning_metrics["uncertainty_flags"]:
                claim_fragments.append(
                    "Confidence is still limited by "
                    + ", ".join(
                        flag.replace("_", " ")
                        for flag in learning_metrics["uncertainty_flags"][:3]
                    )
                    + "."
                )
            learned_summary = (
                f"{label} currently has {len(verified_rows)} verified papers and {len(matched_rows)} total matched papers. "
                f"Cross-source diversity={source_diversity_count}. "
                f"This should be treated as {learning_metrics['learning_maturity_label']}, not mastered knowledge. "
                f"Connected topics include {', '.join(connected_topics[:3]) or 'none yet'}."
            )
            profiles.append({
                "domain_id": domain_id,
                "label": label,
                "query": self._domain_query(domain),
                "matched_paper_count": len(matched_rows),
                "verified_paper_count": len(verified_rows),
                "weak_paper_count": len(weak_rows),
                "recent_paper_count": len(recent_rows),
                "strong_venue_paper_count": len(strong_venue_rows),
                "benchmark_count": benchmark_count,
                "sota_entry_count": sota_count,
                "truth_delta": truth_delta,
                "proficiency_delta": proficiency_delta,
                "source_mix": source_mix,
                "source_diversity_count": source_diversity_count,
                **learning_metrics,
                "connected_topics": connected_topics,
                "trusted_terms": trusted_terms,
                "learned_summary": learned_summary,
                "claim_fragments": claim_fragments[:4],
                "sample_titles": top_titles,
                "top_verified_titles": top_verified_titles,
                "top_verified_paper_ids": top_verified_ids,
            })

        profiles.sort(
            key=lambda item: (
                -float(item.get("truth_delta", 0.0) or 0.0),
                -int(item.get("verified_paper_count", 0) or 0),
                item.get("label", ""),
            )
        )
        return profiles

    def _domain_benchmark_counts(self, domain_id: str) -> tuple[int, int]:
        mapped = _DOMAIN_BENCHMARK_MAP.get(domain_id, [])
        if not mapped:
            return 0, 0
        placeholders = ",".join("?" for _ in mapped)
        benchmark_count = self.graph.conn.execute(
            f"SELECT COUNT(*) FROM benchmarks WHERE domain IN ({placeholders})",
            mapped,
        ).fetchone()[0]
        sota_count = self.graph.conn.execute(
            "SELECT COUNT(*) FROM sota_entries WHERE benchmark_id IN "
            f"(SELECT benchmark_id FROM benchmarks WHERE domain IN ({placeholders}))",
            mapped,
        ).fetchone()[0]
        return int(benchmark_count or 0), int(sota_count or 0)

    @staticmethod
    def _row_is_verified(row: Any) -> bool:
        source = str(row["source"] or "")
        venue_tier = str(row["venue_tier"] or "")
        citation_count = int(row["citation_count"] or 0)
        influential_count = int(row["influential_citation_count"] or 0)
        if source == "semantic_scholar" and venue_tier in {"top", "strong"}:
            return True
        if source == "semantic_scholar" and (citation_count >= 20 or influential_count >= 5):
            return True
        if source == "openalex" and venue_tier in {"top", "strong"}:
            return True
        if source == "openalex" and citation_count >= 25:
            return True
        if source == "crossref" and venue_tier in {"top", "strong"}:
            return True
        if source == "crossref" and citation_count >= 35:
            return True
        return False

    def _write_state(self, payload: dict[str, Any]) -> None:
        payload = normalize_literature_payload(payload)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        learned_payload = payload.get("learned_knowledge", {})
        if isinstance(learned_payload, dict):
            self.learned_path.write_text(json.dumps(learned_payload, indent=2), encoding="utf-8")


def main() -> None:
    workspace = ensure_workspace_layout(resolve_workspace(_REPO), repo_root=_REPO)
    args = set(__import__("sys").argv[1:])
    ingestor = ExternalEvidenceIngestor(workspace)
    if "--daemon" in args:
        ingestor.run_forever()
        return
    force = "--force" in args
    cycle = "auto"
    for option in ("fast", "daily", "weekly", "bootstrap"):
        if f"--{option}" in args:
            cycle = option
            break
    payload = ingestor.run_once(force=force, cycle=cycle)
    print(json.dumps(payload.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
