"""
TAR Research Director
=====================

Prioritizes frontier research areas, expands TAR's domain knowledge registry,
assesses evidence strength, and recommends paper/execution order for both
TAR-Author and the experiment scheduler.

Storage: {workspace}/tar_state/research_director_state.json
"""
from __future__ import annotations

import json
import re
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tar_lab.phase_catalog import iter_phase_catalog_entries
from tar_lab.human_review import sync_human_review_from_director_state
from tar_lab.result_artifacts import (
    read_advisory_verdict,
    read_statistics,
    resolve_canonical_comparison_path,
)
from tar_lab.validation import build_validation_state
from tar_storage import ensure_workspace_layout, resolve_workspace


_REPO = Path(__file__).resolve().parent
_STRICT_REAL_WORLD_FRONTIER_ONLY = True

_DEFAULT_DOMAIN_SPECS: list[dict[str, Any]] = [
    {
        "id": "general_ai",
        "label": "General AI Research",
        "description": "Broad ML/AI research not yet specific enough to classify into a stronger domain.",
        "keywords": ["generic_ml", "machine learning", "artificial intelligence", "research"],
        "seed_priority": 50,
    },
    {
        "id": "continual_learning",
        "label": "Continual Learning",
        "description": "Task, class, and lifelong learning systems that must retain prior knowledge.",
        "keywords": ["continual", "forgetting", "lifelong", "task-incremental", "class-incremental", "memory"],
        "seed_priority": 10,
    },
    {
        "id": "thermodynamics_ml",
        "label": "Thermodynamics in ML",
        "description": "Thermodynamic regime detection, entropy-driven control, and criticality-aware learning.",
        "keywords": ["thermodynamic", "entropy", "sigma-star", "critical point", "regime", "activation entropy"],
        "seed_priority": 12,
    },
    {
        "id": "medical_ai",
        "label": "Medical AI",
        "description": "Clinical, diagnostic, and healthcare ML systems requiring high factual reliability.",
        "keywords": ["medical", "clinical", "healthcare", "patient", "diagnosis", "radiology", "drug", "biomedical"],
        "seed_priority": 35,
    },
    {
        "id": "quantum_ml",
        "label": "Quantum / Quantum-Inspired ML",
        "description": "Quantum optimization, qubit systems, and quantum-inspired learning/control problems.",
        "keywords": ["quantum", "qubit", "hamiltonian", "qaoa", "quantum-inspired", "variational circuit"],
        "seed_priority": 38,
    },
    {
        "id": "quantitative_finance",
        "label": "Quantitative Finance",
        "description": "Portfolio, market, risk, and financial optimization research.",
        "keywords": ["portfolio", "asset", "market", "trading", "risk", "finance", "alpha", "skewness", "kurtosis"],
        "seed_priority": 40,
    },
    {
        "id": "multimodal_ai",
        "label": "Multimodal AI",
        "description": "Vision-language and cross-modal reasoning systems.",
        "keywords": ["multimodal", "vision-language", "text-image", "audio", "video", "cross-modal"],
        "seed_priority": 45,
    },
]

_LITERATURE_DOMAIN_MAP: dict[str, str] = {
    "continual_learning": "continual_learning",
    "thermodynamics_ml": "continual_learning",
    "multimodal_ai": "multimodal",
    "general_ai": "",
    "medical_ai": "",
    "quantum_ml": "",
    "quantitative_finance": "",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jload(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass
class KnowledgeDomain:
    id: str
    label: str
    description: str
    keywords: list[str] = field(default_factory=list)
    seed_priority: int = 50
    status: str = "candidate"
    truth_status: str = "weak"
    strong_evidence_count: int = 0
    weak_signal_count: int = 0
    confidence_score: float = 0.0
    source_count: int = 0
    related_frontier_ids: list[str] = field(default_factory=list)
    sample_signals: list[str] = field(default_factory=list)
    truth_of_knowledge_score: float = 0.0
    domain_proficiency_score: float = 0.0
    active_expansion_score: float = 0.0
    expansion_status: str = "seeded"
    expansion_goal: str = ""
    external_verified_source_count: int = 0
    external_weak_source_count: int = 0
    external_benchmark_count: int = 0
    external_sota_count: int = 0
    external_learning_score: float = 0.0
    external_source_diversity_score: float = 0.0
    last_literature_sync: str = ""
    literature_status: str = "unseeded"
    top_verified_titles: list[str] = field(default_factory=list)
    connected_topics: list[str] = field(default_factory=list)
    learned_claims: list[str] = field(default_factory=list)
    learned_summary: str = ""
    next_action: str = ""


@dataclass
class ActiveResearchPath:
    path_id: str
    title: str
    domain_id: str
    domain_label: str
    path_kind: str
    status: str
    priority_score: float
    novelty_score: float
    evidence_score: float
    problem_statement: str
    why_this_now: str
    novelty_basis: str
    experiment_policy: str
    writing_policy: str
    target_frontier_problem_id: str = ""
    target_frontier_title: str = ""
    target_paper_id: str = ""
    source_gap_ids: list[str] = field(default_factory=list)
    source_problem_ids: list[str] = field(default_factory=list)
    required_experiment_ids: list[str] = field(default_factory=list)
    verified_source_count: int = 0
    weak_source_count: int = 0
    allowed_topics: list[str] = field(default_factory=list)
    blocked_topics: list[str] = field(default_factory=list)


class ResearchDirector:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.state_path = workspace / "tar_state" / "research_director_state.json"
        self.frontier_path = workspace / "tar_state" / "frontier_problems.json"
        self.queue_path = workspace / "tar_state" / "experiment_queue.json"
        self.archive_path = workspace / "tar_state" / "experiment_archive.json"
        self.projects_path = workspace / "tar_state" / "research_projects.json"
        self.comparisons_dir = workspace / "tar_state" / "comparisons"
        self.autonomous_dir = workspace / "tar_state" / "autonomous_research"
        self.evidence_path = workspace / "tar_state" / "literature" / "evidence_ingest_state.json"
        self.literature_db_path = workspace / "tar_state" / "literature" / "literature_graph.db"

    def read_state(self) -> dict:
        data = _jload(self.state_path)
        return data if isinstance(data, dict) else {}

    def update_state(self) -> dict:
        frontier_problems = self._load_frontier_problems()
        experiments = self._load_experiments()
        project_entries = self._load_project_entries()
        external_evidence = self._load_external_evidence_state()

        frontier_directives = self._build_frontier_directives(frontier_problems, experiments)
        knowledge_domains = self._build_knowledge_domains(frontier_directives, project_entries, external_evidence)
        paper_directives = self._build_paper_directives(frontier_directives, experiments)
        active_research_paths = self._build_active_research_paths(
            frontier_directives,
            knowledge_domains,
            paper_directives,
            external_evidence,
        )
        frontier_directives = self._apply_active_paths_to_frontiers(frontier_directives, active_research_paths)
        paper_directives = self._apply_active_paths_to_papers(
            paper_directives,
            active_research_paths,
            knowledge_domains,
        )
        experiment_directives = self._build_experiment_directives(
            frontier_directives,
            paper_directives,
            active_research_paths,
            experiments,
        )
        evidence_directives = self._build_evidence_directives(
            frontier_directives,
            paper_directives,
            knowledge_domains,
            experiments,
            external_evidence,
        )

        frontier_rank = {
            rec["problem_id"]: idx + 1 for idx, rec in enumerate(frontier_directives)
        }
        paper_rank = {
            rec["paper_id"]: idx + 1 for idx, rec in enumerate(paper_directives)
        }
        experiment_rank = {
            rec["experiment_id"]: idx + 1 for idx, rec in enumerate(experiment_directives)
        }
        for rec in frontier_directives:
            rec["scheduler_rank"] = frontier_rank[rec["problem_id"]]
            rec["author_rank"] = paper_rank.get(rec["suggested_paper_id"], len(paper_directives) + 1)
        for rec in paper_directives:
            rec["author_rank"] = paper_rank[rec["paper_id"]]
        for rec in experiment_directives:
            rec["scheduler_rank"] = experiment_rank[rec["experiment_id"]]
            rec["author_rank"] = paper_rank.get(str(rec.get("target_paper_id", "") or ""), len(paper_directives) + 1)

        # LLM annotation: enrich top directives with Claude insights (cached, best-effort)
        try:
            self._annotate_directives_with_llm(
                frontier_directives, experiment_directives, evidence_directives, knowledge_domains
            )
        except Exception:
            pass

        validated_claims = sum(1 for rec in frontier_directives if rec["truth_status"] == "validated")
        weak_claims = sum(1 for rec in frontier_directives if rec["truth_status"] == "weak")
        top_frontier = frontier_directives[0]["problem_id"] if frontier_directives else ""
        top_paper = paper_directives[0]["paper_id"] if paper_directives else ""
        top_path = active_research_paths[0].path_id if active_research_paths else ""
        top_experiment = experiment_directives[0]["experiment_id"] if experiment_directives else ""
        operational_frontier_problem_ids = sorted({
            str(rec.get("frontier_problem_id", "") or "")
            for rec in experiment_directives
            if str(rec.get("frontier_problem_id", "") or "")
            and str(rec.get("status", "") or "") not in {"complete", "archive"}
        } | {
            str(rec.get("problem_id", "") or "")
            for rec in frontier_directives
            if str(rec.get("problem_id", "") or "")
            and (
                rec.get("waiting_on_experiment_ids")
                or str(rec.get("readiness", "") or "") in {"experiment_first", "outline_now", "write_now"}
            )
        })
        actively_expanding = [dom for dom in knowledge_domains if dom.expansion_status in {"active_expansion", "stabilizing"}]
        top_expanding = actively_expanding[0].id if actively_expanding else ""
        active_domain_ids = sorted({
            path.domain_id for path in active_research_paths
            if path.status in {"pursue_now", "incubate"}
        })
        active_frontier_problem_ids = sorted(({
            path.target_frontier_problem_id for path in active_research_paths
            if path.target_frontier_problem_id and path.status in {"pursue_now", "incubate"}
        }) | set(operational_frontier_problem_ids))
        active_paper_ids = sorted(({
            path.target_paper_id for path in active_research_paths
            if path.target_paper_id and path.status in {"pursue_now", "incubate"}
        }) | {
            str(rec.get("target_paper_id", "") or "")
            for rec in experiment_directives
            if str(rec.get("target_paper_id", "") or "")
            and str(rec.get("status", "") or "") not in {"complete", "archive"}
        })
        payload = {
            "timestamp": _now_iso(),
            "summary": {
                "frontier_count": len(frontier_directives),
                "paper_count": len(paper_directives),
                "experiment_count": len(experiment_directives),
                "active_path_count": len(active_research_paths),
                "domain_count": len(knowledge_domains),
                "evidence_task_count": len(evidence_directives),
                "actively_expanding_domain_count": len(actively_expanding),
                "validated_claims": validated_claims,
                "weak_claims": weak_claims,
                "top_frontier_problem_id": top_frontier,
                "top_paper_id": top_paper,
                "top_experiment_id": top_experiment,
                "top_research_path_id": top_path,
                "top_expanding_domain_id": top_expanding,
                "literature_total_papers": int(external_evidence.get("summary", {}).get("literature_total_papers", 0) or 0),
                "external_verified_sources": int(external_evidence.get("summary", {}).get("external_verified_sources", 0) or 0),
                "last_literature_sync": str(external_evidence.get("summary", {}).get("last_literature_sync", "") or ""),
                "active_domain_ids": active_domain_ids,
                "active_frontier_problem_ids": active_frontier_problem_ids,
                "active_paper_ids": active_paper_ids,
                "writing_scope_titles": [path.title for path in active_research_paths[:5]],
            },
            "frontier_directives": frontier_directives,
            "paper_directives": paper_directives,
            "experiment_directives": experiment_directives,
            "evidence_directives": evidence_directives,
            "active_research_paths": [asdict(path) for path in active_research_paths],
            "knowledge_domains": [asdict(d) for d in knowledge_domains],
            "llm_insights": {
                "frontier_syntheses": [
                    {"frontier_id": rec["problem_id"], "synthesis": rec["llm_synthesis"]}
                    for rec in frontier_directives[:3]
                    if rec.get("llm_synthesis")
                ],
                "experiment_evaluations": [
                    {"experiment_id": rec["experiment_id"], "evaluation": rec["llm_evaluation"]}
                    for rec in experiment_directives[:5]
                    if rec.get("llm_evaluation")
                ],
                "claim_verifications": [
                    {"task_id": rec["task_id"], "verification": rec["llm_claim_check"]}
                    for rec in evidence_directives[:3]
                    if rec.get("llm_claim_check")
                ],
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            return payload
        try:
            build_validation_state(self.workspace, persist=True)
        except Exception:
            pass
        try:
            human_review = sync_human_review_from_director_state(self.workspace, payload)
            payload["human_review_summary"] = {
                "proposal_count": len(human_review.get("proposals", [])),
                "question_count": len(human_review.get("questions", [])),
                "claim_review_count": len(human_review.get("claim_reviews", [])),
            }
            self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            return payload
        except Exception:
            pass
        try:
            from tar_project_registry import rebuild_registry_from_director_state

            rebuild_registry_from_director_state(self.workspace, payload)
        except Exception:
            pass
        return payload

    def _load_frontier_problems(self) -> list[dict]:
        data = _jload(self.frontier_path)
        if isinstance(data, dict):
            problems = data.get("problems", [])
            return problems if isinstance(problems, list) else []
        return []

    def _canonical_phase_experiments(self) -> list[dict]:
        synthetic: list[dict] = []
        for entry in iter_phase_catalog_entries():
            result_path = resolve_canonical_comparison_path(
                self.workspace,
                entry.logical_name,
                legacy_filename=entry.legacy_filename,
            )
            if result_path is None or not result_path.exists():
                continue
            result = _jload(result_path)
            if not isinstance(result, dict):
                continue
            advisory = read_advisory_verdict(result)
            verdict = str(advisory.get("label", "") or result.get("verdict", "") or "")
            status = "complete"
            if str(result.get("status", "") or "").upper() == "ERROR" or "ERROR" in verdict.upper():
                status = "failed"
            synthetic.append({
                "id": entry.experiment_id,
                "name": entry.title,
                "title": entry.title,
                "status": status,
                "stage": status,
                "dataset": entry.dataset,
                "method": "phase_result",
                "logical_name": entry.logical_name,
                "legacy_result_filename": entry.legacy_filename,
                "result_path": str(result_path),
                "frontier_problem_id": entry.frontier_problem_id,
                "author_paper_id": entry.target_paper_id,
                "project_id": entry.target_paper_id,
                "paper_title": entry.target_paper_title,
                "domain_profile": entry.primary_domain_id,
                "context": {
                    "why": entry.research_goal,
                    "hypothesis": entry.research_goal,
                    "frontier_problem": entry.frontier_problem_id,
                    "feeds_paper": entry.target_paper_title,
                    "methodology_note": f"Canonical phase result for {entry.title}.",
                },
                "_phase_num": entry.phase_number,
            })
        return synthetic

    def _load_experiments(self) -> list[dict]:
        combined: dict[str, dict] = {}

        queue_data = _jload(self.queue_path)
        if isinstance(queue_data, dict):
            experiments = queue_data.get("experiments", [])
            if isinstance(experiments, list):
                for exp in experiments:
                    if isinstance(exp, dict) and exp.get("id"):
                        combined[str(exp.get("id"))] = exp

        archive_data = _jload(self.archive_path)
        if isinstance(archive_data, dict):
            experiments = archive_data.get("experiments", [])
            if isinstance(experiments, list):
                for exp in experiments:
                    if isinstance(exp, dict) and exp.get("id"):
                        combined.setdefault(str(exp.get("id")), exp)

        for exp in self._canonical_phase_experiments():
            exp_id = str(exp.get("id", "") or "")
            if not exp_id:
                continue
            existing = combined.get(exp_id)
            if not existing:
                combined[exp_id] = exp
                continue
            merged = dict(existing)
            for key, value in exp.items():
                if key == "context":
                    context = dict(merged.get("context", {}) or {})
                    for ckey, cvalue in dict(value or {}).items():
                        if cvalue and not context.get(ckey):
                            context[ckey] = cvalue
                    if context:
                        merged["context"] = context
                    continue
                if merged.get(key) in ("", None, [], {}):
                    merged[key] = value
            combined[exp_id] = merged

        return list(combined.values())

    def _load_project_entries(self) -> list[dict]:
        data = _jload(self.projects_path)
        if isinstance(data, dict):
            entries = data.get("entries", [])
            if isinstance(entries, list):
                return entries[:200]
        return []

    def _load_external_evidence_state(self) -> dict[str, Any]:
        data = _jload(self.evidence_path)
        if not isinstance(data, dict):
            return {}
        try:
            from tar_evidence_ingest import normalize_literature_payload

            return normalize_literature_payload(data)
        except Exception:
            return data

    def _build_frontier_directives(self, problems: list[dict], experiments: list[dict]) -> list[dict]:
        directives: list[dict] = []
        for problem in problems:
            pid = str(problem.get("id", ""))
            problem_title = str(problem.get("industry_problem_title", "") or problem.get("title", pid))
            global_problem_statement = str(problem.get("global_problem_statement", "") or problem.get("description", ""))
            industry_contexts = [
                str(item) for item in problem.get("industry_contexts", [])
                if str(item).strip()
            ]
            solution_family = str(problem.get("solution_family", "TAR/TCL/ASC") or "TAR/TCL/ASC")
            solution_novelty_note = str(
                problem.get(
                    "solution_novelty_note",
                    "TAR, TCL, and ASC are the user's unpublished internal methods under evaluation, not established literature or accepted solutions.",
                ) or ""
            )
            target_venues = [
                str(item) for item in problem.get("target_venues", [])
                if str(item).strip()
            ]
            candidate_datasets = [
                str(item) for item in problem.get("candidate_datasets", [])
                if str(item).strip()
            ]
            candidate_backbones = [
                str(item) for item in problem.get("candidate_backbones", [])
                if str(item).strip()
            ]
            external_baselines = [
                str(item) for item in problem.get("external_baselines", [])
                if str(item).strip()
            ]
            research_guidance = str(problem.get("research_guidance", "") or "")
            linked = [exp for exp in experiments if exp.get("frontier_problem_id") == pid]
            complete = 0
            running = 0
            pending = 0
            significant_hits = 0
            directional_hits = 0
            evidence_notes: list[str] = []
            paper_ids = sorted({
                str(exp.get("author_paper_id") or exp.get("project_id") or "")
                for exp in linked
                if exp.get("author_paper_id") or exp.get("project_id")
            })
            waiting_on = []
            for exp in linked:
                exp_id = str(exp.get("id", ""))
                status = str(exp.get("status", ""))
                stage = str(exp.get("stage", ""))
                if status == "complete" or stage == "complete":
                    complete += 1
                elif status == "running" or stage == "running":
                    running += 1
                    if exp_id:
                        waiting_on.append(exp_id)
                else:
                    pending += 1
                    if exp_id:
                        waiting_on.append(exp_id)
                res = self._load_experiment_result(exp)
                if not res:
                    continue
                stats = read_statistics(res)
                advisory = read_advisory_verdict(res)
                verdict = str(advisory.get("label", "") or res.get("verdict") or "").upper()
                p_val = stats.get("p_val", res.get("p_val"))
                mean_delta = stats.get("mean_delta", res.get("mean_delta"))
                if mean_delta is None or p_val is None:
                    pairwise = stats.get("pairwise") or res.get("pairwise", {})
                    if isinstance(pairwise, dict):
                        for key in ("ewc", "sgd_baseline", "sgd"):
                            candidate = pairwise.get(key, {})
                            if not isinstance(candidate, dict):
                                continue
                            candidate_delta = candidate.get("mean_delta")
                            if isinstance(candidate_delta, (int, float)):
                                mean_delta = candidate_delta
                                p_val = candidate.get("p_val", p_val)
                                break
                if verdict == "BREAKTHROUGH":
                    significant_hits += 1
                    evidence_notes.append(f"{exp_id} reached BREAKTHROUGH.")
                elif verdict == "DIRECTIONAL":
                    directional_hits += 1
                    evidence_notes.append(f"{exp_id} produced directional evidence.")
                elif isinstance(p_val, (int, float)) and isinstance(mean_delta, (int, float)):
                    if float(p_val) < 0.05 and float(mean_delta) < 0:
                        significant_hits += 1
                        evidence_notes.append(f"{exp_id} met significance with beneficial delta.")
                    elif float(mean_delta) < 0:
                        directional_hits += 1
                        evidence_notes.append(f"{exp_id} improved delta without full significance.")

            breakthrough_count = significant_hits
            linked_papers = paper_ids or [
                str(item) for item in problem.get("papers_linked", [])
                if str(item).strip()
            ]
            problem_priority = int(problem.get("priority", 50) or 50)

            impact_score = max(5.0, 110.0 - problem_priority) + breakthrough_count * 12.0
            evidence_score = complete * 14.0 + running * 5.0 + significant_hits * 18.0 + directional_hits * 8.0
            truth_score = breakthrough_count * 20.0 + significant_hits * 18.0 + complete * 8.0 + len(linked_papers) * 6.0
            priority_score = round(impact_score + evidence_score + truth_score, 1)

            if breakthrough_count > 0 and not waiting_on:
                truth_status = "validated"
                evidence_strength = "strong"
            elif significant_hits > 0 or complete > 0:
                truth_status = "supported"
                evidence_strength = "moderate"
            elif running > 0 or pending > 0:
                truth_status = "provisional"
                evidence_strength = "emerging"
            else:
                truth_status = "weak"
                evidence_strength = "weak"

            suggested_paper_id = paper_ids[0] if paper_ids else f"frontier-paper-{pid}"
            suggested_paper_title = problem_title.strip() + " - Frontier Paper"
            if breakthrough_count > 0 and not waiting_on:
                next_action = "Prioritize a frontier paper now; TAR already has a breakthrough-grade result here."
                readiness = "write_now"
                publication_lane = "conference_candidate"
            elif complete > 0 and not waiting_on:
                next_action = "Write paper now and lock down strongest claims."
                readiness = "write_now"
                publication_lane = "conference_candidate"
            elif complete > 0:
                next_action = "Draft methods and partial results now while waiting for final experiments."
                readiness = "outline_now"
                publication_lane = "conference_material_in_progress"
            elif running > 0 or pending > 0:
                next_action = "Prioritize experiments before making strong paper claims."
                readiness = "experiment_first"
                publication_lane = "evidence_building"
            else:
                next_action = "Treat as an exploratory direction until stronger evidence arrives."
                readiness = "landscape_scan"
                publication_lane = "problem_scoping"

            if waiting_on:
                live_status = "active"
            elif truth_status == "validated":
                live_status = "publishing"
            elif complete > 0 or running > 0:
                live_status = "active"
            else:
                live_status = "exploring"

            why_now = (
                f"{complete} complete, {running} running, {pending} pending experiments; "
                f"{breakthrough_count} breakthroughs and {len(linked_papers)} linked papers so far."
            )
            directives.append({
                "problem_id": pid,
                "title": problem_title,
                "domain": problem.get("domain", ""),
                "status": live_status,
                "global_problem_statement": global_problem_statement,
                "industry_contexts": industry_contexts,
                "well_known_problem": bool(problem.get("well_known_problem", True)),
                "solution_family": solution_family,
                "solution_novelty_note": solution_novelty_note,
                "target_venues": target_venues,
                "candidate_datasets": candidate_datasets,
                "candidate_backbones": candidate_backbones,
                "external_baselines": external_baselines,
                "research_guidance": research_guidance,
                "impact_score": round(impact_score, 1),
                "evidence_score": round(evidence_score, 1),
                "truth_score": round(truth_score, 1),
                "priority_score": priority_score,
                "truth_status": truth_status,
                "evidence_strength": evidence_strength,
                "next_action": next_action,
                "readiness": readiness,
                "publication_lane": publication_lane,
                "why_now": why_now,
                "waiting_on_experiment_ids": sorted(set(waiting_on)),
                "linked_experiment_ids": [exp.get("id", "") for exp in linked if exp.get("id")],
                "linked_paper_ids": linked_papers,
                "suggested_paper_id": suggested_paper_id,
                "suggested_paper_title": suggested_paper_title,
                "evidence_notes": evidence_notes[:4],
            })

        directives.sort(
            key=lambda rec: (
                rec["readiness"] not in {"write_now", "outline_now"},
                -rec["priority_score"],
                rec["title"],
            )
        )
        return directives

    def _build_knowledge_domains(
        self,
        frontier_directives: list[dict],
        project_entries: list[dict],
        external_evidence_state: dict[str, Any] | None = None,
    ) -> list[KnowledgeDomain]:
        domains: dict[str, KnowledgeDomain] = {
            spec["id"]: KnowledgeDomain(**spec) for spec in _DEFAULT_DOMAIN_SPECS
        }
        external_profiles = {
            str(rec.get("domain_id", "") or ""): rec
            for rec in (external_evidence_state or {}).get("domain_profiles", [])
            if isinstance(rec, dict) and rec.get("domain_id")
        }
        external_summary = (external_evidence_state or {}).get("summary", {})

        def note_domain(domain_id: str, signal: str, strong: bool, frontier_id: str = "") -> None:
            dom = domains[domain_id]
            dom.source_count += 1
            if strong:
                dom.strong_evidence_count += 1
            else:
                dom.weak_signal_count += 1
            if frontier_id and frontier_id not in dom.related_frontier_ids:
                dom.related_frontier_ids.append(frontier_id)
            if signal and signal not in dom.sample_signals:
                dom.sample_signals.append(signal[:120])

        for directive in frontier_directives:
            domain_id = self._infer_domain_id(" ".join([
                str(directive.get("domain", "")),
                str(directive.get("title", "")),
                " ".join(directive.get("evidence_notes", [])),
            ]))
            note_domain(
                domain_id=domain_id,
                signal=f"{directive['title']} ({directive['truth_status']})",
                strong=directive["truth_status"] in {"validated", "supported"},
                frontier_id=directive["problem_id"],
            )

        for project in project_entries:
            signal_text = " ".join([
                str(project.get("title", "")),
                str(project.get("goal", "")),
                str(project.get("domain_profile", "")),
            ])
            domain_id = self._infer_domain_id(signal_text)
            project_status = str(project.get("status", "") or "").lower()
            paper_status = str(project.get("paper_status", "") or "").lower()
            strong = (
                project_status in {"complete", "validated"}
                and paper_status not in {"blocked", "awaiting_validation", "awaiting_human_review", "revision_requested"}
            )
            note_domain(domain_id=domain_id, signal=str(project.get("title", "")), strong=strong)

        for dom in domains.values():
            profile = external_profiles.get(dom.id, {})
            dom.external_verified_source_count = int(profile.get("verified_paper_count", 0) or 0)
            dom.external_weak_source_count = int(profile.get("weak_paper_count", 0) or 0)
            dom.external_benchmark_count = int(profile.get("benchmark_count", 0) or 0)
            dom.external_sota_count = int(profile.get("sota_entry_count", 0) or 0)
            dom.external_learning_score = float(profile.get("learning_score", 0.0) or 0.0)
            dom.external_source_diversity_score = float(profile.get("source_diversity_score", 0.0) or 0.0)
            dom.last_literature_sync = str(external_summary.get("last_literature_sync", "") or "")
            dom.literature_status = str(external_summary.get("status", "unseeded") or "unseeded")
            dom.top_verified_titles = [
                str(title) for title in profile.get("top_verified_titles", [])
                if str(title).strip()
            ][:3]
            dom.connected_topics = [
                str(topic) for topic in profile.get("connected_topics", [])
                if str(topic).strip()
            ][:5]
            dom.learned_claims = [
                str(claim) for claim in profile.get("claim_fragments", [])
                if str(claim).strip()
            ][:4]
            dom.learned_summary = str(profile.get("learned_summary", "") or "")
            for title in profile.get("sample_titles", [])[:3]:
                title_text = str(title).strip()
                if title_text and title_text not in dom.sample_signals:
                    dom.sample_signals.append(title_text[:120])
            for claim in dom.learned_claims[:2]:
                if claim not in dom.sample_signals:
                    dom.sample_signals.append(claim[:120])
            dom.sample_signals = dom.sample_signals[:4]
            source_bonus = min(20.0, dom.source_count * 4.0)
            frontier_bonus = min(15.0, len(dom.related_frontier_ids) * 5.0)
            external_truth = min(12.0, float(profile.get("truth_delta", 0.0) or 0.0) * 0.35)
            external_confidence = float(
                profile.get("learning_confidence_score", profile.get("learning_score", 0.0)) or 0.0
            )
            confidence_band = str(profile.get("learning_confidence_band", "") or "")
            confidence_penalty = 0.0
            if confidence_band == "provisional":
                confidence_penalty = 6.0
            elif confidence_band == "weak":
                confidence_penalty = 12.0
            elif confidence_band == "seed":
                confidence_penalty = 16.0
            learning_bonus = min(8.0, external_confidence * 0.08)
            diversity_bonus = min(6.0, dom.external_source_diversity_score * 0.06)
            truth_base = (
                dom.strong_evidence_count * 22.0
                + dom.weak_signal_count * 6.0
                + source_bonus
                + frontier_bonus
                + external_truth
                + learning_bonus
                + diversity_bonus
                - confidence_penalty
            )
            dom.truth_of_knowledge_score = round(max(0.0, min(100.0, truth_base)), 1)

            if dom.truth_of_knowledge_score >= 80:
                dom.truth_status = "validated"
                dom.status = "active"
            elif dom.truth_of_knowledge_score >= 55:
                dom.truth_status = "supported"
                dom.status = "active"
            elif dom.truth_of_knowledge_score >= 30:
                dom.truth_status = "provisional"
                dom.status = "emerging"
            elif dom.source_count > 0:
                dom.truth_status = "weak"
                dom.status = "candidate"
            else:
                dom.truth_status = "weak"
                dom.status = "seed"

            proficiency_base = (
                dom.truth_of_knowledge_score * 0.55
                + min(25.0, len(dom.related_frontier_ids) * 7.0)
                + min(20.0, dom.strong_evidence_count * 8.0)
                + min(10.0, dom.source_count * 2.0)
                + min(10.0, float(profile.get("proficiency_delta", 0.0) or 0.0) * 0.35)
                + min(7.0, external_confidence * 0.07)
                + min(5.0, dom.external_source_diversity_score * 0.05)
                - confidence_penalty * 0.7
            )
            dom.domain_proficiency_score = round(max(0.0, min(100.0, proficiency_base)), 1)

            expand_pressure = max(0.0, 70.0 - dom.domain_proficiency_score)
            strategic_bonus = max(0.0, 50.0 - float(dom.seed_priority))
            signal_bonus = min(20.0, dom.weak_signal_count * 5.0 + dom.source_count * 2.0)
            dom.active_expansion_score = round(min(100.0, expand_pressure * 0.5 + strategic_bonus * 0.7 + signal_bonus), 1)
            dom.confidence_score = dom.truth_of_knowledge_score

            if dom.status == "active" and dom.domain_proficiency_score >= 70:
                dom.expansion_status = "stabilizing"
                dom.expansion_goal = "Convert the strong evidence base into reusable experiments, papers, and verified claims."
                dom.next_action = "Exploit validated signals and keep experiments/papers moving."
            elif dom.status in {"emerging", "candidate"} or dom.seed_priority <= 40:
                dom.expansion_status = "active_expansion"
                dom.expansion_goal = (
                    "Raise truth-of-knowledge with verified sources and improve TAR's operating proficiency in this domain."
                )
                if dom.status == "emerging":
                    dom.next_action = "Gather one more strong result before escalating claims."
                elif dom.status == "candidate":
                    dom.next_action = "Collect verified outside evidence before treating this as a priority frontier."
                else:
                    dom.next_action = "Actively expand this domain with verified knowledge before opening major experiments."
            else:
                dom.expansion_status = "monitoring"
                dom.expansion_goal = "Keep this seeded domain visible until stronger evidence or a new strategic reason appears."
                dom.next_action = "Keep as a seeded knowledge domain until real signals arrive."

            if dom.external_verified_source_count > 0 and dom.expansion_status == "active_expansion":
                dom.next_action = (
                    f"Promote the {dom.external_verified_source_count} verified outside sources into usable claims, then open experiments."
                )
            elif dom.external_verified_source_count == 0 and dom.expansion_status == "active_expansion":
                dom.next_action = "Harvest verified outside evidence before escalating this domain."
            if dom.external_source_diversity_score < 30 and dom.expansion_status == "active_expansion":
                dom.next_action = "Broaden reliable literature sources before trusting this domain too strongly."

        ordered = sorted(
            domains.values(),
            key=lambda rec: (
                rec.expansion_status not in {"active_expansion", "stabilizing"},
                -rec.active_expansion_score,
                -rec.truth_of_knowledge_score,
                rec.seed_priority,
                rec.label,
            ),
        )
        return ordered

    def _build_active_research_paths(
        self,
        frontier_directives: list[dict],
        knowledge_domains: list[KnowledgeDomain],
        paper_directives: list[dict],
        external_evidence_state: dict[str, Any] | None = None,
    ) -> list[ActiveResearchPath]:
        domain_by_id = {domain.id: domain for domain in knowledge_domains}
        paper_by_frontier = {
            str(rec.get("frontier_problem_id", "") or ""): rec
            for rec in paper_directives
            if rec.get("frontier_problem_id")
        }
        active_domains = [
            domain for domain in knowledge_domains
            if domain.expansion_status in {"active_expansion", "stabilizing"}
            or domain.truth_status in {"validated", "supported"}
        ][:6]
        active_domain_ids = {domain.id for domain in active_domains}
        represented_domain_ids: set[str] = set()

        paths: list[ActiveResearchPath] = []
        for frontier in frontier_directives:
            domain_id = self._infer_domain_id(" ".join([
                str(frontier.get("domain", "") or ""),
                str(frontier.get("title", "") or ""),
                " ".join(str(note) for note in frontier.get("evidence_notes", []) if str(note)),
            ]))
            if domain_id not in domain_by_id:
                continue
            domain = domain_by_id[domain_id]
            if domain_id not in active_domain_ids and frontier.get("truth_status") not in {"validated", "supported"}:
                continue
            linked_paper = paper_by_frontier.get(str(frontier.get("problem_id", "") or ""), {})
            frontier_title = str(frontier.get("title", frontier.get("problem_id", "")) or "")
            global_problem_statement = str(frontier.get("global_problem_statement", frontier_title) or frontier_title)
            solution_family = str(frontier.get("solution_family", "TAR/TCL/ASC") or "TAR/TCL/ASC")
            solution_novelty_note = str(frontier.get("solution_novelty_note", "") or "")
            readiness = str(frontier.get("readiness", "") or "")
            status = "pursue_now" if readiness in {"write_now", "outline_now", "experiment_first"} else "incubate"
            novelty_score = 85.0 if frontier.get("truth_status") == "validated" else 62.0 if frontier.get("truth_status") == "supported" else 48.0
            writing_policy = (
                f"Anchor every manuscript in the external problem '{frontier_title}'. "
                f"Present {solution_family} as TAR's unpublished internal work under evaluation, never as established literature or an already-proven solution. "
                "Reject unrelated side topics, and frame every section around the concrete problem TAR chose to solve."
            )
            experiment_policy = (
                f"Prioritize experiments that tighten evidence for the global problem '{frontier_title}', "
                f"using {solution_family} as one candidate internal method family under test alongside real external baselines where available. Avoid side experiments that do not "
                "directly improve the chosen problem statement."
            )
            paths.append(ActiveResearchPath(
                path_id=f"path-frontier-{frontier.get('problem_id', '')}",
                title=frontier_title,
                domain_id=domain.id,
                domain_label=domain.label,
                path_kind="frontier_validation",
                status=status,
                priority_score=float(frontier.get("priority_score", 0.0) or 0.0),
                novelty_score=novelty_score,
                evidence_score=float(frontier.get("evidence_score", 0.0) or 0.0),
                problem_statement=(
                    f"Address the globally recognized problem '{frontier_title}' within {domain.label}. "
                    f"Use {solution_family} as TAR's unpublished internal work under evaluation, and keep both experiments and papers "
                    f"tied to verifiable evidence on that real-world problem. {solution_novelty_note}"
                ),
                why_this_now=str(frontier.get("why_now", "") or ""),
                novelty_basis=(
                    "This path is anchored to a real external problem already recognized in the field, while TAR/TCL/ASC remain the novel internal mechanisms under evaluation."
                ),
                experiment_policy=experiment_policy,
                writing_policy=writing_policy,
                target_frontier_problem_id=str(frontier.get("problem_id", "") or ""),
                target_frontier_title=frontier_title,
                target_paper_id=str(linked_paper.get("paper_id", frontier.get("suggested_paper_id", "")) or ""),
                required_experiment_ids=[str(exp_id) for exp_id in frontier.get("waiting_on_experiment_ids", []) if str(exp_id or "")],
                verified_source_count=domain.external_verified_source_count,
                weak_source_count=domain.external_weak_source_count,
                allowed_topics=[frontier_title, domain.label, global_problem_statement],
                blocked_topics=["unrelated papers", "off-domain writing", "unsupported novelty claims", "claims that TAR/TCL/ASC are prior literature"],
            ))
            represented_domain_ids.add(domain.id)

        exploratory_domains = [] if _STRICT_REAL_WORLD_FRONTIER_ONLY else active_domains

        for domain in exploratory_domains:
            candidate = self._pick_novel_problem_for_domain(domain.id)
            if candidate:
                paper_id = f"frontier-paper-{domain.id}-{_slug(candidate['title'])[:48]}"
                verified_sources = domain.external_verified_source_count
                status = "pursue_now" if verified_sources >= 3 or domain.truth_status in {"validated", "supported"} else "incubate"
                writing_policy = (
                    f"Only outline or write papers about novel problems inside {domain.label} that TAR has explicitly selected. "
                    "Keep the manuscript centered on this problem statement and the evidence path TAR intends to close."
                )
                paths.append(ActiveResearchPath(
                    path_id=f"path-problem-{_slug(candidate['title'])[:48]}",
                    title=str(candidate["title"]),
                    domain_id=domain.id,
                    domain_label=domain.label,
                    path_kind="novel_problem",
                    status=status,
                    priority_score=float(candidate["priority_score"]),
                    novelty_score=float(candidate["novelty_score"]),
                    evidence_score=float(candidate["evidence_score"]),
                    problem_statement=str(candidate["problem_statement"]),
                    why_this_now=str(candidate["why_this_now"]),
                    novelty_basis=str(candidate["novelty_basis"]),
                    experiment_policy=str(candidate["experiment_policy"]),
                    writing_policy=writing_policy,
                    target_paper_id=paper_id,
                    source_gap_ids=[str(gap_id) for gap_id in candidate.get("source_gap_ids", []) if str(gap_id)],
                    source_problem_ids=[str(problem_id) for problem_id in candidate.get("source_problem_ids", []) if str(problem_id)],
                    verified_source_count=verified_sources,
                    weak_source_count=domain.external_weak_source_count,
                    allowed_topics=[domain.label, str(candidate["title"])],
                    blocked_topics=["generic surveys", "unselected domains", "papers disconnected from TAR's chosen problem"],
                ))
                represented_domain_ids.add(domain.id)
            elif domain.expansion_status == "active_expansion":
                paths.append(ActiveResearchPath(
                    path_id=f"path-domain-{domain.id}",
                    title=f"Frontier Scan - {domain.label}",
                    domain_id=domain.id,
                    domain_label=domain.label,
                    path_kind="domain_frontier_scan",
                    status="incubate",
                    priority_score=float(domain.active_expansion_score),
                    novelty_score=max(35.0, float(domain.active_expansion_score)),
                    evidence_score=float(domain.external_verified_source_count * 8 + domain.external_benchmark_count * 5),
                    problem_statement=(
                        f"Find a genuinely novel, falsifiable ML problem inside {domain.label} that TAR can own rather than drifting into broad commentary."
                    ),
                    why_this_now=(
                        f"{domain.label} is active in TAR's expansion map but still needs a sharper self-chosen frontier target."
                    ),
                    novelty_basis=(
                        "The domain is active, but TAR has not yet committed to one concrete frontier problem with enough verified support."
                    ),
                    experiment_policy=(
                        "Harvest verified sources first, then nominate one falsifiable problem instead of scattering across unrelated subtopics."
                    ),
                    writing_policy=(
                        f"Do not write broad papers about {domain.label} yet. Only write once TAR has selected one concrete problem in this active field."
                    ),
                    verified_source_count=domain.external_verified_source_count,
                    weak_source_count=domain.external_weak_source_count,
                    allowed_topics=[domain.label],
                    blocked_topics=["unfocused domain overviews", "papers outside active Director fields"],
                ))
                represented_domain_ids.add(domain.id)

        for domain in exploratory_domains:
            if domain.id in represented_domain_ids:
                continue
            paths.append(ActiveResearchPath(
                path_id=f"path-domain-{domain.id}",
                title=f"Frontier Scan - {domain.label}",
                domain_id=domain.id,
                domain_label=domain.label,
                path_kind="domain_frontier_scan",
                status="incubate",
                priority_score=max(25.0, float(domain.active_expansion_score)),
                novelty_score=max(35.0, float(domain.active_expansion_score)),
                evidence_score=float(domain.external_verified_source_count * 5 + domain.external_benchmark_count * 4),
                problem_statement=(
                    f"Identify one concrete, falsifiable ML problem inside {domain.label} that TAR can justify with verified sources "
                    "before opening a major experiment track."
                ),
                why_this_now=(
                    f"{domain.label} has enough verified outside evidence to justify an active frontier scan, but TAR has not "
                    "yet committed to one concrete problem in this field."
                ),
                novelty_basis=(
                    "The domain is live and evidence-backed, but TAR still needs to select a precise problem instead of drifting into broad commentary."
                ),
                experiment_policy=(
                    "Harvest verified evidence first, then nominate one falsifiable problem and one topic-fit experiment instead of spraying generic probes."
                ),
                writing_policy=(
                    f"Do not write broad papers about {domain.label} yet. Only write once TAR has selected one concrete problem in this active field."
                ),
                verified_source_count=domain.external_verified_source_count,
                weak_source_count=domain.external_weak_source_count,
                allowed_topics=[domain.label],
                blocked_topics=["broad commentary", "problem-free papers", "generic surveys"],
            ))

        unique: dict[str, ActiveResearchPath] = {}
        for path in paths:
            unique[path.path_id] = path
        ordered = sorted(
            unique.values(),
            key=lambda rec: (
                rec.status != "pursue_now",
                -rec.priority_score,
                -rec.novelty_score,
                rec.title,
            ),
        )
        selected: list[ActiveResearchPath] = []
        per_domain: dict[str, int] = {}
        domain_frontier_scan_kept: set[str] = set()
        for path in ordered:
            limit = 3 if path.status == "pursue_now" else 2
            if per_domain.get(path.domain_id, 0) >= limit:
                continue
            if path.path_kind == "domain_frontier_scan" and path.domain_id in domain_frontier_scan_kept:
                continue
            selected.append(path)
            per_domain[path.domain_id] = per_domain.get(path.domain_id, 0) + 1
            if path.path_kind == "domain_frontier_scan":
                domain_frontier_scan_kept.add(path.domain_id)
            if len(selected) >= 12:
                break
        return selected

    def _pick_novel_problem_for_domain(self, domain_id: str) -> dict[str, Any] | None:
        if _STRICT_REAL_WORLD_FRONTIER_ONLY:
            return None
        literature_domain = _LITERATURE_DOMAIN_MAP.get(domain_id, "")
        if not literature_domain or not self.literature_db_path.exists():
            return None
        try:
            from literature.gap_detector import GapDetector
            from literature.knowledge_graph import LiteratureKnowledgeGraph

            graph = LiteratureKnowledgeGraph(str(self.literature_db_path))
            try:
                gaps = graph.get_top_gaps(n=5, domain=literature_domain)
                if not gaps:
                    return None
                detector = GapDetector(graph)
                problems = detector.gaps_to_problems(gaps=gaps, top_n=2)
                if not problems:
                    return None
                problem = problems[0]
                novelty_basis = (
                    f"Derived from literature gap(s) {', '.join(problem.gap_ids[:3])} in the active {domain_id} research field."
                )
                return {
                    "title": problem.title,
                    "priority_score": round(float(problem.priority_score) * 100.0, 1),
                    "novelty_score": round(float(problem.priority_score) * 100.0, 1),
                    "evidence_score": round(float(problem.priority_score) * 55.0, 1),
                    "problem_statement": problem.description,
                    "why_this_now": (
                        f"The literature gap detector surfaced this as one of the strongest unresolved problems in {literature_domain}."
                    ),
                    "novelty_basis": novelty_basis,
                    "experiment_policy": problem.proposed_experiment,
                    "source_gap_ids": problem.gap_ids,
                    "source_problem_ids": [problem.problem_id],
                }
            finally:
                graph.close()
        except Exception:
            return None

    def _apply_active_paths_to_frontiers(
        self,
        frontier_directives: list[dict],
        active_research_paths: list[ActiveResearchPath],
    ) -> list[dict]:
        path_by_frontier = {
            path.target_frontier_problem_id: path
            for path in active_research_paths
            if path.target_frontier_problem_id
        }
        active_domain_ids = {path.domain_id for path in active_research_paths if path.status in {"pursue_now", "incubate"}}
        enriched: list[dict] = []
        for rec in frontier_directives:
            entry = dict(rec)
            domain_id = self._infer_domain_id(" ".join([
                str(rec.get("domain", "") or ""),
                str(rec.get("title", "") or ""),
            ]))
            path = path_by_frontier.get(str(rec.get("problem_id", "") or ""))
            entry["active_domain_id"] = domain_id
            entry["active_path_id"] = path.path_id if path else ""
            entry["path_kind"] = path.path_kind if path else ""
            if path:
                entry["path_status"] = "active" if path.status == "pursue_now" else "incubating"
                entry["director_focus"] = path.problem_statement
                entry["writing_policy"] = path.writing_policy
            elif domain_id in active_domain_ids:
                entry["path_status"] = "domain_watch"
                entry["director_focus"] = (
                    "This frontier lives inside an active ML field, but TAR has not explicitly selected it as a current research path."
                )
                entry["writing_policy"] = (
                    "Keep this frontier visible for planning, but do not write papers on it until the Director explicitly selects it."
                )
            else:
                entry["path_status"] = "out_of_scope"
                entry["director_focus"] = entry.get("next_action", "")
                entry["writing_policy"] = "Only write on active Director-selected frontier paths."
            enriched.append(entry)
        return enriched

    def _apply_active_paths_to_papers(
        self,
        paper_directives: list[dict],
        active_research_paths: list[ActiveResearchPath],
        knowledge_domains: list[KnowledgeDomain],
    ) -> list[dict]:
        active_domain_ids = {path.domain_id for path in active_research_paths if path.status in {"pursue_now", "incubate"}}
        path_by_paper = {
            path.target_paper_id: path
            for path in active_research_paths
            if path.target_paper_id
        }
        path_by_frontier = {
            path.target_frontier_problem_id: path
            for path in active_research_paths
            if path.target_frontier_problem_id
        }
        enriched: list[dict] = []
        seen_papers: set[str] = set()
        for rec in paper_directives:
            entry = dict(rec)
            paper_id = str(entry.get("paper_id", "") or "")
            frontier_id = str(entry.get("frontier_problem_id", "") or "")
            domain_id = self._infer_domain_id(" ".join([
                str(entry.get("frontier_problem_title", "") or ""),
                str(entry.get("title", "") or ""),
            ]))
            path = path_by_paper.get(paper_id) or path_by_frontier.get(frontier_id)
            if path:
                scope_status = "active" if path.status == "pursue_now" else "incubating"
            elif domain_id in active_domain_ids:
                scope_status = "domain_watch"
            else:
                scope_status = "out_of_scope"
            entry["active_domain_id"] = domain_id
            entry["active_path_id"] = path.path_id if path else ""
            entry["path_kind"] = path.path_kind if path else ""
            entry["scope_status"] = scope_status
            entry["director_focus"] = path.problem_statement if path else ""
            entry["writing_policy"] = path.writing_policy if path else "Do not write outside TAR's active research paths."
            if path:
                entry["director_recommendation"] = path.writing_policy
                entry["recommendation"] = path.writing_policy
                if path.path_kind == "domain_frontier_scan":
                    entry["readiness"] = "hold"
                    entry["recommendation"] = (
                        "The Director is still scanning this active field for a concrete frontier problem. Do not draft a paper yet."
                    )
                elif path.path_kind == "novel_problem":
                    if path.status == "pursue_now" and path.verified_source_count >= 3:
                        entry["readiness"] = "outline_now"
                    else:
                        entry["readiness"] = "hold"
                elif path.status != "pursue_now":
                    entry["readiness"] = "hold"
                    entry["recommendation"] = (
                        "This problem is incubating, but the Director has not promoted it into the active writing lane yet."
                    )
            if scope_status in {"domain_watch", "out_of_scope"}:
                entry["readiness"] = "hold"
                entry["recommendation"] = (
                    "Outside TAR's current self-chosen research paths. Do not write this now; keep writing focused on the Director's selected problems."
                )
                if scope_status == "domain_watch":
                    entry["recommendation"] = (
                        "This topic is inside an active ML field, but TAR has not chosen it as a current problem to solve. Hold writing."
                    )
            seen_papers.add(paper_id)
            enriched.append(entry)

        for path in active_research_paths:
            if path.path_kind != "novel_problem" or not path.target_paper_id or path.target_paper_id in seen_papers:
                continue
            enriched.append({
                "paper_id": path.target_paper_id,
                "title": path.title,
                "frontier_problem_id": path.target_frontier_problem_id,
                "frontier_problem_title": path.target_frontier_title,
                "dataset": "",
                "priority_score": round(path.priority_score, 1),
                "complete_count": 0,
                "running_count": 0,
                "pending_count": 0,
                "waiting_for_experiments": [],
                "linked_experiment_ids": [],
                "linked_result_paths": [],
                "recommendation": path.writing_policy,
                "readiness": "outline_now" if path.status == "pursue_now" and path.verified_source_count >= 3 else "hold",
                "truth_status": "provisional" if path.verified_source_count >= 1 else "weak",
                "why_now": path.why_this_now,
                "author_rank": 999,
                "active_domain_id": path.domain_id,
                "active_path_id": path.path_id,
                "path_kind": path.path_kind,
                "scope_status": "active" if path.status == "pursue_now" else "incubating",
                "director_focus": path.problem_statement,
                "writing_policy": path.writing_policy,
                "director_recommendation": path.writing_policy,
                "source_gap_ids": path.source_gap_ids[:],
                "source_problem_ids": path.source_problem_ids[:],
            })

        enriched.sort(
            key=lambda rec: (
                rec.get("scope_status") != "active",
                rec.get("readiness") not in {"write_now", "outline_now"},
                -float(rec.get("priority_score", 0.0) or 0.0),
                str(rec.get("title", "") or ""),
            )
        )
        return enriched

    def _build_paper_directives(self, frontier_directives: list[dict], experiments: list[dict]) -> list[dict]:
        by_paper: dict[str, dict] = {}
        by_frontier = {rec["problem_id"]: rec for rec in frontier_directives}

        for exp in experiments:
            paper_id = str(exp.get("author_paper_id") or exp.get("project_id") or "").strip()
            if not paper_id:
                continue
            frontier_id = str(exp.get("frontier_problem_id") or "")
            if not frontier_id and paper_id.startswith("director-"):
                continue
            frontier = by_frontier.get(frontier_id, {})
            entry = by_paper.setdefault(paper_id, {
                "paper_id": paper_id,
                "title": (exp.get("context") or {}).get("feeds_paper") or paper_id.replace("_", " ").replace("-", " ").title(),
                "frontier_problem_id": frontier_id,
                "frontier_problem_title": frontier.get("title", ""),
                "global_problem_statement": frontier.get("global_problem_statement", ""),
                "industry_contexts": frontier.get("industry_contexts", []),
                "solution_family": frontier.get("solution_family", "TAR/TCL/ASC"),
                "solution_novelty_note": frontier.get("solution_novelty_note", ""),
                "target_venues": frontier.get("target_venues", []),
                "candidate_datasets": frontier.get("candidate_datasets", []),
                "candidate_backbones": frontier.get("candidate_backbones", []),
                "external_baselines": frontier.get("external_baselines", []),
                "research_guidance": frontier.get("research_guidance", ""),
                "dataset": str(exp.get("dataset", "") or ""),
                "priority_score": 0.0,
                "complete_count": 0,
                "running_count": 0,
                "pending_count": 0,
                "waiting_for_experiments": [],
                "linked_experiment_ids": [],
                "linked_result_paths": [],
                "recommendation": "Hold until stronger evidence accumulates.",
                "readiness": "hold",
                "publication_lane": "problem_scoping",
                "truth_status": frontier.get("truth_status", "weak"),
                "why_now": frontier.get("why_now", ""),
            })
            status = str(exp.get("status", ""))
            stage = str(exp.get("stage", ""))
            if status == "complete" or stage == "complete":
                entry["complete_count"] += 1
            elif status == "running" or stage == "running":
                entry["running_count"] += 1
                if exp.get("id"):
                    entry["waiting_for_experiments"].append(exp["id"])
            else:
                entry["pending_count"] += 1
                if exp.get("id"):
                    entry["waiting_for_experiments"].append(exp["id"])
            if exp.get("id"):
                entry["linked_experiment_ids"].append(str(exp["id"]))
            result_path = str(exp.get("result_path", "") or "")
            if result_path:
                entry["linked_result_paths"].append(result_path)
            entry["priority_score"] = max(entry["priority_score"], float(frontier.get("priority_score", 0.0)))

        for frontier in frontier_directives:
            paper_id = frontier["suggested_paper_id"]
            if paper_id in by_paper:
                continue
            readiness = "exploratory"
            recommendation = "Draft a frontier position paper only after stronger evidence exists."
            if frontier["readiness"] == "write_now":
                readiness = "write_now"
                recommendation = "Write the frontier paper now; this area already has breakthrough-grade support."
            elif frontier["readiness"] == "outline_now":
                readiness = "outline_now"
                recommendation = "Outline the frontier paper now and attach the finished experiment evidence."
            by_paper[paper_id] = {
                "paper_id": paper_id,
                "title": frontier["suggested_paper_title"],
                "frontier_problem_id": frontier["problem_id"],
                "frontier_problem_title": frontier["title"],
                "global_problem_statement": frontier.get("global_problem_statement", ""),
                "industry_contexts": frontier.get("industry_contexts", []),
                "solution_family": frontier.get("solution_family", "TAR/TCL/ASC"),
                "solution_novelty_note": frontier.get("solution_novelty_note", ""),
                "target_venues": frontier.get("target_venues", []),
                "candidate_datasets": frontier.get("candidate_datasets", []),
                "candidate_backbones": frontier.get("candidate_backbones", []),
                "external_baselines": frontier.get("external_baselines", []),
                "research_guidance": frontier.get("research_guidance", ""),
                "dataset": "",
                "priority_score": frontier["priority_score"],
                "complete_count": 0,
                "running_count": 0,
                "pending_count": 0,
                "waiting_for_experiments": frontier["waiting_on_experiment_ids"][:],
                "linked_experiment_ids": frontier["linked_experiment_ids"][:],
                "linked_result_paths": [],
                "recommendation": recommendation,
                "readiness": readiness,
                "publication_lane": frontier.get("publication_lane", "problem_scoping"),
                "truth_status": frontier["truth_status"],
                "why_now": frontier["why_now"],
            }

        paper_directives: list[dict] = []
        for entry in by_paper.values():
            if not str(entry.get("frontier_problem_id", "") or "") and str(entry.get("paper_id", "") or "").startswith("director-"):
                continue
            total_complete = entry["complete_count"]
            total_waiting = entry["running_count"] + entry["pending_count"]
            if total_complete > 0 and total_waiting == 0:
                entry["readiness"] = "write_now"
                entry["recommendation"] = (
                    "Prioritize a full rewrite now. Keep only scientifically backed claims, cite exact validated result files, "
                    "and include evidence-backed figures/diagrams generated from the completed runs."
                )
                entry["publication_lane"] = "conference_candidate"
            elif total_complete > 0:
                entry["readiness"] = "outline_now"
                entry["recommendation"] = (
                    "Rewrite from current validated evidence and generate figures for the completed runs, "
                    "but keep any claim that depends on unfinished experiments explicitly provisional."
                )
                entry["publication_lane"] = "conference_material_in_progress"
            elif entry["running_count"] > 0:
                entry["readiness"] = "prepare_now"
                entry["recommendation"] = (
                    "Prepare the full rewrite scaffold and the validated-results figure plan, "
                    "but do not promote unvalidated claims until the running experiments finish."
                )
                entry["publication_lane"] = "evidence_building"
            elif entry["pending_count"] > 0:
                entry["readiness"] = "blocked"
                entry["recommendation"] = (
                    "Do not advance scientific claims yet. Keep only the rewrite checklist, evidence checklist, "
                    "and figure requirements tied to the queued experiments."
                )
                entry["publication_lane"] = "evidence_building"
            entry["waiting_for_experiments"] = sorted(set(entry["waiting_for_experiments"]))
            entry["linked_experiment_ids"] = sorted(set(entry["linked_experiment_ids"]))
            entry["linked_result_paths"] = sorted(set(entry["linked_result_paths"]))
            paper_directives.append(entry)

        paper_directives.sort(
            key=lambda rec: (
                rec["readiness"] not in {"write_now", "outline_now"},
                -rec["priority_score"],
                rec["title"],
            )
        )
        return paper_directives

    def _frontier_experiment_catalog(
        self,
        frontier: dict[str, Any],
        paper: dict[str, Any],
        path: ActiveResearchPath | None,
    ) -> list[dict[str, Any]]:
        frontier_id = str(frontier.get("problem_id", "") or "")
        frontier_title = str(frontier.get("title", frontier_id) or frontier_id)
        target_paper_id = str(
            paper.get("paper_id", "")
            or frontier.get("suggested_paper_id", "")
            or f"frontier-paper-{frontier_id}"
        )
        target_paper_title = str(
            paper.get("title", "")
            or frontier.get("suggested_paper_title", "")
            or target_paper_id.replace("_", " ").replace("-", " ").title()
        )
        target_venues = [str(item) for item in frontier.get("target_venues", []) if str(item).strip()]
        candidate_datasets = [str(item) for item in frontier.get("candidate_datasets", []) if str(item).strip()]
        candidate_backbones = [str(item) for item in frontier.get("candidate_backbones", []) if str(item).strip()]
        external_baselines = [str(item) for item in frontier.get("external_baselines", []) if str(item).strip()]
        research_guidance = str(frontier.get("research_guidance", "") or "")

        base_common = {
            "frontier_problem_id": frontier_id,
            "global_problem_statement": str(frontier.get("global_problem_statement", "") or ""),
            "solution_family": str(frontier.get("solution_family", "TAR/TCL/ASC") or "TAR/TCL/ASC"),
            "solution_novelty_note": str(frontier.get("solution_novelty_note", "") or ""),
            "target_paper_id": target_paper_id,
            "target_paper_title": target_paper_title,
            "active_path_id": path.path_id if path else str(frontier.get("active_path_id", "") or ""),
            "path_kind": path.path_kind if path else str(frontier.get("path_kind", "") or ""),
            "path_status": str(frontier.get("path_status", "") or ""),
            "target_venues": target_venues,
            "candidate_datasets": candidate_datasets,
            "candidate_backbones": candidate_backbones,
            "external_baselines": external_baselines,
            "research_guidance": research_guidance,
            "internal_method_role": (
                "TAR/TCL/ASC are the user's unpublished internal methods under evaluation. "
                "They are not assumed solutions and must be tested against real external baselines."
            ),
        }

        if frontier_id == "fp-scale-up":
            return [
                {
                    **base_common,
                    "experiment_id": "phase16_scale_up",
                    "title": "Phase 16 - CIFAR-100 Scale-up",
                    "proposal_origin": "suite",
                    "proposal_kind": "resume_suite",
                    "hypothesis_name": "scale_up_validation",
                    "dataset": "split_cifar100",
                    "method": "tcl",
                    "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                    "seeds": [42, 0, 1],
                    "estimated_runtime_h": 24.0,
                    "hardware_budget": {"vram_gb": 3.2, "cpu_cores": 4},
                    "depends_on": [],
                    "backbone": "resnet18",
                    "epochs": 40,
                    "mechanism_focus": (
                        "Test whether the user's unpublished TAR/TCL/ASC methods still retain old tasks when the "
                        "stream becomes materially harder and class diversity increases."
                    ),
                    "experiment_goal": (
                        f"Advance evidence on the global problem '{frontier_title}' on Split-CIFAR-100 while comparing "
                        "the internal TAR/TCL/ASC methods against real external continual-learning baselines."
                    ),
                    "description": (
                        f"Scale-up suite on {frontier_title}. This is a harder 100-class benchmark chosen for the "
                        "real scaling question rather than convenience, and it keeps external baselines in the loop."
                    ),
                    "research_strategy": (
                        "Use the hardest runnable benchmark that directly answers the scale-up question. Treat TAR/TCL/ASC "
                        "as candidate internal methods and compare them against established baselines before claiming robustness."
                    ),
                    "config_overrides": {
                        "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                        "external_baselines": external_baselines or ["ewc", "sgd_baseline", "experience_replay", "lwf"],
                    },
                    "priority_bias": 34.0,
                },
                {
                    **base_common,
                    "experiment_id": "phase17_tinyimagenet",
                    "title": "Phase 17 - TinyImageNet Scale-up",
                    "proposal_origin": "suite",
                    "proposal_kind": "follow_on_suite",
                    "hypothesis_name": "scale_up_validation",
                    "dataset": "split_tinyimagenet",
                    "method": "tcl",
                    "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                    "seeds": [42, 0, 1],
                    "estimated_runtime_h": 36.0,
                    "hardware_budget": {"vram_gb": 3.3, "cpu_cores": 4},
                    "depends_on": ["phase16_scale_up"],
                    "backbone": "resnet18",
                    "epochs": 40,
                    "mechanism_focus": (
                        "Push the user's unpublished TAR/TCL/ASC methods onto a larger visual stream to see whether the thermodynamic regime signal "
                        "remains useful when image size and class count both grow."
                    ),
                    "experiment_goal": (
                        f"Complete the highest-cost validation step for the global problem '{frontier_title}'."
                    ),
                    "description": (
                        f"Scale-up suite on {frontier_title}. Split-TinyImageNet is the strongest currently supported "
                        "generalization check in the queue, and it keeps external baselines attached."
                    ),
                    "research_strategy": (
                        "Escalate to a larger dataset only after the lower-cost scale-up run is complete, and preserve the "
                        "same external baseline comparison so claims remain grounded."
                    ),
                    "config_overrides": {
                        "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                        "external_baselines": external_baselines or ["ewc", "sgd_baseline", "experience_replay", "lwf"],
                    },
                    "priority_bias": 20.0,
                },
            ]

        if frontier_id == "fp-regime-detection-accuracy":
            return [
                {
                    **base_common,
                    "experiment_id": "director-regime-detection-accuracy-regime-probe",
                    "title": "Director Follow-up - Thermodynamic Regime Detection Accuracy",
                    "proposal_origin": "director",
                    "proposal_kind": "frontier_probe",
                    "hypothesis_name": "director_regime_probe",
                    "dataset": "split_cifar10",
                    "method": "tcl",
                    "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                    "seeds": [42, 0, 1],
                    "estimated_runtime_h": 6.0,
                    "hardware_budget": {"vram_gb": 2.5, "cpu_cores": 4},
                    "depends_on": [],
                    "backbone": "resnet18",
                    "epochs": 40,
                    "mechanism_focus": (
                        "Adjust the user's unpublished TAR/TCL/ASC regime controller so sigma-star calibration and resets become more reliable."
                    ),
                    "experiment_goal": (
                        f"Use a fast, falsifiable probe to improve evidence on the real-world problem '{frontier_title}' "
                        "before escalating to heavier datasets."
                    ),
                    "description": (
                        f"Director-selected probe on {frontier_title}. This low-cost run checks whether the internal "
                        "thermodynamic controller adds value relative to external baselines instead of assuming it does."
                    ),
                    "research_strategy": (
                        "Use a smaller supported benchmark for quick diagnosis, but keep richer candidate datasets available for "
                        "later validation if the signal holds."
                    ),
                    "config_overrides": {
                        "tcl_governor_enabled": True,
                        "tcl_penalty_lambda": 0.015,
                        "tcl_alpha": 0.55,
                        "tcl_reset_on_task_boundary": True,
                        "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                        "external_baselines": external_baselines or ["ewc", "si", "sgd_baseline"],
                    },
                    "priority_bias": 18.0,
                },
                {
                    **base_common,
                    "experiment_id": "director-regime-detection-accuracy-sigma-calibration-probe",
                    "title": "Director Follow-up - Sigma-Star Calibration Sensitivity",
                    "proposal_origin": "director",
                    "proposal_kind": "frontier_probe",
                    "hypothesis_name": "director_sigma_probe",
                    "dataset": "split_cifar10",
                    "method": "tcl",
                    "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                    "seeds": [42, 0, 1],
                    "estimated_runtime_h": 6.0,
                    "hardware_budget": {"vram_gb": 2.5, "cpu_cores": 4},
                    "depends_on": [],
                    "backbone": "resnet18",
                    "epochs": 40,
                    "mechanism_focus": (
                        "Test whether sigma-star calibration in the user's unpublished TAR/TCL/ASC regime controller "
                        "remains accurate when the penalty lambda is raised to sharpen the detection threshold."
                    ),
                    "experiment_goal": (
                        f"Complement the regime-probe on '{frontier_title}' by testing a sharper calibration regime "
                        "while keeping external baselines for honest comparison."
                    ),
                    "description": (
                        f"Director-selected sigma calibration probe on {frontier_title}. "
                        "Tests whether a higher penalty sharpens or destabilises regime detection."
                    ),
                    "research_strategy": (
                        "Use the same cheap benchmark as the regime-probe but shift to a higher penalty to test "
                        "whether the internal regime detector's calibration remains reliable under stronger regularisation."
                    ),
                    "config_overrides": {
                        "tcl_governor_enabled": True,
                        "tcl_penalty_lambda": 0.025,
                        "tcl_alpha": 0.55,
                        "tcl_reset_on_task_boundary": True,
                        "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                        "external_baselines": external_baselines or ["ewc", "si", "sgd_baseline"],
                    },
                    "priority_bias": 15.0,
                },
            ]

        if frontier_id == "fp-hyperparameter-robustness":
            return [
                {
                    **base_common,
                    "experiment_id": "director-hyperparameter-robustness-lambda-probe",
                    "title": "Director Follow-up - TCL Hyperparameter Robustness",
                    "proposal_origin": "director",
                    "proposal_kind": "frontier_probe",
                    "hypothesis_name": "director_lambda_probe",
                    "dataset": "split_cifar10",
                    "method": "tcl",
                    "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                    "seeds": [42, 0, 1],
                    "estimated_runtime_h": 6.0,
                    "hardware_budget": {"vram_gb": 2.5, "cpu_cores": 4},
                    "depends_on": [],
                    "backbone": "resnet18",
                    "epochs": 40,
                    "mechanism_focus": (
                        "Stress-test the user's unpublished TAR/TCL/ASC anchor strength to see whether the mechanism remains useful without narrow tuning."
                    ),
                    "experiment_goal": (
                        f"Use a low-cost but falsifiable probe to strengthen evidence on '{frontier_title}' while still "
                        "anchoring the claim against external baselines."
                    ),
                    "description": (
                        f"Director-selected probe on {frontier_title}. This tests whether the internal method remains useful "
                        "under tougher settings instead of only looking good near its home hyperparameters."
                    ),
                    "research_strategy": (
                        "Probe sensitivity on a cheap supported benchmark first, but keep additional datasets and backbones in the "
                        "design metadata so robustness work can expand beyond CIFAR-10."
                    ),
                    "config_overrides": {
                        "tcl_governor_enabled": True,
                        "tcl_penalty_lambda": 0.02,
                        "tcl_alpha": 0.50,
                        "tcl_reset_on_task_boundary": True,
                        "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                        "external_baselines": external_baselines or ["ewc", "si", "sgd_baseline"],
                    },
                    "priority_bias": 16.0,
                },
                {
                    **base_common,
                    "experiment_id": "director-hyperparameter-robustness-alpha-probe",
                    "title": "Director Follow-up - TCL Alpha Sensitivity",
                    "proposal_origin": "director",
                    "proposal_kind": "frontier_probe",
                    "hypothesis_name": "director_alpha_probe",
                    "dataset": "split_cifar10",
                    "method": "tcl",
                    "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                    "seeds": [42, 0, 1],
                    "estimated_runtime_h": 6.0,
                    "hardware_budget": {"vram_gb": 2.5, "cpu_cores": 4},
                    "depends_on": [],
                    "backbone": "resnet18",
                    "epochs": 40,
                    "mechanism_focus": (
                        "Stress-test the thermal blending coefficient (alpha) in the user's unpublished TAR/TCL/ASC methods "
                        "to see whether robustness holds across different mixing ratios — orthogonal to the lambda probe."
                    ),
                    "experiment_goal": (
                        f"Complement the lambda sensitivity probe on '{frontier_title}' by varying the alpha "
                        "blending term independently while keeping external baselines for honest comparison."
                    ),
                    "description": (
                        f"Director-selected alpha sensitivity probe on {frontier_title}. "
                        "Tests the orthogonal hyperparameter axis to the completed lambda probe."
                    ),
                    "research_strategy": (
                        "Run an orthogonal hyperparameter probe on the same cheap benchmark used for the lambda probe, "
                        "testing whether the mechanism degrades when the blending coefficient is raised."
                    ),
                    "config_overrides": {
                        "tcl_governor_enabled": True,
                        "tcl_penalty_lambda": 0.015,
                        "tcl_alpha": 0.70,
                        "tcl_reset_on_task_boundary": True,
                        "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                        "external_baselines": external_baselines or ["ewc", "si", "sgd_baseline"],
                    },
                    "priority_bias": 14.0,
                },
            ]

        if frontier_id == "fp-catastrophic-forgetting":
            return [
                {
                    **base_common,
                    "experiment_id": "director-catastrophic-forgetting-carryover-probe",
                    "title": "Director Follow-up - Catastrophic Forgetting in Sequential Task Learning",
                    "proposal_origin": "director",
                    "proposal_kind": "frontier_probe",
                    "hypothesis_name": "director_carryover_probe",
                    "dataset": "split_cifar10",
                    "method": "tcl",
                    "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                    "seeds": [42, 0, 1],
                    "estimated_runtime_h": 6.0,
                    "hardware_budget": {"vram_gb": 2.5, "cpu_cores": 4},
                    "depends_on": [],
                    "backbone": "resnet18",
                    "epochs": 40,
                    "mechanism_focus": (
                        "Carry thermal state across task boundaries so the user's unpublished TAR/TCL/ASC methods can be tested as one candidate attack on forgetting."
                    ),
                    "experiment_goal": (
                        f"Advance evidence on the real-world problem '{frontier_title}' with a direct forgetting-reduction probe "
                        "that still measures the internal method against real external baselines."
                    ),
                    "description": (
                        f"Director-selected probe on {frontier_title}. This tests whether cross-task thermal carry-over improves "
                        "retention without assuming the internal method is already the right answer."
                    ),
                    "research_strategy": (
                        "Use a fast supported benchmark to ask one narrow forgetting question at a time, while retaining enough "
                        "baseline structure to decide whether the internal method is genuinely useful."
                    ),
                    "config_overrides": {
                        "tcl_governor_enabled": True,
                        "tcl_penalty_lambda": 0.01,
                        "tcl_alpha": 0.50,
                        "tcl_reset_on_task_boundary": False,
                        "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                        "external_baselines": external_baselines or ["ewc", "si", "sgd_baseline", "experience_replay"],
                    },
                    "priority_bias": 17.0,
                },
                {
                    **base_common,
                    "experiment_id": "director-catastrophic-forgetting-boundary-reset-probe",
                    "title": "Director Follow-up - Boundary-Reset Forgetting Test",
                    "proposal_origin": "director",
                    "proposal_kind": "frontier_probe",
                    "hypothesis_name": "director_boundary_probe",
                    "dataset": "split_cifar10",
                    "method": "tcl",
                    "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                    "seeds": [42, 0, 1],
                    "estimated_runtime_h": 6.0,
                    "hardware_budget": {"vram_gb": 2.5, "cpu_cores": 4},
                    "depends_on": [],
                    "backbone": "resnet18",
                    "epochs": 40,
                    "mechanism_focus": (
                        "Test whether explicit boundary resets in the user's unpublished TAR/TCL/ASC thermal controller "
                        "reduce forgetting compared to the thermal carry-over strategy."
                    ),
                    "experiment_goal": (
                        f"Advance evidence on the real-world problem '{frontier_title}' by comparing boundary-reset "
                        "against carry-over as orthogonal strategies for reducing catastrophic forgetting."
                    ),
                    "description": (
                        f"Director-selected boundary-reset probe on {frontier_title}. "
                        "Tests the complementary strategy to the carryover probe to triangulate the best forgetting defence."
                    ),
                    "research_strategy": (
                        "Run the orthogonal boundary-reset configuration on the same cheap benchmark as the carryover probe "
                        "so the two probes together answer whether to carry or reset thermal state across task boundaries."
                    ),
                    "config_overrides": {
                        "tcl_governor_enabled": True,
                        "tcl_penalty_lambda": 0.02,
                        "tcl_alpha": 0.50,
                        "tcl_reset_on_task_boundary": True,
                        "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                        "external_baselines": external_baselines or ["ewc", "si", "sgd_baseline", "experience_replay"],
                    },
                    "priority_bias": 15.0,
                },
            ]

        if frontier_id == "fp-class-incremental":
            return [{
                **base_common,
                "experiment_id": "phase15_class_incremental_search",
                "title": "Phase 15 - Class-Incremental Search",
                "proposal_origin": "suite",
                "proposal_kind": "frontier_probe",
                "hypothesis_name": "class_incremental_search",
                "dataset": "split_cifar10",
                "method": "tcl",
                "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                "seeds": [42, 0, 1],
                "estimated_runtime_h": 8.0,
                "hardware_budget": {"vram_gb": 2.5, "cpu_cores": 4},
                "depends_on": [],
                "backbone": "resnet18",
                "epochs": 40,
                "mechanism_focus": (
                    "Evaluate whether the user's unpublished TAR/TCL/ASC methods survive the shift "
                    "from task-ID-assisted to class-incremental learning without task boundaries."
                ),
                "experiment_goal": (
                    f"Advance evidence on the real-world problem '{frontier_title}' by testing "
                    "the internal TAR/TCL/ASC method family in the class-incremental setting "
                    "against external baselines that do not rely on task labels."
                ),
                "description": (
                    f"Class-incremental probe for {frontier_title}. Tests whether thermodynamic "
                    "regime detection adds value when task identity is unavailable at inference time."
                ),
                "research_strategy": (
                    "Use the lightest supported benchmark (Split-CIFAR-10) to ask one specific falsifiable "
                    "question about class-incremental generalisation before escalating to harder settings."
                ),
                "config_overrides": {
                    "setting": "class_incremental",
                    "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                    "external_baselines": external_baselines or ["ewc", "si", "sgd_baseline", "experience_replay"],
                },
                "priority_bias": 15.0,
            }]

        return []

    def _active_path_experiment_catalog(self, path: ActiveResearchPath) -> list[dict[str, Any]]:
        if _STRICT_REAL_WORLD_FRONTIER_ONLY:
            return []
        if path.path_kind not in {"novel_problem", "domain_frontier_scan"}:
            return []
        if path.status not in {"pursue_now", "incubate"}:
            return []

        supported_profiles = {
            "continual_learning": {
                "dataset": "split_cifar10",
                "backbone": "resnet18",
                "hardware_budget": {"vram_gb": 2.5, "cpu_cores": 4},
                "external_baselines": ["ewc", "si", "sgd_baseline", "experience_replay"],
                "target_venues": ["NeurIPS", "ICML", "AISTATS"],
            },
            "thermodynamics_ml": {
                "dataset": "split_cifar10",
                "backbone": "resnet18",
                "hardware_budget": {"vram_gb": 2.5, "cpu_cores": 4},
                "external_baselines": ["ewc", "si", "sgd_baseline"],
                "target_venues": ["NeurIPS", "ICML"],
            },
            "general_ai": {
                "dataset": "split_cifar10",
                "backbone": "resnet18",
                "hardware_budget": {"vram_gb": 2.5, "cpu_cores": 4},
                "external_baselines": ["ewc", "sgd_baseline", "experience_replay"],
                "target_venues": ["NeurIPS", "ICML"],
            },
        }
        profile = supported_profiles.get(path.domain_id)
        if not profile:
            return []

        problem_slug = _slug(path.title)[:28] or _slug(path.domain_label)[:28] or "path"
        kind_slug = "novel" if path.path_kind == "novel_problem" else "scan"
        exp_id = f"director-{_slug(path.domain_id)}-{kind_slug}-{problem_slug}-probe"
        return [{
            "experiment_id": exp_id,
            "title": f"Director Path Probe - {path.title}",
            "proposal_origin": "director",
            "proposal_kind": "active_path_probe",
            "hypothesis_name": "director_frontier_probe",
            "dataset": str(profile["dataset"]),
            "method": "tcl",
            "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
            "seeds": [42, 0, 1],
            "estimated_runtime_h": 6.0,
            "hardware_budget": dict(profile["hardware_budget"]),
            "depends_on": [],
            "backbone": str(profile["backbone"]),
            "epochs": 40,
            "frontier_problem_id": path.target_frontier_problem_id,
            "global_problem_statement": path.problem_statement,
            "solution_family": "TAR/TCL/ASC",
            "solution_novelty_note": (
                "TAR, TCL, and ASC are unpublished internal mechanisms under evaluation rather than established literature."
            ),
            "target_paper_id": path.target_paper_id,
            "target_paper_title": path.title if path.target_paper_id else "",
            "active_path_id": path.path_id,
            "path_kind": path.path_kind,
            "path_status": path.status,
            "target_venues": list(profile["target_venues"]),
            "candidate_datasets": [str(profile["dataset"])],
            "candidate_backbones": [str(profile["backbone"])],
            "external_baselines": list(profile["external_baselines"]),
            "research_guidance": path.experiment_policy,
            "internal_method_role": (
                "TAR/TCL/ASC are the user's unpublished internal methods under evaluation. "
                "Treat them as one candidate family under test rather than the assumed answer."
            ),
            "mechanism_focus": path.experiment_policy,
            "experiment_goal": path.problem_statement,
            "description": (
                f"Director-selected active-path probe for {path.title}. "
                "This converts a live incubating path into a concrete falsifiable experiment instead of leaving it as passive planning."
            ),
            "research_strategy": (
                "Use the lightest supported benchmark that still asks a real falsifiable question about this active path. "
                "Escalate only if the probe yields credible evidence."
            ),
            "config_overrides": {
                "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                "external_baselines": list(profile["external_baselines"]),
            },
            "priority_bias": 14.0 if path.path_kind == "novel_problem" else 10.0,
        }]

    def _build_experiment_directives(
        self,
        frontier_directives: list[dict],
        paper_directives: list[dict],
        active_research_paths: list[ActiveResearchPath],
        experiments: list[dict],
    ) -> list[dict]:
        path_by_frontier = {
            path.target_frontier_problem_id: path
            for path in active_research_paths
            if path.target_frontier_problem_id
        }
        paper_by_frontier = {
            str(rec.get("frontier_problem_id", "") or ""): rec
            for rec in paper_directives
            if str(rec.get("frontier_problem_id", "") or "")
        }
        experiment_by_id = {
            str(exp.get("id", "") or ""): exp
            for exp in experiments
            if exp.get("id")
        }
        complete_ids = {
            exp_id for exp_id, exp in experiment_by_id.items()
            if str(exp.get("status", "") or "") == "complete"
            or str(exp.get("stage", "") or "") == "complete"
        }

        directives: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        tracked_frontiers = [
            rec for rec in frontier_directives
            if (
                str(rec.get("path_status", "") or "") in {"active", "incubating", "domain_watch"}
                or rec.get("waiting_on_experiment_ids")
                or rec.get("linked_experiment_ids")
                or str(rec.get("readiness", "") or "") in {"experiment_first", "outline_now", "write_now"}
            )
        ]

        for frontier in tracked_frontiers:
            frontier_id = str(frontier.get("problem_id", "") or "")
            if not frontier_id:
                continue
            paper = paper_by_frontier.get(frontier_id, {})
            path = path_by_frontier.get(frontier_id)
            base_priority = float(path.priority_score if path else frontier.get("priority_score", 0.0) or 0.0)
            paper_readiness = str(paper.get("readiness", frontier.get("readiness", "")) or "")
            paper_boost = 18.0 if paper_readiness == "write_now" else 12.0 if paper_readiness == "outline_now" else 6.0 if paper_readiness == "prepare_now" else 0.0

            def _experiment_status(record: dict[str, Any] | None) -> str:
                if not record:
                    return "proposed"
                stage = str(record.get("stage", "") or "")
                status = str(record.get("status", "") or "")
                if status == "running" or stage == "running":
                    return "running"
                if stage == "stalled":
                    return "stalled"
                if status == "complete" or stage == "complete":
                    return "complete"
                if status in {"failed", "skipped"} or stage == "failed":
                    return "failed"
                return "queued"

            def _intent_for(status: str, unmet_deps: list[str]) -> str:
                if status == "running":
                    return "maintain_running"
                if status == "stalled":
                    return "resume_now"
                if status == "complete":
                    return "archive"
                if status == "failed":
                    return "retry_now"
                if unmet_deps:
                    return "hold_dependency"
                if status == "queued":
                    return "queue_now"
                return "propose_now"

            def _priority_for(exp_id: str, status: str, unmet_deps: list[str], bias: float = 0.0) -> float:
                status_boost = {
                    "running": 52.0,
                    "stalled": 58.0,
                    "queued": 40.0,
                    "proposed": 36.0,
                    "complete": -28.0,
                    "failed": 42.0,
                }.get(status, 0.0)
                dependency_penalty = 14.0 if unmet_deps else 0.0
                special_boost = 0.0
                if exp_id == "phase16_scale_up":
                    special_boost = 24.0
                elif exp_id == "phase17_tinyimagenet":
                    special_boost = 14.0
                return round(base_priority + paper_boost + status_boost + bias + special_boost - dependency_penalty, 1)

            for exp in experiments:
                if str(exp.get("frontier_problem_id", "") or "") != frontier_id:
                    continue
                exp_id = str(exp.get("id", "") or "")
                if not exp_id or exp_id in seen_ids:
                    continue
                status = _experiment_status(exp)
                if status == "complete":
                    continue
                depends_on = [str(dep) for dep in exp.get("depends_on", []) or [] if str(dep or "")]
                unmet_deps = [dep for dep in depends_on if dep not in complete_ids]
                intent = _intent_for(status, unmet_deps)
                priority_score = _priority_for(exp_id, status, unmet_deps)
                directives.append({
                    "experiment_id": exp_id,
                    "title": str(exp.get("name", exp_id) or exp_id),
                    "status": status,
                    "scheduler_intent": intent,
                    "priority_score": priority_score,
                    "frontier_problem_id": frontier_id,
                    "frontier_problem_title": str(frontier.get("title", "") or ""),
                    "global_problem_statement": str(frontier.get("global_problem_statement", "") or ""),
                    "solution_family": str(frontier.get("solution_family", "TAR/TCL/ASC") or "TAR/TCL/ASC"),
                    "solution_novelty_note": str(frontier.get("solution_novelty_note", "") or ""),
                    "target_paper_id": str(exp.get("author_paper_id", "") or paper.get("paper_id", "")),
                    "target_paper_title": str(paper.get("title", "") or exp.get("author_paper_id", "") or ""),
                    "active_path_id": path.path_id if path else str(frontier.get("active_path_id", "") or ""),
                    "path_kind": path.path_kind if path else str(frontier.get("path_kind", "") or ""),
                    "path_status": str(frontier.get("path_status", "") or ""),
                    "dataset": str(exp.get("dataset", "") or ""),
                    "backbone": str(exp.get("backbone", "resnet18") or "resnet18"),
                    "epochs": int(exp.get("epochs", 40) or 40),
                    "method": str(exp.get("method", "tcl") or "tcl"),
                    "seeds": [int(seed) for seed in exp.get("seeds", []) or [] if str(seed).strip()],
                    "hardware_budget": exp.get("hardware_budget", {}),
                    "candidate_datasets": ((exp.get("runtime_context") or {}).get("candidate_datasets") or frontier.get("candidate_datasets", [])),
                    "candidate_backbones": ((exp.get("runtime_context") or {}).get("candidate_backbones") or frontier.get("candidate_backbones", [])),
                    "external_baselines": (
                        ((exp.get("runtime_context") or {}).get("external_baselines"))
                        or ((exp.get("config_overrides") or {}).get("external_baselines"))
                        or frontier.get("external_baselines", [])
                    ),
                    "comparison_methods": (
                        ((exp.get("runtime_context") or {}).get("comparison_methods"))
                        or ((exp.get("config_overrides") or {}).get("comparison_methods"))
                        or [str(exp.get("method", "tcl") or "tcl")]
                    ),
                    "research_strategy": (
                        ((exp.get("runtime_context") or {}).get("research_strategy"))
                        or frontier.get("research_guidance", "")
                    ),
                    "internal_method_role": (
                        ((exp.get("runtime_context") or {}).get("internal_method_role"))
                        or "TAR/TCL/ASC are unpublished internal methods under evaluation, not assumed solutions."
                    ),
                    "depends_on": depends_on,
                    "blocked_by_experiment_ids": unmet_deps,
                    "proposal_origin": "queue",
                    "proposal_kind": "existing_experiment",
                    "experiment_goal": str(
                        ((exp.get("context") or {}).get("why"))
                        or frontier.get("next_action", "")
                        or f"Advance evidence on {frontier.get('title', frontier_id)}."
                    ),
                    "mechanism_focus": str(
                        ((exp.get("context") or {}).get("hypothesis"))
                        or exp.get("description", "")
                        or f"Test TAR/TCL/ASC as one candidate internal method family against the real-world problem {frontier.get('title', frontier_id)}."
                    ),
                    "why_now": (
                        f"{intent.replace('_', ' ')} for the real-world problem '{frontier.get('title', frontier_id)}'. "
                        f"TAR is evaluating the unpublished internal {frontier.get('solution_family', 'TAR/TCL/ASC')} work against real external baselines."
                    ),
                    "target_venues": [str(item) for item in frontier.get("target_venues", []) if str(item).strip()],
                    "estimated_runtime_h": float(exp.get("estimated_runtime_h", 0.0) or 0.0),
                })
                seen_ids.add(exp_id)

            for proposal in self._frontier_experiment_catalog(frontier, paper, path):
                exp_id = str(proposal.get("experiment_id", "") or "")
                if not exp_id or exp_id in seen_ids:
                    continue
                existing = experiment_by_id.get(exp_id, {})
                status = _experiment_status(existing)
                if status == "complete":
                    continue
                depends_on = [str(dep) for dep in proposal.get("depends_on", []) or [] if str(dep or "")]
                unmet_deps = [dep for dep in depends_on if dep not in complete_ids]
                intent = _intent_for(status, unmet_deps)
                if status == "proposed" and unmet_deps:
                    intent = "hold_dependency"
                priority_score = _priority_for(
                    exp_id,
                    status,
                    unmet_deps,
                    float(proposal.get("priority_bias", 0.0) or 0.0),
                )
                directives.append({
                    **proposal,
                    "status": status,
                    "scheduler_intent": intent,
                    "priority_score": priority_score,
                    "blocked_by_experiment_ids": unmet_deps,
                    "why_now": (
                        f"{intent.replace('_', ' ')} for the global problem '{frontier.get('title', frontier_id)}'. "
                        f"This is a TAR-directed test of the unpublished internal {frontier.get('solution_family', 'TAR/TCL/ASC')} work against real external baselines."
                    ),
                })
                seen_ids.add(exp_id)

            # LLM follow-up proposals — fires only when the static catalog is
            # exhausted for this frontier and there are completed experiments to
            # learn from. Capped at 2 new proposals per frontier per cycle.
            n_pending_for_frontier = sum(
                1 for d in directives
                if str(d.get("frontier_problem_id", "") or "") == frontier_id
                and str(d.get("status", "") or "") not in {"running", "complete", "failed"}
            )
            completed_for_frontier = [
                exp for exp in experiments
                if str(exp.get("frontier_problem_id", "") or "") == frontier_id
                and (
                    str(exp.get("status", "") or "") == "complete"
                    or str(exp.get("stage", "") or "") == "complete"
                )
            ]
            frontier_title = str(frontier.get("title", frontier_id) or frontier_id)
            if n_pending_for_frontier == 0 and completed_for_frontier:
                try:
                    llm_proposals = self._llm_propose_followup_experiments(
                        frontier, completed_for_frontier, seen_ids
                    )
                except Exception as exc:
                    print(f"[Director] LLM follow-up proposals failed for {frontier_id}: {exc}", flush=True)
                    llm_proposals = []
                for proposal in llm_proposals:
                    exp_id = str(proposal.get("experiment_id", "") or "")
                    if not exp_id or exp_id in seen_ids:
                        continue
                    status = "proposed"
                    unmet: list[str] = []
                    intent = "propose_now"
                    priority_score = _priority_for(exp_id, status, unmet, 0.0)
                    directives.append({
                        "experiment_id": exp_id,
                        "title": str(proposal.get("title", "") or f"LLM Follow-up - {frontier_title}"),
                        "dataset": str(proposal.get("dataset", "") or "split_cifar10"),
                        "backbone": str(proposal.get("backbone", "resnet18") or "resnet18"),
                        "method": "tcl",
                        "seeds": [42, 0, 1],
                        "config_overrides": dict(proposal.get("config_overrides") or {}),
                        "estimated_runtime_h": float(proposal.get("estimated_runtime_h", 6.0) or 6.0),
                        "hardware_budget": {"vram_gb": 2.5, "cpu_cores": 4},
                        "epochs": 40,
                        "depends_on": [],
                        "description": str(proposal.get("why", "") or "LLM-proposed follow-up experiment."),
                        "experiment_goal": str(proposal.get("why", "") or ""),
                        "mechanism_focus": str(proposal.get("hypothesis", "") or ""),
                        "status": status,
                        "scheduler_intent": intent,
                        "priority_score": priority_score,
                        "frontier_problem_id": frontier_id,
                        "frontier_problem_title": frontier_title,
                        "global_problem_statement": str(frontier.get("global_problem_statement", "") or ""),
                        "solution_family": str(frontier.get("solution_family", "TAR/TCL/ASC") or "TAR/TCL/ASC"),
                        "solution_novelty_note": str(frontier.get("solution_novelty_note", "") or ""),
                        "target_paper_id": str(
                            paper.get("paper_id", "")
                            or frontier.get("suggested_paper_id", "")
                            or f"frontier-paper-{frontier_id}"
                        ),
                        "active_path_id": path.path_id if path else str(frontier.get("active_path_id", "") or ""),
                        "path_kind": path.path_kind if path else str(frontier.get("path_kind", "") or ""),
                        "path_status": str(frontier.get("path_status", "") or ""),
                        "candidate_datasets": [str(d) for d in frontier.get("candidate_datasets", []) if str(d).strip()],
                        "candidate_backbones": [str(b) for b in frontier.get("candidate_backbones", []) if str(b).strip()],
                        "external_baselines": [str(b) for b in frontier.get("external_baselines", []) if str(b).strip()],
                        "comparison_methods": ["tcl", "ewc", "sgd_baseline"],
                        "research_strategy": str(frontier.get("research_guidance", "") or ""),
                        "internal_method_role": (
                            "TAR/TCL/ASC are the user's unpublished internal methods under evaluation. "
                            "They are not assumed solutions and must be tested against real external baselines."
                        ),
                        "proposal_origin": "director",
                        "proposal_kind": "llm_follow_up",
                        "blocked_by_experiment_ids": [],
                        "why_now": (
                            f"LLM-proposed follow-up for '{frontier_title}'. "
                            "Static catalog exhausted; all prior experiments complete or queued."
                        ),
                        "target_venues": [str(item) for item in frontier.get("target_venues", []) if str(item).strip()],
                    })
                    seen_ids.add(exp_id)

        for path in active_research_paths:
            if path.path_kind not in {"novel_problem", "domain_frontier_scan"}:
                continue
            for proposal in self._active_path_experiment_catalog(path):
                exp_id = str(proposal.get("experiment_id", "") or "")
                if not exp_id or exp_id in seen_ids:
                    continue
                existing = experiment_by_id.get(exp_id, {})
                status = _experiment_status(existing)
                if status == "complete":
                    continue
                depends_on = [str(dep) for dep in proposal.get("depends_on", []) or [] if str(dep or "")]
                unmet_deps = [dep for dep in depends_on if dep not in complete_ids]
                intent = _intent_for(status, unmet_deps)
                if status == "proposed" and unmet_deps:
                    intent = "hold_dependency"
                priority_score = round(
                    float(path.priority_score or 0.0)
                    + float(proposal.get("priority_bias", 0.0) or 0.0)
                    + {
                        "running": 52.0,
                        "stalled": 58.0,
                        "queued": 40.0,
                        "proposed": 34.0,
                        "failed": 38.0,
                    }.get(status, 0.0)
                    - (14.0 if unmet_deps else 0.0),
                    1,
                )
                directives.append({
                    **proposal,
                    "status": status,
                    "scheduler_intent": intent,
                    "priority_score": priority_score,
                    "blocked_by_experiment_ids": unmet_deps,
                    "frontier_problem_title": path.target_frontier_title,
                    "why_now": (
                        f"{intent.replace('_', ' ')} on the active research path '{path.title}'. "
                        "This keeps TAR turning evidence-backed incubation into concrete probes."
                    ),
                })
                seen_ids.add(exp_id)

        intent_rank = {
            "maintain_running": 0,
            "resume_now": 1,
            "retry_now": 2,
            "queue_now": 3,
            "propose_now": 4,
            "hold_dependency": 5,
            "rethink": 6,
            "archive": 7,
        }
        directives.sort(
            key=lambda rec: (
                intent_rank.get(str(rec.get("scheduler_intent", "") or ""), 9),
                -float(rec.get("priority_score", 0.0) or 0.0),
                str(rec.get("title", "") or ""),
            )
        )
        return directives

    def _build_evidence_directives(
        self,
        frontier_directives: list[dict],
        paper_directives: list[dict],
        knowledge_domains: list[KnowledgeDomain],
        experiments: list[dict],
        external_evidence_state: dict[str, Any] | None = None,
    ) -> list[dict]:
        directives: list[dict] = []
        exp_by_id = {
            str(exp.get("id", "") or ""): exp
            for exp in experiments
            if exp.get("id")
        }
        paper_by_frontier = {
            str(rec.get("frontier_problem_id", "") or ""): rec
            for rec in paper_directives
            if rec.get("frontier_problem_id")
        }

        for frontier in frontier_directives:
            frontier_id = str(frontier.get("problem_id", "") or "")
            waiting = [str(exp_id) for exp_id in frontier.get("waiting_on_experiment_ids", []) if str(exp_id or "")]
            linked_ids = [str(exp_id) for exp_id in frontier.get("linked_experiment_ids", []) if str(exp_id or "")]
            linked_paths = [
                str(exp.get("result_path", "") or "")
                for exp_id in linked_ids
                for exp in [exp_by_id.get(exp_id, {})]
                if str(exp.get("result_path", "") or "")
            ]
            truth_status = str(frontier.get("truth_status", "weak") or "weak")
            paper = paper_by_frontier.get(frontier_id, {})

            status = "verify_now"
            next_action = "Audit completed result files and reject unsupported claims."
            if waiting:
                status = "collect_now"
                next_action = "Finish the queued/running experiments before upgrading the claim."
            elif truth_status == "validated":
                status = "archive_now"
                next_action = "Lock the verified evidence into the paper and registry."
            elif truth_status in {"supported", "provisional"}:
                status = "verify_now"
                next_action = "Cross-check the finished runs against baselines and keep only factual claims."

            verification_standard = (
                "Require negative delta against the relevant baseline plus statistical support where available. "
                "Reject any claim that depends only on a single weak directional run."
            )
            if truth_status == "validated":
                verification_standard = (
                    "Preserve only the statistically supported or breakthrough-grade claims and attach the exact result files."
                )
            elif waiting:
                verification_standard = (
                    "Do not escalate claims until the queued experiments finish and the result JSONs can be audited."
                )

            title = str(frontier.get("title", frontier_id) or frontier_id)
            claim = (
                f"Verify whether TAR's current evidence really supports the frontier claim: {title}."
            )
            directives.append({
                "task_id": f"evidence-{frontier_id}",
                "task_type": "frontier_verification",
                "status": status,
                "priority_score": round(float(frontier.get("priority_score", 0.0) or 0.0) + (12.0 if waiting else 0.0), 1),
                "frontier_problem_id": frontier_id,
                "frontier_title": title,
                "paper_id": str(paper.get("paper_id", frontier.get("suggested_paper_id", "")) or ""),
                "paper_title": str(paper.get("title", frontier.get("suggested_paper_title", "")) or ""),
                "truth_status": truth_status,
                "claim_under_test": claim,
                "required_experiment_ids": waiting,
                "linked_experiment_ids": linked_ids,
                "linked_result_paths": linked_paths,
                "verification_standard": verification_standard,
                "next_action": next_action,
                "agent_brief": (
                    "Collect verified evidence only. Read the linked experiment outputs, compare each run against its baseline, "
                    "record exact numbers, and downgrade any claim that lacks replication or significance."
                ),
            })

        for domain in knowledge_domains:
            if domain.status not in {"seed", "candidate"} and domain.expansion_status not in {"active_expansion", "stabilizing"}:
                continue
            status = "harvest_now"
            if domain.external_verified_source_count >= 3:
                status = "verify_now"
            if domain.truth_status == "validated":
                status = "archive_now"
            directives.append({
                "task_id": f"domain-{domain.id}",
                "task_type": "domain_expansion",
                "status": status,
                "priority_score": round(max(1.0, 80.0 - float(domain.seed_priority)) + float(domain.active_expansion_score), 1),
                "frontier_problem_id": "",
                "frontier_title": domain.label,
                "paper_id": "",
                "paper_title": "",
                "truth_status": domain.truth_status,
                "claim_under_test": f"Assess whether TAR should open a verified research track in {domain.label}.",
                "required_experiment_ids": [],
                "linked_experiment_ids": [],
                "linked_result_paths": [],
                "verification_standard": (
                    "Treat this as weak information until TAR has at least three verified outside sources and one falsifiable experiment proposal."
                ),
                "next_action": (
                    f"{domain.next_action} Verified sources={domain.external_verified_source_count}, "
                    f"weak sources={domain.external_weak_source_count}, benchmarks={domain.external_benchmark_count}."
                ),
                "agent_brief": (
                    "Expand the knowledge base carefully: gather strong sources, flag hype or speculative claims as weak, "
                    "and only propose experiments that can be verified with concrete evidence."
                ),
                "verified_source_count": domain.external_verified_source_count,
                "weak_source_count": domain.external_weak_source_count,
                "benchmark_count": domain.external_benchmark_count,
                "last_literature_sync": domain.last_literature_sync,
                "top_verified_titles": domain.top_verified_titles[:],
            })

        directives.sort(
            key=lambda rec: (
                rec["status"] not in {"collect_now", "verify_now"},
                -float(rec.get("priority_score", 0.0) or 0.0),
                str(rec.get("frontier_title", "") or rec.get("task_id", "")),
            )
        )
        return directives

    def _llm_propose_followup_experiments(
        self,
        frontier: dict[str, Any],
        completed_exps: list[dict],
        seen_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Ask Claude for follow-up experiments when the static catalog is exhausted."""
        from tar_lab.llm_bridge import propose_followup_experiments
        frontier_id = str(frontier.get("problem_id", "") or "")
        frontier_title = str(frontier.get("title", frontier_id) or frontier_id)
        summaries: list[str] = []
        for exp in completed_exps:
            exp_id = str(exp.get("id", "") or "")
            dataset = str(exp.get("dataset", "") or "")
            backbone = str(exp.get("backbone", "resnet18") or "resnet18")
            result = self._load_experiment_result(exp)
            forgetting = ""
            if result:
                for key in ("tcl_forgetting", "forgetting", "mean_forgetting"):
                    if result.get(key) is not None:
                        forgetting = f"forgetting={result[key]:.4f}"
                        break
            summaries.append(f"{exp_id}: {dataset}, {backbone}{(', ' + forgetting) if forgetting else ''}")
        return propose_followup_experiments(
            self.workspace,
            frontier_id=frontier_id,
            frontier_title=frontier_title,
            global_problem_statement=str(frontier.get("global_problem_statement", "") or ""),
            candidate_datasets=[str(d) for d in frontier.get("candidate_datasets", []) if str(d).strip()],
            candidate_backbones=[str(b) for b in frontier.get("candidate_backbones", []) if str(b).strip()],
            external_baselines=[str(b) for b in frontier.get("external_baselines", []) if str(b).strip()],
            completed_summaries=summaries,
            exclude_ids=seen_ids,
            max_proposals=2,
        )

    def _infer_domain_id(self, text: str) -> str:
        hay = text.lower()
        best_id = "general_ai"
        best_hits = 0
        for spec in _DEFAULT_DOMAIN_SPECS:
            hits = sum(1 for kw in spec["keywords"] if kw in hay)
            if hits > best_hits:
                best_hits = hits
                best_id = spec["id"]
        if best_hits == 0:
            if "generic_ml" in hay or "ml" in hay:
                return "general_ai"
            return "general_ai"
        return best_id

    def _load_experiment_result(self, exp: dict) -> dict:
        candidates = []
        rp = str(exp.get("result_path", "") or "")
        if rp:
            candidates.append(Path(rp))
        exp_id = str(exp.get("id", "") or "")
        logical_name = str(exp.get("logical_name", "") or "")
        legacy_result_filename = str(exp.get("legacy_result_filename", "") or "")
        if not logical_name and exp_id:
            for entry in iter_phase_catalog_entries():
                if entry.experiment_id == exp_id:
                    logical_name = entry.logical_name
                    legacy_result_filename = entry.legacy_filename
                    break
        if logical_name:
            resolved = resolve_canonical_comparison_path(
                self.workspace,
                logical_name,
                legacy_filename=legacy_result_filename or None,
            )
            if resolved is not None:
                candidates.append(resolved)
        if exp_id:
            candidates.append(self.workspace / "tar_state" / "experiments" / exp_id / "result.json")
        for path in candidates:
            if not path.exists():
                continue
            data = _jload(path)
            if isinstance(data, dict):
                if "result" in data and isinstance(data["result"], dict):
                    return data["result"]
                return data
        return {}


    def _annotate_directives_with_llm(
        self,
        frontier_directives: list[dict],
        experiment_directives: list[dict],
        evidence_directives: list[dict],
        knowledge_domains: list[KnowledgeDomain],
    ) -> None:
        """
        Enrich top directives with Claude-generated insights.
        Annotations are cached; this is a no-op when Anthropic key is absent.
        Mutates the passed lists in-place by adding llm_* fields.

        Limits: top 3 frontiers, top 4 experiments (non-running), top 3 evidence tasks.
        """
        from tar_lab.llm_bridge import (
            synthesize_frontier_literature,
            evaluate_experiment_proposal,
            verify_result_claim,
        )
        domain_by_id = {dom.id: dom for dom in knowledge_domains}

        def _dom_summary(domain_id: str) -> tuple[str, list[str]]:
            dom = domain_by_id.get(domain_id)
            if not dom:
                return "", []
            return dom.learned_summary, dom.top_verified_titles[:]

        # 3 a. Frontier literature synthesis
        for rec in frontier_directives[:3]:
            frontier_id = str(rec.get("problem_id", "") or "")
            if not frontier_id:
                continue
            inferred_domain = self._infer_domain_id(" ".join([
                str(rec.get("domain", "") or ""),
                str(rec.get("title", "") or ""),
            ]))
            learned_summary, top_titles = _dom_summary(inferred_domain)
            synthesis = synthesize_frontier_literature(
                self.workspace,
                frontier_id,
                str(rec.get("title", frontier_id) or frontier_id),
                evidence_notes=list(rec.get("evidence_notes", []) or []),
                domain_learned_summary=learned_summary,
                top_verified_titles=top_titles,
                truth_status=str(rec.get("truth_status", "weak") or "weak"),
            )
            rec["llm_synthesis"] = synthesis

        # 3 b. Experiment proposal evaluation (skip running/complete)
        evaluated = 0
        for rec in experiment_directives:
            if evaluated >= 4:
                break
            status = str(rec.get("status", "") or "")
            if status in {"running", "complete"}:
                continue
            frontier_id = str(rec.get("frontier_problem_id", "") or "")
            frontier_rec = next(
                (f for f in frontier_directives if f.get("problem_id") == frontier_id), {}
            )
            inferred_domain = self._infer_domain_id(
                str(rec.get("frontier_problem_title", "") or rec.get("title", "") or "")
            )
            learned_summary, _ = _dom_summary(inferred_domain)
            evaluation = evaluate_experiment_proposal(
                self.workspace,
                str(rec.get("experiment_id", "") or ""),
                frontier_title=str(rec.get("frontier_problem_title", "") or ""),
                experiment_goal=str(rec.get("experiment_goal", "") or ""),
                mechanism_focus=str(rec.get("mechanism_focus", "") or ""),
                dataset=str(rec.get("dataset", "") or ""),
                method=str(rec.get("method", "") or ""),
                status=status,
                evidence_notes=list(frontier_rec.get("evidence_notes", []) or []),
            )
            rec["llm_evaluation"] = evaluation
            evaluated += 1

        # 3 c. Claim verification for evidence directives
        for rec in evidence_directives[:3]:
            frontier_id = str(rec.get("frontier_problem_id", "") or "")
            claim_text = str(rec.get("claim_under_test", "") or "")
            if not frontier_id or not claim_text:
                continue
            frontier_rec = next(
                (f for f in frontier_directives if f.get("problem_id") == frontier_id), {}
            )
            inferred_domain = self._infer_domain_id(
                str(frontier_rec.get("domain", "") or frontier_rec.get("title", "") or "")
            )
            learned_summary, _ = _dom_summary(inferred_domain)
            verification = verify_result_claim(
                self.workspace,
                frontier_id,
                claim_text,
                evidence_notes=list(frontier_rec.get("evidence_notes", []) or []),
                truth_status=str(rec.get("truth_status", "weak") or "weak"),
                domain_learned_summary=learned_summary,
            )
            rec["llm_claim_check"] = verification


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Update TAR research-director state.")
    parser.add_argument(
        "--workspace",
        default=str(resolve_workspace(_REPO)),
        help="Workspace path containing tar_state/",
    )
    args = parser.parse_args()
    workspace = ensure_workspace_layout(Path(args.workspace), repo_root=_REPO)
    state = ResearchDirector(workspace).update_state()
    summary = state.get("summary", {})
    print(
        f"[ResearchDirector] frontier={summary.get('frontier_count', 0)} "
        f"papers={summary.get('paper_count', 0)} domains={summary.get('domain_count', 0)} "
        f"top_frontier={summary.get('top_frontier_problem_id', '')}"
    )


if __name__ == "__main__":
    main()
