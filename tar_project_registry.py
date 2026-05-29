"""
TAR Project Registry
======================
Single source of truth for every research project TAR produces.
Prevents confusion between whitepapers, research tracks, and experiments.

Each project has a unique slug, human-readable name, field classification,
status, and pointers to its paper and data artefacts.

Storage: {workspace}/tar_state/project_registry.json
Index:   {workspace}/tar_state/project_index.json  (lightweight, quick queries)
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from tar_lab.human_review import approved_paper_ids
from tar_lab.validation import validate_paper_evidence
from tar_storage import ensure_workspace_layout, resolve_workspace

# ── taxonomy ──────────────────────────────────────────────────────────────────
# Maps (field, subfield) -> keyword triggers used for auto-classification.
FIELD_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "continual_learning": {
        "forgetting_mitigation": [
            "forgetting", "catastrophic", "anchor", "ewc", "penalty", "stability",
            "consolidation", "regulariz", "synaptic",
        ],
        "mechanism_design": [
            "mechanism", "deep_anchor", "graduated", "carryover", "thermal",
            "observer", "hypothesis",
        ],
        "evaluation_methodology": [
            "publishability", "jaf", "metric", "benchmark", "assessment",
            "collapse", "si_collapse",
        ],
        "class_incremental": [
            "class_incremental", "class-incremental", "task_free", "boundary",
        ],
        "scale_up": [
            "cifar100", "cifar-100", "tinyimagenet", "tiny-imagenet", "scale",
            "multi_dataset",
        ],
        "hyperparameter_study": [
            "sweep", "lambda", "ablation", "sensitivity", "grid",
        ],
    },
    "thermodynamics_ml": {
        "phase_transitions": [
            "phase_transition", "criticality", "ordered", "disordered",
            "regime", "thermal", "rho", "sigma_star",
        ],
        "entropy_dynamics": [
            "entropy", "activation_entropy", "sigma", "thermodynamic",
            "temperature", "heat",
        ],
    },
    "optimization": {
        "lr_scheduling": ["lr_scale", "governor", "learning_rate", "schedule"],
        "regularization": ["regularization", "weight_decay", "l2"],
    },
}

# ── status codes ──────────────────────────────────────────────────────────────
STATUS_PENDING  = "pending"
STATUS_PLANNED  = "planned"
STATUS_RUNNING  = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED   = "failed"
STATUS_ARCHIVED = "archived"

# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class ResearchProject:
    id: str                          # unique slug, e.g. "tcl-deep-anchor-v1"
    name: str                        # "Deep Anchor: Stable Sigma-Star for TCL"
    field: str                       # top-level field from FIELD_TAXONOMY
    subfield: str                    # subfield within that field
    keywords: list[str]              # free-text search tags
    status: str                      # pending | running | complete | failed | archived
    phase_source: str                # e.g. "phase10", "autonomous_research", "phase16"
    dataset: str                     # split_cifar10 | split_cifar100 | split_tinyimagenet
    abstract: str                    # 1-3 sentence summary
    paper_dir: str                   # absolute path to LaTeX output dir
    paper_pdf: str                   # absolute path to compiled PDF (empty if not yet compiled)
    data_paths: list[str]            # result JSON files
    authors: list[str]               # author names
    affiliation: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""
    project_type: str = "paper"
    readiness: str = ""
    paper_status: str = ""
    director_priority_score: float = 0.0
    director_truth_status: str = "weak"
    director_recommendation: str = ""
    frontier_problem_ids: list[str] = field(default_factory=list)
    linked_experiment_ids: list[str] = field(default_factory=list)
    waiting_for_experiment_ids: list[str] = field(default_factory=list)
    active_domain_id: str = ""
    active_path_id: str = ""
    scope_status: str = ""
    director_focus: str = ""


# ── classifier ────────────────────────────────────────────────────────────────
def classify_research(
    title: str = "",
    hypothesis_name: str = "",
    phase_source: str = "",
    evidence_keys: list[str] | None = None,
    extra_text: str = "",
) -> tuple[str, str, list[str]]:
    """
    Return (field, subfield, keywords) by matching text against the taxonomy.
    Falls back to 'continual_learning' / 'forgetting_mitigation' if no clear match.
    """
    corpus = " ".join([
        title.lower(),
        hypothesis_name.lower().replace("_", " "),
        phase_source.lower(),
        " ".join(evidence_keys or []).lower(),
        extra_text.lower(),
    ])

    scores: dict[tuple[str, str], int] = {}
    matched_keywords: list[str] = []

    for top_field, subfields in FIELD_TAXONOMY.items():
        for subfield, triggers in subfields.items():
            score = 0
            for trigger in triggers:
                if trigger in corpus:
                    score += 1
                    if trigger not in matched_keywords:
                        matched_keywords.append(trigger)
            if score > 0:
                scores[(top_field, subfield)] = score

    if scores:
        best = max(scores, key=lambda k: scores[k])
        top_field, subfield = best
    else:
        top_field, subfield = "continual_learning", "forgetting_mitigation"

    # Add explicit keywords from hypothesis / title
    for word in re.findall(r"[a-z][a-z_]+", corpus):
        if len(word) > 4 and word not in matched_keywords:
            matched_keywords.append(word)

    return top_field, subfield, matched_keywords[:20]


def generate_project_id(
    hypothesis_name: str = "",
    phase_source: str = "",
    dataset: str = "split_cifar10",
) -> str:
    """
    Generate a stable, human-readable project slug.
    E.g.: 'tcl-deep-anchor-cifar10-v1'
    """
    parts = ["tcl"]
    if hypothesis_name:
        parts.append(hypothesis_name.lower().replace("_", "-"))
    elif phase_source:
        parts.append(phase_source.lower().replace("_", "-"))
    ds_short = {
        "split_cifar10":      "cifar10",
        "split_cifar100":     "cifar100",
        "split_tinyimagenet": "tinyimagenet",
    }.get(dataset, dataset.replace("_", "-")[:12])
    parts.append(ds_short)
    base_slug = "-".join(p for p in parts if p)
    return f"{base_slug}-v1"


def generate_project_name(
    hypothesis_name: str = "",
    phase_source: str = "",
    dataset: str = "split_cifar10",
    title: str = "",
) -> str:
    """Convert internal identifiers to a clean human-readable project name."""
    if title:
        return title

    _hyp_names = {
        "deep_anchor":             "Deep Anchor: Extended Sigma-Star Calibration",
        "graduated_penalty":       "Graduated Penalty: Continuous Regime-Based Anchoring",
        "strict_consolidation":    "Strict Consolidation: Tighter Thermal Regime Thresholds",
        "thermal_carryover":       "Thermal Carry-Over: Inter-Task Sigma-Star Persistence",
        "high_penalty_conservative": "High-Penalty Conservative: Aggressive TCL Anchoring",
    }
    _phase_names = {
        "phase10": "TCL vs. EWC/SI/SGD — Four-Way Baseline Comparison",
        "phase11": "TCL Ablation — Governor and Penalty Component Analysis",
        "phase12": "EWC Lambda Sensitivity — Hyperparameter Robustness",
        "phase13": "SI Collapse Robustness — Hyperparameter Sweep",
        "phase14": "TCL Publishability and Contribution Positioning",
        "phase15": "Class-Incremental Mechanism Search",
        "phase16": "TCL Scale-Up — Split-CIFAR-100 Multi-Dataset Validation",
        "phase17": "TCL Scale-Up — Split-TinyImageNet Multi-Dataset Validation",
    }

    _ds_names = {
        "split_cifar10":      "Split-CIFAR-10",
        "split_cifar100":     "Split-CIFAR-100",
        "split_tinyimagenet": "Split-TinyImageNet",
    }
    ds_label = _ds_names.get(dataset, dataset)

    if hypothesis_name in _hyp_names:
        return f"{_hyp_names[hypothesis_name]} ({ds_label})"
    if phase_source in _phase_names:
        return _phase_names[phase_source]
    raw = hypothesis_name or phase_source or "Unknown"
    return f"TAR Research: {raw.replace('_', ' ').title()} ({ds_label})"


# ── registry ──────────────────────────────────────────────────────────────────
class ProjectRegistry:
    """JSON-backed registry of all TAR research projects."""

    def __init__(self, workspace: Path):
        self.workspace  = workspace
        self._path      = workspace / "tar_state" / "project_registry.json"
        self._index_path = workspace / "tar_state" / "project_index.json"
        self._projects: dict[str, ResearchProject] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                for pid, rec in raw.items():
                    clean = {
                        k: v for k, v in rec.items()
                        if k in ResearchProject.__dataclass_fields__
                    }
                    self._projects[pid] = ResearchProject(**clean)
            except Exception:
                pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {pid: asdict(p) for pid, p in self._projects.items()}
        try:
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            return
        # Write lightweight index for quick lookups
        index = [
            {
                "id":          p.id,
                "name":        p.name,
                "field":       p.field,
                "subfield":    p.subfield,
                "status":      p.status,
                "phase_source": p.phase_source,
                "dataset":     p.dataset,
                "has_pdf":     bool(p.paper_pdf),
                "readiness":   p.readiness,
                "paper_status": p.paper_status,
                "director_truth_status": p.director_truth_status,
                "created_at":  p.created_at,
            }
            for p in sorted(self._projects.values(), key=lambda x: x.created_at)
        ]
        try:
            self._index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        except OSError:
            pass

    def register(
        self,
        project: ResearchProject,
        *,
        overwrite_existing: bool = False,
    ) -> ResearchProject:
        """Register or update a project."""
        existing = self._projects.get(project.id)
        if existing:
            project.created_at = existing.created_at
            if not overwrite_existing:
                for field_name in ResearchProject.__dataclass_fields__:
                    if field_name in {"created_at", "updated_at"}:
                        continue
                    incoming = getattr(project, field_name)
                    prior = getattr(existing, field_name)
                    if incoming in ("", [], 0.0) and prior not in ("", [], 0.0):
                        setattr(project, field_name, prior)
        project.updated_at = datetime.now(timezone.utc).isoformat()
        self._projects[project.id] = project
        self._save()
        print(f"[Registry] Registered: {project.id} — {project.name}", flush=True)
        return project

    def delete(self, project_id: str) -> None:
        if project_id in self._projects:
            del self._projects[project_id]
            self._save()

    def reset_director_paper_projects(self) -> None:
        reset_ids = [
            project.id
            for project in self._projects.values()
            if project.project_type == "paper"
            and project.phase_source == "research_director"
        ]
        for project_id in reset_ids:
            self._projects.pop(project_id, None)
        if reset_ids:
            self._save()

    def prune_invalid_director_papers(self) -> None:
        invalid_ids = [
            project.id
            for project in self._projects.values()
            if _is_invalid_director_paper_project(project)
        ]
        for project_id in invalid_ids:
            self._projects.pop(project_id, None)
        if invalid_ids:
            self._save()

    def update_status(self, project_id: str, status: str, notes: str = "") -> None:
        p = self._projects.get(project_id)
        if p:
            p.status     = status
            p.notes      = notes or p.notes
            p.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()

    def update_pdf(self, project_id: str, pdf_path: str) -> None:
        p = self._projects.get(project_id)
        if p:
            p.paper_pdf  = pdf_path
            p.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()

    def get(self, project_id: str) -> ResearchProject | None:
        return self._projects.get(project_id)

    def list_all(self) -> list[ResearchProject]:
        return sorted(self._projects.values(), key=lambda p: p.created_at)

    def list_by_field(self, field: str) -> list[ResearchProject]:
        return [p for p in self._projects.values() if p.field == field]

    def list_by_status(self, status: str) -> list[ResearchProject]:
        return [p for p in self._projects.values() if p.status == status]

    def print_summary(self) -> None:
        """Print a formatted summary of all projects to stdout."""
        projects = self.list_all()
        print(f"\n{'='*72}")
        print(f"  TAR PROJECT REGISTRY  ({len(projects)} projects)")
        print(f"{'='*72}")
        if not projects:
            print("  (empty)")
        for p in projects:
            pdf_mark = "[PDF]" if p.paper_pdf else "     "
            print(f"  {pdf_mark} {p.id:<36s} {p.status:<10s} {p.field}/{p.subfield}")
            print(f"         {p.name}")
        print(f"{'='*72}\n")


# ── convenience: build from a phase JSON result ────────────────────────────────
def register_phase_result(
    workspace: Path,
    phase_source: str,
    data: dict,
    paper_dir: str = "",
    paper_pdf: str = "",
    authors: list[str] | None = None,
    affiliation: str = "Independent Research",
    extra_notes: str = "",
) -> ResearchProject:
    """
    Build and register a project entry from a phase result dict.
    Called automatically by tar_author.py after paper generation.
    """
    dataset   = data.get("dataset", "split_cifar10")
    verdict   = data.get("verdict", "")
    hyp_name  = data.get("hypothesis_name", "")

    field, subfield, keywords = classify_research(
        title=paper_dir,
        hypothesis_name=hyp_name,
        phase_source=phase_source,
        evidence_keys=list(data.keys()),
        extra_text=verdict,
    )

    project_id = generate_project_id(
        hypothesis_name=hyp_name,
        phase_source=phase_source,
        dataset=dataset,
    )
    name = generate_project_name(
        hypothesis_name=hyp_name,
        phase_source=phase_source,
        dataset=dataset,
    )

    abstract = verdict[:300] if verdict else f"TAR research project: {phase_source}"

    registry = ProjectRegistry(workspace)
    project = ResearchProject(
        id=project_id,
        name=name,
        field=field,
        subfield=subfield,
        keywords=keywords,
        status=STATUS_COMPLETE if verdict and "ERROR" not in verdict else STATUS_FAILED,
        phase_source=phase_source,
        dataset=dataset,
        abstract=abstract,
        paper_dir=paper_dir,
        paper_pdf=paper_pdf,
        data_paths=[],
        authors=authors or ["Christopher Gardner"],
        affiliation=affiliation,
        notes=extra_notes,
    )
    return registry.register(project)


def _is_supported_director_paper_directive(directive: dict[str, Any]) -> bool:
    if not isinstance(directive, dict):
        return False
    paper_id = str(directive.get("paper_id", "") or "")
    frontier_id = str(directive.get("frontier_problem_id", "") or "")
    scope_status = str(directive.get("scope_status", "") or "").strip().lower()
    if not paper_id:
        return False
    if not frontier_id and (
        scope_status in {"domain_watch", "out_of_scope", "incubating", "unscoped"}
        or paper_id.startswith("director-")
    ):
        return False
    return True


def _is_invalid_director_paper_project(project: ResearchProject) -> bool:
    if project.project_type != "paper" or project.phase_source != "research_director":
        return False
    if project.id.startswith("director-"):
        return True
    return (
        not project.frontier_problem_ids
        and project.scope_status in {"domain_watch", "out_of_scope", "incubating", "unscoped"}
    )


def sync_director_paper_projects(
    workspace: Path,
    director_state: dict[str, Any],
    author_state: dict[str, Any] | None = None,
) -> list[ResearchProject]:
    """
    Materialize Research Director paper directives into the registry immediately,
    so planned or in-progress papers appear before a final PDF exists.
    """
    registry = ProjectRegistry(workspace)
    registry.prune_invalid_director_papers()
    raw_paper_directives = director_state.get("paper_directives", []) if isinstance(director_state, dict) else []
    paper_directives = [
        directive
        for directive in raw_paper_directives
        if _is_supported_director_paper_directive(directive)
    ]
    frontier_directives = {
        str(rec.get("problem_id", "") or ""): rec
        for rec in director_state.get("frontier_directives", [])
        if isinstance(director_state, dict) and rec.get("problem_id")
    }
    current_paper = (author_state or {}).get("current_paper", {}) if isinstance(author_state, dict) else {}
    active_paper_id = str(current_paper.get("project_id", "") or "")
    active_paper_status = str(current_paper.get("status", "") or "")
    human_approved_papers = approved_paper_ids(workspace)

    queue_path = workspace / "tar_state" / "experiment_queue.json"
    queue_data: dict[str, Any] = {}
    if queue_path.exists():
        try:
            queue_data = json.loads(queue_path.read_text(encoding="utf-8"))
        except Exception:
            queue_data = {}
    queue_experiments = queue_data.get("experiments", []) if isinstance(queue_data, dict) else []
    exp_by_id = {
        str(exp.get("id", "") or ""): exp
        for exp in queue_experiments
        if exp.get("id")
    }
    archive_path = workspace / "tar_state" / "experiment_archive.json"
    archive_data: dict[str, Any] = {}
    if archive_path.exists():
        try:
            archive_data = json.loads(archive_path.read_text(encoding="utf-8"))
        except Exception:
            archive_data = {}
    archive_by_id = {
        str(exp.get("id", "") or ""): exp
        for exp in (archive_data.get("experiments", []) if isinstance(archive_data, dict) else [])
        if isinstance(exp, dict) and exp.get("id")
    }
    live_paper_ids = {
        str(rec.get("paper_id", "") or "")
        for rec in paper_directives
        if str(rec.get("paper_id", "") or "")
    }
    stale_director_ids = [
        project.id
        for project in registry.list_all()
        if project.phase_source == "research_director"
        and project.project_type == "paper"
        and project.id not in live_paper_ids
    ]
    for project_id in stale_director_ids:
        registry.delete(project_id)

    def _experiment_complete(exp_id: str) -> bool:
        for rec in (exp_by_id.get(exp_id, {}), archive_by_id.get(exp_id, {})):
            if not isinstance(rec, dict):
                continue
            status = str(rec.get("status", "") or "").lower()
            stage = str(rec.get("stage", "") or "").lower()
            if status in {"complete", "skipped"} or stage in {"complete", "skipped"}:
                return True
            result_path = str(rec.get("result_path", "") or "")
            if result_path:
                result_file = Path(result_path)
                if result_file.exists():
                    try:
                        raw = json.loads(result_file.read_text(encoding="utf-8"))
                    except Exception:
                        raw = {}
                    verdict = str(raw.get("verdict", "") or "").upper()
                    status_hint = str(raw.get("status", "") or "").upper()
                    if raw and verdict != "ERROR" and status_hint != "ERROR":
                        return True
        direct_path = workspace / "tar_state" / "experiments" / exp_id / "result.json"
        if not direct_path.exists():
            return False
        try:
            raw = json.loads(direct_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        verdict = str(raw.get("verdict", "") or "").upper()
        status_hint = str(raw.get("status", "") or "").upper()
        return bool(raw) and verdict != "ERROR" and status_hint != "ERROR"

    linked: list[ResearchProject] = []
    for directive in paper_directives:
        paper_id = str(directive.get("paper_id", "") or "")
        if not paper_id:
            continue

        frontier_id = str(directive.get("frontier_problem_id", "") or "")
        frontier = frontier_directives.get(frontier_id, {})
        linked_experiment_ids = [
            str(exp_id) for exp_id in directive.get("linked_experiment_ids", [])
            if str(exp_id or "")
        ]
        waiting_for = [
            str(exp_id) for exp_id in directive.get("waiting_for_experiments", [])
            if str(exp_id or "") and not _experiment_complete(str(exp_id or ""))
        ]
        datasets: list[str] = []
        data_paths: list[str] = []
        for exp_id in linked_experiment_ids:
            exp = exp_by_id.get(exp_id, {})
            dataset_value = str(exp.get("dataset", "") or "").strip()
            if dataset_value and dataset_value not in datasets:
                datasets.append(dataset_value)
            result_path = str(exp.get("result_path", "") or "")
            if result_path and result_path not in data_paths:
                data_paths.append(result_path)
        if not datasets:
            candidate_datasets = [
                str(item).strip()
                for item in directive.get("candidate_datasets", []) or frontier.get("candidate_datasets", []) or []
                if str(item).strip()
            ]
            datasets = sorted(dict.fromkeys(candidate_datasets))
        if len(datasets) == 1:
            dataset = datasets[0]
        elif len(datasets) > 1:
            dataset = "multi_dataset"
        else:
            dataset = "validated_result_pending"

        title = str(directive.get("title", "") or paper_id.replace("_", " ").replace("-", " ").title())
        field, subfield, keywords = classify_research(
            title=title,
            phase_source="research_director",
            evidence_keys=[frontier_id, dataset],
            extra_text=" ".join([
                str(frontier.get("title", "") or ""),
                str(frontier.get("domain", "") or ""),
                str(directive.get("recommendation", "") or ""),
            ]),
        )

        paper_dir = workspace / "paper" / paper_id
        tex_path = paper_dir / "main.tex"
        pdf_path = paper_dir / "main.pdf"
        readiness = str(directive.get("readiness", "") or "planned")
        evidence_status = validate_paper_evidence(
            workspace,
            paper_id=paper_id,
            linked_experiment_ids=linked_experiment_ids,
            waiting_for_experiment_ids=waiting_for,
        )
        evidence_ready = bool(evidence_status.get("evidence_ready"))
        human_approved = paper_id in human_approved_papers
        if waiting_for:
            status = STATUS_PLANNED
            paper_status = "blocked"
        elif not evidence_ready:
            status = STATUS_PLANNED
            paper_status = "awaiting_validation"
        elif paper_id == active_paper_id and active_paper_status not in {"", "idle"}:
            status = STATUS_RUNNING
            paper_status = active_paper_status
        elif pdf_path.exists():
            status = STATUS_COMPLETE
            paper_status = "draft_compiled"
        elif human_approved:
            status = STATUS_PLANNED
            paper_status = "approved_for_rewrite"
        else:
            status = STATUS_PLANNED
            paper_status = "awaiting_human_review"

        project = ResearchProject(
            id=paper_id,
            name=title,
            field=field,
            subfield=subfield,
            keywords=keywords,
            status=status,
            phase_source="research_director",
            dataset=dataset,
            abstract=str(directive.get("recommendation", "") or directive.get("why_now", "") or title)[:300],
            paper_dir=str(paper_dir),
            paper_pdf=str(pdf_path) if pdf_path.exists() else "",
            data_paths=data_paths,
            authors=["Christopher Gardner", "TAR (Thermodynamic Autonomous Researcher)"],
            affiliation="Independent Research",
            notes=str(directive.get("why_now", "") or ""),
            project_type="paper",
            readiness=readiness,
            paper_status=paper_status,
            director_priority_score=float(directive.get("priority_score", 0.0) or 0.0),
            director_truth_status=str(directive.get("truth_status", "weak") or "weak"),
            director_recommendation=str(directive.get("recommendation", "") or ""),
            frontier_problem_ids=[frontier_id] if frontier_id else [],
            linked_experiment_ids=linked_experiment_ids,
            waiting_for_experiment_ids=waiting_for,
            active_domain_id=str(directive.get("active_domain_id", "") or ""),
            active_path_id=str(directive.get("active_path_id", "") or ""),
            scope_status=str(directive.get("scope_status", "") or ""),
            director_focus=str(directive.get("director_focus", "") or ""),
        )
        project.notes = "\n".join(
            bit for bit in [
                project.notes,
                f"evidence_ready={evidence_ready}",
                f"human_approved={human_approved}",
                f"validation_issues={evidence_status.get('issues', [])}",
            ]
            if str(bit).strip()
        )
        linked.append(registry.register(project, overwrite_existing=True))
        if frontier_id:
            try:
                from tar_frontier import FrontierRegistry

                FrontierRegistry(workspace).link_paper(frontier_id, paper_id)
            except Exception:
                pass
    registry.prune_invalid_director_papers()

    return linked


def rebuild_registry_from_director_state(
    workspace: Path,
    director_state: dict[str, Any],
    author_state: dict[str, Any] | None = None,
) -> list[ResearchProject]:
    """
    Rebuild the director-managed paper slice of the registry from current truth.

    This intentionally replaces stale Research Director paper entries rather than
    merge-forwarding historical fields, so registry state follows the validated
    evidence and active directives after reruns or audit corrections.
    """
    registry = ProjectRegistry(workspace)
    registry.reset_director_paper_projects()
    return sync_director_paper_projects(
        workspace=workspace,
        director_state=director_state,
        author_state=author_state,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    _REPO = Path(__file__).resolve().parent
    ws = ensure_workspace_layout(resolve_workspace(_REPO), repo_root=_REPO)
    reg = ProjectRegistry(ws)
    reg.print_summary()
