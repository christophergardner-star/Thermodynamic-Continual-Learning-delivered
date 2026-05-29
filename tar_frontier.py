"""
TAR Frontier Problem Registry
==============================
Tracks the real-world research problems TAR is actively investigating.
Each problem can have multiple experiments and papers linked to it.

TAR can work on more than one frontier problem simultaneously. The registry
provides the human-language context shown in the dashboard modal and used
by the scheduler to group related work.

Storage: {workspace}/tar_state/frontier_problems.json
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tar_storage import ensure_workspace_layout, resolve_workspace

# ── status codes ──────────────────────────────────────────────────────────────
FP_EXPLORING  = "exploring"   # TAR is investigating the problem space
FP_ACTIVE     = "active"      # experiments actively running
FP_PUBLISHING = "publishing"  # results found, paper being written
FP_COMPLETE   = "complete"    # paper published / archived

# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class FrontierProblem:
    id: str
    title: str
    domain: str                          # e.g. "continual_learning"
    description: str                     # 2-3 sentence problem statement
    why_important: str                   # why solving this matters
    tcl_approach: str                    # how TCL/thermodynamic framing is relevant
    industry_problem_title: str = ""
    global_problem_statement: str = ""
    industry_contexts: list[str] = field(default_factory=list)
    well_known_problem: bool = True
    solution_family: str = "TAR/TCL/ASC"
    solution_novelty_note: str = (
        "TAR, TCL, and ASC are the user's unpublished internal methods under evaluation. "
        "They must be treated as novel work from this project, not as established literature "
        "or pre-validated solutions."
    )
    target_venues: list[str] = field(default_factory=list)
    candidate_datasets: list[str] = field(default_factory=list)
    candidate_backbones: list[str] = field(default_factory=list)
    external_baselines: list[str] = field(default_factory=list)
    research_guidance: str = ""
    status: str = FP_ACTIVE
    experiments_linked: list[str] = field(default_factory=list)
    papers_linked: list[str] = field(default_factory=list)
    breakthroughs_found: int = 0
    adverse_count: int = 0       # experiments that returned ADVERSE verdict
    null_count: int = 0          # experiments that returned NULL verdict
    truth_status: str = "weak"   # weak | provisional | supported | validated | falsified
    priority: int = 50
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""


# ── pre-seeded problems ───────────────────────────────────────────────────────
_DEFAULT_PROBLEMS: list[dict] = [
    {
        "id":          "fp-catastrophic-forgetting",
        "title":       "Reliable Long-Horizon Learning Without Catastrophic Forgetting",
        "domain":      "continual_learning",
        "description": (
            "Production ML systems must absorb new data over long horizons without "
            "erasing previously learned capability. Catastrophic forgetting remains "
            "a major barrier to deploying lifelong learning systems in practice."
        ),
        "why_important": (
            "This is a well-known external problem across robotics, healthcare, "
            "autonomous systems, industrial inspection, and adaptive enterprise ML. "
            "If it is not solved, real systems keep retraining from scratch instead "
            "of safely accumulating knowledge."
        ),
        "tcl_approach": (
            "TCL uses activation entropy (sigma) relative to a calibrated critical "
            "point (sigma-star) to detect whether a layer is in an ordered, critical, "
            "or disordered thermodynamic regime. A D_PR-weighted L2 anchor selectively "
            "stabilises ordered layers, providing plasticity-stability control without "
            "explicit task labels."
        ),
        "industry_problem_title": "Reliable Long-Horizon Learning Without Catastrophic Forgetting",
        "global_problem_statement": (
            "Widely deployed ML systems need to keep learning from new tasks, classes, "
            "or environments without destroying previously validated performance."
        ),
        "industry_contexts": ["robotics", "autonomous systems", "medical ai", "industrial inspection", "finance"],
        "target_venues": ["NeurIPS", "ICML", "ICLR"],
        "candidate_datasets": ["split_cifar10", "split_cifar100", "split_tinyimagenet", "permuted_mnist"],
        "candidate_backbones": ["resnet18", "vit_tiny", "mlp"],
        "external_baselines": ["ewc", "si", "sgd_baseline", "experience_replay", "agem", "lwf"],
        "research_guidance": (
            "Treat TAR/TCL/ASC as unpublished internal candidate methods. Choose the smallest dataset "
            "and backbone that can genuinely falsify the forgetting claim, and always compare against "
            "real external continual-learning baselines."
        ),
        "status":   FP_ACTIVE,
        "priority": 10,
    },
    {
        "id":          "fp-regime-detection-accuracy",
        "title":       "Reliable Change-State Detection in Non-Stationary ML Systems",
        "domain":      "thermodynamics_ml",
        "description": (
            "Adaptive ML systems need reliable internal state detection so they can "
            "tell when to protect prior knowledge and when to adapt. Weak state-change "
            "detection causes instability, false adaptation, and brittle deployment behavior."
        ),
        "why_important": (
            "This is a real-world control problem for non-stationary ML in MLOps, "
            "autonomous platforms, recommender systems, and industrial AI. Reliable "
            "change-state detection determines whether continuous adaptation is safe."
        ),
        "tcl_approach": (
            "Investigating sigma-star calibration window length, exponential smoothing "
            "of sigma estimates, and cross-task sigma-star carry-over as mechanisms "
            "to improve regime boundary accuracy."
        ),
        "industry_problem_title": "Reliable Change-State Detection in Non-Stationary ML Systems",
        "global_problem_statement": (
            "Real adaptive systems need trustworthy signals for stability, drift, and "
            "adaptation state so they can react correctly under distribution shift."
        ),
        "industry_contexts": ["mlops", "autonomous systems", "recommenders", "industrial control"],
        "target_venues": ["NeurIPS", "ICML", "AISTATS"],
        "candidate_datasets": ["split_cifar10", "split_cifar100", "permuted_mnist"],
        "candidate_backbones": ["resnet18", "mlp"],
        "external_baselines": ["ewc", "si", "sgd_baseline"],
        "research_guidance": (
            "Do not assume the user's internal methods solve this control problem. Use quick diagnostic "
            "experiments plus external baselines to check whether the regime signal is actually reliable."
        ),
        "status":   FP_ACTIVE,
        "priority": 20,
    },
    {
        "id":          "fp-class-incremental",
        "title":       "Continuous Class Expansion Without Task Labels",
        "domain":      "continual_learning",
        "description": (
            "Real deployed systems often see new categories appear without clean task "
            "boundaries. Models must expand class knowledge continuously without being "
            "handed a task identity at inference time."
        ),
        "why_important": (
            "This is a globally recognized deployment problem in vision platforms, "
            "medical imaging, security, and retail catalogs. Solving it is much closer "
            "to the real operating condition of production ML than task-incremental evaluation."
        ),
        "tcl_approach": (
            "Phase 15 found tcl_stability_bias beats EWC on forgetting (p=0.012, "
            "d=5.26) in class-incremental setting. The thermal regime signal is "
            "task-label-free by design, making TCL a natural fit."
        ),
        "industry_problem_title": "Continuous Class Expansion Without Task Labels",
        "global_problem_statement": (
            "Operational ML systems need to recognize newly emerging categories without "
            "relying on explicit task IDs or manual task segmentation."
        ),
        "industry_contexts": ["vision platforms", "medical imaging", "security", "retail catalogs"],
        "target_venues": ["CVPR", "ICCV", "ECCV", "NeurIPS"],
        "candidate_datasets": ["split_cifar10", "split_cifar100", "split_tinyimagenet"],
        "candidate_backbones": ["resnet18", "vit_tiny", "clip_vit_b32"],
        "external_baselines": ["icarl", "l2p", "dualprompt", "ewc", "sgd_baseline"],
        "research_guidance": (
            "Keep this anchored to the real class-incremental deployment problem. Compare TAR/TCL/ASC "
            "against established class-incremental baselines and select datasets/backbones that match "
            "the question rather than defaulting to one house benchmark."
        ),
        "status":     FP_PUBLISHING,
        "priority":   15,
        "breakthroughs_found": 1,
    },
    {
        "id":          "fp-scale-up",
        "title":       "Scalable Continual Adaptation on Realistic Visual Streams",
        "domain":      "continual_learning",
        "description": (
            "A continual-learning method is only practically useful if it remains stable "
            "as datasets, class counts, and visual difficulty increase. Small-benchmark wins "
            "do not prove real deployment scalability."
        ),
        "why_important": (
            "This is a real-world adoption problem for robotics, surveillance, edge vision, "
            "and factory inspection. Industry needs methods that hold up beyond toy continual-learning benchmarks."
        ),
        "tcl_approach": (
            "Phase 16 (CIFAR-100) and Phase 17 (TinyImageNet) run identical TCL "
            "with ResNet-18. The regime detector is dataset-agnostic; only the "
            "number of tasks and classes changes."
        ),
        "industry_problem_title": "Scalable Continual Adaptation on Realistic Visual Streams",
        "global_problem_statement": (
            "Continual adaptation systems must keep working as tasks become visually richer, "
            "class spaces expand, and streams become closer to production-scale difficulty."
        ),
        "industry_contexts": ["robotics", "surveillance", "edge vision", "factory inspection"],
        "target_venues": ["CVPR", "ICCV", "NeurIPS"],
        "candidate_datasets": ["split_cifar100", "split_tinyimagenet", "imagenet_subset"],
        "candidate_backbones": ["resnet18", "resnet34", "vit_tiny"],
        "external_baselines": ["ewc", "sgd_baseline", "experience_replay", "lwf"],
        "research_guidance": (
            "Use harder datasets and stronger visual backbones whenever they are the right test for the "
            "scaling question. TAR/TCL/ASC should be one candidate family in a scale-up comparison, not "
            "the assumed answer."
        ),
        "status":   FP_ACTIVE,
        "priority": 25,
    },
    {
        "id":          "fp-hyperparameter-robustness",
        "title":       "Low-Tuning Continual Learning for Production ML",
        "domain":      "continual_learning",
        "description": (
            "Production continual-learning systems should not require expensive manual retuning "
            "for every dataset or environment. Robustness to optimizer and control settings is a "
            "major real-world adoption constraint."
        ),
        "why_important": (
            "This is a well-known MLOps and platform problem: if a method is too sensitive "
            "to settings, it is hard to deploy, monitor, and trust at scale."
        ),
        "tcl_approach": (
            "Phases 12-13 already swept EWC lambda and SI xi. The autonomous research "
            "hypotheses (graduated_penalty, high_penalty_conservative) directly probe "
            "TCL's penalty sensitivity in the regime-detection context."
        ),
        "industry_problem_title": "Low-Tuning Continual Learning for Production ML",
        "global_problem_statement": (
            "Deployment-ready continual learning should preserve performance without "
            "requiring repeated expensive hyperparameter tuning across tasks and domains."
        ),
        "industry_contexts": ["mlops", "enterprise ai", "platform ml", "edge deployment"],
        "target_venues": ["ICML", "NeurIPS", "KDD"],
        "candidate_datasets": ["split_cifar10", "split_cifar100", "split_tinyimagenet"],
        "candidate_backbones": ["resnet18", "vit_tiny"],
        "external_baselines": ["ewc", "si", "sgd_baseline"],
        "research_guidance": (
            "Probe robustness with more than one dataset/backbone when possible, and evaluate whether "
            "TAR/TCL/ASC remains useful relative to established baselines rather than only against its "
            "own nearby variants."
        ),
        "status":   FP_ACTIVE,
        "priority": 30,
    },
]
_DEFAULT_PROBLEM_BY_ID: dict[str, dict[str, Any]] = {
    str(rec["id"]): rec for rec in _DEFAULT_PROBLEMS
}


# ── registry ──────────────────────────────────────────────────────────────────
class FrontierRegistry:
    """JSON-backed registry of frontier research problems."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self._path     = workspace / "tar_state" / "frontier_problems.json"
        self._problems: dict[str, FrontierProblem] = {}
        self._load()
        self._seed_defaults()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                for rec in raw.get("problems", []):
                    p = FrontierProblem(**{
                        k: v for k, v in rec.items()
                        if k in FrontierProblem.__dataclass_fields__
                    })
                    self._problems[p.id] = p
            except Exception:
                pass

    def _seed_defaults(self) -> None:
        changed = False
        legacy_titles = {
            "fp-catastrophic-forgetting": "Catastrophic Forgetting in Sequential Task Learning",
            "fp-regime-detection-accuracy": "Thermodynamic Regime Detection Accuracy",
            "fp-class-incremental": "Class-Incremental Learning Without Task Boundaries",
            "fp-scale-up": "TCL Scalability to Complex Datasets",
            "fp-hyperparameter-robustness": "TCL Hyperparameter Robustness",
        }
        for d in _DEFAULT_PROBLEMS:
            current = self._problems.get(d["id"])
            if current is None:
                self._problems[d["id"]] = FrontierProblem(**{
                    k: v for k, v in d.items()
                    if k in FrontierProblem.__dataclass_fields__
                })
                changed = True
                continue

            for key in (
                "industry_problem_title",
                "global_problem_statement",
                "industry_contexts",
                "solution_family",
                "solution_novelty_note",
                "target_venues",
                "candidate_datasets",
                "candidate_backbones",
                "external_baselines",
                "research_guidance",
            ):
                if not getattr(current, key, None):
                    setattr(current, key, d.get(key, getattr(current, key)))
                    changed = True
            if not current.description:
                current.description = d["description"]
                changed = True
            if not current.why_important:
                current.why_important = d["why_important"]
                changed = True
            if current.title == legacy_titles.get(d["id"], ""):
                current.title = d["title"]
                changed = True
            current.well_known_problem = True
        if changed:
            self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "problems": [asdict(p) for p in sorted(
                self._problems.values(), key=lambda x: x.priority)],
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def reset_to_real_world_defaults(self, *, preserve_links: bool = False) -> None:
        """
        Rebuild the frontier registry from the known real-world ML problem set.

        This intentionally drops speculative or stale frontier entries so the
        module remains a registry of externally grounded problems rather than
        an open-ended idea generator.
        """
        prior = self._problems
        rebuilt: dict[str, FrontierProblem] = {}
        for default in _DEFAULT_PROBLEMS:
            problem_id = str(default["id"])
            preserved = prior.get(problem_id)
            payload = {
                k: v for k, v in default.items()
                if k in FrontierProblem.__dataclass_fields__
            }
            if preserve_links and preserved is not None:
                payload["experiments_linked"] = preserved.experiments_linked[:]
                payload["papers_linked"] = preserved.papers_linked[:]
                payload["breakthroughs_found"] = int(preserved.breakthroughs_found)
                payload["adverse_count"]        = int(preserved.adverse_count)
                payload["null_count"]           = int(preserved.null_count)
                payload["truth_status"]         = str(preserved.truth_status or "weak")
                payload["status"] = str(preserved.status or payload.get("status", FP_ACTIVE))
                payload["created_at"] = preserved.created_at
            rebuilt[problem_id] = FrontierProblem(**payload)
        self._problems = rebuilt
        self._save()

    # ── mutations ─────────────────────────────────────────────────────────────
    def register(self, problem: FrontierProblem) -> FrontierProblem:
        if not problem.well_known_problem:
            raise ValueError(
                "Frontier problems must be real-world, externally grounded ML problems. "
                "Speculative or self-invented frontier entries are not allowed."
            )
        required_text = {
            "industry_problem_title": problem.industry_problem_title or problem.title,
            "global_problem_statement": problem.global_problem_statement,
            "why_important": problem.why_important,
            "research_guidance": problem.research_guidance,
        }
        missing_text = [key for key, value in required_text.items() if not str(value or "").strip()]
        if missing_text:
            raise ValueError(
                f"Frontier problem '{problem.id}' is missing required real-world grounding fields: {missing_text}"
            )
        if not problem.external_baselines or not problem.candidate_datasets or not problem.candidate_backbones:
            raise ValueError(
                f"Frontier problem '{problem.id}' must name external baselines, candidate datasets, and candidate backbones."
            )
        if problem.id not in _DEFAULT_PROBLEM_BY_ID and not str(problem.domain or "").strip():
            raise ValueError(
                f"Custom frontier problem '{problem.id}' must declare an explicit ML domain."
            )
        existing = self._problems.get(problem.id)
        if existing:
            problem.created_at = existing.created_at
        problem.updated_at = datetime.now(timezone.utc).isoformat()
        self._problems[problem.id] = problem
        self._save()
        return problem

    def link_experiment(self, problem_id: str, experiment_id: str) -> None:
        p = self._problems.get(problem_id)
        if p and experiment_id not in p.experiments_linked:
            p.experiments_linked.append(experiment_id)
            p.updated_at = datetime.now(timezone.utc).isoformat()
            if p.status == FP_EXPLORING:
                p.status = FP_ACTIVE
            self._save()

    def link_paper(self, problem_id: str, paper_id: str) -> None:
        p = self._problems.get(problem_id)
        if p and paper_id not in p.papers_linked:
            p.papers_linked.append(paper_id)
            p.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()

    def record_breakthrough(self, problem_id: str) -> None:
        p = self._problems.get(problem_id)
        if p:
            p.breakthroughs_found += 1
            p.status = FP_PUBLISHING
            p.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()

    def record_adverse(self, problem_id: str) -> None:
        p = self._problems.get(problem_id)
        if p:
            p.adverse_count += 1
            p.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()

    def record_null(self, problem_id: str) -> None:
        p = self._problems.get(problem_id)
        if p:
            p.null_count += 1
            p.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()

    def update_truth_status(self, problem_id: str, truth_status: str) -> None:
        p = self._problems.get(problem_id)
        if p and p.truth_status != truth_status:
            p.truth_status = truth_status
            p.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()

    def set_status(self, problem_id: str, status: str) -> None:
        p = self._problems.get(problem_id)
        if p:
            p.status = status
            p.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()

    # ── queries ───────────────────────────────────────────────────────────────
    def get(self, problem_id: str) -> FrontierProblem | None:
        return self._problems.get(problem_id)

    def list_active(self) -> list[FrontierProblem]:
        return sorted(
            [p for p in self._problems.values() if p.status != FP_COMPLETE],
            key=lambda x: x.priority,
        )

    def list_all(self) -> list[FrontierProblem]:
        return sorted(self._problems.values(), key=lambda x: x.priority)

    def for_experiment(self, experiment_id: str) -> FrontierProblem | None:
        for p in self._problems.values():
            if experiment_id in p.experiments_linked:
                return p
        return None

    # ── human summary ─────────────────────────────────────────────────────────
    @staticmethod
    def human_summary(p: FrontierProblem) -> str:
        problem_title = p.industry_problem_title or p.title
        lines = [
            f"Problem: {problem_title}",
            f"Domain: {p.domain.replace('_', ' ').title()}",
            "",
            p.description,
            "",
            f"Global problem: {p.global_problem_statement or p.description}",
            "",
            f"Why it matters: {p.why_important}",
            "",
            f"TAR's approach: {p.tcl_approach}",
            "",
            f"Internal work under evaluation: {p.solution_family}",
            p.solution_novelty_note,
        ]
        if p.industry_contexts:
            lines += ["", f"Industry contexts: {', '.join(p.industry_contexts)}"]
        if p.target_venues:
            lines += ["", f"Likely venues: {', '.join(p.target_venues)}"]
        if p.external_baselines:
            lines += ["", f"External baselines: {', '.join(p.external_baselines)}"]
        if p.candidate_datasets:
            lines += ["", f"Candidate datasets: {', '.join(p.candidate_datasets)}"]
        if p.candidate_backbones:
            lines += ["", f"Candidate backbones: {', '.join(p.candidate_backbones)}"]
        if p.research_guidance:
            lines += ["", f"Research guidance: {p.research_guidance}"]
        if p.breakthroughs_found:
            lines += ["", f"Breakthroughs found so far: {p.breakthroughs_found}"]
        return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ws = ensure_workspace_layout(resolve_workspace(Path(__file__).resolve().parent))
    reg = FrontierRegistry(ws)
    print(f"\nFrontier Problems ({len(reg.list_all())} total)\n{'='*60}")
    for p in reg.list_all():
        bk = f"  [{p.breakthroughs_found} BK]" if p.breakthroughs_found else ""
        print(f"  [{p.status.upper():10s}] P{p.priority:2d}  {p.id}{bk}")
        print(f"          {p.title}")
        print(f"          {len(p.experiments_linked)} experiments, {len(p.papers_linked)} papers")
    print()
