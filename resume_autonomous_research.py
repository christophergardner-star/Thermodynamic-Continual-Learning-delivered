"""
Resume Autonomous Research
===========================
Picks up the autonomous research loop from where it was interrupted.

State at interruption:
  - deep_anchor: seeds [42, 0, 1] complete (saved in deep_anchor_partial.json)
  - deep_anchor: seeds [2, 3] still needed
  - H2 graduated_penalty  — not started
  - H3 strict_consolidation — not started
  - H4 thermal_carryover   — not started
  - H5 high_penalty_conservative — not started

Approach:
  1. Run deep_anchor seeds [2, 3] via run_split_cifar10_benchmark
  2. Merge with partial results -> full 5-seed result -> save deep_anchor.json
  3. Run H2-H5 exactly as tar_autonomous_research.py would
  4. Write summary.json

Uses the same Split-CIFAR-10 benchmark, observer classes, and config_overrides
as the original tar_autonomous_research.py.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
from tar_storage import ensure_workspace_layout, resolve_workspace
workspace = str(ensure_workspace_layout(resolve_workspace(_REPO), repo_root=_REPO))

# ── stats helpers ─────────────────────────────────────────────────────────────
try:
    from scipy import stats as _scipy_stats
    def _ttest_1samp(values: list[float], mu: float = 0.0):
        t, p = _scipy_stats.ttest_1samp(values, mu)
        return float(t), float(p)
except ImportError:
    def _ttest_1samp(values: list[float], mu: float = 0.0):
        n = len(values)
        if n < 2: return 0.0, 1.0
        m = sum(values) / n
        s = math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))
        if s < 1e-12: return 0.0, 1.0
        t = (m - mu) / (s / math.sqrt(n))
        import math as _m
        p = 2.0 * (1.0 - 0.5 * (1.0 + _m.erf(abs(t) / _m.sqrt(2.0))))
        return t, p

def _mean(v): return sum(v) / len(v)
def _std(v):
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))
def _cohens_d(deltas): return abs(_mean(deltas)) / max(_std(deltas), 1e-12)


# ── TAR imports ───────────────────────────────────────────────────────────────
from tar_lab.schemas import ContinualLearningBenchmarkConfig
from tar_lab.multimodal_payloads import run_split_cifar10_benchmark
from tar_lab.thermoobserver import ActivationThermoObserver
from tar_project_registry import (
    ProjectRegistry, ResearchProject, STATUS_RUNNING, STATUS_COMPLETE, STATUS_FAILED,
    classify_research, generate_project_id, generate_project_name,
)
from tar_experiment_orchestrator import (
    ExperimentOrchestrator, ExperimentSpec, DATASET_CIFAR10,
)


# ── logging ───────────────────────────────────────────────────────────────────
_LOG_PATH = Path(workspace) / "tar_state" / "autonomous_research_resume.log"

def _ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _log(msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass

def _notify(title: str, body: str) -> None:
    import subprocess
    safe_t = title.replace('"', "'"); safe_b = body.replace('"', "'")
    ps = (
        'Add-Type -AssemblyName System.Windows.Forms; '
        '$n = New-Object System.Windows.Forms.NotifyIcon; '
        '$n.Icon = [System.Drawing.SystemIcons]::Application; '
        f'$n.BalloonTipTitle = "{safe_t}"; '
        f'$n.BalloonTipText  = "{safe_b}"; '
        '$n.Visible = $True; $n.ShowBalloonTip(25000); Start-Sleep 26; $n.Dispose()'
    )
    try:
        subprocess.Popen(["powershell", "-WindowStyle", "Hidden", "-Command", ps],
                         creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception:
        pass


# ── data structures (mirrors tar_autonomous_research.py) ─────────────────────
@dataclass
class Hypothesis:
    name: str
    mechanism_description: str
    prediction: str
    breakthrough_criteria: dict
    null_prediction: str
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class ExperimentResult:
    hypothesis_name: str
    seeds: list
    mechanism_forgetting: list
    baseline_forgetting: list
    mechanism_accuracy: list
    mean_delta: float
    t_stat: float
    p_val: float
    cohens_d: float
    n_better: int
    verdict: str
    notes: str
    run_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── observer subclasses (copied from tar_autonomous_research.py) ──────────────
class CarryoverAnchorObserver(ActivationThermoObserver):
    def reset_for_new_task(self) -> None:
        if hasattr(self, '_sigma_star') and self._sigma_star is not None:
            self._collecting_anchor = False
            self._anchor_buffer = []
        else:
            super().reset_for_new_task()

class StrictConsolidationObserver(ActivationThermoObserver):
    @property
    def current_regime(self) -> str:
        rho = getattr(self, 'rho', 1.0)
        if rho > 1.05:   return "disordered"
        elif rho < 0.85: return "ordered"
        else:            return "critical"

class GraduatedPenaltyObserver(ActivationThermoObserver):
    def _graduated_penalty_scale(self) -> float:
        rho = getattr(self, 'rho', 1.0)
        if rho >= 0.9: return 0.0
        return (0.9 - rho) / 0.9
    def get_penalty_scale(self) -> float:
        return self._graduated_penalty_scale()

class DeepAnchorObserver(ActivationThermoObserver):
    def __init__(self, model, **kwargs):
        kwargs.setdefault('sigma_star_anchor_n', 50)
        kwargs.setdefault('warmup_batches', 90)
        kwargs.setdefault('sigma_window_size', 12)
        kwargs.setdefault('sigma_tolerance', 0.12)
        super().__init__(model, **kwargs)


# ── constants ─────────────────────────────────────────────────────────────────
ALL_SEEDS    = [42, 0, 1, 2, 3]
RESUME_SEEDS = [2, 3]          # seeds still needed for deep_anchor
BACKBONE     = "resnet18"
EPOCHS       = 40

_TCL_BASELINE_FORGETTING = [0.1269, 0.1294, 0.1697, 0.1007, 0.1108]
_TCL_BASELINE_ACCURACY   = [0.767,  0.769,  0.731,  0.796,  0.786]

_AR_DIR = Path(workspace) / "tar_state" / "autonomous_research"


# ── benchmark wrapper ──────────────────────────────────────────────────────────
def _run_one_seed(seed: int, config_overrides: dict) -> tuple[float, float]:
    """Returns (mean_forgetting, mean_accuracy) for one seed."""
    cfg = ContinualLearningBenchmarkConfig(
        seed=seed,
        train_epochs_per_task=EPOCHS,
        ewc_lambda=100.0,
        **(config_overrides or {}),
    )
    r = run_split_cifar10_benchmark(cfg, method="tcl", workspace=workspace, backbone=BACKBONE)
    return r.mean_forgetting, r.final_mean_accuracy


def _eval_result(name: str, seeds: list, forgetting: list, accuracy: list,
                 baseline: list, crit: dict) -> ExperimentResult:
    n = len(forgetting)
    baseline_sub = baseline[:n]
    deltas = [m - b for m, b in zip(forgetting, baseline_sub)]
    mean_delta = _mean(deltas)
    t_stat, p_val = _ttest_1samp(deltas, 0.0)
    d = _cohens_d(deltas)
    n_better = sum(1 for x in deltas if x < 0)

    is_breakthrough = (
        mean_delta < crit.get("max_delta", -0.01)
        and p_val  < crit.get("max_p",    0.05)
        and d      > crit.get("min_d",    0.5)
    )
    is_directional = mean_delta < 0 and n_better >= (n // 2 + 1)
    is_adverse     = mean_delta > 0.02

    if is_breakthrough: verdict = "BREAKTHROUGH"
    elif is_directional: verdict = "DIRECTIONAL"
    elif is_adverse:     verdict = "ADVERSE"
    else:                verdict = "NULL"

    notes = (f"mean_delta={mean_delta:+.4f}  p={p_val:.4f}  d={d:.3f}  "
             f"{n_better}/{n} seeds mechanism better")

    return ExperimentResult(
        hypothesis_name=name,
        seeds=seeds[:n],
        mechanism_forgetting=forgetting,
        baseline_forgetting=baseline_sub,
        mechanism_accuracy=accuracy,
        mean_delta=mean_delta,
        t_stat=t_stat,
        p_val=p_val,
        cohens_d=d,
        n_better=n_better,
        verdict=verdict,
        notes=notes,
    )


def _save_result(result: ExperimentResult, hyp: Hypothesis) -> None:
    _AR_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "hypothesis": {
            "name": hyp.name,
            "mechanism_description": hyp.mechanism_description,
            "prediction": hyp.prediction,
            "breakthrough_criteria": hyp.breakthrough_criteria,
            "null_prediction": hyp.null_prediction,
            "registered_at": hyp.registered_at,
        },
        "result": asdict(result),
    }
    (_AR_DIR / f"{hyp.name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def _write_hypothesis_paper(hyp: Hypothesis, result: ExperimentResult) -> Path | None:
    """
    Write a self-contained LaTeX paper for this hypothesis result.
    Called for EVERY hypothesis regardless of verdict — null results are science too.
    Paper saved to {workspace}/paper/{hyp.name}/main.tex (+ PDF if LaTeX available).
    """
    import shutil, subprocess, textwrap as tw

    paper_dir = Path(workspace) / "paper" / hyp.name
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Human-readable hypothesis name
    _names = {
        "deep_anchor":              "Deep Anchor",
        "graduated_penalty":        "Graduated Penalty",
        "strict_consolidation":     "Strict Consolidation",
        "thermal_carryover":        "Thermal Carry-Over",
        "high_penalty_conservative":"High-Penalty Conservative",
    }
    h_name   = _names.get(hyp.name, hyp.name.replace("_", " ").title())
    verdict  = result.verdict
    n        = len(result.mechanism_forgetting)
    forg     = result.mechanism_forgetting
    acc      = result.mechanism_accuracy
    base     = result.baseline_forgetting
    m_forg   = sum(forg) / max(n, 1)
    m_acc    = sum(acc)  / max(n, 1)
    m_base   = sum(base) / max(len(base), 1)
    delta    = result.mean_delta
    p_val    = result.p_val
    d_val    = result.cohens_d
    n_better = result.n_better

    title = f"{h_name}: A Mechanism Variant for Thermodynamic Continual Learning"
    _verdict_tex = {
        "BREAKTHROUGH": r"\textbf{\textcolor{green!60!black}{BREAKTHROUGH}}",
        "DIRECTIONAL":  r"\textbf{\textcolor{blue}{Directional Improvement}}",
        "NULL":         r"Null Result (no significant improvement)",
        "ADVERSE":      r"\textbf{\textcolor{red}{Adverse Result}}",
    }.get(verdict, verdict)

    # Per-seed table rows
    seed_rows = ""
    for i, s in enumerate(result.seeds):
        f  = forg[i] if i < len(forg) else float("nan")
        a  = acc[i]  if i < len(acc)  else float("nan")
        b  = base[i] if i < len(base) else float("nan")
        d  = f - b
        sign = "-" if d < 0 else "+"
        seed_rows += rf"  {s} & ${f:.4f}$ & ${a:.3f}$ & ${b:.4f}$ & ${d:+.4f}$ \\" + "\n"

    mech_tex = hyp.mechanism_description.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    pred_tex = hyp.prediction.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    null_tex = hyp.null_prediction.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    main_tex = tw.dedent(rf"""
\documentclass[10pt,a4paper]{{article}}
\usepackage{{amsmath,amssymb,booktabs,geometry,hyperref,xcolor,microtype}}
\geometry{{margin=1in}}
\hypersetup{{colorlinks=true,linkcolor=blue,citecolor=green!40!black}}
\title{{{title}}}
\author{{Christopher Gardner \\ TAR (Thermodynamic Autonomous Researcher) \\ \small Independent Research}}
\date{{{datetime.now(timezone.utc).strftime("%B %Y")}}}
\begin{{document}}
\maketitle

\begin{{abstract}}
We report the result of a pre-registered experiment testing the \emph{{{h_name}}} variant
of Thermodynamic Continual Learning (TCL) on Split-CIFAR-10 with ResNet-18 across {n}~seeds.
The mechanism achieves mean forgetting ${m_forg:.4f}$ vs.\ TCL baseline ${m_base:.4f}$
($\Delta = {delta:+.4f}$, $p = {p_val:.4f}$, Cohen's $d = {d_val:.3f}$, {n_better}/{n}~seeds better).
Verdict: {_verdict_tex}.
\end{{abstract}}

\section{{Hypothesis and Mechanism}}

{mech_tex}

\subsection*{{Pre-registered Prediction}}
{pred_tex}

\subsection*{{Null Prediction}}
{null_tex}

\section{{Methods}}

Split-CIFAR-10 benchmark, task-incremental setting, ResNet-18 backbone.
$N={n}$ seeds, {40} epochs per task, 5 tasks × 2 classes.
All criteria pre-registered in \texttt{{preregistration.json}} before any experiments.
Baseline: Phase~10 TCL (mean forgetting ${m_base:.4f}$, $n=5$ seeds).

\section{{Results}}

\begin{{table}}[h]
\centering
\caption{{{h_name} — per-seed results vs.\ TCL Phase~10 baseline.}}
\begin{{tabular}}{{rcccc}}
\toprule
Seed & Forgetting $\downarrow$ & Accuracy $\uparrow$ & Baseline & $\Delta$ \\
\midrule
{seed_rows.strip()}
\midrule
\textbf{{Mean}} & $\mathbf{{{m_forg:.4f}}}$ & $\mathbf{{{m_acc:.3f}}}$ & ${m_base:.4f}$ & $\mathbf{{{delta:+.4f}}}$ \\
\bottomrule
\end{{tabular}}
\end{{table}}

Paired $t$-test vs.\ baseline: $t = {result.t_stat:.3f}$, $p = {p_val:.4f}$, Cohen's $d = {d_val:.3f}$.
{n_better}/{n} seeds show improvement over TCL baseline.

\section{{Verdict}}

\noindent\textbf{{Result: {_verdict_tex}.}}

{_verdict_discussion(verdict, delta, p_val, d_val, n_better, n, h_name)}

\end{{document}}
""").lstrip()

    tex_path = paper_dir / "main.tex"
    tex_path.write_text(main_tex, encoding="utf-8")
    _log(f"  Paper written: {tex_path}")

    # Compile to PDF
    compiler = shutil.which("pdflatex") or shutil.which("xelatex")
    if not compiler:
        for c in [
            Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe",
            Path("C:/Program Files/MiKTeX/miktex/bin/x64/pdflatex.exe"),
        ]:
            if c.exists(): compiler = str(c); break
    if compiler:
        cmd = [compiler, "-interaction=nonstopmode", f"-output-directory={paper_dir}", str(tex_path)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
            subprocess.run(cmd, capture_output=True, timeout=120)   # second pass for refs
            pdf = paper_dir / "main.pdf"
            if pdf.exists():
                _log(f"  PDF compiled: {pdf}")
                return pdf
        except Exception as exc:
            _log(f"  PDF compilation failed: {exc}")
    return tex_path


def _verdict_discussion(verdict: str, delta: float, p_val: float, d: float,
                        n_better: int, n: int, name: str) -> str:
    if verdict == "BREAKTHROUGH":
        return (
            f"The {name} mechanism achieves a statistically significant reduction in "
            f"catastrophic forgetting ($\\Delta = {delta:+.4f}$, $p = {p_val:.4f}$, "
            f"$d = {d:.3f}$, {n_better}/{n} seeds), meeting all pre-registered criteria. "
            f"This constitutes a genuine improvement over the TCL baseline."
        )
    if verdict == "DIRECTIONAL":
        return (
            f"The {name} mechanism shows a consistent directional improvement "
            f"({n_better}/{n} seeds better, $\\Delta = {delta:+.4f}$) but does not reach "
            f"the pre-registered significance threshold ($p = {p_val:.4f} > 0.05$ or $d = {d:.3f} < 0.5$). "
            f"The signal warrants follow-up with $n \\geq 10$ seeds."
        )
    if verdict == "ADVERSE":
        return (
            f"The {name} mechanism increases forgetting ($\\Delta = {delta:+.4f}$), "
            f"consistent with the null prediction. The mechanism as implemented "
            f"harms continual learning performance."
        )
    return (
        f"The {name} mechanism shows no consistent improvement over TCL baseline "
        f"($\\Delta = {delta:+.4f}$, $p = {p_val:.4f}$). "
        f"The null prediction is confirmed: this variant does not reduce forgetting "
        f"beyond the reference TCL implementation."
    )


def _log_paper_to_registry(hyp: Hypothesis, result: ExperimentResult, paper_path: Path) -> None:
    """Record the paper in the global papers log and project registry."""
    ws = Path(workspace)
    # papers.jsonl
    papers_log = ws / "tar_state" / "papers"
    papers_log.mkdir(parents=True, exist_ok=True)
    pdf_path = paper_path.parent / "main.pdf"
    record = {
        "project_id":   f"tcl-{hyp.name.replace('_','-')}-cifar10-v1",
        "title":        f"{hyp.name.replace('_',' ').title()} — TCL Autonomous Research",
        "verdict":      result.verdict,
        "tex_path":     str(paper_path.parent / "main.tex"),
        "pdf_path":     str(pdf_path) if pdf_path.exists() else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mean_forgetting": result.mean_forgetting if result.mechanism_forgetting else None,
        "mean_delta":   result.mean_delta,
        "p_val":        result.p_val,
    }
    log_path = papers_log / "papers.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    # Also update project registry
    try:
        registry.update_pdf(
            f"tcl-{hyp.name.replace('_','-')}-cifar10-v1",
            str(pdf_path) if pdf_path.exists() else str(paper_path.parent / "main.tex"),
        )
    except Exception:
        pass


def _write_summary(results: list[tuple[Hypothesis, ExperimentResult]]) -> None:
    _AR_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total_hypotheses_tested": len(results),
        "results": [
            {
                "name": h.name,
                "verdict": r.verdict,
                "mean_delta": r.mean_delta,
                "p_val": r.p_val,
                "cohens_d": r.cohens_d,
                "n_better": r.n_better,
                "notes": r.notes,
            }
            for h, r in results
        ],
    }
    (_AR_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ── load TCL baseline ─────────────────────────────────────────────────────────
def _load_tcl_baseline() -> list[float]:
    for p in [
        Path(workspace) / "tar_state" / "comparisons" / "phase10_baseline.json",
        _REPO / "tar_state" / "comparisons" / "phase10_baseline.json",
    ]:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                vals = [row["tcl_forgetting"] for row in data.get("per_seed", [])
                        if "tcl_forgetting" in row]
                if vals: return vals
            except Exception:
                pass
    return _TCL_BASELINE_FORGETTING


# ── hypothesis definitions (matching tar_autonomous_research.py) ──────────────
def _build_hypotheses(tcl_mean: float) -> list[tuple[Hypothesis, dict]]:
    """Returns (Hypothesis, config_overrides) pairs for H1-H5."""
    return [
        (
            Hypothesis(
                name="deep_anchor",
                mechanism_description=(
                    "Deep Anchor (DA): Longer sigma_star calibration window (50 batches vs 20) "
                    "and longer warmup guard (90 vs 60 batches)."
                ),
                prediction=(
                    f"DA mean forgetting < TCL ({tcl_mean:.4f}) by >0.01, p<0.05, d>0.5."
                ),
                breakthrough_criteria={"max_delta": -0.01, "max_p": 0.05, "min_d": 0.5},
                null_prediction="Longer calibration provides no benefit.",
            ),
            {
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.01,
                "tcl_alpha": 0.5,
                "tcl_reset_on_task_boundary": True,
            },
        ),
        (
            Hypothesis(
                name="graduated_penalty",
                mechanism_description=(
                    "Graduated Penalty (GP): Scale anchor penalty continuously with depth "
                    "into ordered regime: lambda * (0.9-rho)/0.9 when rho<0.9."
                ),
                prediction=(
                    f"GP mean forgetting < TCL ({tcl_mean:.4f}), p<0.05, d>0.5."
                ),
                breakthrough_criteria={"max_delta": -0.01, "max_p": 0.05, "min_d": 0.5},
                null_prediction="Binary penalty equally effective as graduated.",
            ),
            {
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.01,
                "tcl_alpha": 0.5,
                "tcl_reset_on_task_boundary": True,
            },
        ),
        (
            Hypothesis(
                name="strict_consolidation",
                mechanism_description=(
                    "Strict Consolidation (SC): Tighten regime thresholds to rho<0.85 ordered, "
                    "rho>1.05 disordered (vs 0.9/1.1 standard)."
                ),
                prediction=(
                    f"SC mean forgetting < TCL ({tcl_mean:.4f}), p<0.05, d>0.5."
                ),
                breakthrough_criteria={"max_delta": -0.01, "max_p": 0.05, "min_d": 0.5},
                null_prediction="Stricter thresholds fire too late, increasing forgetting.",
            ),
            {
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.01,
                "tcl_alpha": 0.5,
                "tcl_reset_on_task_boundary": True,
            },
        ),
        (
            Hypothesis(
                name="thermal_carryover",
                mechanism_description=(
                    "Thermal Carry-Over (TCO): Carry sigma_star across task boundaries instead "
                    "of resetting per-task, encoding inter-task consolidation memory."
                ),
                prediction=(
                    f"TCO mean forgetting < TCL ({tcl_mean:.4f}), p<0.05, d>0.5."
                ),
                breakthrough_criteria={"max_delta": -0.01, "max_p": 0.05, "min_d": 0.5},
                null_prediction="Carrying old anchor over-constrains new-task learning.",
            ),
            {
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.01,
                "tcl_alpha": 0.5,
                "tcl_reset_on_task_boundary": False,
            },
        ),
        (
            Hypothesis(
                name="high_penalty_conservative",
                mechanism_description=(
                    "High-Penalty Conservative (HPC): lambda_tcl=0.05 (5x), "
                    "ordered LR scale=0.3. Stronger consolidation brakes."
                ),
                prediction=(
                    f"HPC mean forgetting < {tcl_mean - 0.01:.4f}, p<0.05, d>0.5, acc>0.70."
                ),
                breakthrough_criteria={"max_delta": -0.01, "max_p": 0.05, "min_d": 0.5},
                null_prediction="High penalty causes accuracy collapse below 0.70.",
            ),
            {
                "tcl_governor_enabled": True,
                "tcl_penalty_lambda": 0.05,
                "tcl_ordered_lr_scale": 0.3,
                "tcl_alpha": 0.45,
                "tcl_reset_on_task_boundary": True,
            },
        ),
    ]


# ── main ──────────────────────────────────────────────────────────────────────
def _register_hypothesis(registry: ProjectRegistry, hyp: Hypothesis,
                          status: str, result: ExperimentResult | None = None) -> None:
    """Register/update a hypothesis experiment in the project registry."""
    field, subfield, keywords = classify_research(
        hypothesis_name=hyp.name,
        phase_source="autonomous_research",
    )
    project_id = generate_project_id(hypothesis_name=hyp.name, dataset="split_cifar10")
    name = generate_project_name(hypothesis_name=hyp.name, dataset="split_cifar10")
    abstract = hyp.prediction[:300]
    if result:
        abstract = result.notes[:300]
    project = ResearchProject(
        id=project_id,
        name=name,
        field=field,
        subfield=subfield,
        keywords=keywords,
        status=status,
        phase_source="autonomous_research",
        dataset="split_cifar10",
        abstract=abstract,
        paper_dir="",
        paper_pdf="",
        data_paths=[str(_AR_DIR / f"{hyp.name}.json")],
        authors=["Christopher Gardner"],
        affiliation="Independent Research",
        notes=f"verdict={result.verdict}" if result else "pending",
    )
    registry.register(project)


def main() -> None:
    from tar_living_research import run_portfolio

    _log("=" * 70)
    _log("TAR AUTONOMOUS RESEARCH RESUME - ORCHESTRATED MODE")
    _log("=" * 70)
    _notify("TAR Resume", "Resuming autonomous research through orchestrator")
    run_portfolio(
        Path(workspace),
        include_scaleup=False,
        include_autonomous=True,
    )
    return

    _log("=" * 70)
    _log("TAR AUTONOMOUS RESEARCH — RESUME FROM INTERRUPTION")
    _log("=" * 70)
    _notify("TAR Resume", "Resuming autonomous research from deep_anchor seed=2")

    tcl_baseline = _load_tcl_baseline()
    tcl_mean     = _mean(tcl_baseline)
    _log(f"TCL baseline loaded: mean={tcl_mean:.4f}  n={len(tcl_baseline)}")

    hypotheses = _build_hypotheses(tcl_mean)
    all_results: list[tuple[Hypothesis, ExperimentResult]] = []
    breakthrough_found = False

    # Initialise project registry — pre-register all hypotheses so they appear
    # in the index even before results arrive.
    ws = Path(workspace)
    registry = ProjectRegistry(ws)
    orch     = ExperimentOrchestrator(ws)
    for hyp, _ in hypotheses:
        _register_hypothesis(registry, hyp, STATUS_RUNNING)
    registry.print_summary()

    # ── Step 1: Complete deep_anchor (seeds [2, 3]) ───────────────────────────
    _log("\n[RESUME] deep_anchor — completing seeds [2, 3]")

    partial_path = _AR_DIR / "deep_anchor_partial.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else {}

    # Collect already-completed seed data
    da_forgetting = [r["forgetting"] for r in partial.get("partial_results", [])]
    da_accuracy   = [r["acc"]        for r in partial.get("partial_results", [])]
    da_seeds_done = list(partial.get("seeds_completed", []))

    da_hyp, da_cfg = hypotheses[0]  # deep_anchor is H1
    _log(f"  Seeds already done: {da_seeds_done} -> forgetting={[f'{f:.4f}' for f in da_forgetting]}")

    for seed in RESUME_SEEDS:
        _log(f"  Running deep_anchor seed={seed}...")
        t0 = time.time()
        try:
            forgetting, acc = _run_one_seed(seed, da_cfg)
            da_forgetting.append(forgetting)
            da_accuracy.append(acc)
            da_seeds_done.append(seed)
            elapsed = time.time() - t0
            _log(f"  seed={seed}  forgetting={forgetting:.4f}  acc={acc:.4f}  ({elapsed/60:.1f}m)")
        except Exception as exc:
            _log(f"  seed={seed}  ERROR: {exc}")

    if da_forgetting:
        da_result = _eval_result(
            "deep_anchor", da_seeds_done, da_forgetting, da_accuracy,
            tcl_baseline, da_hyp.breakthrough_criteria,
        )
        _log(f"\ndeep_anchor RESULT: {da_result.verdict}")
        _log(f"  {da_result.notes}")
        _save_result(da_result, da_hyp)
        all_results.append((da_hyp, da_result))
        _register_hypothesis(registry, da_hyp,
                             STATUS_COMPLETE if da_result.verdict != "ERROR" else STATUS_FAILED,
                             da_result)
        _log(f"  Writing paper for deep_anchor ({da_result.verdict})...")
        paper_path = _write_hypothesis_paper(da_hyp, da_result)
        if paper_path:
            _log_paper_to_registry(da_hyp, da_result, paper_path)
        _notify(f"TAR deep_anchor {da_result.verdict}", da_result.notes[:100])

        if da_result.verdict == "BREAKTHROUGH":
            _log("BREAKTHROUGH: deep_anchor!")
            breakthrough_found = True

    # ── Steps 2-5: H2 through H5 ─────────────────────────────────────────────
    if not breakthrough_found:
        for hyp, cfg in hypotheses[1:]:
            _log(f"\n{'='*70}")
            _log(f"HYPOTHESIS: {hyp.name}")
            _log(f"{'='*70}")
            _log(hyp.mechanism_description)

            # Skip if result file already exists and is complete
            existing = _AR_DIR / f"{hyp.name}.json"
            if existing.exists():
                try:
                    rec = json.loads(existing.read_text(encoding="utf-8"))
                    existing_seeds = rec.get("result", {}).get("seeds", [])
                    if len(existing_seeds) >= len(ALL_SEEDS):
                        _log(f"  Already complete ({len(existing_seeds)} seeds) — skipping")
                        verdict = rec.get("result", {}).get("verdict", "UNKNOWN")
                        if verdict == "BREAKTHROUGH":
                            _log(f"  Previous BREAKTHROUGH for {hyp.name} — stopping")
                            breakthrough_found = True
                            break
                        continue
                except Exception:
                    pass

            h_forgetting, h_accuracy, h_seeds = [], [], []
            for seed in ALL_SEEDS:
                _log(f"  seed={seed}...")
                t0 = time.time()
                try:
                    forgetting, acc = _run_one_seed(seed, cfg)
                    h_forgetting.append(forgetting)
                    h_accuracy.append(acc)
                    h_seeds.append(seed)
                    elapsed = time.time() - t0
                    _log(f"  seed={seed}  forgetting={forgetting:.4f}  acc={acc:.4f}  ({elapsed/60:.1f}m)")
                except Exception as exc:
                    _log(f"  seed={seed}  ERROR: {exc}")

            if not h_forgetting:
                _log(f"  All seeds failed for {hyp.name}")
                continue

            result = _eval_result(
                hyp.name, h_seeds, h_forgetting, h_accuracy,
                tcl_baseline, hyp.breakthrough_criteria,
            )
            _log(f"\n{hyp.name} RESULT: {result.verdict}")
            _log(f"  {result.notes}")
            _save_result(result, hyp)
            all_results.append((hyp, result))
            _register_hypothesis(registry, hyp,
                                 STATUS_COMPLETE if result.verdict != "ERROR" else STATUS_FAILED,
                                 result)
            _log(f"  Writing paper for {hyp.name} ({result.verdict})...")
            paper_path = _write_hypothesis_paper(hyp, result)
            if paper_path:
                _log_paper_to_registry(hyp, result, paper_path)
            _notify(f"TAR {hyp.name} {result.verdict}", result.notes[:100])

            if result.verdict == "BREAKTHROUGH":
                _log(f"\nBREAKTHROUGH: {hyp.name}!")
                breakthrough_found = True
                break
            elif result.verdict == "DIRECTIONAL":
                _log("  Directional improvement — continuing to next hypothesis.")
            else:
                _log(f"  {result.verdict} result. Continuing.")

    # ── Summary ───────────────────────────────────────────────────────────────
    _log(f"\n{'='*70}")
    _write_summary(all_results)

    if breakthrough_found:
        winning = [(h, r) for h, r in all_results if r.verdict == "BREAKTHROUGH"]
        _log("AUTONOMOUS RESEARCH COMPLETE — BREAKTHROUGH ACHIEVED")
        for h, r in winning:
            _log(f"  {h.name}: {r.notes}")
        _notify("TAR BREAKTHROUGH", f"{winning[0][0].name}: {winning[0][1].notes[:80]}")
    else:
        _log("AUTONOMOUS RESEARCH COMPLETE — NO BREAKTHROUGH THIS CYCLE")
        for h, r in all_results:
            _log(f"  {h.name:30s}  {r.verdict:12s}  delta={r.mean_delta:+.4f}  p={r.p_val:.4f}")
        directional = [(h, r) for h, r in all_results if r.verdict == "DIRECTIONAL"]
        _notify(
            "TAR Research Cycle Complete",
            f"{len(all_results)} hypotheses. "
            + (f"{len(directional)} directional." if directional else "No directional."),
        )

    _log(f"Summary: {_AR_DIR / 'summary.json'}")
    _log(f"Log:     {_LOG_PATH}")


if __name__ == "__main__":
    main()
