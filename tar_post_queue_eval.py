"""
TAR Post-Queue Evaluation
=========================
Reads all phase results, autonomous research output, and ASC model status.
Generates a structured decision report with:
  1. Summary of each phase outcome against pre-registered criteria
  2. Key findings ranked by scientific significance
  3. Recommended Queue 2 configuration
  4. arXiv submission readiness assessment
  5. Autonomous research breakthrough status

Output:
  tar_state/post_queue_eval/report.json   — machine-readable
  tar_state/post_queue_eval/report.txt    — human-readable summary
  tar_state/post_queue_eval/queue2_config.json — ready for run_queue2.py

Run: python tar_post_queue_eval.py [--workspace E:/TAR/...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tar_storage import ensure_workspace_layout, resolve_workspace

# ---------------------------------------------------------------------------
# Repo root (the directory this script lives in)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent


# ===========================================================================
# Helper utilities
# ===========================================================================

def _ts() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _mean(v: list[float]) -> float:
    """Safe mean — returns 0.0 for empty lists."""
    if not v:
        return 0.0
    return sum(v) / len(v)


def _std(v: list[float]) -> float:
    """Safe population std — returns 0.0 for lists shorter than 2."""
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _load_json(p: Path) -> dict | None:
    """Load a JSON file; return None on any error."""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_all_phases(workspace: Path) -> dict[int, dict]:
    """
    Scan workspace/tar_state/comparisons/ for files matching phase*.json.
    Returns a mapping {phase_number: data_dict}.
    Handles both 'phase15_class_incremental_search.json' (named) and
    numeric phase files such as 'phase10.json'.
    """
    comparisons_dir = workspace / "tar_state" / "comparisons"
    results: dict[int, dict] = {}
    if not comparisons_dir.is_dir():
        return results

    for p in comparisons_dir.glob("phase*.json"):
        data = _load_json(p)
        if data is None:
            continue
        # Try to extract phase number from the filename
        m = re.search(r"phase(\d+)", p.stem, re.IGNORECASE)
        if m:
            num = int(m.group(1))
            results[num] = data
    return results


def _load_autonomous_research(workspace: Path) -> dict[str, Any]:
    """
    Read tar_state/autonomous_research/ directory.
    Returns a dict with keys: preregistration, results, found (bool).
    """
    ar_dir = workspace / "tar_state" / "autonomous_research"
    out: dict[str, Any] = {"found": False, "preregistration": None, "results": []}
    if not ar_dir.is_dir():
        return out

    # Preregistration
    prereg_path = ar_dir / "preregistration.json"
    if prereg_path.exists():
        out["preregistration"] = _load_json(prereg_path)
        out["found"] = True

    # Any results JSONs
    result_files = sorted(ar_dir.glob("*.json"))
    for rf in result_files:
        if rf.name == "preregistration.json":
            continue
        d = _load_json(rf)
        if d:
            out["results"].append(d)
            out["found"] = True

    return out


def _load_asc_model(workspace: Path) -> dict[str, Any]:
    """
    Check whether a fine-tuned ASC model artefact exists under
    workspace/training_artifacts/asc_finetune/.
    Returns a dict with: exists (bool), path (str|None), details (dict|None).
    """
    model_dir = workspace / "training_artifacts" / "asc_finetune"
    if not model_dir.is_dir():
        return {"exists": False, "path": None, "details": None}

    # Look for a model checkpoint or metadata file
    possible = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.bin")) + \
               list(model_dir.glob("config.json")) + list(model_dir.glob("model_card.json"))

    if not possible:
        return {"exists": False, "path": str(model_dir), "details": None}

    details_path = model_dir / "model_card.json"
    details = _load_json(details_path) if details_path.exists() else None

    return {
        "exists": True,
        "path": str(model_dir),
        "details": details,
    }


# ===========================================================================
# Phase evaluators
# ===========================================================================

def _eval_phase10(data: dict) -> dict:
    """
    Phase 10 — 4-way baseline comparison.
    Outcome B criteria: TCL vs SGD p<0.05, Cohen's d>0.5, all 5 seeds present.
    """
    phase = 10
    try:
        comparisons = data.get("comparisons", {})
        tcl_vs_sgd = comparisons.get("tcl_vs_sgd", {})
        tcl_vs_ewc = comparisons.get("tcl_vs_ewc", {})

        p_sgd = tcl_vs_sgd.get("p_value", 1.0)
        d_sgd = tcl_vs_sgd.get("effect_size", 0.0)
        n_seeds = data.get("n_seeds", data.get("seeds_completed", 0))
        p_ewc = tcl_vs_ewc.get("p_value", 1.0)

        outcome_b_met = (p_sgd < 0.05) and (d_sgd > 0.5) and (n_seeds >= 5)

        if outcome_b_met:
            outcome = "OUTCOME_B_MET"
            significance = "HIGH"
            key_finding = (
                f"TCL significantly outperforms SGD baseline "
                f"(p={p_sgd:.3f}, d={d_sgd:.2f}) across all {n_seeds} seeds."
            )
            recommendation = "Proceed to scale-up (CIFAR-100). Core result is publishable."
        elif p_sgd < 0.05:
            outcome = "PARTIAL_B"
            significance = "MEDIUM"
            key_finding = (
                f"TCL beats SGD (p={p_sgd:.3f}) but effect size marginal (d={d_sgd:.2f}) "
                f"or insufficient seeds ({n_seeds})."
            )
            recommendation = "Increase seeds or refine hyperparameters before scale-up."
        else:
            outcome = "OUTCOME_A_NULL"
            significance = "LOW"
            key_finding = (
                f"No significant improvement over SGD (p={p_sgd:.3f}, d={d_sgd:.2f})."
            )
            recommendation = "Re-examine TCL mechanism. Check ablation (Phase 11) for cause."

        return {
            "phase": phase,
            "outcome": outcome,
            "significance": significance,
            "key_finding": key_finding,
            "recommendation": recommendation,
            "details": {
                "p_vs_sgd": p_sgd,
                "d_vs_sgd": d_sgd,
                "p_vs_ewc": p_ewc,
                "n_seeds": n_seeds,
            },
        }
    except Exception as exc:
        return _eval_error(phase, exc)


def _eval_phase11(data: dict) -> dict:
    """
    Phase 11 — Ablation study.
    Checks: penalty_only < sgd_baseline (p<0.05), governor contribution to variance stability.
    """
    phase = 11
    try:
        ablation = data.get("ablation_results", data.get("results", {}))

        penalty_vs_sgd = ablation.get("penalty_only_vs_sgd", {})
        p_pen = penalty_vs_sgd.get("p_value", 1.0)
        significant_penalty = p_pen < 0.05

        # Governor variance contribution
        full_stds = ablation.get("full_tcl_std_across_seeds", None)
        penalty_only_stds = ablation.get("penalty_only_std_across_seeds", None)

        governor_note = ""
        if full_stds is not None and penalty_only_stds is not None:
            governor_note = (
                f"governor stabilises variance (full std={full_stds:.3f} "
                f"vs penalty-only std={penalty_only_stds:.3f})"
            )
        else:
            # Try alternate key names
            gov = data.get("governor_contribution", {})
            if gov:
                governor_note = f"governor contribution metric={gov.get('metric', 'n/a')}"

        if significant_penalty:
            outcome = "PENALTY_DOMINANT"
            significance = "HIGH"
            key_finding = (
                f"Penalty component alone beats SGD (p={p_pen:.3f}). "
                + (governor_note or "Governor role requires further analysis.")
            )
            recommendation = "Both penalty and governor contribute. Report full ablation in paper."
        else:
            outcome = "GOVERNOR_ESSENTIAL"
            significance = "MEDIUM"
            key_finding = (
                f"Penalty-only insufficient (p={p_pen:.3f}); governor required for improvement. "
                + governor_note
            )
            recommendation = "Governor is the key mechanism — emphasise in mechanism section."

        return {
            "phase": phase,
            "outcome": outcome,
            "significance": significance,
            "key_finding": key_finding,
            "recommendation": recommendation,
            "details": {
                "p_penalty_vs_sgd": p_pen,
                "governor_note": governor_note,
            },
        }
    except Exception as exc:
        return _eval_error(phase, exc)


def _eval_phase12(data: dict) -> dict:
    """
    Phase 12 — EWC lambda sweep.
    Checks: TCL beats best EWC at any tested lambda (p<0.05).
    """
    phase = 12
    try:
        sweep = data.get("lambda_sweep", data.get("ewc_sweep", {}))
        tcl_beats_ewc_lambdas: list[float] = []
        ewc_collapse_lambdas: list[float] = []

        # sweep may be a list of dicts or a dict keyed by lambda value
        items: list[dict] = []
        if isinstance(sweep, list):
            items = sweep
        elif isinstance(sweep, dict):
            items = [{"lambda": k, **v} for k, v in sweep.items()]

        best_p = 1.0
        best_lam = None
        for entry in items:
            lam = entry.get("lambda", entry.get("lam", "?"))
            p = entry.get("p_tcl_vs_ewc", entry.get("p_value", 1.0))
            ewc_collapsed = entry.get("ewc_collapsed", entry.get("catastrophic_forgetting", False))
            if ewc_collapsed:
                ewc_collapse_lambdas.append(lam)
            if p < 0.05:
                tcl_beats_ewc_lambdas.append(lam)
            if p < best_p:
                best_p = p
                best_lam = lam

        # Fallback: summary fields at top level
        if not items:
            best_p = data.get("best_p_tcl_vs_ewc", data.get("p_value", 1.0))
            best_lam = data.get("best_lambda", "unknown")
            ewc_collapse_lambdas = data.get("collapsed_lambdas", [])
            if best_p < 0.05:
                tcl_beats_ewc_lambdas = [best_lam]

        if tcl_beats_ewc_lambdas:
            outcome = "TCL_ROBUST"
            significance = "HIGH"
            key_finding = (
                f"TCL outperforms EWC at lambda={tcl_beats_ewc_lambdas} (best p={best_p:.3f}). "
                + (f"EWC collapses at lambda={ewc_collapse_lambdas}." if ewc_collapse_lambdas else "")
            )
            recommendation = "Include EWC sweep table in supplementary material."
        else:
            outcome = "EWC_COMPETITIVE"
            significance = "MEDIUM"
            key_finding = (
                f"TCL does not significantly beat best EWC (best p={best_p:.3f} at lambda={best_lam}). "
                + (f"EWC collapses at high lambda={ewc_collapse_lambdas}." if ewc_collapse_lambdas else "")
            )
            recommendation = "Frame TCL as more robust/stable rather than strictly superior to EWC."

        return {
            "phase": phase,
            "outcome": outcome,
            "significance": significance,
            "key_finding": key_finding,
            "recommendation": recommendation,
            "details": {
                "best_p_vs_ewc": best_p,
                "best_lambda": best_lam,
                "beats_ewc_at": tcl_beats_ewc_lambdas,
                "ewc_collapse_at": ewc_collapse_lambdas,
            },
        }
    except Exception as exc:
        return _eval_error(phase, exc)


def _eval_phase13(data: dict) -> dict:
    """
    Phase 13 — SI / degenerate-dynamics study.
    Checks verdict_key for ALL_DEGENERATE / PARTIAL_RECOVERY / FULL_RECOVERY.
    """
    phase = 13
    try:
        verdict_key = data.get("verdict_key", data.get("verdict", "UNKNOWN")).upper()
        c_values_tested = data.get("c_values_tested", data.get("c_values", []))
        recovery_rate = data.get("recovery_rate", data.get("partial_recovery_fraction", None))

        if verdict_key == "ALL_DEGENERATE":
            outcome = "ALL_DEGENERATE"
            significance = "HIGH"
            key_finding = (
                f"SI dynamics degenerate universally across all tested c values "
                f"({c_values_tested}). TCL governor prevents collapse."
            )
            recommendation = (
                "Strong negative result against SI. Highlight in paper as motivation "
                "for thermodynamic governor design."
            )
        elif verdict_key == "PARTIAL_RECOVERY":
            frac = f" ({recovery_rate:.0%})" if recovery_rate is not None else ""
            outcome = "PARTIAL_RECOVERY"
            significance = "MEDIUM"
            key_finding = (
                f"Partial SI recovery{frac} at some c values. "
                "TCL outperforms degenerate SI configurations."
            )
            recommendation = "Report as partial evidence; include c-value sensitivity analysis."
        elif verdict_key == "FULL_RECOVERY":
            outcome = "FULL_RECOVERY"
            significance = "LOW"
            key_finding = (
                f"SI achieves full recovery at optimal c. "
                "TCL advantage over SI is narrower than expected."
            )
            recommendation = "Re-examine whether TCL provides unique benefit over tuned SI."
        else:
            outcome = f"UNKNOWN_VERDICT_{verdict_key}"
            significance = "LOW"
            key_finding = f"Verdict key not recognised: '{verdict_key}'. Manual review required."
            recommendation = "Inspect Phase 13 output JSON for unexpected format."

        return {
            "phase": phase,
            "outcome": outcome,
            "significance": significance,
            "key_finding": key_finding,
            "recommendation": recommendation,
            "details": {
                "verdict_key": verdict_key,
                "c_values_tested": c_values_tested,
                "recovery_rate": recovery_rate,
            },
        }
    except Exception as exc:
        return _eval_error(phase, exc)


def _eval_phase14(data: dict) -> dict:
    """
    Phase 14 — Paper positioning / publication audit.
    Extracts publishability_status and positioning recommendation.
    """
    phase = 14
    try:
        pub_status = data.get(
            "publishability_status",
            data.get("publication_status", data.get("status", "UNKNOWN")),
        ).upper()
        positioning = data.get(
            "positioning_recommendation",
            data.get("positioning", data.get("recommendation", "")),
        )
        venue = data.get("recommended_venue", data.get("target_venue", "unspecified"))
        missing_items = data.get("missing_items", data.get("gaps", []))

        if pub_status in ("PUBLISHABLE", "READY", "STRONG"):
            outcome = "PUBLISHABLE"
            significance = "HIGH"
            key_finding = (
                f"Paper assessed as publishable (status={pub_status}). "
                f"Target venue: {venue}."
            )
            recommendation = positioning or "Submit to target venue. Address minor missing items."
        elif pub_status in ("CONDITIONAL", "NEAR_READY", "WEAK"):
            outcome = "CONDITIONAL"
            significance = "MEDIUM"
            key_finding = (
                f"Conditional publishability (status={pub_status}). "
                f"Missing items: {missing_items}."
            )
            recommendation = positioning or "Address missing items before submission."
        else:
            outcome = "NOT_READY"
            significance = "LOW"
            key_finding = (
                f"Paper not ready for submission (status={pub_status}). "
                f"Missing items: {missing_items}."
            )
            recommendation = positioning or "Significant revisions required. Re-run Phase 14 after changes."

        return {
            "phase": phase,
            "outcome": outcome,
            "significance": significance,
            "key_finding": key_finding,
            "recommendation": recommendation,
            "details": {
                "publishability_status": pub_status,
                "recommended_venue": venue,
                "missing_items": missing_items,
            },
        }
    except Exception as exc:
        return _eval_error(phase, exc)


def _eval_phase15(data: dict) -> dict:
    """
    Phase 15 — Class-incremental search.
    Checks: external_breakthrough_candidate, best_delta_vs_strong_baseline < -0.01, p < 0.05.
    """
    phase = 15
    try:
        breakthrough = data.get("external_breakthrough_candidate", False)
        delta = data.get("best_delta_vs_strong_baseline", data.get("delta_vs_strong_baseline", 0.0))
        p = data.get("p_value_vs_strong_baseline", data.get("p_value", 1.0))
        d = data.get("effect_size_vs_strong_baseline", data.get("effect_size", 0.0))
        best_candidate = data.get("best_candidate_name", data.get("best_method", "unknown"))
        pub_status = data.get("publishability_status", "UNKNOWN")

        is_significant = (p < 0.05) and (delta < -0.01)

        if breakthrough and is_significant:
            outcome = "BREAKTHROUGH"
            significance = "HIGH"
            key_finding = (
                f"External breakthrough candidate: {best_candidate} "
                f"(delta={delta:+.4f}, p={p:.4f}, d={d:.3f}). "
                f"Publishability: {pub_status}."
            )
            recommendation = (
                "Immediate priority: deep-dive on class-incremental setting. "
                "Consider Phase 16 = CI Fisher hybrid on CIFAR-100."
            )
        elif is_significant:
            outcome = "SIGNIFICANT_CI_RESULT"
            significance = "HIGH"
            key_finding = (
                f"Significant CI result: {best_candidate} "
                f"(delta={delta:+.4f}, p={p:.4f}, d={d:.3f})."
            )
            recommendation = "Report CI result prominently. Queue 2 should scale this up."
        elif p < 0.05:
            outcome = "MARGINAL_CI"
            significance = "MEDIUM"
            key_finding = (
                f"Marginal CI improvement: {best_candidate} "
                f"(delta={delta:+.4f}, p={p:.4f}). Effect size small (d={d:.3f})."
            )
            recommendation = "Include as exploratory result. More seeds/epochs needed."
        else:
            outcome = "NULL_CI"
            significance = "LOW"
            key_finding = (
                f"No significant CI advantage (delta={delta:+.4f}, p={p:.4f})."
            )
            recommendation = "Do not prioritise CI in Queue 2. Focus on task-incremental scale-up."

        return {
            "phase": phase,
            "outcome": outcome,
            "significance": significance,
            "key_finding": key_finding,
            "recommendation": recommendation,
            "details": {
                "breakthrough": breakthrough,
                "delta": delta,
                "p_value": p,
                "effect_size": d,
                "best_candidate": best_candidate,
                "publishability_status": pub_status,
            },
        }
    except Exception as exc:
        return _eval_error(phase, exc)


def _eval_error(phase: int, exc: Exception) -> dict:
    """Return a safe error dict when a phase evaluator fails."""
    return {
        "phase": phase,
        "outcome": "EVAL_ERROR",
        "significance": "UNKNOWN",
        "key_finding": f"Evaluator raised exception: {exc}",
        "recommendation": f"Inspect phase {phase} JSON manually.",
        "details": {"error": str(exc)},
    }


# Dispatch table — add new phases here
_PHASE_EVALUATORS = {
    10: _eval_phase10,
    11: _eval_phase11,
    12: _eval_phase12,
    13: _eval_phase13,
    14: _eval_phase14,
    15: _eval_phase15,
}


# ===========================================================================
# Queue 2 recommendation
# ===========================================================================

def _recommend_queue2(
    phase_evals: list[dict], auto_result: dict
) -> dict[str, Any]:
    """
    Decide what to run in Queue 2 based on phase outcomes and autonomous research.
    Returns a dict with key 'recommended_phases', each entry containing:
      {phase, script, priority, rationale, estimated_hours}
    """
    recommended: list[dict] = []
    priority_counter = 1

    evals_by_phase = {e["phase"]: e for e in phase_evals}

    p15 = evals_by_phase.get(15, {})
    p10 = evals_by_phase.get(10, {})
    p11 = evals_by_phase.get(11, {})

    # --- Rule 1: Phase 15 breakthrough -> CI deep-dive ---
    if p15.get("outcome") in ("BREAKTHROUGH", "SIGNIFICANT_CI_RESULT"):
        recommended.append({
            "phase": 16,
            "script": "phase16_ci_fisher_hybrid.py",
            "priority": priority_counter,
            "rationale": (
                "Phase 15 shows a significant class-incremental result. "
                "Phase 16 scales this with a Fisher-information hybrid regulariser on CIFAR-100."
            ),
            "estimated_hours": 24,
        })
        priority_counter += 1

    # --- Rule 2: Phase 10/11 strong -> task-incremental scale-up ---
    p10_strong = p10.get("outcome") in ("OUTCOME_B_MET",)
    p11_strong = p11.get("outcome") in ("PENALTY_DOMINANT", "GOVERNOR_ESSENTIAL")
    if p10_strong or p11_strong:
        if not any(r["phase"] == 16 for r in recommended):
            recommended.append({
                "phase": 16,
                "script": "phase16_cifar100_scaleup.py",
                "priority": priority_counter,
                "rationale": (
                    "Phase 10 demonstrates significant TCL advantage. "
                    "CIFAR-100 scale-up validates generality."
                ),
                "estimated_hours": 24,
            })
            priority_counter += 1

        recommended.append({
            "phase": 17,
            "script": "phase17_tinyimagenet.py",
            "priority": priority_counter,
            "rationale": (
                "If Phase 16 (CIFAR-100) succeeds, TinyImageNet provides a third "
                "benchmark for a strong multi-dataset paper."
            ),
            "estimated_hours": 48,
        })
        priority_counter += 1

    # --- Rule 3: Autonomous research breakthrough ---
    ar_breakthroughs = [
        r for r in auto_result.get("results", [])
        if r.get("breakthrough_candidate", False) or r.get("significance", "") == "HIGH"
    ]
    if ar_breakthroughs:
        ar_dir = ar_breakthroughs[0].get("direction", "autonomous-discovered direction")
        recommended.append({
            "phase": 18,
            "script": "phase18_autonomous_followup.py",
            "priority": priority_counter,
            "rationale": (
                f"Autonomous research identified a potentially significant direction: "
                f"'{ar_dir}'. Phase 18 formalises this as a controlled experiment."
            ),
            "estimated_hours": 16,
        })
        priority_counter += 1

    # --- Fallback: nothing strong -> diagnostic ---
    if not recommended:
        recommended.append({
            "phase": 16,
            "script": "phase16_diagnostic_rerun.py",
            "priority": 1,
            "rationale": (
                "No strong signals from Queue 1. Queue 2 should re-examine "
                "hyperparameter sensitivity before attempting scale-up."
            ),
            "estimated_hours": 8,
        })

    return {
        "recommended_phases": recommended,
        "total_estimated_hours": sum(r["estimated_hours"] for r in recommended),
        "generated_at": _ts(),
    }


# ===========================================================================
# arXiv readiness assessment
# ===========================================================================

def _assess_arxiv_readiness(
    phase_evals: list[dict], paper_dir: Path
) -> dict[str, Any]:
    """
    Checks whether the paper is ready for arXiv submission.
    """
    missing: list[str] = []

    # Required PDF
    pdf = paper_dir / "main.pdf"
    if not pdf.exists():
        missing.append("paper/main.pdf not found — run LaTeX build")

    # Required TeX source sections
    required_sections = [
        "abstract.tex",
        "s1_introduction.tex",
        "s2_background.tex",
        "s3_method.tex",
        "s4_experiments.tex",
    ]
    for sec in required_sections:
        if not (paper_dir / sec).exists():
            missing.append(f"paper/{sec} missing")

    # Phase data coverage
    evals_by_phase = {e["phase"]: e for e in phase_evals}
    for required_phase in (10, 11):
        if required_phase not in evals_by_phase:
            missing.append(f"Phase {required_phase} results absent — required for experiments section")
        elif evals_by_phase[required_phase].get("outcome") == "EVAL_ERROR":
            missing.append(
                f"Phase {required_phase} evaluation failed — verify JSON and re-run"
            )

    # Build readiness verdict
    ready = len(missing) == 0

    if ready:
        recommendation = (
            "All required components present. Run final proofread, then submit to arXiv. "
            "Suggested categories: cs.LG, cs.NE."
        )
    elif len(missing) <= 2:
        recommendation = (
            f"Minor issues only ({len(missing)} item(s)). Address, rebuild PDF, then submit."
        )
    else:
        recommendation = (
            f"{len(missing)} items missing. Complete experimental sections before arXiv submission."
        )

    return {
        "ready": ready,
        "missing": missing,
        "n_missing": len(missing),
        "recommendation": recommendation,
    }


# ===========================================================================
# Report text formatter
# ===========================================================================

def _format_report_txt(
    report: dict,
    phase_evals: list[dict],
    q2: dict,
    arxiv: dict,
    ar: dict,
    asc: dict,
) -> str:
    """Render the machine report dict as a human-readable text document."""

    _SIG_LABEL = {"HIGH": "[HIGH]", "MEDIUM": "[MED] ", "LOW": "[LOW] ", "UNKNOWN": "[???] "}

    lines: list[str] = []

    def h1(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("=" * len(title))

    def h2(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    def wrap(text: str, indent: int = 2) -> None:
        prefix = " " * indent
        for ln in textwrap.wrap(text, width=78, initial_indent=prefix, subsequent_indent=prefix):
            lines.append(ln)

    # Title
    h1("TAR POST-QUEUE EVALUATION REPORT")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Phases evaluated: {sorted(report['phases_evaluated'])}")
    lines.append("=" * 42)

    # Phase outcomes
    h2("PHASE OUTCOMES")
    if not phase_evals:
        lines.append("  No phase data found.")
    else:
        col_w = 38
        for ev in sorted(phase_evals, key=lambda e: e["phase"]):
            ph = ev["phase"]
            outcome = ev["outcome"]
            det = ev.get("details", {})

            # Build a compact inline stats string
            stat_parts: list[str] = []
            for k in ("p_vs_sgd", "p_value", "p_penalty_vs_sgd", "best_p_vs_ewc"):
                v = det.get(k)
                if v is not None:
                    stat_parts.append(f"p={v:.3f}")
                    break
            for k in ("d_vs_sgd", "effect_size"):
                v = det.get(k)
                if v is not None:
                    stat_parts.append(f"d={v:.2f}")
                    break

            stat_str = f" ({', '.join(stat_parts)})" if stat_parts else ""
            label = f"Phase {ph} ({_phase_short_name(ph)}):"
            lines.append(f"  {label:<{col_w}} {outcome}{stat_str}")

    # Key findings
    h2("KEY FINDINGS (ranked by significance)")
    key_findings: list[dict] = report.get("key_findings", [])
    if not key_findings:
        lines.append("  No findings extracted.")
    else:
        for i, kf in enumerate(key_findings, 1):
            sig = _SIG_LABEL.get(kf.get("significance", "UNKNOWN"), "[???] ")
            lines.append(f"  {i}. {sig} Phase {kf['phase']}: {kf['key_finding']}")
            wrap(f"Recommendation: {kf['recommendation']}", indent=8)

    # Recommended Queue 2
    h2("RECOMMENDED QUEUE 2")
    if not q2.get("recommended_phases"):
        lines.append("  No recommendations generated.")
    else:
        lines.append(f"  Total estimated GPU-hours: ~{q2['total_estimated_hours']}h")
        lines.append("")
        for rec in q2["recommended_phases"]:
            lines.append(
                f"  Priority {rec['priority']}: Phase {rec['phase']} — {rec['script']}"
                f"  (~{rec['estimated_hours']}h)"
            )
            wrap(f"Rationale: {rec['rationale']}", indent=6)

    # arXiv readiness
    h2("ARXIV READINESS")
    status_str = "READY" if arxiv["ready"] else "NOT READY"
    lines.append(f"  Status: {status_str}  ({arxiv['n_missing']} issue(s))")
    if arxiv["missing"]:
        lines.append("  Missing / issues:")
        for item in arxiv["missing"]:
            lines.append(f"    - {item}")
    wrap(arxiv["recommendation"], indent=2)

    # Autonomous research
    h2("AUTONOMOUS RESEARCH STATUS")
    if ar.get("found"):
        results = ar.get("results", [])
        breakthroughs = [r for r in results if r.get("breakthrough_candidate")]
        lines.append(
            f"  Research artefacts found: {len(results)} result file(s). "
            f"Breakthrough candidates: {len(breakthroughs)}."
        )
        if ar.get("preregistration"):
            hyp = ar["preregistration"].get("hypothesis", "")
            if hyp:
                lines.append(f"  Pre-registered hypothesis: {hyp}")
    else:
        lines.append("  No autonomous research artefacts found.")

    # ASC model
    h2("ASC MODEL STATUS")
    if asc["exists"]:
        lines.append(f"  Fine-tuned ASC model found at: {asc['path']}")
        if asc.get("details"):
            for k, v in asc["details"].items():
                lines.append(f"    {k}: {v}")
    else:
        lines.append("  No fine-tuned ASC model found at training_artifacts/asc_finetune/.")
        lines.append("  The base ASC (if any) will be used for Queue 2 guidance.")

    lines.append("")
    lines.append("— end of report —")
    lines.append("")
    return "\n".join(lines)


def _phase_short_name(n: int) -> str:
    names = {
        10: "4-way baseline",
        11: "ablation",
        12: "EWC sweep",
        13: "SI dynamics",
        14: "paper audit",
        15: "class-incremental",
    }
    return names.get(n, f"phase-{n}")


# ===========================================================================
# Main report generator
# ===========================================================================

def _generate_report(workspace: Path) -> dict:
    """
    Orchestrates all loading, evaluation, and report writing.
    Returns the machine-readable report dict.
    """
    out_dir = workspace / "tar_state" / "post_queue_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    all_phases = _load_all_phases(workspace)
    auto_result = _load_autonomous_research(workspace)
    asc = _load_asc_model(workspace)

    # ---- Evaluate each phase ----
    phase_evals: list[dict] = []
    for phase_num, data in sorted(all_phases.items()):
        evaluator = _PHASE_EVALUATORS.get(phase_num)
        if evaluator:
            ev = evaluator(data)
        else:
            # Generic pass-through for unknown phases
            ev = {
                "phase": phase_num,
                "outcome": data.get("outcome", data.get("verdict", "UNKNOWN")),
                "significance": "UNKNOWN",
                "key_finding": f"No evaluator registered for Phase {phase_num}.",
                "recommendation": "Review manually.",
                "details": {},
            }
        phase_evals.append(ev)

    # ---- Key findings (sort by significance) ----
    sig_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}
    key_findings = sorted(
        [
            {
                "phase": ev["phase"],
                "significance": ev["significance"],
                "key_finding": ev["key_finding"],
                "recommendation": ev["recommendation"],
            }
            for ev in phase_evals
        ],
        key=lambda x: (sig_order.get(x["significance"], 9), x["phase"]),
    )

    # ---- Queue 2 recommendation ----
    q2 = _recommend_queue2(phase_evals, auto_result)

    # ---- arXiv readiness ----
    paper_dir = workspace / "paper"
    arxiv = _assess_arxiv_readiness(phase_evals, paper_dir)

    # ---- Assemble machine report ----
    report = {
        "generated_at": _ts(),
        "workspace": str(workspace),
        "phases_evaluated": sorted(all_phases.keys()),
        "key_findings": key_findings,
        "phase_details": {str(ev["phase"]): ev for ev in phase_evals},
        "queue2_recommendation": q2,
        "arxiv_readiness": arxiv,
        "autonomous_research_status": {
            "found": auto_result.get("found", False),
            "n_result_files": len(auto_result.get("results", [])),
            "preregistration_present": auto_result.get("preregistration") is not None,
            "breakthrough_candidates": [
                r for r in auto_result.get("results", [])
                if r.get("breakthrough_candidate")
            ],
        },
        "asc_model_status": asc,
    }

    # ---- Write machine report ----
    report_json_path = out_dir / "report.json"
    report_json_path.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"[eval] report.json -> {report_json_path}")

    # ---- Write human report ----
    report_txt = _format_report_txt(report, phase_evals, q2, arxiv, auto_result, asc)
    report_txt_path = out_dir / "report.txt"
    report_txt_path.write_text(report_txt, encoding="utf-8")
    print(f"[eval] report.txt  -> {report_txt_path}")
    print()
    print(report_txt)

    # ---- Write queue2_config.json ----
    q2_config = {
        "generated_at": _ts(),
        "source_report": str(report_json_path),
        "phases": q2["recommended_phases"],
        "total_estimated_hours": q2["total_estimated_hours"],
        "arxiv_ready": arxiv["ready"],
        "notes": (
            "Auto-generated by tar_post_queue_eval.py. "
            "Pass this file to run_queue2.py with --config queue2_config.json"
        ),
    }
    q2_config_path = out_dir / "queue2_config.json"
    q2_config_path.write_text(
        json.dumps(q2_config, indent=2, default=str), encoding="utf-8"
    )
    print(f"[eval] queue2_config.json -> {q2_config_path}")

    return report


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="TAR Post-Queue Evaluation — generates decision report after all phases complete."
    )
    parser.add_argument(
        "--workspace",
        default=str(resolve_workspace(_REPO)),
        help=(
            "Root workspace directory (default: TAR_WORKSPACE env var, "
            "or the directory containing this script)."
        ),
    )
    args = parser.parse_args()

    workspace = ensure_workspace_layout(Path(args.workspace).resolve(), repo_root=_REPO)
    if not workspace.is_dir():
        print(f"[eval] ERROR: workspace does not exist: {workspace}", file=sys.stderr)
        return 1

    print(f"[eval] TAR Post-Queue Evaluation")
    print(f"[eval] workspace  = {workspace}")
    print(f"[eval] started at = {_ts()}")
    print()

    try:
        _generate_report(workspace)
    except Exception as exc:
        print(f"[eval] FATAL: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
