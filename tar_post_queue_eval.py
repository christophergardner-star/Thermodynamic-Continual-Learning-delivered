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
  tar_state/post_queue_eval/queue2_config.json — review artifact for bounded manifest planning

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
from tar_lab.result_artifacts import load_latest_phase_comparisons

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
    Load the latest canonical phase comparison for each phase number, falling
    back to legacy fixed-name files when no canonical append-only artifact has
    been registered yet.
    """
    return load_latest_phase_comparisons(workspace)


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

def _pairwise_block(data: dict, *keys: str) -> dict:
    """Return the first present pairwise sub-block from the canonical schema
    (data['pairwise'][key]) falling back to the legacy comparisons schema."""
    pairwise = data.get("pairwise", {})
    if isinstance(pairwise, dict):
        for key in keys:
            block = pairwise.get(key)
            if isinstance(block, dict) and block:
                return block
    return {}


def _eval_phase10(data: dict) -> dict:
    """
    Phase 10 — 4-way baseline comparison.
    Outcome B criteria: TCL vs SGD p<0.05, Cohen's d>0.5, all 5 seeds present.

    Canonical schema: data['pairwise']['sgd_baseline'|'ewc'] with
    {mean_delta, p_val, cohens_d, n_tcl_better}. The legacy
    data['comparisons']['tcl_vs_sgd'] schema is kept as a fallback — the old
    reader silently defaulted to p=1.0/d=0.0 on canonical files, reporting a
    false NULL for the system's single Bonferroni-significant result.
    """
    phase = 10
    try:
        comparisons = data.get("comparisons", {})
        tcl_vs_sgd = comparisons.get("tcl_vs_sgd", {}) or _pairwise_block(data, "sgd_baseline", "sgd")
        tcl_vs_ewc = comparisons.get("tcl_vs_ewc", {}) or _pairwise_block(data, "ewc")

        p_sgd = tcl_vs_sgd.get("p_value", tcl_vs_sgd.get("p_val", 1.0))
        d_sgd = abs(tcl_vs_sgd.get("effect_size", tcl_vs_sgd.get("cohens_d", 0.0)))
        n_seeds = data.get("n_seeds", data.get("seeds_completed", 0)) or len(data.get("seeds", []) or [])
        p_ewc = tcl_vs_ewc.get("p_value", tcl_vs_ewc.get("p_val", 1.0))

        # Direction guard (mirror phase11/12): TCL is "better" only when its
        # forgetting delta vs SGD is negative. Without this, a significant
        # WRONG-direction result (TCL worse) would be reported as OUTCOME_B_MET
        # "TCL significantly outperforms SGD" — a false positive. When no delta
        # is present, don't block (legacy comparisons schema had none).
        delta_sgd = tcl_vs_sgd.get("mean_delta")
        tcl_better = delta_sgd is None or float(delta_sgd) < 0

        outcome_b_met = (p_sgd < 0.05) and (d_sgd > 0.5) and (n_seeds >= 5) and tcl_better

        if outcome_b_met:
            outcome = "OUTCOME_B_MET"
            significance = "HIGH"
            key_finding = (
                f"TCL significantly outperforms SGD baseline "
                f"(p={p_sgd:.3f}, d={d_sgd:.2f}) across all {n_seeds} seeds."
            )
            recommendation = "Proceed to scale-up (CIFAR-100). Core result is publishable."
        elif p_sgd < 0.05 and tcl_better:
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
        # Canonical schema: data['pairwise'] holds full_tcl-vs-X blocks for
        # X in {sgd, governor_only, penalty_only}, each {mean_delta, p_val,
        # cohens_d, n_full_tcl_better}. The old reader looked for a
        # 'penalty_only_vs_sgd' block that does not exist in canonical files,
        # defaulted p=1.0 and concluded GOVERNOR_ESSENTIAL — the OPPOSITE of
        # the recorded finding (honest inventory: governor never fires,
        # penalty is the entire mechanism).
        full_vs_penalty = _pairwise_block(data, "penalty_only")
        full_vs_sgd = _pairwise_block(data, "sgd")
        verdict_key = str(data.get("verdict_key", "") or "").upper()

        p_full_vs_pen = full_vs_penalty.get("p_val", full_vs_penalty.get("p_value", 1.0))
        p_full_vs_sgd = full_vs_sgd.get("p_val", full_vs_sgd.get("p_value", 1.0))
        delta_full_vs_pen = full_vs_penalty.get("mean_delta")

        # Legacy fallback (pre-canonical files only)
        if not full_vs_penalty and not full_vs_sgd:
            ablation = data.get("ablation_results", data.get("results", {}))
            penalty_vs_sgd = ablation.get("penalty_only_vs_sgd", {})
            p_full_vs_pen = penalty_vs_sgd.get("p_value", 1.0)

        full_beats_penalty = (
            p_full_vs_pen < 0.05
            and (delta_full_vs_pen is None or float(delta_full_vs_pen) < 0)
        )

        if full_beats_penalty:
            outcome = "GOVERNOR_CONTRIBUTES"
            significance = "MEDIUM"
            key_finding = (
                f"Full TCL significantly beats penalty-only (p={p_full_vs_pen:.3f}, uncorrected) "
                f"— the governor adds measurable benefit beyond the penalty."
            )
            recommendation = "Report both components; verify against Bonferroni before any mechanistic claim."
        else:
            outcome = "PENALTY_SUFFICIENT"
            significance = "MEDIUM"
            key_finding = (
                f"Full TCL is NOT significantly better than penalty-only "
                f"(p={p_full_vs_pen:.3f}, uncorrected) — consistent with the honest "
                f"inventory finding that the penalty is the entire mechanism and the "
                f"governor never fires. Full-vs-SGD p={p_full_vs_sgd:.3f} (uncorrected)."
            )
            recommendation = (
                "Frame the penalty as the mechanism. Do not attribute improvement to the "
                "governor. Mechanistic ablation (pre-registered) is the confirmatory step."
            )

        return {
            "phase": phase,
            "outcome": outcome,
            "significance": significance,
            "key_finding": key_finding,
            "recommendation": recommendation,
            "details": {
                "p_full_vs_penalty_only": p_full_vs_pen,
                "p_full_vs_sgd": p_full_vs_sgd,
                "self_reported_verdict_key": verdict_key,
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
        # Canonical schema: data['pairwise_tcl_vs_ewc'] keyed by lambda value,
        # each {mean_delta, p_val, cohens_d, n_tcl_better}. Legacy
        # 'lambda_sweep'/'ewc_sweep' kept as fallback (old reader defaulted to
        # p=1.0 on canonical files).
        sweep = data.get("lambda_sweep", data.get("ewc_sweep", {}))
        if not sweep:
            canonical = data.get("pairwise_tcl_vs_ewc", {})
            if isinstance(canonical, dict) and canonical:
                sweep = {
                    lam: {
                        "p_value": block.get("p_val", block.get("p_value", 1.0)),
                        "mean_delta": block.get("mean_delta"),
                        "cohens_d": block.get("cohens_d"),
                    }
                    for lam, block in canonical.items()
                    if isinstance(block, dict)
                }
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
            delta = entry.get("mean_delta")
            tcl_direction_ok = delta is None or float(delta) < 0
            if p < 0.05 and tcl_direction_ok:
                tcl_beats_ewc_lambdas.append(lam)
            if p < best_p and tcl_direction_ok:
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
    p11_strong = p11.get("outcome") in (
        "PENALTY_DOMINANT", "GOVERNOR_ESSENTIAL",  # legacy keys
        "PENALTY_SUFFICIENT", "GOVERNOR_CONTRIBUTES",
    )
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
    lines.append("")
    wrap(
        "ADVISORY REPORT: outcomes below are heuristic labels, NOT verified "
        "claims. tar_state/honest_evidence_inventory.json is the statistical "
        "source of truth; where they disagree, the inventory wins. "
        "All p-values shown are uncorrected unless stated otherwise. "
        "Cite the statistics, not the labels.", indent=0,
    )

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
            for k in ("p_vs_sgd", "p_value", "p_full_vs_penalty_only", "p_penalty_vs_sgd", "best_p_vs_ewc"):
                v = det.get(k)
                if v is not None:
                    stat_parts.append(f"p={v:.3f} uncorr.")
                    break
            for k in ("d_vs_sgd", "effect_size"):
                v = det.get(k)
                if v is not None:
                    stat_parts.append(f"d={v:.2f}")
                    break

            stat_str = f" ({', '.join(stat_parts)})" if stat_parts else ""
            label = f"Phase {ph} ({_phase_short_name(ph)}):"
            lines.append(f"  {label:<{col_w}} {outcome}{stat_str}")
            honest = ev.get("honest_inventory") or []
            for rec in honest:
                lines.append(
                    f"  {'':<{col_w}} inventory: {rec.get('experiment_id')} -> "
                    f"{rec.get('honest_verdict')}"
                )

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
            # Generic pass-through for unknown phases.
            # TRUTH-LOCK: never echo the file's free-text 'verdict' (which may
            # embed stale, uncorrected statistical claims) as the outcome.
            # Report only the short verdict_key, explicitly marked advisory.
            verdict_key = str(data.get("verdict_key", "") or data.get("outcome", "") or "UNKNOWN")
            self_reported = str(data.get("verdict", "") or "")
            ev = {
                "phase": phase_num,
                "outcome": f"SELF_REPORTED:{verdict_key}",
                "significance": "UNKNOWN",
                "key_finding": (
                    f"No evaluator registered for Phase {phase_num}. The phase file's own "
                    f"label is advisory, not verified — cite the statistics, not the label."
                ),
                "recommendation": "Review manually against honest_evidence_inventory.json.",
                "details": {"self_reported_verdict": self_reported[:200]},
            }
        phase_evals.append(ev)

    # ---- Honest-inventory cross-check (source of truth) ----
    honest_by_phase: dict[int, list[dict]] = {}
    try:
        inv = _load_json(workspace / "tar_state" / "honest_evidence_inventory.json") or {}
        for rec in inv.get("results", []):
            if not isinstance(rec, dict):
                continue
            m = re.match(r"phase(\d+)", str(rec.get("experiment_id", "") or ""))
            if not m:
                continue
            honest_by_phase.setdefault(int(m.group(1)), []).append({
                "experiment_id": rec.get("experiment_id"),
                "honest_verdict": rec.get("honest_verdict"),
            })
    except Exception:
        honest_by_phase = {}
    for ev in phase_evals:
        honest = honest_by_phase.get(ev.get("phase"))
        if honest:
            ev["honest_inventory"] = honest

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
            "Legacy queue2 execution has been retired. Use this artifact for human review "
            "and bounded manifest planning instead."
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
