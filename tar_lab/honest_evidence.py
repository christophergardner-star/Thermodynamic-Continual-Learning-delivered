"""
Honest-evidence helpers (TRUTH-LOCK public surface).
====================================================

Single source of truth for mapping tar_state/honest_evidence_inventory.json
onto every public-facing surface (website research.json, dashboard
/api/breakthroughs, post-queue report).  The inventory is the ONLY place a
public evidence_strength may come from; anything without an inventory record
is "none"/unverified — never an invented "moderate".

Verdict scale (from the inventory):
    PUBLICATION_ALLOWED > DIRECTIONAL > EXPLORATION_GRADE > FALSIFIED
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VERDICT_RANK: dict[str, int] = {
    "PUBLICATION_ALLOWED": 3,
    "DIRECTIONAL": 2,
    "EXPLORATION_GRADE": 1,
    "FALSIFIED": 0,
}

VERDICT_TO_STRENGTH: dict[str, str] = {
    "PUBLICATION_ALLOWED": "strong",
    "DIRECTIONAL": "directional",
    "EXPLORATION_GRADE": "weak",
    "FALSIFIED": "none",
}


def load_inventory(workspace: Path) -> dict[str, Any]:
    """Load honest_evidence_inventory.json; {} on any error (fail-quiet read)."""
    try:
        path = Path(workspace) / "tar_state" / "honest_evidence_inventory.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def inventory_records(workspace: Path) -> list[dict[str, Any]]:
    inv = load_inventory(workspace)
    results = inv.get("results", [])
    return [rec for rec in results if isinstance(rec, dict)]


def experiment_verdict_map(workspace: Path) -> dict[str, dict[str, Any]]:
    """Map experiment_id -> {verdict, detail, citation_caveat} from the inventory."""
    out: dict[str, dict[str, Any]] = {}
    for rec in inventory_records(workspace):
        exp_id = str(rec.get("experiment_id", "") or "")
        verdict = str(rec.get("honest_verdict", "") or "").upper()
        if not exp_id or verdict not in VERDICT_RANK:
            continue
        out[exp_id] = {
            "verdict": verdict,
            "detail": str(rec.get("honest_verdict_detail", "") or ""),
            "citation_caveat": rec.get("paper_citation_caveat"),
        }
    return out


def frontier_best_verdict_map(workspace: Path) -> dict[str, str]:
    """Map frontier_problem_id (falling back to experiment_id) -> best honest verdict."""
    best: dict[str, str] = {}
    for rec in inventory_records(workspace):
        fid = str(rec.get("frontier_problem_id", "") or rec.get("experiment_id", "") or "")
        verdict = str(rec.get("honest_verdict", "") or "").upper()
        if not fid or verdict not in VERDICT_RANK:
            continue
        prev = best.get(fid, "")
        if not prev or VERDICT_RANK.get(verdict, -1) > VERDICT_RANK.get(prev, -1):
            best[fid] = verdict
    return best


def honest_strength_for_frontier(workspace: Path, frontier_id: str) -> str:
    """Inventory-backed evidence_strength for a frontier; 'none' when unrecorded."""
    verdict = frontier_best_verdict_map(workspace).get(str(frontier_id or ""), "")
    return VERDICT_TO_STRENGTH.get(verdict, "none")


def match_inventory_record(
    candidate_key: str, verdicts: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """
    Best-effort match of a public artifact id onto an inventory experiment_id.

    Handles the id schemes used by the public surfaces:
      "ar-high_penalty_conservative" -> "hpc_autonomous_high_penalty_conservative"
      "phase17"                      -> any experiment_id starting "phase17"
      exact ids pass through unchanged.
    Returns the matched record dict (with 'verdict') or None.
    """
    key = str(candidate_key or "").strip()
    if not key:
        return None
    if key in verdicts:
        return verdicts[key]
    if key.startswith("ar-"):
        alias = "hpc_autonomous_" + key[3:].replace("-", "_")
        if alias in verdicts:
            return verdicts[alias]
    # Prefix match (phaseNN -> phaseNN_*): a phase can hold several sub-comparisons
    # (e.g. phase10_*_tcl_vs_sgd PUBLICATION_ALLOWED and phase10_*_tcl_vs_ewc
    # DIRECTIONAL). We cannot tell which sub-comparison a public card's stats
    # belong to, so pick the LOWEST-ranked verdict — never attach a stronger
    # verdict than the weakest matching record justifies (anti-over-claim).
    matches = [rec for exp_id, rec in verdicts.items() if exp_id.startswith(key + "_") or exp_id == key]
    if not matches and key.startswith("phase"):
        matches = [rec for exp_id, rec in verdicts.items() if exp_id.startswith(key)]
    if not matches:
        return None
    return min(matches, key=lambda rec: VERDICT_RANK.get(rec.get("verdict", ""), 99))
