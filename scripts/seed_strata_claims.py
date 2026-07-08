"""
Seed STRATA (github.com/christophergardner-star/CF) method-hypotheses as first-class
TAR research frontiers, so the autonomous solution loop investigates them.

CONTEXT. STRATA is a SIBLING continual-learning project (the same author). Its rigorous,
pre-registered core is a forgetting-ATTRIBUTION methodology (the "trichotomy": substrate
overlap / timescale collision / basis correlation) — NOT the thermodynamic "organism",
which STRATA itself informally FALSIFIED (a classic substrate beat it), mirroring TAR's own
falsification of TCL. That attribution methodology is a model-agnostic set of numpy probes
and is the highest-value STRATA integration, but it is a code PORT (probes run on model
checkpoints), not a frontier — see docs / the follow-on task.

WHAT THIS SCRIPT DOES (fully wired, no code changes needed): it seeds STRATA's testable
METHOD-hypotheses as `tar_anomaly::` gaps + preregistered joint criteria, exactly like
scripts/seed_si_anomaly.py. The director then registers each as a frontier, the widened
proposer (NOT the tcl-only path) composes a candidate from the whole CL design space, and
the incorruptible tester screens it (n=5, kill-only) → survivors escalate to n=20 confirm.

HONESTY (STRATA is the author's OWN project — TAR must NOT self-validate it):
  * These frontiers test STRATA's *hypotheses*, against REAL external baselines auto-sourced
    by frontier_problem_from_gap (ewc/si/replay/agem/lwf). TAR does NOT run STRATA's code and
    a STRATA-derived candidate is source-tagged tar_novel (excluded from the NoveltyGate bar).
  * Every claim carries a min_seed_acc COLLAPSE guard: a method may not "prevent forgetting"
    by learning nothing (the exact anti-cheat the SI seed introduced).
  * inventory records are paper_citation_allowed=False — an OPEN target, never a citable win.
  * Seeding creates a FALSIFIABLE target; the daemon routinely returns NULL/ADVERSE/COLLAPSED.

Three writes per claim (idempotent, upsert): honest_evidence_inventory.json, the
literature_graph.db research_gaps row, and the autonomous_research/preregistration.json
joint criterion (joined to director-minted specs by frontier_problem_id).

SAFETY: dry-run by default (writes NOTHING). --apply only when you intend these to become
live autonomous targets. Composite scores are set BELOW the SI-stability anomaly (0.9075) so
TAR's own grounded anomaly stays the #1 CL target and the STRATA claims queue right behind it
(only the single top gap per domain is worked per cycle — reprioritize by editing the scores).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# Real Split-CIFAR-10 benchmark id used across the suite (task-incremental; methods actually
# learn here, unlike class-incremental CIFAR-100 which collapses to chance).
_BENCHMARK_ID = "benchmark:402cf4341499b795"

# STRATA's own reported evidence files are NOT in TAR's tar_state; we cite the STRATA repo so
# the inventory record is honest about provenance (external claim under test, not a TAR result).
_STRATA_REPO = "github.com/christophergardner-star/CF (STRATA)"

# Each claim: a STRATA method-hypothesis expressed as a falsifiable TAR frontier.
# criteria keys are ONLY those the orchestrator recognizes
# ({max_delta,max_p,min_d,max_forgetting_std,min_mean_acc,min_seed_acc,
#   variance_ratio_alpha,si_forgetting_std,si_n_seeds}); min_seed_acc is the anti-cheat.
_CLAIMS = [
    {
        "gap_id": "tar_anomaly::strata_replayfree_forgetting_prevention",
        "hypothesis_name": "strata_replayfree_forgetting_prevention",
        "inventory_id": "strata_replayfree_claim",
        "title": "Replay-free continual learning can prevent catastrophic forgetting",
        "method_names": ["strata"],
        "impact": 0.92, "novelty": 0.90, "tractability": 0.85,   # composite ~0.896 (< SI 0.9075)
        "statement": (
            "STRATA (replay-free, label-light CL) claims catastrophic forgetting can be prevented "
            "WITHOUT a replay buffer. TARGET on Split-CIFAR-10: a replay-free candidate that "
            "significantly reduces forgetting vs the baseline AND genuinely learns (no collapse), "
            "tested against real external baselines incl. experience replay. JOINT criterion: "
            "reduce forgetting (max_delta<=-0.01, p<=0.05, d>=0.5), mean-acc >= 0.65, and NO seed "
            "below 0.55 acc (else 'no forgetting' is an artifact of learning nothing)."
        ),
        "criteria": {
            "max_delta": -0.01, "max_p": 0.05, "min_d": 0.5,
            "min_mean_acc": 0.65, "min_seed_acc": 0.55, "max_forgetting_std": 0.02,
        },
    },
    {
        "gap_id": "tar_anomaly::strata_sparse_routing_no_interference",
        "hypothesis_name": "strata_sparse_routing_no_interference",
        "inventory_id": "strata_sparse_routing_claim",
        "title": "Sparse routing prevents task interference",
        "method_names": ["sparse_routing"],
        "impact": 0.90, "novelty": 0.90, "tractability": 0.82,   # composite ~0.880
        "statement": (
            "STRATA claims sparse (kWTA) routing shields sequential tasks from interference. "
            "TARGET on Split-CIFAR-10: a parameter-isolation / architectural candidate that "
            "reduces forgetting AND is cross-seed stable AND genuinely learns, vs real external "
            "baselines. JOINT criterion: reduce forgetting (max_delta<=-0.01, p<=0.05, d>=0.5), "
            "cross-seed forgetting-std <= 0.015, mean-acc >= 0.65, NO seed below 0.55 acc."
        ),
        "criteria": {
            "max_delta": -0.01, "max_p": 0.05, "min_d": 0.5,
            "min_mean_acc": 0.65, "min_seed_acc": 0.55, "max_forgetting_std": 0.015,
        },
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_inventory(ws: Path, apply: bool) -> list[str]:
    out: list[str] = []
    path = ws / "tar_state" / "honest_evidence_inventory.json"
    if not path.exists():
        return [f"  [inventory] SKIP: {path} not found"]
    inv = json.loads(path.read_text(encoding="utf-8"))
    results = inv.get("results", [])
    changed = False
    for c in _CLAIMS:
        if any(isinstance(r, dict) and r.get("experiment_id") == c["inventory_id"] for r in results):
            out.append(f"  [inventory] {c['inventory_id']} already present (skip)")
            continue
        results.append({
            "experiment_id": c["inventory_id"],
            "label": f"STRATA claim under test: {c['title']}",
            "dataset": "split_cifar10",
            "method_comparison": "strata_hypothesis_vs_external_baselines",
            "n_seeds": 5,
            "trust_tier": "external_claim_under_test",
            "honest_verdict": "CLAIM_UNDER_TEST",
            "honest_verdict_detail": c["statement"],
            "evidence_sources": [_STRATA_REPO],
            "paper_citation_allowed": False,
            "paper_citation_caveat": ("External sibling-project claim seeded as an OPEN falsifiable "
                                      "target; NOT a TAR result and never citable as one."),
        })
        changed = True
        out.append(f"  [inventory] {'APPLIED: added' if apply else 'DRY-RUN: would add'} {c['inventory_id']}")
    if apply and changed:
        inv["results"] = results
        path.write_text(json.dumps(inv, indent=2), encoding="utf-8")
    return out


def seed_gaps(ws: Path, apply: bool) -> list[str]:
    from literature.knowledge_graph import LiteratureKnowledgeGraph
    from literature.schemas import ResearchGap
    out: list[str] = []
    db = ws / "tar_state" / "literature" / "literature_graph.db"
    if not db.exists():
        return [f"  [gap] SKIP: {db} not found"]
    g = LiteratureKnowledgeGraph(str(db)) if apply else None
    try:
        for c in _CLAIMS:
            gap = ResearchGap(
                gap_id=c["gap_id"],
                gap_type="theoretical",
                title=c["title"],
                description=c["statement"],
                domain="continual_learning",
                benchmark_id=_BENCHMARK_ID,
                method_names=list(c["method_names"]),
                impact_score=c["impact"], novelty_score=c["novelty"],
                tractability_score=c["tractability"],
            )
            gap.recompute_composite()
            if apply:
                g.upsert_gap(gap)
                out.append(f"  [gap] APPLIED: upserted {c['gap_id']} (composite={round(gap.composite_score,4)})")
            else:
                out.append(f"  [gap] DRY-RUN: would upsert {c['gap_id']} (composite={round(gap.composite_score,4)})")
    finally:
        if g is not None:
            g.close()
    return out


def seed_prereg(ws: Path, apply: bool) -> list[str]:
    from tar_frontier import _frontier_slug
    out: list[str] = []
    path = ws / "tar_state" / "autonomous_research" / "preregistration.json"
    prereg = {}
    if path.exists():
        try:
            prereg = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            prereg = {}
    hyps = prereg.get("hypotheses", [])
    changed = False
    for c in _CLAIMS:
        # Deterministic join key — the ONLY stable link to director-minted specs (see seed_si_anomaly).
        fpid = f"fp-gap-{_frontier_slug(c['gap_id'])}"
        crit = dict(c["criteria"])
        existing = next((h for h in hyps if isinstance(h, dict) and h.get("name") == c["hypothesis_name"]), None)
        if existing is not None:
            if existing.get("criteria") == crit and existing.get("frontier_problem_id") == fpid:
                out.append(f"  [prereg] {c['hypothesis_name']} up to date (skip)")
                continue
            if apply:
                existing["criteria"] = crit
                existing["frontier_problem_id"] = fpid
                existing["updated_at"] = _now()
                changed = True
            out.append(f"  [prereg] {'APPLIED: updated' if apply else 'DRY-RUN: would update'} {c['hypothesis_name']}")
            continue
        hyps.append({
            "name": c["hypothesis_name"],
            "registered_at": _now(),
            "prediction": c["statement"],
            "criteria": crit,
            "frontier_problem_id": fpid,
            "author_paper_id": "",
            "source": "strata_external_claim",
        })
        changed = True
        out.append(f"  [prereg] {'APPLIED: added' if apply else 'DRY-RUN: would add'} {c['hypothesis_name']}")
    if apply and changed:
        prereg["hypotheses"] = hyps
        prereg.setdefault("registered_at", _now())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prereg, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed STRATA method-hypotheses as TAR frontiers.")
    ap.add_argument("--workspace", default=r"E:\TAR\Thermodynamic-Continual-Learning-delivered")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args()
    ws = Path(args.workspace)
    print(f"[seed_strata_claims] {'APPLY' if args.apply else 'DRY-RUN'} on {ws}")
    for line in seed_inventory(ws, args.apply):
        print(line)
    for line in seed_gaps(ws, args.apply):
        print(line)
    for line in seed_prereg(ws, args.apply):
        print(line)
    if not args.apply:
        print("[seed_strata_claims] DRY-RUN complete. Re-run with --apply to make these live "
              "autonomous frontiers (composites are below the SI anomaly, so they queue behind it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
