"""
Seed the SI-stability anomaly as a first-class research object (solution-loop Phase 2).

TAR generated this anomaly from its own honest results: on Split-CIFAR-10 (same seeds)
Synaptic Intelligence is EITHER the best + by far the most stable method in the suite
(forgetting 0.0474 +/- 0.00753, 3-9x tighter cross-seed std than any other) OR exactly at
chance (acc 0.500, std 0.0), controlled solely by the damping constant c in {0.01} vs
{0.1, 0.5}. Nothing in the suite explains the cliff, and honest_evidence_inventory.json has
NO SI record. This script makes the anomaly a research target the widened proposer can
attack, with a JOINT success criterion that forbids "stability by not learning".

Writes THREE things (idempotent):
  1. honest_evidence_inventory.json  -> an SI anomaly record (evidence of both sides)
  2. literature_graph.db research_gaps -> the anomaly gap (top composite so the director picks it)
  3. autonomous_research/preregistration.json -> the joint-criterion prereg entry

SAFETY: dry-run by default (writes NOTHING). Run with --apply ONLY at activation, AFTER the
daemon has been restarted onto the new solution-loop code, so the WIDENED proposer (not the
old tcl-only path) consumes the gap. Numbers are cited to real comparison files; verify them
against the sources before --apply.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# Split-CIFAR-10 benchmark id used across the suite (Continual Learning On Split-CIFAR-10).
_BENCHMARK_ID = "benchmark:402cf4341499b795"
_GAP_ID = "tar_anomaly::si_stability_without_collapse"

# Evidence (cite: tar_state/comparisons/phase18_tcl_canonical_fullprotocol__20260605T093223Z.json;
# phase10_controlled_rerun_20260509T132155Z.json; phase13_si_sweep__20260512T061410Z.json)
_SI_FORGETTING_MEAN = 0.0474
_SI_FORGETTING_STD = 0.00753
_SI_ACC_MEAN = 0.79412
_COLLAPSE_MIN_SEED_ACC = 0.55   # phase13 collapse_threshold

_ANOMALY_STATEMENT = (
    "SI-stability-without-collapse: On Split-CIFAR-10 (ResNet-18, task-incremental, 5 seeds) "
    "Synaptic Intelligence at c=0.01 is the best AND by far the most stable method "
    f"(forgetting {_SI_FORGETTING_MEAN}+/-{_SI_FORGETTING_STD}, cross-seed std 3-9x tighter than "
    "EWC/SGD/TCL; beats TCL on all 5 seeds, d=2.5-4.2), yet at c in {0.1,0.5} it collapses to "
    "chance accuracy (0.500, std 0.0) on all seeds. No mechanism in the suite explains the cliff. "
    "TARGET: a method that achieves SI-level stability WITHOUT the collapse failure mode. "
    "A candidate must meet the JOINT criterion: forgetting-std <= SI's, mean-acc >= SI's, AND "
    "no seed below 0.55 accuracy (else the 'stability' is an artifact of learning nothing)."
)

_SI_N_SEEDS = 5   # SI reference seed count (for the F-test degrees of freedom)

_JOINT_CRITERIA = {
    # Stability is tested by a real one-sided F-test (variance_ratio): is the candidate's
    # cross-seed forgetting variance NOT significantly greater than SI's? The absolute
    # max_forgetting_std is kept only as the scipy-unavailable fallback.
    "variance_ratio_alpha": 0.05,
    "si_forgetting_std": _SI_FORGETTING_STD,
    "si_n_seeds": _SI_N_SEEDS,
    "max_forgetting_std": _SI_FORGETTING_STD,   # fallback bound if the F-test can't run
    "min_mean_acc": _SI_ACC_MEAN,
    "min_seed_acc": _COLLAPSE_MIN_SEED_ACC,
    "max_delta": -0.01,   # must reduce forgetting vs the TCL baseline used by _build_result
    "max_p": 0.05,
    "min_d": 0.5,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_inventory(ws: Path, apply: bool) -> str:
    path = ws / "tar_state" / "honest_evidence_inventory.json"
    if not path.exists():
        return f"  [inventory] SKIP: {path} not found"
    inv = json.loads(path.read_text(encoding="utf-8"))
    results = inv.get("results", [])
    if any(isinstance(r, dict) and r.get("experiment_id") == "si_stability_anomaly" for r in results):
        return "  [inventory] already present (idempotent skip)"
    record = {
        "experiment_id": "si_stability_anomaly",
        "label": "SI stability-vs-collapse anomaly (Split-CIFAR-10)",
        "dataset": "split_cifar10",
        "method_comparison": "si_c_sweep",
        "n_seeds": 5,
        "trust_tier": "trusted_manual_controlled",
        "honest_verdict": "ANOMALY_OPEN",
        "honest_verdict_detail": _ANOMALY_STATEMENT,
        "evidence_sources": [
            "comparisons/phase18_tcl_canonical_fullprotocol__20260605T093223Z.json",
            "comparisons/phase10_controlled_rerun_20260509T132155Z.json",
            "comparisons/phase13_si_sweep__20260512T061410Z.json",
        ],
        "paper_citation_allowed": False,
        "paper_citation_caveat": "Open anomaly; not a result to cite. Seeds a solution-finding target.",
    }
    if apply:
        results.append(record)
        inv["results"] = results
        path.write_text(json.dumps(inv, indent=2), encoding="utf-8")
        return "  [inventory] APPLIED: added si_stability_anomaly record"
    return "  [inventory] DRY-RUN: would add si_stability_anomaly record"


def seed_gap(ws: Path, apply: bool) -> str:
    from literature.knowledge_graph import LiteratureKnowledgeGraph
    from literature.schemas import ResearchGap
    db = ws / "tar_state" / "literature" / "literature_graph.db"
    if not db.exists():
        return f"  [gap] SKIP: {db} not found"
    gap = ResearchGap(
        gap_id=_GAP_ID,
        gap_type="theoretical",   # empirical anomaly without a mechanistic explanation
        title="SI-level stability without the collapse failure mode",
        description=_ANOMALY_STATEMENT,
        domain="continual_learning",
        benchmark_id=_BENCHMARK_ID,
        method_names=["si"],
        impact_score=0.95, novelty_score=0.90, tractability_score=0.85,
    )
    gap.recompute_composite()   # ~0.91 -> should be top of the open CL gaps
    if not apply:
        return (f"  [gap] DRY-RUN: would upsert {_GAP_ID} "
                f"(composite={gap.composite_score}, gap_type=theoretical)")
    g = LiteratureKnowledgeGraph(str(db))
    try:
        g.upsert_gap(gap)
    finally:
        g.close()
    return f"  [gap] APPLIED: upserted {_GAP_ID} (composite={gap.composite_score})"


def seed_prereg(ws: Path, apply: bool) -> str:
    path = ws / "tar_state" / "autonomous_research" / "preregistration.json"
    prereg = {}
    if path.exists():
        try:
            prereg = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            prereg = {}
    hyps = prereg.get("hypotheses", [])
    # HANDSHAKE: the orchestrator loads these criteria for an experiment by matching
    # spec.id / spec.name / spec.frontier_problem_id (tar_experiment_orchestrator
    # _load_prereg_criteria). The director mints DYNAMIC spec ids/names for the gap
    # probe, so the only stable join key is frontier_problem_id, which the director
    # derives deterministically from the gap_id as f"fp-gap-{_frontier_slug(gap_id)}"
    # (tar_frontier.frontier_problem_from_gap). Reuse the SAME slug fn so the key can
    # never drift; without this the joint collapse-guard criteria silently never load.
    from tar_frontier import _frontier_slug
    _frontier_pid = f"fp-gap-{_frontier_slug(_GAP_ID)}"
    # UPSERT (not skip): refresh criteria + join key if the hypothesis already exists, so
    # re-running --apply picks up criteria changes (e.g. the variance-ratio keys).
    for h in hyps:
        if isinstance(h, dict) and h.get("name") == "si_stability_without_collapse":
            if (h.get("criteria") == dict(_JOINT_CRITERIA)
                    and h.get("frontier_problem_id") == _frontier_pid):
                return "  [prereg] already up to date (idempotent skip)"
            if apply:
                h["criteria"] = dict(_JOINT_CRITERIA)
                h["frontier_problem_id"] = _frontier_pid
                h["updated_at"] = _now()
                path.write_text(json.dumps(prereg, indent=2), encoding="utf-8")
                return "  [prereg] APPLIED: updated si_stability_without_collapse criteria + join key"
            return "  [prereg] DRY-RUN: would UPDATE existing si_stability_without_collapse criteria"
    entry = {
        "name": "si_stability_without_collapse",
        "registered_at": _now(),
        "prediction": ("A composed candidate can match SI's cross-seed stability "
                       "(forgetting-std <= 0.00753) at mean-acc >= 0.794 with no seed below 0.55 acc."),
        "criteria": dict(_JOINT_CRITERIA),
        "frontier_problem_id": _frontier_pid,   # deterministic join key (see above)
        "author_paper_id": "",
        "source": "solution_loop_anomaly",
    }
    if apply:
        hyps.append(entry)
        prereg["hypotheses"] = hyps
        prereg.setdefault("registered_at", _now())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prereg, indent=2), encoding="utf-8")
        return "  [prereg] APPLIED: added si_stability_without_collapse joint criterion"
    return f"  [prereg] DRY-RUN: would add joint criterion {json.dumps(_JOINT_CRITERIA)}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed the SI-stability anomaly (solution-loop Phase 2).")
    ap.add_argument("--workspace", default=r"E:\TAR\Thermodynamic-Continual-Learning-delivered")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args()
    ws = Path(args.workspace)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[seed_si_anomaly] {mode} on {ws}")
    print(seed_inventory(ws, args.apply))
    print(seed_gap(ws, args.apply))
    print(seed_prereg(ws, args.apply))
    if not args.apply:
        print("[seed_si_anomaly] DRY-RUN complete. Re-run with --apply AFTER restarting the daemon "
              "onto the solution-loop code (so the widened proposer consumes the gap).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
