"""Phase 3.1a self-improvement: method-refinement PROPOSER (advisory).

Reads the outcome registry (Phase 1.3) + experiment archive and, for under-performing or
failed method x dataset combinations, proposes TARGETED variants — e.g. "scale the penalty
strength", "add LR annealing", "address the recorded failure mode". The proposals are a
durable, inspectable JSON registry: tar_state/method_refinement/variant_proposals.json.

This module is deliberately PROPOSE-ONLY and APPEND-ONLY-SAFE:
- It generates NO method code, runs NO experiment, and adopts NOTHING.
- It NEVER touches a canonical algorithm or the knowledge graph (read-only on outcomes).
- Each proposal records that any future actuation must (a) create a NEW variant file via
  append_only_guard.guarded_create_new (never overwrite the parent or a canonical method),
  (b) run under the RAIL-3 manifest + sandbox, (c) be pre-registered with multiple-
  comparisons correction (no variant-shopping), and (d) be human-approved before adoption.
Off-switch: tar_state/method_refinement.disabled.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_REL = Path("method_refinement") / "variant_proposals.json"
DISABLE_FLAG = "method_refinement.disabled"

_HIGH_FORGETTING = 0.10           # combos above this are candidates for a regularisation tweak
_MAX_PROPOSALS_PER_COMBO = 2

_ACTUATION_RAILS = (
    "Advisory only — NOT generated, NOT run, NOT adopted. Any future actuation MUST: "
    "create a NEW variant file via append_only_guard (never overwrite the parent or any "
    "canonical method); run under the RAIL-3 manifest + sandbox; be pre-registered with "
    "multiple-comparisons correction; and be human-approved (BETTER on >=2 datasets) "
    "before adoption."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state(workspace, *parts) -> Path:
    return Path(workspace) / "tar_state" / Path(*parts)


def is_disabled(workspace) -> bool:
    return _state(workspace, DISABLE_FLAG).exists()


def _variants_for(method: str, dataset: str, md: dict, by_exp: dict) -> list[dict]:
    """Heuristic, deterministic variant suggestions (descriptions, not code)."""
    out: list[dict] = []
    fmean = md.get("forgetting_mean")
    failed = int(md.get("failed", 0) or 0)

    if fmean is not None and float(fmean) > _HIGH_FORGETTING:
        out.append({
            "variant_id": f"{method}-{dataset}-stronger-penalty",
            "suggested_change": {"param": "penalty_strength", "direction": "increase", "factor": 1.5},
            "rationale": f"mean forgetting {float(fmean):.3f} > {_HIGH_FORGETTING}; a stronger "
                         f"consolidation penalty may reduce forgetting.",
        })
        out.append({
            "variant_id": f"{method}-{dataset}-lr-annealing",
            "suggested_change": {"param": "lr_schedule", "direction": "add_linear_decay"},
            "rationale": "complementary lever: anneal LR across tasks to trade plasticity for stability.",
        })

    if failed > 0:
        # Surface the recorded failure diagnosis (Phase 1.3) so the fix is grounded.
        diag = ""
        for eid, rec in by_exp.items():
            if rec.get("method") == method and rec.get("dataset") == dataset and rec.get("failure_diagnosis"):
                diag = str(rec.get("failure_diagnosis", ""))[:200]
                break
        out.append({
            "variant_id": f"{method}-{dataset}-fix-failure",
            "suggested_change": {"param": "address_failure_mode"},
            "rationale": f"{failed} operational failure(s) recorded. Diagnosis: {diag or '(none)'}",
        })

    return out[:_MAX_PROPOSALS_PER_COMBO + 1]


def rebuild_variant_proposals(workspace) -> dict:
    """Build and persist advisory variant proposals from the outcome registry. Read-only."""
    try:
        from tar_lab import outcome_learner
        priors = outcome_learner.rebuild_outcome_priors(workspace)
    except Exception:
        priors = {}
    by_md = priors.get("by_method_dataset", {}) or {}
    by_exp = priors.get("by_experiment", {}) or {}

    proposals: list[dict] = []
    for key, md in by_md.items():
        method = str(md.get("method", "") or "")
        dataset = str(md.get("dataset", "") or "")
        if not method or not dataset:
            continue
        for v in _variants_for(method, dataset, md, by_exp):
            proposals.append({
                "proposal_id": v["variant_id"],
                "parent_method": method,
                "dataset": dataset,
                "suggested_change": v["suggested_change"],
                "rationale": v["rationale"],
                "evidence": {"n": md.get("n"), "completed": md.get("completed"),
                             "failed": md.get("failed"), "forgetting_mean": md.get("forgetting_mean"),
                             "reproducible": md.get("reproducible")},
                "status": "proposed",
                "actuation_rails": _ACTUATION_RAILS,
            })

    registry = {
        "generated_at": _now(),
        "advisory_only": True,
        "append_only_guarantee": (
            "Canonical methods are immutable to TAR. Variants would be NEW files only; "
            "the parent and all canonical methods are never overwritten."
        ),
        "summary": {"proposals": len(proposals),
                    "methods": sorted({p["parent_method"] for p in proposals})},
        "proposals": proposals,
    }
    try:
        p = _state(workspace, *REGISTRY_REL.parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass
    return registry


def load_variant_proposals(workspace) -> dict:
    p = _state(workspace, *REGISTRY_REL.parts)
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _main():
    from tar_storage import resolve_workspace
    ws = resolve_workspace(Path(__file__).resolve().parent.parent)
    print(json.dumps(rebuild_variant_proposals(ws).get("summary", {}), indent=2))


if __name__ == "__main__":
    _main()
