"""Phase 3.1b self-improvement (B5): close the method-synthesis loop on CPU.

method_synthesizer is a complete LLM -> AST -> sandbox -> minibench pipeline, but it NEVER
fired from a novel-idea path: its only caller was a missing-method fallback that AUTO-ADOPTS
the result into the live METHOD_REGISTRY. This driver turns a TOP method_refinement variant
proposal into a synthesised, validated CLMethod CANDIDATE and records it PENDING HUMAN
APPROVAL — it never adopts anything.

RAIL-3 + append-only + human-gated, by construction:
  * Candidates are written to a QUARANTINE dir (tar_state/synthesized_methods_pending/) that
    load_generated_methods() never scans (it globs only tar_state/synthesized_methods/*.py),
    so a validated-but-unapproved method can never be auto-loaded on the next cycle.
  * Adoption is a SEPARATE, explicit human step (approve_pending_candidate), which uses
    append_only_guard to create a NEW file in the live dir — never overwriting a canonical
    method. The synthesis driver itself adopts NOTHING.
  * OFF unless tar_state/method_synthesis.enabled exists. Synthesis calls an LLM + sandbox +
    minibench, so it is operator/launcher-invoked and never runs in the live director cycle.

Import-light on purpose (no torch at module load): method_synthesizer + method_refinement are
imported lazily inside functions, and synthesize_fn is injectable so tests run fully offline.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ENABLE_FLAG = "method_synthesis.enabled"
QUARANTINE_REL = Path("synthesized_methods_pending")
PENDING_REGISTRY_REL = QUARANTINE_REL / "pending_review.json"
ADOPT_REL = Path("synthesized_methods")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state(workspace, *parts) -> Path:
    return Path(workspace) / "tar_state" / Path(*parts)


def is_enabled(workspace) -> bool:
    return _state(workspace, ENABLE_FLAG).exists()


def _jload(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# proposal -> idea
# ---------------------------------------------------------------------------

def _change_to_english(change: Any) -> str:
    if not isinstance(change, dict):
        return "apply the suggested refinement"
    param = str(change.get("param", "") or "")
    if param == "address_failure_mode":
        return "address the recorded operational failure mode"
    direction = str(change.get("direction", "") or "")
    factor = change.get("factor")
    parts: list[str] = []
    if direction:
        parts.append(direction.replace("_", " "))
    if param:
        parts.append(param)
    if factor is not None:
        parts.append(f"(factor {factor})")
    return " ".join(parts) if parts else "apply the suggested refinement"


def proposal_to_idea(proposal: dict) -> str:
    """Deterministically turn a method_refinement variant proposal into a plain-English idea
    for the synthesizer. Asks for a genuinely NEW mechanism, not a hyperparameter tweak."""
    parent = str(proposal.get("parent_method", "") or "a continual-learning method")
    dataset = str(proposal.get("dataset", "") or "a continual-learning benchmark")
    change = _change_to_english(proposal.get("suggested_change", {}))
    rationale = str(proposal.get("rationale", "") or "")
    idea = (
        f"Implement a NEW, distinct continual-learning method (a fresh CLMethod with its own "
        f"registration key) inspired by improving '{parent}' on {dataset}. "
        f"Core refinement: {change}. "
    )
    if rationale:
        idea += f"Motivation: {rationale} "
    idea += (
        "It must be a genuinely different mechanism, not a hyperparameter tweak of the parent, "
        "and must implement the CLMethod interface."
    )
    return idea


def _select_proposals(registry: dict, max_candidates: int) -> list[dict]:
    """Pick the top still-'proposed' variants — those with the worst recorded forgetting
    (most in need of a new method) first; deterministic tiebreak on proposal_id."""
    proposals = registry.get("proposals", []) if isinstance(registry, dict) else []
    candidates = [
        p for p in proposals
        if isinstance(p, dict) and str(p.get("status", "")) == "proposed"
    ]

    def _key(p: dict):
        ev = p.get("evidence", {}) or {}
        try:
            fm = float(ev.get("forgetting_mean"))
        except (TypeError, ValueError):
            fm = -1.0
        return (-fm, str(p.get("proposal_id", "")))

    candidates.sort(key=_key)
    return candidates[: max(0, int(max_candidates))]


# ---------------------------------------------------------------------------
# pending-candidate registry (quarantine, human-gated)
# ---------------------------------------------------------------------------

def load_pending_candidates(workspace) -> dict:
    return _jload(_state(workspace, *PENDING_REGISTRY_REL.parts)) or {}


def _record_pending(workspace, new_candidates: list[dict]) -> list[dict]:
    """Append-only-safe merge into the pending registry: an existing candidate_id is
    preserved untouched (so a human decision survives re-runs); only new ones are added."""
    existing_doc = load_pending_candidates(workspace)
    existing = existing_doc.get("candidates", []) if isinstance(existing_doc, dict) else []
    by_id: dict[str, dict] = {
        str(c.get("candidate_id")): c for c in existing
        if isinstance(c, dict) and c.get("candidate_id")
    }
    for c in new_candidates:
        cid = str(c.get("candidate_id", ""))
        if cid and cid not in by_id:
            by_id[cid] = c
    merged = sorted(by_id.values(), key=lambda c: str(c.get("candidate_id", "")))
    doc = {
        "generated_at": _now(),
        "rail": 3,
        "advisory_only": True,
        "summary": {
            "total": len(merged),
            "pending": sum(1 for c in merged if c.get("status") == "pending_human_approval"),
        },
        "candidates": merged,
    }
    p = _state(workspace, *PENDING_REGISTRY_REL.parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(p)
    return merged


def run_synthesis_from_proposals(
    workspace,
    *,
    max_candidates: int = 1,
    synthesize_fn: Optional[Callable[..., dict]] = None,
    log_fn: Callable[[str], Any] = print,
) -> dict:
    """B5: synthesise + validate a NEW CLMethod from the top variant proposal(s), recording
    each validated result as PENDING HUMAN APPROVAL in the quarantine registry. Adopts NOTHING.

    OFF unless method_synthesis.enabled exists. synthesize_fn defaults to
    method_synthesizer.synthesize_and_validate_method (lazily imported); tests inject a fake.
    Returns: enabled / attempted / validated / candidates / errors.
    """
    base = {"enabled": False, "attempted": 0, "validated": 0, "candidates": [], "errors": []}
    if not is_enabled(workspace):
        return {**base, "reason": "method_synthesis.enabled absent (RAIL-3: human opt-in required)"}

    from tar_lab.method_refinement_engine import load_variant_proposals
    registry = load_variant_proposals(workspace)
    proposals = _select_proposals(registry, max_candidates)
    if not proposals:
        return {**base, "enabled": True, "reason": "no 'proposed' variant proposals to synthesise"}

    if synthesize_fn is None:
        from tar_lab.method_synthesizer import synthesize_and_validate_method as synthesize_fn

    quarantine = _state(workspace, *QUARANTINE_REL.parts)
    quarantine.mkdir(parents=True, exist_ok=True)

    summary = {**base, "enabled": True}
    new_candidates: list[dict] = []
    for prop in proposals:
        idea = proposal_to_idea(prop)
        summary["attempted"] += 1
        try:
            result = synthesize_fn(idea, str(workspace), out_dir=quarantine, log_fn=log_fn)
        except Exception as exc:  # synthesis must never crash the loop
            summary["errors"].append(f"{prop.get('proposal_id', '')}: {str(exc)[:180]}")
            continue
        if not isinstance(result, dict) or not result.get("success"):
            err = (result or {}).get("error", "unknown") if isinstance(result, dict) else "no result"
            summary["errors"].append(f"{prop.get('proposal_id', '')}: {str(err)[:180]}")
            continue
        summary["validated"] += 1
        new_candidates.append({
            "candidate_id": f"synth-{prop.get('proposal_id', '')}",
            "source_proposal_id": str(prop.get("proposal_id", "")),
            "parent_method": str(prop.get("parent_method", "")),
            "dataset": str(prop.get("dataset", "")),
            "idea": idea,
            "method_key": str(result.get("method_key", "")),
            "class_name": str(result.get("class_name", "")),
            "pending_path": result.get("saved_path"),
            "description": str(result.get("description", "")),
            "status": "pending_human_approval",
            "rail": 3,
            "proposed_at": _now(),
            "note": (
                "Validated (AST + sandbox + minibench) but NOT adopted. Quarantined from "
                "load_generated_methods. A human must review and call approve_pending_candidate "
                "before it can be loaded into METHOD_REGISTRY and run."
            ),
        })

    summary["candidates"] = _record_pending(workspace, new_candidates)
    return summary


def approve_pending_candidate(workspace, candidate_id: str, *, approved_by: str) -> dict:
    """HUMAN-INVOKED adoption: move a pending candidate's validated file into the live
    synthesized_methods/ dir (where load_generated_methods picks it up next cycle).

    Uses append_only_guard to create a NEW file — never overwrites a canonical/existing
    method. Refuses unless the candidate exists and is still pending. Returns a result dict;
    raises nothing on the normal refusal paths.
    """
    doc = load_pending_candidates(workspace)
    candidates = doc.get("candidates", []) if isinstance(doc, dict) else []
    target = next((c for c in candidates if str(c.get("candidate_id", "")) == str(candidate_id)), None)
    if target is None:
        return {"approved": False, "reason": "candidate_not_found"}
    if target.get("status") != "pending_human_approval":
        return {"approved": False, "reason": f"not_pending (status={target.get('status')})"}
    if not approved_by or not str(approved_by).strip():
        return {"approved": False, "reason": "approved_by required (human attribution)"}

    src = Path(str(target.get("pending_path") or ""))
    if not src.exists():
        return {"approved": False, "reason": "candidate_file_missing"}
    method_key = str(target.get("method_key") or src.stem)
    dest = _state(workspace, *ADOPT_REL.parts) / f"{method_key}.py"

    try:
        from tar_lab import append_only_guard
        append_only_guard.guarded_create_new(
            dest, src.read_text(encoding="utf-8"), require_additive_area=False
        )
    except Exception as exc:
        return {"approved": False, "reason": f"append_only_guard refused: {str(exc)[:160]}"}

    target["status"] = "approved_adopted"
    target["approved_by"] = str(approved_by)
    target["approved_at"] = _now()
    target["adopted_path"] = str(dest)
    p = _state(workspace, *PENDING_REGISTRY_REL.parts)
    doc["candidates"] = candidates
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return {"approved": True, "method_key": method_key, "adopted_path": str(dest)}
