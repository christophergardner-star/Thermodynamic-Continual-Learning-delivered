from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tar_lab.manifest import compute_manifest_hash


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def human_review_state_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "human_review_state.json"


def _default_state() -> dict[str, Any]:
    return {
        "schema": "tar_human_review_v1",
        "updated_at": _now_iso(),
        "proposals": [],
        "questions": [],
        "claim_reviews": [],
        "history": [],
    }


def load_human_review_state(workspace: Path) -> dict[str, Any]:
    path = human_review_state_path(workspace)
    if not path.exists():
        return _default_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_state()
    if not isinstance(raw, dict):
        return _default_state()
    raw.setdefault("schema", "tar_human_review_v1")
    raw.setdefault("updated_at", _now_iso())
    raw.setdefault("proposals", [])
    raw.setdefault("questions", [])
    raw.setdefault("claim_reviews", [])
    raw.setdefault("history", [])
    return raw


def save_human_review_state(workspace: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = human_review_state_path(workspace)
    payload = dict(payload or {})
    payload["schema"] = "tar_human_review_v1"
    payload["updated_at"] = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _existing_index(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {
        str(item.get(key, "") or ""): item
        for item in items
        if isinstance(item, dict) and str(item.get(key, "") or "")
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest_dir() -> Path:
    return _repo_root() / "manifests" / "human_review"


def _build_manifest_for_proposal(proposal: dict[str, Any]) -> tuple[str, str]:
    experiment_id = str(proposal.get("experiment_id", "") or "").strip()
    if not experiment_id:
        raise ValueError("proposal has no experiment_id")
    stamp = _now_stamp()
    manifest_id = f"human-review-{experiment_id}-{stamp}"
    manifest_path = _manifest_dir() / f"{experiment_id}__{stamp}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "manifest_id": manifest_id,
        "manifest_schema": "tar_execution_manifest_v1",
        "created_at": _now_iso(),
        "authorised_by": os.environ.get("USERNAME", "human-review"),
        "purpose": (
            f"Human-approved bounded experiment for {experiment_id}. "
            "Prepared from TAR human review state; execution still requires committed manifest verification."
        ),
        "frontier_problem_id": str(proposal.get("frontier_problem_id", "") or ""),
        "target_paper_id": str(proposal.get("target_paper_id", "") or ""),
        "human_review_review_id": str(proposal.get("review_id", "") or ""),
        "global_time_limit_h": 48.0,
        "experiments": [
            {
                "experiment_id": experiment_id,
                "name": str(proposal.get("title", "") or experiment_id),
                "allowed_datasets": [
                    str(proposal.get("dataset", "") or "").strip()
                ] if str(proposal.get("dataset", "") or "").strip() else [],
                "allowed_methods": [
                    str(proposal.get("method", "") or "").strip()
                ] if str(proposal.get("method", "") or "").strip() else [],
                "allowed_seeds": [
                    int(seed) for seed in proposal.get("seeds", []) or []
                    if isinstance(seed, int) or str(seed).strip().isdigit()
                ],
                "time_limit_h": 48.0,
                "run_limit": 1,
                "notes": str(proposal.get("human_notes", "") or proposal.get("why_now", "") or ""),
            }
        ],
        "content_hash": "UNSIGNED",
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["content_hash"] = compute_manifest_hash(manifest_path)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_id, str(manifest_path)


def sync_human_review_from_director_state(workspace: Path, director_state: dict[str, Any]) -> dict[str, Any]:
    state = load_human_review_state(workspace)
    existing_proposals = _existing_index(state.get("proposals", []), "review_id")
    existing_questions = _existing_index(state.get("questions", []), "question_id")
    existing_claims = _existing_index(state.get("claim_reviews", []), "review_id")

    proposals: list[dict[str, Any]] = []
    for directive in director_state.get("experiment_directives", []) if isinstance(director_state, dict) else []:
        if not isinstance(directive, dict):
            continue
        experiment_id = str(directive.get("experiment_id", "") or "")
        if not experiment_id:
            continue
        status = str(directive.get("status", "") or "")
        if status in {"complete", "archive"}:
            continue
        review_id = f"proposal:{experiment_id}"
        existing = existing_proposals.get(review_id, {})
        proposals.append({
            "review_id": review_id,
            "status": str(existing.get("status", "") or "awaiting_human_review"),
            "decision": str(existing.get("decision", "") or ""),
            "updated_at": str(existing.get("updated_at", "") or _now_iso()),
            "created_at": str(existing.get("created_at", "") or _now_iso()),
            "frontier_problem_id": str(directive.get("frontier_problem_id", "") or ""),
            "frontier_problem_title": str(directive.get("frontier_problem_title", "") or ""),
            "domain_id": str(directive.get("active_domain_id", "") or ""),
            "experiment_id": experiment_id,
            "title": str(directive.get("title", "") or experiment_id),
            "target_paper_id": str(directive.get("target_paper_id", "") or ""),
            "scheduler_intent": str(directive.get("scheduler_intent", "") or ""),
            "proposal_kind": str(directive.get("proposal_kind", "") or ""),
            "proposal_origin": str(directive.get("proposal_origin", "") or ""),
            "dataset": str(directive.get("dataset", "") or ""),
            "method": str(directive.get("method", "") or ""),
            "seeds": list(directive.get("seeds", []) or []),
            "priority_score": float(directive.get("priority_score", 0.0) or 0.0),
            "why_now": str(directive.get("why_now", "") or ""),
            "experiment_goal": str(directive.get("experiment_goal", "") or ""),
            "mechanism_focus": str(directive.get("mechanism_focus", "") or ""),
            "global_problem_statement": str(directive.get("global_problem_statement", "") or ""),
            "external_baselines": list(directive.get("external_baselines", []) or []),
            "validation_plan": {
                "run_validation": True,
                "result_validation": True,
                "claim_validation": True,
                "figure_validation": bool(directive.get("target_paper_id")),
            },
            "blocked_by_experiment_ids": list(directive.get("blocked_by_experiment_ids", []) or []),
            "human_notes": str(existing.get("human_notes", "") or ""),
            "build_manifest_authorised": bool(existing.get("build_manifest_authorised", False)),
        })

    claim_reviews: list[dict[str, Any]] = []
    for directive in director_state.get("paper_directives", []) if isinstance(director_state, dict) else []:
        if not isinstance(directive, dict):
            continue
        paper_id = str(directive.get("paper_id", "") or "")
        if not paper_id:
            continue
        review_id = f"claim:{paper_id}"
        existing = existing_claims.get(review_id, {})
        claim_reviews.append({
            "review_id": review_id,
            "paper_id": paper_id,
            "title": str(directive.get("title", "") or paper_id),
            "status": str(existing.get("status", "") or "awaiting_human_review"),
            "decision": str(existing.get("decision", "") or ""),
            "created_at": str(existing.get("created_at", "") or _now_iso()),
            "updated_at": str(existing.get("updated_at", "") or _now_iso()),
            "truth_status": str(directive.get("truth_status", "weak") or "weak"),
            "readiness": str(directive.get("readiness", "planned") or "planned"),
            "scope_status": str(directive.get("scope_status", "") or ""),
            "frontier_problem_id": str(directive.get("frontier_problem_id", "") or ""),
            "waiting_for_experiments": list(directive.get("waiting_for_experiments", []) or []),
            "recommendation": str(directive.get("recommendation", "") or ""),
            "human_notes": str(existing.get("human_notes", "") or ""),
        })

    questions: list[dict[str, Any]] = []
    for directive in director_state.get("paper_directives", []) if isinstance(director_state, dict) else []:
        if not isinstance(directive, dict):
            continue
        paper_id = str(directive.get("paper_id", "") or "")
        waiting = [
            str(item) for item in directive.get("waiting_for_experiments", []) or []
            if str(item or "")
        ]
        if not paper_id or not waiting:
            continue
        question_id = f"question:paper:{paper_id}:waiting_for"
        existing = existing_questions.get(question_id, {})
        questions.append({
            "question_id": question_id,
            "component": "tar_research_director",
            "frontier_problem_id": str(directive.get("frontier_problem_id", "") or ""),
            "question_type": "rerun_required",
            "question_text": (
                f"Paper '{paper_id}' is blocked by {len(waiting)} experiment(s). "
                "Should TAR keep this paper blocked pending more evidence, or narrow the claim scope?"
            ),
            "why_this_blocks_progress": "Paper claims cannot advance until the required experiment set is complete.",
            "recommended_default": "hold_pending_more_evidence",
            "options": [
                "hold_pending_more_evidence",
                "narrow_claim_scope",
                "deprioritise_paper",
            ],
            "deadline_hint": "",
            "linked_proposal_ids": [f"claim:{paper_id}"],
            "status": str(existing.get("status", "") or "awaiting_human_answer"),
            "answer": str(existing.get("answer", "") or ""),
            "answer_notes": str(existing.get("answer_notes", "") or ""),
            "created_at": str(existing.get("created_at", "") or _now_iso()),
            "updated_at": str(existing.get("updated_at", "") or _now_iso()),
        })

    for frontier in director_state.get("frontier_directives", []) if isinstance(director_state, dict) else []:
        if not isinstance(frontier, dict):
            continue
        frontier_id = str(frontier.get("problem_id", "") or "")
        truth_status = str(frontier.get("truth_status", "") or "")
        if not frontier_id or truth_status == "validated":
            continue
        question_id = f"question:frontier:{frontier_id}:scientific_scope"
        existing = existing_questions.get(question_id, {})
        questions.append({
            "question_id": question_id,
            "component": "tar_research_director",
            "frontier_problem_id": frontier_id,
            "question_type": "scientific_scope",
            "question_text": (
                f"Frontier '{frontier_id}' is not yet validated. Should TAR keep proposing bounded experiments here, "
                "or pause this frontier until stronger external/problem evidence is gathered?"
            ),
            "why_this_blocks_progress": "TAR must not overstate weak frontier evidence or silently widen claims.",
            "recommended_default": "keep_proposing_bounded_experiments",
            "options": [
                "keep_proposing_bounded_experiments",
                "pause_this_frontier",
                "collect_more_external_evidence_first",
            ],
            "deadline_hint": "",
            "linked_proposal_ids": [
                item["review_id"]
                for item in proposals
                if str(item.get("frontier_problem_id", "") or "") == frontier_id
            ],
            "status": str(existing.get("status", "") or "awaiting_human_answer"),
            "answer": str(existing.get("answer", "") or ""),
            "answer_notes": str(existing.get("answer_notes", "") or ""),
            "created_at": str(existing.get("created_at", "") or _now_iso()),
            "updated_at": str(existing.get("updated_at", "") or _now_iso()),
        })

    state["proposals"] = proposals
    state["questions"] = questions
    state["claim_reviews"] = claim_reviews
    return save_human_review_state(workspace, state)


def record_review_decision(
    workspace: Path,
    *,
    review_id: str,
    decision: str,
    human_notes: str = "",
    build_manifest_authorised: bool = False,
) -> dict[str, Any] | None:
    state = load_human_review_state(workspace)
    updated = None
    for bucket_name in ("proposals", "claim_reviews"):
        bucket = state.get(bucket_name, [])
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if not isinstance(item, dict) or str(item.get("review_id", "") or "") != review_id:
                continue
            item["decision"] = decision
            item["human_notes"] = human_notes
            item["build_manifest_authorised"] = bool(build_manifest_authorised)
            item["status"] = {
                "approve": "approved",
                "approve_and_build_manifest": "approved_manifest_ready",
                "request_revision": "revision_requested",
                "reject": "rejected",
                "pause_this_frontier": "paused",
                "approve_claim_scope": "approved",
                "approve_paper_rewrite": "approved",
                "approve_figure_set": "approved",
                "hold_pending_more_evidence": "held",
            }.get(decision, decision or "reviewed")
            item["updated_at"] = _now_iso()
            if bucket_name == "proposals" and build_manifest_authorised:
                manifest_id, manifest_path = _build_manifest_for_proposal(item)
                item["manifest_id"] = manifest_id
                item["manifest_path"] = manifest_path
            updated = item
            state.setdefault("history", []).append({
                "timestamp": _now_iso(),
                "kind": bucket_name[:-1],
                "review_id": review_id,
                "decision": decision,
                "human_notes": human_notes,
                "manifest_path": str(item.get("manifest_path", "") or ""),
            })
            break
    if updated is None:
        return None
    save_human_review_state(workspace, state)
    return updated


def answer_human_question(
    workspace: Path,
    *,
    question_id: str,
    answer: str,
    answer_notes: str = "",
) -> dict[str, Any] | None:
    state = load_human_review_state(workspace)
    for item in state.get("questions", []):
        if not isinstance(item, dict) or str(item.get("question_id", "") or "") != question_id:
            continue
        item["answer"] = answer
        item["answer_notes"] = answer_notes
        item["status"] = "answered"
        item["updated_at"] = _now_iso()
        state.setdefault("history", []).append({
            "timestamp": _now_iso(),
            "kind": "question",
            "question_id": question_id,
            "answer": answer,
            "answer_notes": answer_notes,
        })
        save_human_review_state(workspace, state)
        return item
    return None


class _UniversalApproval:
    """Sentinel returned in autonomous mode — all experiments are within scope."""
    def __contains__(self, item: object) -> bool:
        return True
    def __bool__(self) -> bool:
        return True


def approved_experiment_ids(workspace: Path) -> "set[str] | _UniversalApproval":
    from tar_validation_mode import load_state as _load_vs
    vs = _load_vs(workspace) or {}
    if vs.get("active"):
        # Stabilisation/validation mode: only explicitly approved experiments run.
        state = load_human_review_state(workspace)
        approved: set[str] = set()
        for proposal in state.get("proposals", []):
            if not isinstance(proposal, dict):
                continue
            if str(proposal.get("status", "") or "") not in {"approved", "approved_manifest_ready"}:
                continue
            experiment_id = str(proposal.get("experiment_id", "") or "")
            if experiment_id:
                approved.add(experiment_id)
        return approved
    # Autonomous mode: Director scope constraint (_STRICT_REAL_WORLD_FRONTIER_ONLY)
    # is the gate. All Director-proposed experiments are within scope by design.
    return _UniversalApproval()


def approved_paper_ids(workspace: Path) -> set[str]:
    state = load_human_review_state(workspace)
    approved: set[str] = set()
    for review in state.get("claim_reviews", []):
        if not isinstance(review, dict):
            continue
        if str(review.get("status", "") or "") != "approved":
            continue
        paper_id = str(review.get("paper_id", "") or "")
        if paper_id:
            approved.add(paper_id)
    return approved
