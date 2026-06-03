"""Phase 1.2 self-improvement: durable human-feedback priors.

Turns one-time human vetoes/rejections of proposed experiments into a persistent
preference prior, so the Research Director stops re-proposing experiments (and
morphologies) that humans consistently reject.

Design (matches the self-improvement safety envelope):
- READ-ONLY aggregator over the human-review history (human_review_state.json
  proposals + director_proposals.json veto window). It never mutates those files.
- Produces a durable, inspectable prior: tar_state/human_feedback_learned_priors.json
  (90-day sliding window — unlike the 48h director priority overlay which evaporates).
- The Director applies the prior as a SCORE PENALTY ONLY (never a boost).
- Operator off-switch: tar_state/human_feedback_priors.disabled (presence disables it).
- Only EXPLICIT human actions count. Passive auto-approval (24h veto window expiry)
  is NOT counted as a human approval.

Signals counted:
  veto    = director_proposals status "vetoed"  OR  human_review proposal status "rejected"
  approve = director_proposals status "approved" OR human_review status in
            {"approved", "approved_manifest_ready"}
  ignored = "auto_approved", "pending_veto", "awaiting_human_review", ...
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PRIORS_FILE = "human_feedback_learned_priors.json"
DISABLE_FLAG = "human_feedback_priors.disabled"
WINDOW_DAYS = 90

# Score sunk applied to an experiment a human EXPLICITLY vetoed/rejected (net negative):
# large enough to sort it below any fresh proposal. Reversible — re-approval clears it
# on the next rebuild. Not a hard delete (the directive still exists, auditable).
_EXACT_VETO_PENALTY = 1000.0
# Softer penalty on NEW experiments belonging to a frontier humans keep vetoing.
_FRONTIER_PENALTY_MAX = 40.0
_FRONTIER_MIN_SAMPLES = 3
_FRONTIER_MIN_VETO_RATE = 0.6

_VETO_HR_STATUSES = {"rejected"}
_APPROVE_HR_STATUSES = {"approved", "approved_manifest_ready"}
_VETO_DP_STATUSES = {"vetoed"}
_APPROVE_DP_STATUSES = {"approved"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _path(workspace, name: str) -> Path:
    return Path(workspace) / "tar_state" / name


def is_disabled(workspace) -> bool:
    return _path(workspace, DISABLE_FLAG).exists()


def _parse_dt(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _within_window(ts: str, now: datetime) -> bool:
    dt = _parse_dt(ts)
    if dt is None:
        return True  # no timestamp -> don't drop (conservative)
    try:
        return (now - dt).days <= WINDOW_DAYS
    except Exception:
        return True


def _jload(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def rebuild_priors(workspace) -> dict:
    """Read the human-review history and (re)write the durable prior. Returns the prior dict.

    Read-only on the source state files; writes only human_feedback_learned_priors.json.
    """
    now = _now()
    by_exp: dict[str, dict] = {}
    by_frontier: dict[str, dict] = {}

    def _bump(bucket: dict, key: str, kind: str, frontier: str = "", reason: str = "", exp_id: str = ""):
        if not key:
            return
        rec = bucket.setdefault(key, {"vetoes": 0, "approves": 0, "frontier": frontier,
                                      "last_at": "", "reasons": [], "experiments": []})
        rec[kind] = int(rec.get(kind, 0)) + 1
        if frontier and not rec.get("frontier"):
            rec["frontier"] = frontier
        rec["last_at"] = now.isoformat()
        if reason:
            rec.setdefault("reasons", [])
            if reason not in rec["reasons"]:
                rec["reasons"].append(reason)
        if exp_id:
            rec.setdefault("experiments", [])
            if exp_id not in rec["experiments"]:
                rec["experiments"].append(exp_id)

    # 1. human_review_state.json proposals
    hr = _jload(_path(workspace, "human_review_state.json")) or {}
    for prop in (hr.get("proposals", []) if isinstance(hr, dict) else []):
        if not isinstance(prop, dict):
            continue
        ts = str(prop.get("updated_at", "") or prop.get("created_at", "") or "")
        if not _within_window(ts, now):
            continue
        status = str(prop.get("status", "") or "")
        exp_id = str(prop.get("experiment_id", "") or "")
        frontier = str(prop.get("frontier_problem_id", "") or "")
        if status in _VETO_HR_STATUSES:
            _bump(by_exp, exp_id, "vetoes", frontier, "human rejected via review")
            _bump(by_frontier, frontier, "vetoes", frontier, exp_id=exp_id)
        elif status in _APPROVE_HR_STATUSES:
            _bump(by_exp, exp_id, "approves", frontier)
            _bump(by_frontier, frontier, "approves", frontier, exp_id=exp_id)

    # 2. director_proposals.json (veto window)
    dp = _jload(_path(workspace, "director_proposals.json")) or []
    for entry in (dp if isinstance(dp, list) else []):
        if not isinstance(entry, dict):
            continue
        ts = str(entry.get("vetoed_at", "") or entry.get("proposed_at", "") or "")
        if not _within_window(ts, now):
            continue
        status = str(entry.get("status", "") or "")
        exp_id = str(entry.get("experiment_id", "") or "")
        frontier = str(entry.get("frontier_problem_id", "") or "")
        if status in _VETO_DP_STATUSES:
            _bump(by_exp, exp_id, "vetoes", frontier, str(entry.get("veto_reason", "") or "human veto"))
            _bump(by_frontier, frontier, "vetoes", frontier, exp_id=exp_id)
        elif status in _APPROVE_DP_STATUSES:
            _bump(by_exp, exp_id, "approves", frontier)
            _bump(by_frontier, frontier, "approves", frontier, exp_id=exp_id)

    priors = {
        "generated_at": now.isoformat(),
        "window_days": WINDOW_DAYS,
        "by_experiment": by_exp,
        "by_frontier": by_frontier,
    }
    try:
        p = _path(workspace, PRIORS_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(priors, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass
    return priors


def load_priors(workspace) -> dict:
    return _jload(_path(workspace, PRIORS_FILE)) or {}


def penalty_for(priors: dict, experiment_id: str, frontier_id: str = "") -> tuple[float, str]:
    """Return (penalty_points, reason) to SUBTRACT from an experiment's priority score.

    Penalty is always >= 0 (penalty-only; never a boost). 0.0 when there is no
    qualifying veto signal.
    """
    if not isinstance(priors, dict):
        return 0.0, ""
    by_exp = priors.get("by_experiment", {}) or {}
    by_frontier = priors.get("by_frontier", {}) or {}

    # Exact: this exact experiment was explicitly vetoed/rejected (net negative).
    rec = by_exp.get(str(experiment_id)) if experiment_id else None
    if isinstance(rec, dict):
        v = int(rec.get("vetoes", 0) or 0)
        a = int(rec.get("approves", 0) or 0)
        if v >= 1 and v > a:
            return _EXACT_VETO_PENALTY, f"explicitly human-vetoed ({v} veto(s), {a} approve(s))"

    # Frontier-level: humans keep vetoing on this frontier -> soft penalty on new proposals.
    frec = by_frontier.get(str(frontier_id)) if frontier_id else None
    if isinstance(frec, dict):
        v = int(frec.get("vetoes", 0) or 0)
        a = int(frec.get("approves", 0) or 0)
        total = v + a
        if total >= _FRONTIER_MIN_SAMPLES:
            rate = v / total if total else 0.0
            if rate >= _FRONTIER_MIN_VETO_RATE:
                pen = round(_FRONTIER_PENALTY_MAX * rate, 1)
                return pen, f"frontier veto-rate {rate:.0%} over {total} decisions"
    return 0.0, ""
