"""
Solution-finding loop: the DATA-LEVEL learning loop (Ingredient 4).

When a candidate mechanism is killed (its terminal verdict is not positive, or it
fails its preregistered joint criterion), we append a record to an append-only
kill-ledger keyed by a deterministic CONFIG FINGERPRINT. The proposer then (a) is
shown the killed regions as prose, and (b) is HARD-BLOCKED from re-proposing a
fingerprint that was already killed (deterministic pruning — not prompt-hope).

No GPU / weights self-improvement here: the loop "gets better at proposing" purely
by pruning the search from recorded outcomes. Everything is fail-quiet.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_KILLED_VERDICTS = frozenset({"NULL", "ADVERSE", "COLLAPSED", "ERROR"})

# A solution-loop anomaly gap has gap_id "tar_anomaly::<name>"; the director derives
# its frontier id deterministically as f"fp-gap-{_frontier_slug(gap_id)}" -> it always
# begins "fp-gap-tar-anomaly". Such a frontier must NOT be probed with the static
# gap-probe (which defaults method="tcl" — re-testing the falsified incumbent on the
# anomaly); the widened proposer generates its FIRST candidate instead.
_ANOMALY_FRONTIER_PREFIX = "fp-gap-tar-anomaly"


def is_anomaly_frontier(frontier_id: str) -> bool:
    """True for a solution-loop anomaly frontier (derived from a tar_anomaly:: gap)."""
    return str(frontier_id or "").startswith(_ANOMALY_FRONTIER_PREFIX)


def _ledger_path(workspace: Path) -> Path:
    return Path(workspace) / "tar_state" / "solution_loop" / "kill_ledger.jsonl"


def candidate_fingerprint(method: str, config_overrides: dict | None = None,
                          mechanism_class: str = "") -> str:
    """Deterministic fingerprint of a candidate's design-space region.

    method + mechanism_class + the sorted, rounded config overrides. Two proposals
    that differ only in irrelevant ordering hash identically; a genuinely different
    HP setting hashes differently (so re-tuning a killed mechanism is allowed, but
    re-proposing the SAME config is blocked)."""
    norm: dict[str, Any] = {}
    for k, v in sorted((config_overrides or {}).items()):
        if isinstance(v, float):
            norm[k] = round(v, 6)
        elif isinstance(v, (int, str, bool)) or v is None:
            norm[k] = v
        else:
            norm[k] = str(v)
    payload = json.dumps(
        {"method": str(method or "").strip().lower(),
         "mechanism_class": str(mechanism_class or "").strip().lower(),
         "config": norm},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def record_kill(workspace: Path, *, experiment_id: str, method: str,
                config_overrides: dict | None, mechanism_class: str,
                verdict: str, kill_reason: str,
                criteria_failed: list[str] | None = None) -> str:
    """Append a kill record; returns the fingerprint. Fail-quiet."""
    fp = candidate_fingerprint(method, config_overrides, mechanism_class)
    rec = {
        "fingerprint": fp,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": str(experiment_id or ""),
        "method": str(method or ""),
        "mechanism_class": str(mechanism_class or ""),
        "config_overrides": dict(config_overrides or {}),
        "verdict": str(verdict or ""),
        "kill_reason": str(kill_reason or "")[:500],
        "criteria_failed": list(criteria_failed or []),
    }
    try:
        p = _ledger_path(workspace)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass
    return fp


def _load_records(workspace: Path) -> list[dict]:
    p = _ledger_path(workspace)
    if not p.exists():
        return []
    out: list[dict] = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    except OSError:
        return []
    return out


def load_killed_fingerprints(workspace: Path) -> set[str]:
    """Fingerprints that have been killed (any recorded kill entry)."""
    return {str(r.get("fingerprint", "")) for r in _load_records(workspace) if r.get("fingerprint")}


def is_killed(workspace: Path, method: str, config_overrides: dict | None = None,
              mechanism_class: str = "") -> bool:
    """Deterministic pruning check: has this exact candidate region been killed?"""
    fp = candidate_fingerprint(method, config_overrides, mechanism_class)
    return fp in load_killed_fingerprints(workspace)


def render_kill_ledger_block(workspace: Path, *, max_entries: int = 40) -> str:
    """Prose summary of killed regions for the proposer (so it avoids them)."""
    recs = _load_records(workspace)[-max_entries:]
    lines: list[str] = []
    for r in recs:
        mech = f" [{r.get('mechanism_class')}]" if r.get("mechanism_class") else ""
        lines.append(f"- {r.get('method', '?')}{mech}: {r.get('verdict', '?')} — "
                     f"{(r.get('kill_reason') or '').strip()[:120]}")
    return "\n".join(lines)
