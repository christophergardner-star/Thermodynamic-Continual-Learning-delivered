"""Phase 2.2 self-improvement: operational self-tuning (advisory).

TAR records operational failures (watchdog restarts, health-check failures, stale/failed
runtime leases) but never AGGREGATES them, so the same failures recur (e.g. the dashboard
service restarted 81 times). This read-only aggregator turns those isolated signals into
recurring-pattern RECOMMENDATIONS, written to
tar_state/operational/operational_recommendations.json.

SAFETY RAIL (#4): ADVISORY ONLY. It never tunes anything itself and never touches
governance/execution code. It may only SUGGEST operational-parameter changes (timeouts,
cooldowns, thresholds) for a human to approve and apply. Off-switch:
tar_state/operational.disabled.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_REL = Path("operational") / "operational_recommendations.json"
DISABLE_FLAG = "operational.disabled"

_RESTART_HIGH = 20
_RESTART_MED = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state(workspace, *parts) -> Path:
    return Path(workspace) / "tar_state" / Path(*parts)


def is_disabled(workspace) -> bool:
    return _state(workspace, DISABLE_FLAG).exists()


def _jload(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def rebuild_operational_recommendations(workspace) -> dict:
    """Aggregate operational signals into advisory recommendations. Read-only on sources."""
    recs: list[dict] = []

    # --- 1. Watchdog restart pressure ---
    wd = _jload(_state(workspace, "watchdog_state.json")) or {}
    services = wd.get("services", {}) if isinstance(wd, dict) else {}
    if isinstance(services, dict):
        for name, svc in services.items():
            if not isinstance(svc, dict):
                continue
            rc = int(svc.get("restart_count", 0) or 0)
            reason = str(svc.get("reason", "") or "")
            if rc >= _RESTART_HIGH:
                sev = "high"
            elif rc >= _RESTART_MED:
                sev = "medium"
            else:
                continue
            # Suggested params are advisory ONLY (params, not code/governance).
            suggested = []
            if name == "dashboard":
                suggested.append({"param": "dashboard.stale_after_s", "direction": "increase",
                                  "rationale": "reduce false-positive http_unhealthy restarts"})
            recs.append({
                "pattern": f"{name}_restart_pressure",
                "severity": sev,
                "evidence": f"watchdog restart_count={rc} (cumulative), last reason='{reason}'",
                "recommendation": (
                    f"'{name}' has restarted {rc} times. Investigate the root cause "
                    f"(memory growth / blocked dependency / health-endpoint flakiness) before "
                    f"re-enabling autonomy. Any parameter change below is a SUGGESTION for human "
                    f"approval — TAR does not apply it."
                ),
                "suggested_param_tuning": suggested,
                "affected_service": name,
            })

    # --- 2. Health-check failures ---
    hr = _jload(_state(workspace, "health_report.json")) or {}
    failed = [str(c.get("name", "")) for c in (hr.get("checks", []) or [])
              if isinstance(c, dict) and str(c.get("status", "")) == "fail"]
    if failed:
        recs.append({
            "pattern": "health_check_failures",
            "severity": "high",
            "evidence": f"failed checks: {failed}",
            "recommendation": "Resolve failing health checks before enabling execution; "
                              "these gate a safe autonomy resume.",
            "suggested_param_tuning": [],
            "affected_service": "health",
        })

    # --- 3. Runtime-lease failure/stale rate ---
    led = _jload(_state(workspace, "runtime_ledger.json")) or {}
    leases = led.get("leases", []) if isinstance(led, dict) else []
    reasons: dict[str, int] = {}
    for l in leases:
        if isinstance(l, dict) and str(l.get("status", "")) in {"failed", "stale", "released"}:
            cr = str(l.get("completion_reason", "") or "").split(";")[0][:60] or "(none)"
            reasons[cr] = reasons.get(cr, 0) + 1
    recurring = {k: v for k, v in reasons.items() if v >= 3}
    if recurring:
        recs.append({
            "pattern": "recurring_lease_termination_reason",
            "severity": "medium",
            "evidence": f"lease termination reasons seen >=3x: {recurring}",
            "recommendation": "A recurring lease-termination reason suggests a systematic "
                              "issue (e.g. repeated dead-PID cleanup, duplicate-runtime conflict). "
                              "Review the dominant reason.",
            "suggested_param_tuning": [],
            "affected_service": "orchestrator",
        })

    registry = {
        "generated_at": _now(),
        "advisory_only": True,
        "summary": {"recommendations": len(recs),
                    "high": sum(1 for r in recs if r["severity"] == "high"),
                    "medium": sum(1 for r in recs if r["severity"] == "medium")},
        "recommendations": recs,
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


def load_recommendations(workspace) -> dict:
    return _jload(_state(workspace, *REGISTRY_REL.parts)) or {}


def _main():
    from tar_storage import resolve_workspace
    ws = resolve_workspace(Path(__file__).resolve().parent.parent)
    reg = rebuild_operational_recommendations(ws)
    print(json.dumps(reg, indent=2))


if __name__ == "__main__":
    _main()
