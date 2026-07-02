"""Autonomy Ramp Controller.

Gates TAR's transition from CONFIRMATORY mode (running the pre-registered Phase 2/3
GPU experiments) to FULL AUTONOMY (director-generated experiments). The transition is
deliberately SAFE: full autonomy only engages after
  (a) all confirmatory runs are terminal,
  (b) safety gates pass (health + evidence integrity), and
  (c) a human gives explicit final confirmation.

Stages:
  confirmatory     - phase2/3 runs still pending/running
  verifying        - all phase2/3 terminal; running safety gates
  awaiting_confirm - gates passed; waiting for human final go (TAR notifies, does NOT auto-promote)
  full_autonomy    - human confirmed; director-generated experiments may run
  hold             - a safety gate failed; held with a reason for human attention

Design notes:
- OPT-IN for fresh installs: when NO ramp was ever configured (no state file, no
  configured-sentinel), is_full_autonomy() returns True so existing behaviour is
  unchanged. Once a ramp HAS been configured, missing/corrupt state FAILS CLOSED:
  deleting autonomy_ramp.json can never ungate autonomy.
- The controller NEVER promotes itself past awaiting_confirm without human confirmation
  (confirm_promotion() or the autonomy_ramp_confirm.flag). When reauth_required_at is
  set, confirmations recorded BEFORE that checkpoint are stale and do not count — the
  human must re-affirm (fresh confirm_promotion() / fresh flag).
- evaluate_ramp() is cheap during confirmatory (only a queue/archive scan); the heavier
  health gate runs only once the confirmatory runs are all terminal.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

RAMP_FILE = "autonomy_ramp.json"
CONFIRM_FLAG = "autonomy_ramp_confirm.flag"
# Durable sentinel: written the first time a ramp is configured. Its presence
# makes a MISSING ramp file fail CLOSED (deleting the state file must never
# ungate autonomy). Without it, a fresh install keeps the documented opt-in
# behaviour (no ramp configured -> ungated).
CONFIGURED_SENTINEL = "autonomy_ramp_configured.flag"

# The pre-registered Phase 2/3 confirmatory runs that must complete before full autonomy.
# Identified by orchestrator runner_key (the reliable manual-vs-generated discriminator).
PHASE2_RUNNER_KEYS = [
    "hpc_replication_phase2",
    "hp_selection",
    "mechanistic_ablation_7c",
    "phase16_cifar100_rerun",
    "phase17_tinyimagenet_rerun",
    "hpc_lambda_momentum_abl",
]
_TERMINAL_STATES = {"complete", "failed", "skipped", "archived"}

STAGE_CONFIRMATORY = "confirmatory"
STAGE_VERIFYING = "verifying"
STAGE_AWAITING_CONFIRM = "awaiting_confirm"
STAGE_FULL_AUTONOMY = "full_autonomy"
STAGE_HOLD = "hold"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value) -> "datetime | None":
    """Parse an ISO-8601 timestamp; None for empty/invalid. Naive -> UTC."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip())
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _path(workspace, name: str) -> Path:
    return Path(workspace) / "tar_state" / name


def load_ramp_state(workspace):
    p = _path(workspace, RAMP_FILE)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_ramp_state(workspace, state: dict) -> None:
    state["updated_at"] = _now()
    p = _path(workspace, RAMP_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(p)


def init_ramp(workspace, runner_keys=None) -> dict:
    """Start the ramp in confirmatory mode. The human kicks this off once."""
    state = {
        "enabled": True,
        "stage": STAGE_CONFIRMATORY,
        "phase2_runner_keys": list(runner_keys or PHASE2_RUNNER_KEYS),
        "gate_report": {},
        "blocked_reason": "",
        "confirmed_by_human": False,
        "promoted_at": "",
        "created_at": _now(),
    }
    save_ramp_state(workspace, state)
    try:
        _path(workspace, CONFIGURED_SENTINEL).write_text(_now(), encoding="utf-8")
    except Exception:
        pass
    return state


def disable_ramp(workspace) -> None:
    st = load_ramp_state(workspace)
    if st:
        st["enabled"] = False
        save_ramp_state(workspace, st)


def ramp_active(workspace) -> bool:
    st = load_ramp_state(workspace)
    return bool(st and st.get("enabled"))


def is_full_autonomy(workspace) -> bool:
    """Whether director-generated experiments are allowed to run.

    FAIL-CLOSED semantics:
    - Ramp file present and parseable: only stage==full_autonomy permits
      generated experiments; enabled=False is the explicit human opt-out
      (disable_ramp) and ungated.
    - Ramp file present but CORRUPT: False. Corruption must never ungate.
    - Ramp file MISSING but the configured-sentinel exists: False. Deleting
      the state file must never ungate a previously configured ramp.
    - Never configured (no file, no sentinel): True — the documented opt-in,
      backwards-compatible default for fresh installs.
    """
    p = _path(workspace, RAMP_FILE)
    if not p.exists():
        return not _path(workspace, CONFIGURED_SENTINEL).exists()
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(st, dict):
        return False
    if not st.get("enabled"):
        return True
    return st.get("stage") == STAGE_FULL_AUTONOMY


def _phase2_status(workspace, runner_keys):
    """Scan queue + archive; return (all_terminal: bool, detail: dict[key->status])."""
    statuses: dict[str, dict] = {}
    for fname in ("experiment_archive.json", "experiment_queue.json"):
        p = _path(workspace, fname)
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for e in (data.get("experiments", []) if isinstance(data, dict) else []):
            if not isinstance(e, dict):
                continue
            rk = str(e.get("runner_key", "") or "")
            if rk in runner_keys:
                st = str(e.get("status", "") or "")
                stg = str(e.get("stage", "") or "")
                terminal = st in _TERMINAL_STATES or stg in _TERMINAL_STATES
                prev = statuses.get(rk)
                # Prefer a terminal sighting (archive wins over a stale queue 'running').
                if prev is None or (terminal and not prev.get("terminal")):
                    statuses[rk] = {
                        "status": st, "stage": stg, "terminal": terminal,
                        "result_path": str(e.get("result_path", "") or ""),
                    }
    all_terminal = True
    detail: dict[str, str] = {}
    for rk in runner_keys:
        s = statuses.get(rk)
        if not s:
            all_terminal = False
            detail[rk] = "not_found"
        else:
            detail[rk] = s["status"] or s["stage"] or "?"
            if not s["terminal"]:
                all_terminal = False
    return all_terminal, detail


def _health_gate(workspace):
    """Run the system health check; pass only if no checks FAIL (warns are allowed)."""
    try:
        from tar_health_check import HealthChecker
        report = HealthChecker(Path(workspace)).run_all_checks()
        failed = [c.name for c in report.checks if c.status == "fail"]
        return (report.failed == 0), {
            "pass": report.failed == 0,
            "failed": failed,
            "summary": {"passed": report.passed, "failed": report.failed,
                        "warned": report.warned, "skipped": report.skipped},
        }
    except Exception as exc:
        return False, {"pass": False, "error": str(exc)}


def _evidence_gate(workspace, runner_keys):
    """Soft-verify that each terminal confirmatory run produced a result artifact."""
    all_terminal, detail = _phase2_status(workspace, runner_keys)
    missing_results = []
    # Re-scan for result_path presence on terminal entries.
    statuses: dict[str, str] = {}
    for fname in ("experiment_archive.json", "experiment_queue.json"):
        p = _path(workspace, fname)
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for e in (data.get("experiments", []) if isinstance(data, dict) else []):
            rk = str(e.get("runner_key", "") or "")
            if rk in runner_keys and str(e.get("status", "")) == "complete":
                statuses[rk] = str(e.get("result_path", "") or "")
    for rk in runner_keys:
        rp = statuses.get(rk)
        # Only require a result for runs that COMPLETED (a 'failed' run legitimately has none).
        if rp is not None and rp and not Path(rp).exists():
            missing_results.append(rk)
    return (len(missing_results) == 0), {"pass": len(missing_results) == 0,
                                         "missing_result_files": missing_results}


def evaluate_ramp(workspace) -> dict | None:
    """Advance the ramp state machine one step. Safe to call every daemon cycle.

    NEVER promotes past awaiting_confirm without an explicit human confirmation.
    Returns the updated state (or None if no ramp is configured).
    """
    st = load_ramp_state(workspace)
    if not st or not st.get("enabled"):
        return st
    if st.get("stage") == STAGE_FULL_AUTONOMY:
        return st  # already promoted; nothing to do

    runner_keys = st.get("phase2_runner_keys", PHASE2_RUNNER_KEYS)

    # Cheap gate first: are all confirmatory runs terminal?
    all_terminal, detail = _phase2_status(workspace, runner_keys)
    report = {"phase2_terminal": {"pass": all_terminal, "detail": detail}}
    if not all_terminal:
        st["stage"] = STAGE_CONFIRMATORY
        st["blocked_reason"] = "Phase 2/3 confirmatory runs are not all complete yet."
        st["gate_report"] = report
        save_ramp_state(workspace, st)
        return st

    # Confirmatory complete -> run the heavier safety gates.
    health_ok, health_report = _health_gate(workspace)
    evidence_ok, evidence_report = _evidence_gate(workspace, runner_keys)
    report["health"] = health_report
    report["evidence"] = evidence_report
    st["gate_report"] = report

    if not (health_ok and evidence_ok):
        st["stage"] = STAGE_HOLD
        st["blocked_reason"] = (
            "Confirmatory runs complete but a safety gate failed — held for human review. "
            f"health={health_report} evidence={evidence_report}"
        )
        save_ramp_state(workspace, st)
        return st

    # All gates pass. Promote ONLY with explicit human confirmation.
    # REAUTH GUARD: when reauth_required_at is set, any confirmation recorded
    # BEFORE that checkpoint is stale and does not count — the capability set
    # changed and the human must re-affirm against it. Enforced HERE, at the
    # single promotion choke point, so every confirm path (CLI, flag file,
    # dashboard POST, operator) inherits it.
    reauth_at = _parse_iso(st.get("reauth_required_at"))
    confirmed_at = _parse_iso(st.get("confirmed_at"))
    flag_path = _path(workspace, CONFIRM_FLAG)
    flag_at = None
    if flag_path.exists():
        flag_at = _parse_iso(flag_path.read_text(encoding="utf-8").strip()) if flag_path.stat().st_size else None
        if flag_at is None:
            try:
                flag_at = datetime.fromtimestamp(flag_path.stat().st_mtime, tz=timezone.utc)
            except OSError:
                flag_at = None

    if reauth_at is not None:
        human_ok = bool(
            (confirmed_at is not None and confirmed_at >= reauth_at)
            or (flag_at is not None and flag_at >= reauth_at)
        )
    else:
        human_ok = bool(st.get("confirmed_by_human")) or flag_path.exists()

    if human_ok:
        st["stage"] = STAGE_FULL_AUTONOMY
        st["promoted_at"] = _now()
        st["confirmed_by_human"] = True
        st["blocked_reason"] = ""
        if reauth_at is not None:
            st["reauth_cleared_at"] = _now()
            st["reauth_required_at"] = ""
    else:
        st["stage"] = STAGE_AWAITING_CONFIRM
        _reauth_note = str(st.get("reauth_note", "") or "")
        st["blocked_reason"] = (
            "All confirmatory runs complete and safety gates passed. "
            "AWAITING HUMAN FINAL CONFIRMATION to enable full autonomy — run "
            "`python tar_autonomy_ramp.py confirm` or create tar_state/autonomy_ramp_confirm.flag."
            + (
                " RE-AUTHORIZATION REQUIRED: a prior confirmation predates the reauth "
                f"checkpoint ({st.get('reauth_required_at')}). {_reauth_note}"
                if reauth_at is not None else ""
            )
        )
    save_ramp_state(workspace, st)
    return st


def confirm_promotion(workspace) -> dict:
    """Human action: give the final go. Only effective once the gates have passed.

    Records confirmed_at so the reauth guard in evaluate_ramp can distinguish a
    FRESH confirmation (valid — the human re-affirmed against the current
    capability set) from a stale one predating reauth_required_at.
    """
    st = load_ramp_state(workspace) or init_ramp(workspace)
    st["confirmed_by_human"] = True
    st["confirmed_at"] = _now()
    save_ramp_state(workspace, st)
    try:
        _path(workspace, CONFIRM_FLAG).write_text(_now(), encoding="utf-8")
    except Exception:
        pass
    return evaluate_ramp(workspace)


def status_line(workspace) -> str:
    st = load_ramp_state(workspace)
    if not st:
        return "autonomy ramp: not configured (full autonomy ungated)"
    if not st.get("enabled"):
        return "autonomy ramp: disabled"
    return f"autonomy ramp: stage={st.get('stage')} — {st.get('blocked_reason') or 'ok'}"


def _main():
    import sys
    from tar_storage import resolve_workspace
    ws = resolve_workspace(Path(__file__).resolve().parent)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "init":
        print(json.dumps(init_ramp(ws), indent=2))
    elif cmd == "evaluate":
        print(json.dumps(evaluate_ramp(ws), indent=2))
    elif cmd == "confirm":
        print(json.dumps(confirm_promotion(ws), indent=2))
    elif cmd == "disable":
        disable_ramp(ws); print("ramp disabled")
    else:
        print(status_line(ws))


if __name__ == "__main__":
    _main()
