"""
Safety regression tests for the autonomy ramp.

T2.4 — is_full_autonomy must FAIL CLOSED: a configured ramp whose state file
is deleted or corrupted must never ungate director-generated experiments.

T2.5 — reauth_required_at must be enforced at the promotion choke point:
confirmations recorded BEFORE the reauth checkpoint are stale and must not
promote; a FRESH confirm_promotion() must.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import tar_autonomy_ramp as ramp


def _ws(tmp_path: Path) -> Path:
    (tmp_path / "tar_state").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _ramp_path(ws: Path) -> Path:
    return ws / "tar_state" / ramp.RAMP_FILE


def _write_terminal_queue(ws: Path, runner_keys: list[str]) -> None:
    """Make every confirmatory run look terminal so gates can be reached."""
    (ws / "tar_state" / "experiment_archive.json").write_text(json.dumps({
        "experiments": [
            {"runner_key": rk, "status": "complete", "result_path": ""}
            for rk in runner_keys
        ]
    }), encoding="utf-8")


# ---------------------------------------------------------------- T2.4 ------

def test_never_configured_stays_opt_in(tmp_path):
    ws = _ws(tmp_path)
    assert ramp.is_full_autonomy(ws) is True  # documented fresh-install default


def test_missing_file_after_configuration_fails_closed(tmp_path):
    ws = _ws(tmp_path)
    ramp.init_ramp(ws)
    assert ramp.is_full_autonomy(ws) is False  # confirmatory stage
    _ramp_path(ws).unlink()                     # simulate deletion attack/accident
    assert ramp.is_full_autonomy(ws) is False   # sentinel keeps it closed


def test_corrupt_file_fails_closed(tmp_path):
    ws = _ws(tmp_path)
    ramp.init_ramp(ws)
    _ramp_path(ws).write_text("{not valid json", encoding="utf-8")
    assert ramp.is_full_autonomy(ws) is False
    _ramp_path(ws).write_text(json.dumps(["wrong", "shape"]), encoding="utf-8")
    assert ramp.is_full_autonomy(ws) is False


def test_malformed_dict_without_enabled_key_fails_closed(tmp_path):
    ws = _ws(tmp_path)
    ramp.init_ramp(ws)
    # A parseable dict that never went through init_ramp/disable_ramp (no
    # 'enabled' key) must NOT be read as an opt-out — fail closed.
    _ramp_path(ws).write_text(json.dumps({"stage": "full_autonomy"}), encoding="utf-8")
    assert ramp.is_full_autonomy(ws) is False


def test_explicit_disable_is_the_only_ungated_path(tmp_path):
    ws = _ws(tmp_path)
    ramp.init_ramp(ws)
    ramp.disable_ramp(ws)  # deliberate human opt-out
    assert ramp.is_full_autonomy(ws) is True


# ---------------------------------------------------------------- T2.5 ------

def test_stale_confirmation_does_not_promote_when_reauth_required(tmp_path):
    ws = _ws(tmp_path)
    st = ramp.init_ramp(ws, runner_keys=["rk1"])
    _write_terminal_queue(ws, ["rk1"])

    past = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    st["confirmed_by_human"] = True          # stale authorization…
    st["confirmed_at"] = past                # …recorded before the checkpoint
    st["reauth_required_at"] = (
        datetime.now(timezone.utc) - timedelta(days=1)
    ).isoformat()
    st["reauth_note"] = "capability set changed; re-affirm"
    ramp.save_ramp_state(ws, st)

    # health gate would normally run; force it deterministic-pass
    out = _evaluate_with_pass_gates(ws)
    assert out["stage"] == ramp.STAGE_AWAITING_CONFIRM
    assert "RE-AUTHORIZATION REQUIRED" in out["blocked_reason"]
    assert ramp.is_full_autonomy(ws) is False


def test_fresh_confirm_clears_reauth_and_promotes(tmp_path):
    ws = _ws(tmp_path)
    st = ramp.init_ramp(ws, runner_keys=["rk1"])
    _write_terminal_queue(ws, ["rk1"])
    st["reauth_required_at"] = (
        datetime.now(timezone.utc) - timedelta(days=1)
    ).isoformat()
    ramp.save_ramp_state(ws, st)

    # Fresh human confirm (records confirmed_at = now, after the checkpoint).
    _confirm_with_pass_gates(ws)
    out = ramp.load_ramp_state(ws)
    assert out["stage"] == ramp.STAGE_FULL_AUTONOMY
    assert out["reauth_required_at"] == ""
    assert out["reauth_cleared_at"]
    assert ramp.is_full_autonomy(ws) is True


def test_touched_confirm_flag_does_not_bypass_reauth(tmp_path):
    # A contentless confirm flag (e.g. created by a stray filesystem touch)
    # must NOT satisfy a reauth checkpoint — only a timestamped confirmation at/
    # after the checkpoint counts. Guards against the mtime-spoof vector.
    ws = _ws(tmp_path)
    st = ramp.init_ramp(ws, runner_keys=["rk1"])
    _write_terminal_queue(ws, ["rk1"])
    st["reauth_required_at"] = (
        datetime.now(timezone.utc) - timedelta(days=1)
    ).isoformat()
    ramp.save_ramp_state(ws, st)
    # Empty flag file "touched" now (fresh mtime, but no content timestamp).
    (ws / "tar_state" / ramp.CONFIRM_FLAG).write_text("", encoding="utf-8")
    out = _evaluate_with_pass_gates(ws)
    assert out["stage"] == ramp.STAGE_AWAITING_CONFIRM
    assert ramp.is_full_autonomy(ws) is False


def test_legacy_flow_without_reauth_still_promotes(tmp_path):
    ws = _ws(tmp_path)
    ramp.init_ramp(ws, runner_keys=["rk1"])
    _write_terminal_queue(ws, ["rk1"])
    _confirm_with_pass_gates(ws)
    assert ramp.load_ramp_state(ws)["stage"] == ramp.STAGE_FULL_AUTONOMY


# ------------------------------------------------ WS2.1a artifact gate ------
# A phase-2 confirmatory run launched from the dashboard/sequencer never creates
# an experiment_queue/archive entry, so the queue-only scan reported it forever
# 'not_found' and the ramp gate stuck. _phase2_status now also honours a completed
# comparison artifact — but strictly: it must be complete, un-quarantined, and (vs
# a reauth checkpoint) recent. These tests pin that it neither over- nor under-fires.

def test_dashboard_artifact_counts_as_terminal(tmp_path):
    ws = _ws(tmp_path)
    since = datetime.now(timezone.utc) - timedelta(days=2)
    _write_artifact(ws, "hpc_replication_", datetime.now(timezone.utc),
                    {"verdict": "REPLICATION_SUCCESS", "per_seed_results": [1, 2]})
    all_terminal, detail = ramp._phase2_status(
        ws, ["hpc_replication_phase2"], since=since)
    assert all_terminal is True
    assert detail["hpc_replication_phase2"] == "complete_artifact"


def test_stale_artifact_before_reauth_is_ignored(tmp_path):
    ws = _ws(tmp_path)
    since = datetime.now(timezone.utc) - timedelta(days=1)
    stale = datetime.now(timezone.utc) - timedelta(days=30)
    _write_artifact(ws, "hpc_replication_", stale, {"verdict": "REPLICATION_SUCCESS"})
    all_terminal, detail = ramp._phase2_status(
        ws, ["hpc_replication_phase2"], since=since)
    assert all_terminal is False
    assert detail["hpc_replication_phase2"] == "not_found"


def test_quarantined_artifact_is_ignored(tmp_path):
    ws = _ws(tmp_path)
    since = datetime.now(timezone.utc) - timedelta(days=2)
    _write_artifact(ws, "hpc_replication_", datetime.now(timezone.utc),
                    {"verdict": "REPLICATION_SUCCESS"}, quarantined=True)
    all_terminal, detail = ramp._phase2_status(
        ws, ["hpc_replication_phase2"], since=since)
    assert all_terminal is False
    assert detail["hpc_replication_phase2"] == "not_found"


def test_incomplete_artifact_is_ignored(tmp_path):
    ws = _ws(tmp_path)
    since = datetime.now(timezone.utc) - timedelta(days=2)
    _write_artifact(ws, "hpc_replication_", datetime.now(timezone.utc),
                    {"note": "run started", "per_seed_results": []})
    all_terminal, detail = ramp._phase2_status(
        ws, ["hpc_replication_phase2"], since=since)
    assert all_terminal is False
    assert detail["hpc_replication_phase2"] == "not_found"


def test_evaluate_ramp_reaches_awaiting_confirm_via_artifacts(tmp_path):
    ws = _ws(tmp_path)
    st = ramp.init_ramp(ws, runner_keys=["hpc_replication_phase2"])
    st["reauth_required_at"] = (
        datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    ramp.save_ramp_state(ws, st)
    _write_artifact(ws, "hpc_replication_", datetime.now(timezone.utc),
                    {"verdict": "REPLICATION_SUCCESS", "per_seed_results": [1]})
    out = _evaluate_with_pass_gates(ws)
    assert out["gate_report"]["phase2_terminal"]["pass"] is True
    # gate is satisfied, but promotion still needs a FRESH human confirm.
    assert out["stage"] == ramp.STAGE_AWAITING_CONFIRM
    assert ramp.is_full_autonomy(ws) is False


# ------------------------------------------------------------- helpers ------

def _write_artifact(ws, prefix, when, payload, quarantined=False):
    comp = ws / "tar_state" / "comparisons"
    comp.mkdir(parents=True, exist_ok=True)
    p = comp / f"{prefix}{when.strftime('%Y%m%dT%H%M%SZ')}.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    if quarantined:
        idx = comp / "canonical_results_index.jsonl"
        with idx.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"result_path": str(p), "quarantined": True}) + "\n")
    return p


def _pass_health(_ws):
    return True, {"pass": True, "failed": [], "summary": {}}


def _evaluate_with_pass_gates(ws):
    orig = ramp._health_gate
    ramp._health_gate = _pass_health
    try:
        return ramp.evaluate_ramp(ws)
    finally:
        ramp._health_gate = orig


def _confirm_with_pass_gates(ws):
    orig = ramp._health_gate
    ramp._health_gate = _pass_health
    try:
        return ramp.confirm_promotion(ws)
    finally:
        ramp._health_gate = orig
