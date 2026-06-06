"""Workstream B6 — wire the two dormant advisory learners (rebuild + load contract).

operational_learner + authoring_learner had no caller (their rebuild_* was never invoked),
so they produced nothing. B6 wires them into the director cycle (opt-in) and surfaces them
at /api/self_improvement. These tests pin the rebuild -> file -> load round-trip + the summary
shape the director/dashboard rely on. Both learners are read-only + advisory (operational
never tunes/executes; authoring is style-only) — proven append-only elsewhere.
"""
import json
from pathlib import Path

from tar_lab import operational_learner as opl
from tar_lab import authoring_learner as aul


def _ws(tmp_path: Path) -> Path:
    (tmp_path / "tar_state").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_operational_rebuild_flags_restart_pressure(tmp_path):
    ws = _ws(tmp_path)
    (ws / "tar_state" / "watchdog_state.json").write_text(json.dumps({
        "services": {"dashboard": {"restart_count": 25, "reason": "http_unhealthy"}}
    }), encoding="utf-8")

    reg = opl.rebuild_operational_recommendations(ws)
    assert reg["advisory_only"] is True
    assert reg["summary"]["high"] >= 1
    rec = next(r for r in reg["recommendations"] if r["pattern"] == "dashboard_restart_pressure")
    assert rec["severity"] == "high"
    assert rec["suggested_param_tuning"]  # advisory param suggestion present

    # round-trips through the file the dashboard reads
    loaded = opl.load_recommendations(ws)
    assert loaded["summary"] == reg["summary"]
    assert (ws / "tar_state" / "operational" / "operational_recommendations.json").exists()


def test_operational_quiet_when_no_pressure(tmp_path):
    ws = _ws(tmp_path)
    (ws / "tar_state" / "watchdog_state.json").write_text(json.dumps({
        "services": {"daemon": {"restart_count": 1, "reason": "ok"}}
    }), encoding="utf-8")
    reg = opl.rebuild_operational_recommendations(ws)
    assert reg["summary"]["recommendations"] == 0  # below thresholds -> nothing


def test_authoring_rebuild_learns_accept_vs_cut(tmp_path):
    ws = _ws(tmp_path)
    (ws / "tar_state" / "human_review_state.json").write_text(json.dumps({
        "claim_reviews": [
            {"status": "approved", "decision": "approve_claim_scope", "human_notes": "clear, bounded scope"},
            {"status": "revision_requested", "decision": "", "human_notes": "overclaims beyond evidence"},
        ],
        "history": [],
    }), encoding="utf-8")

    mem = aul.rebuild_style_memory(ws)
    assert mem["style_only_never_facts"] is True
    s = mem["summary"]
    assert s["claim_decisions"] == 2 and s["accepted"] == 1 and s["cut_or_revised"] == 1
    assert s["accept_rate"] == 0.5
    assert "clear, bounded scope" in mem["accepted_style_notes"]
    assert "overclaims beyond evidence" in mem["cut_style_notes"]

    loaded = aul.load_style_memory(ws)
    assert loaded["summary"] == s
    assert (ws / "tar_state" / "authoring" / "authoring_style_memory.json").exists()


def test_loads_empty_before_any_rebuild(tmp_path):
    ws = _ws(tmp_path)
    # the dashboard route must be safe before either learner has ever run
    assert opl.load_recommendations(ws) == {}
    assert aul.load_style_memory(ws) == {}
