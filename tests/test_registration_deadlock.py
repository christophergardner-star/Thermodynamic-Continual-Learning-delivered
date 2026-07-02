"""
Regression test for the director-proposal registration deadlock.

Before the fix, register_director_proposal was only ever called by the
scheduler for specs already IN the orchestrator queue — but a spec only
entered the queue once approved, and approval could only begin after
registration. A genuinely-new director proposal could therefore never start
its 24h veto clock and could never run, even with the autonomy ramp released.

The fix registers in-scope, not-yet-approved proposals at proposal time
(tar_living_research._register_spec_proposal, called from
_ensure_director_seeded_queue). These tests verify the pipeline semantics at
the human_review level (torch-free) and the helper itself (imports the
daemon module, so it is skipped automatically if torch cannot load — the
known Windows torch-DLL-under-pytest flake while the live daemon holds torch).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tar_lab.human_review import (
    VetoWindowApproval,
    approved_experiment_ids,
    load_director_proposals,
    register_director_proposal,
)


def _make_ws(tmp_path: Path) -> Path:
    (tmp_path / "tar_state").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_registration_starts_veto_clock_and_auto_approves(tmp_path):
    ws = _make_ws(tmp_path)

    assert register_director_proposal(
        ws, experiment_id="director-gap-probe-1", name="gap probe",
        frontier_id="fp-gap-x", priority=132, context_why="deadlock regression",
    ) is True

    proposals = {p["experiment_id"]: p for p in load_director_proposals(ws)}
    entry = proposals["director-gap-probe-1"]
    assert entry["status"] == "pending_veto"

    # Not approved while the window is open.
    assert "director-gap-probe-1" not in approved_experiment_ids(ws)

    # Expire the window (simulate 24h passing) -> auto-approved.
    path = ws / "tar_state" / "director_proposals.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[0]["auto_approve_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    ).isoformat()
    path.write_text(json.dumps(data), encoding="utf-8")

    assert "director-gap-probe-1" in approved_experiment_ids(ws)
    refreshed = {p["experiment_id"]: p for p in load_director_proposals(ws)}
    assert refreshed["director-gap-probe-1"]["status"] == "auto_approved"


def test_reregistration_is_idempotent_and_never_resets_clock(tmp_path):
    ws = _make_ws(tmp_path)
    register_director_proposal(ws, experiment_id="exp-a", name="a")
    first = load_director_proposals(ws)[0]["auto_approve_at"]

    # Second registration (as happens every seeding cycle) must be a no-op.
    assert register_director_proposal(ws, experiment_id="exp-a", name="a") is False
    assert load_director_proposals(ws)[0]["auto_approve_at"] == first
    assert len(load_director_proposals(ws)) == 1


def test_veto_blocks_approval_permanently(tmp_path):
    ws = _make_ws(tmp_path)
    register_director_proposal(ws, experiment_id="exp-bad", name="bad idea")
    assert VetoWindowApproval(ws).veto("exp-bad", reason="not in scope")

    # Even after the window would have expired, vetoed stays blocked.
    path = ws / "tar_state" / "director_proposals.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[0]["auto_approve_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    ).isoformat()
    path.write_text(json.dumps(data), encoding="utf-8")
    assert "exp-bad" not in approved_experiment_ids(ws)


def test_seeding_helper_registers_unapproved_spec(tmp_path):
    tlr = pytest.importorskip(
        "tar_living_research",
        reason="torch DLL unavailable under pytest while the live daemon holds torch",
    )
    ws = _make_ws(tmp_path)
    spec = SimpleNamespace(
        id="director-fp-gap-probe-9",
        name="gap probe 9",
        frontier_problem_id="fp-gap-9",
        priority=132,
        description="registration-deadlock regression spec",
    )
    tlr._register_spec_proposal(ws, spec)

    proposals = {p["experiment_id"]: p for p in load_director_proposals(ws)}
    assert "director-fp-gap-probe-9" in proposals
    entry = proposals["director-fp-gap-probe-9"]
    assert entry["status"] == "pending_veto"
    assert entry["priority"] == 132
    assert entry["frontier_problem_id"] == "fp-gap-9"
