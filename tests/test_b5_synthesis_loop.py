"""Workstream B5 — close the method-synthesis loop on CPU (RAIL-3, human-gated).

method_synthesizer never fired from a novel-idea path (its only caller auto-adopts). The
synthesis_loop driver turns a top variant proposal into a VALIDATED candidate recorded
PENDING HUMAN APPROVAL — and crucially NEVER auto-adopts: candidates are quarantined where
load_generated_methods (which globs only synthesized_methods/*.py) cannot pick them up.

These tests inject a fake synthesize_fn so they run fully offline (no LLM, no torch).
"""
import json
from pathlib import Path

from tar_lab import synthesis_loop as sl


def _write_proposals(ws: Path, proposals):
    p = ws / "tar_state" / "method_refinement" / "variant_proposals.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"proposals": proposals}), encoding="utf-8")


def _proposal(pid="tcl-split_cifar10-stronger-penalty", fm=0.15):
    return {
        "proposal_id": pid, "parent_method": "tcl", "dataset": "split_cifar10",
        "suggested_change": {"param": "penalty_strength", "direction": "increase", "factor": 1.5},
        "rationale": "mean forgetting too high", "status": "proposed",
        "evidence": {"forgetting_mean": fm},
    }


def _fake_synth(idea, workspace, *, out_dir, log_fn=print):
    """Stand-in for synthesize_and_validate_method: writes a candidate file to out_dir."""
    p = Path(out_dir) / "novel_cl_method.py"
    p.write_text("# synthesized candidate\nclass NovelCL:\n    pass\n", encoding="utf-8")
    return {"success": True, "method_key": "novel_cl_method", "class_name": "NovelCL",
            "saved_path": str(p), "description": "a novel CL method", "error": None}


def _fail_synth(idea, workspace, *, out_dir, log_fn=print):
    return {"success": False, "method_key": "", "class_name": "", "saved_path": None,
            "description": "", "error": "sandbox VALIDATION_FAILED"}


def _enable(ws: Path):
    (ws / "tar_state").mkdir(parents=True, exist_ok=True)
    (ws / "tar_state" / sl.ENABLE_FLAG).write_text("on", encoding="utf-8")


# ---- gating ---------------------------------------------------------------------

def test_disabled_by_default(tmp_path):
    _write_proposals(tmp_path, [_proposal()])
    out = sl.run_synthesis_from_proposals(tmp_path, synthesize_fn=_fake_synth)
    assert out["enabled"] is False and out["attempted"] == 0
    assert "method_synthesis.enabled" in out["reason"]
    # nothing written
    assert sl.load_pending_candidates(tmp_path) == {}


def test_enabled_but_no_proposals(tmp_path):
    _enable(tmp_path)
    _write_proposals(tmp_path, [])
    out = sl.run_synthesis_from_proposals(tmp_path, synthesize_fn=_fake_synth)
    assert out["enabled"] is True and out["attempted"] == 0
    assert "no 'proposed'" in out["reason"]


# ---- the loop: synthesise -> pending (NEVER adopt) ------------------------------

def test_validated_candidate_is_recorded_pending_and_quarantined(tmp_path):
    _enable(tmp_path)
    _write_proposals(tmp_path, [_proposal()])
    out = sl.run_synthesis_from_proposals(tmp_path, synthesize_fn=_fake_synth)

    assert out["enabled"] is True and out["attempted"] == 1 and out["validated"] == 1
    cand = out["candidates"][0]
    assert cand["status"] == "pending_human_approval" and cand["rail"] == 3
    assert cand["source_proposal_id"] == "tcl-split_cifar10-stronger-penalty"

    # SAFETY: the candidate file is in the QUARANTINE dir, NOT the live (auto-loaded) one
    quarantine = tmp_path / "tar_state" / "synthesized_methods_pending"
    live = tmp_path / "tar_state" / "synthesized_methods"
    assert (quarantine / "novel_cl_method.py").exists()
    assert not live.exists() or not (live / "novel_cl_method.py").exists()

    # round-trips through the registry
    loaded = sl.load_pending_candidates(tmp_path)
    assert loaded["summary"]["pending"] == 1


def test_synthesis_failure_records_no_candidate(tmp_path):
    _enable(tmp_path)
    _write_proposals(tmp_path, [_proposal()])
    out = sl.run_synthesis_from_proposals(tmp_path, synthesize_fn=_fail_synth)
    assert out["validated"] == 0 and out["candidates"] == []
    assert out["errors"] and "VALIDATION_FAILED" in out["errors"][0]


def test_idempotent_no_duplicate_candidates(tmp_path):
    _enable(tmp_path)
    _write_proposals(tmp_path, [_proposal()])
    sl.run_synthesis_from_proposals(tmp_path, synthesize_fn=_fake_synth)
    out2 = sl.run_synthesis_from_proposals(tmp_path, synthesize_fn=_fake_synth)
    assert len(out2["candidates"]) == 1  # stable candidate_id -> no duplicate


def test_top_proposal_is_worst_forgetting_first(tmp_path):
    _enable(tmp_path)
    _write_proposals(tmp_path, [_proposal("low", fm=0.05), _proposal("high", fm=0.40)])
    out = sl.run_synthesis_from_proposals(tmp_path, max_candidates=1, synthesize_fn=_fake_synth)
    assert out["candidates"][0]["source_proposal_id"] == "high"


# ---- human-gated adoption -------------------------------------------------------

def test_approve_moves_candidate_into_live_registry_dir(tmp_path):
    _enable(tmp_path)
    _write_proposals(tmp_path, [_proposal()])
    out = sl.run_synthesis_from_proposals(tmp_path, synthesize_fn=_fake_synth)
    cid = out["candidates"][0]["candidate_id"]

    res = sl.approve_pending_candidate(tmp_path, cid, approved_by="Christopher Gardner")
    assert res["approved"] is True
    live_file = tmp_path / "tar_state" / "synthesized_methods" / "novel_cl_method.py"
    assert live_file.exists()  # now load_generated_methods would pick it up
    # status updated + re-approval refused
    again = sl.approve_pending_candidate(tmp_path, cid, approved_by="x")
    assert again["approved"] is False and "not_pending" in again["reason"]


def test_approve_requires_human_attribution(tmp_path):
    _enable(tmp_path)
    _write_proposals(tmp_path, [_proposal()])
    out = sl.run_synthesis_from_proposals(tmp_path, synthesize_fn=_fake_synth)
    cid = out["candidates"][0]["candidate_id"]
    res = sl.approve_pending_candidate(tmp_path, cid, approved_by="")
    assert res["approved"] is False and "approved_by" in res["reason"]


def test_approve_unknown_candidate(tmp_path):
    _enable(tmp_path)
    res = sl.approve_pending_candidate(tmp_path, "synth-nope", approved_by="x")
    assert res["approved"] is False and res["reason"] == "candidate_not_found"


# ---- idea construction ----------------------------------------------------------

def test_proposal_to_idea_asks_for_a_new_mechanism(tmp_path):
    idea = sl.proposal_to_idea(_proposal())
    assert "tcl" in idea and "split_cifar10" in idea
    assert "NEW" in idea and "not a hyperparameter tweak" in idea
    assert "increase penalty_strength" in idea
