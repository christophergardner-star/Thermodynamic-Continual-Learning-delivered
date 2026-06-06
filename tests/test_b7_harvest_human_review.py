"""Workstream B7 — harvest REAL self-improvement signals from human review.

The operator-LoRA loop trained only on 29 hand-authored signals. Every human review
decision is free, real supervision. harvest_human_review_signals() turns each decided
review entry into a TrainingSignalRecord whose gold response is the human's OWN decision +
notes (no fabrication), routed through curate_signal() (anchor/overclaim rejection intact),
idempotent per review_id.
"""
import json
from pathlib import Path

from tar_lab.self_improvement import SelfImprovementEngine


def _anchor(engine: SelfImprovementEngine, tmp_path: Path):
    pack_dir = tmp_path / "eval_artifacts" / "external_validation"
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "run_manifest.json").write_text(json.dumps({"pack": "sealed"}), encoding="utf-8")
    (pack_dir / "predictions.jsonl").write_text(
        json.dumps({"item_id": "anchor-item-001"}) + "\n", encoding="utf-8")
    engine.initialize_anchor_pack(
        pack_path="eval_artifacts/external_validation",
        run_manifest_path=str(pack_dir / "run_manifest.json"),
        baseline_mean_score=0.5, baseline_overclaim_rate=0.0)


def _write_review_state(tmp_path: Path):
    state = {
        "claim_reviews": [
            {"review_id": "claim:paper-x", "paper_id": "paper-x",
             "title": "Conservative TCL reduces forgetting", "decision": "approve_claim_scope",
             "status": "approved"},
        ],
        "proposals": [
            {"review_id": "proposal:probe-1", "frontier_problem_id": "fp-cf",
             "decision": "approve_and_build_manifest", "status": "approved_manifest_ready"},
        ],
        "history": [
            {"review_id": "proposal:probe-2", "kind": "proposal",
             "decision": "hold_pending_more_evidence", "human_notes": "need n>=10 seeds"},
            {"review_id": "proposal:probe-1", "kind": "proposal",  # DUPLICATE -> deduped
             "decision": "approve_and_build_manifest"},
            {"review_id": "proposal:probe-3", "kind": "proposal",  # no decision -> skipped
             "decision": "", "human_notes": ""},
        ],
    }
    p = tmp_path / "tar_state" / "human_review_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state), encoding="utf-8")


def test_harvests_real_decisions_with_correct_kinds(tmp_path):
    engine = SelfImprovementEngine(str(tmp_path))
    _anchor(engine, tmp_path)
    _write_review_state(tmp_path)

    summary = engine.harvest_human_review_signals()
    assert summary["harvested"] == 3
    assert summary["skipped_no_decision"] == 1            # probe-3 had no decision
    assert summary["by_kind"] == {
        "claim_verdict": 1,        # claim review
        "research_decision": 1,    # approved proposal
        "evidence_assessment": 1,  # hold_pending_more_evidence
    }
    assert len(engine.list_signals()) == 3


def test_gold_response_is_the_real_human_decision_no_fabrication(tmp_path):
    engine = SelfImprovementEngine(str(tmp_path))
    _anchor(engine, tmp_path)
    _write_review_state(tmp_path)
    engine.harvest_human_review_signals()

    by_source = {s.source_id: s for s in engine.list_signals()}
    sig = by_source["proposal:probe-2"]
    gold = json.loads(sig.gold_response)
    assert gold["decision"] == "hold_pending_more_evidence"   # the human's own decision
    assert gold["rationale"] == "need n>=10 seeds"            # the human's own notes
    assert sig.kind == "evidence_assessment"
    assert sig.quality_score == 0.9                           # notes present -> higher


def test_idempotent_reharvest_no_duplicates(tmp_path):
    engine = SelfImprovementEngine(str(tmp_path))
    _anchor(engine, tmp_path)
    _write_review_state(tmp_path)
    engine.harvest_human_review_signals()
    engine.harvest_human_review_signals()  # second pass
    assert len(engine.list_signals()) == 3  # stable signal_id -> overwrite, not duplicate


def test_safe_without_anchor_pack(tmp_path):
    engine = SelfImprovementEngine(str(tmp_path))
    _write_review_state(tmp_path)  # no anchor initialized
    summary = engine.harvest_human_review_signals()
    assert summary["error"] == "anchor_pack_not_initialized"
    assert summary["harvested"] == 0


def test_safe_when_state_missing(tmp_path):
    engine = SelfImprovementEngine(str(tmp_path))
    _anchor(engine, tmp_path)  # anchor exists, but no human_review_state.json
    summary = engine.harvest_human_review_signals()
    assert summary["error"] == "state_unreadable"
    assert summary["harvested"] == 0
