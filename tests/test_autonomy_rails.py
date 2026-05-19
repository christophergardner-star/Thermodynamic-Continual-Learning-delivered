from __future__ import annotations

import json
from pathlib import Path

from tar_lab.human_review import (
    answer_human_question,
    approved_experiment_ids,
    approved_paper_ids,
    load_human_review_state,
    record_review_decision,
    sync_human_review_from_director_state,
)
from tar_lab.runtime_ledger import RuntimeLeaseError, acquire_runtime_lease, active_runtime_leases
from tar_lab.validation import (
    TRUST_CORRECTED_INTERNAL,
    TRUST_TRUSTED_RERUN,
    build_validation_state,
    classify_trust_tier,
)


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "tar_state" / "comparisons").mkdir(parents=True, exist_ok=True)
    return ws


def test_runtime_ledger_refuses_duplicate_experiment_leases(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    acquire_runtime_lease(
        ws,
        component_id="orchestrator:phase12",
        component_kind="experiment",
        experiment_id="phase12_ewc_sweep",
        conflict_keys=["experiment:phase12_ewc_sweep"],
        stale_timeout_s=3600.0,
    )
    try:
        acquire_runtime_lease(
            ws,
            component_id="rerun_chain:phase12",
            component_kind="rerun_phase",
            experiment_id="phase12_ewc_sweep",
            conflict_keys=["experiment:phase12_ewc_sweep"],
            stale_timeout_s=3600.0,
        )
        assert False, "expected duplicate runtime lease refusal"
    except RuntimeLeaseError:
        pass
    assert len(active_runtime_leases(ws)) == 1


def test_human_review_sync_and_approval_flow(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    director_state = {
        "experiment_directives": [
            {
                "experiment_id": "phase15_class_incremental_search",
                "title": "Phase 15 - Class-Incremental Search",
                "frontier_problem_id": "fp-class-incremental",
                "frontier_problem_title": "Class-Incremental Learning",
                "dataset": "split_cifar10",
                "method": "tcl",
                "seeds": [42, 0, 1],
                "priority_score": 91.0,
                "experiment_goal": "Test class-incremental survival.",
                "scheduler_intent": "propose_now",
                "proposal_kind": "frontier_probe",
                "proposal_origin": "director",
            }
        ],
        "paper_directives": [
            {
                "paper_id": "frontier-paper-fp-class-incremental",
                "title": "Continuous Class Expansion Without Task Labels",
                "frontier_problem_id": "fp-class-incremental",
                "truth_status": "weak",
                "readiness": "planned",
                "waiting_for_experiments": ["phase15_class_incremental_search"],
            }
        ],
        "frontier_directives": [
            {
                "problem_id": "fp-class-incremental",
                "truth_status": "weak",
            }
        ],
    }
    state = sync_human_review_from_director_state(ws, director_state)
    assert len(state["proposals"]) == 1
    assert len(state["claim_reviews"]) == 1
    assert len(state["questions"]) >= 1

    record_review_decision(
        ws,
        review_id="proposal:phase15_class_incremental_search",
        decision="approve_and_build_manifest",
        human_notes="Run this after the current validation lane completes.",
        build_manifest_authorised=True,
    )
    record_review_decision(
        ws,
        review_id="claim:frontier-paper-fp-class-incremental",
        decision="approve_paper_rewrite",
        human_notes="Rewrite only after validated evidence arrives.",
    )
    answer_human_question(
        ws,
        question_id="question:paper:frontier-paper-fp-class-incremental:waiting_for",
        answer="hold_pending_more_evidence",
        answer_notes="Do not narrow the claim until reruns land.",
    )

    synced = load_human_review_state(ws)
    proposal = synced["proposals"][0]
    assert "phase15_class_incremental_search" in approved_experiment_ids(ws)
    assert "frontier-paper-fp-class-incremental" in approved_paper_ids(ws)
    assert any(item.get("status") == "answered" for item in synced["questions"])
    manifest_path = Path(str(proposal.get("manifest_path", "") or ""))
    assert manifest_path.exists()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["content_hash"] != "UNSIGNED"
    assert manifest_payload["experiments"][0]["experiment_id"] == "phase15_class_incremental_search"


def test_validation_trust_tiers_distinguish_env_backed_vs_corrected(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    comp = ws / "tar_state" / "comparisons"
    trusted = comp / "phase12_ewc_sweep__20260512T000000Z.json"
    trusted_env = comp / "phase12_ewc_sweep__20260512T000000Z_env.json"
    corrected = comp / "phase13_si_sweep_corrected__20260512T000000Z.json"
    trusted.write_text(json.dumps({"aggregate": {"tcl": {"forgetting_mean": 0.11}}}), encoding="utf-8")
    trusted_env.write_text(json.dumps({"artifact_schema": "tar_env_snapshot_v1"}), encoding="utf-8")
    corrected.write_text(json.dumps({"aggregate": {"tcl": {"forgetting_mean": 0.12}}}), encoding="utf-8")
    index = comp / "canonical_results_index.jsonl"
    index.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "logical_name": "phase12_ewc_sweep",
                        "phase_number": 12,
                        "result_path": str(trusted),
                        "env_path": str(trusted_env),
                        "created_at": "2026-05-12T00:00:00+00:00",
                    }
                ),
                json.dumps(
                    {
                        "logical_name": "phase13_si_sweep",
                        "phase_number": 13,
                        "result_path": str(corrected),
                        "env_path": "",
                        "created_at": "2026-05-12T00:00:00+00:00",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    state = build_validation_state(ws, persist=False)
    trusts = {
        rec["trust"]["logical_name"]: rec["trust"]["trust_tier"]
        for rec in state["results"]
    }
    assert trusts["phase12_ewc_sweep"] == TRUST_TRUSTED_RERUN
    assert trusts["phase13_si_sweep"] == TRUST_CORRECTED_INTERNAL

    manual = classify_trust_tier(
        ws,
        record={
            "logical_name": "phase12_ewc_sweep",
            "result_path": str(trusted),
            "env_path": str(trusted_env),
        },
        result_payload={"aggregate": {"tcl": {"forgetting_mean": 0.11}}},
    )
    assert manual["publication_allowed"] is True
