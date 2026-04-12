from __future__ import annotations

import json
from pathlib import Path

from build_tar_master_dataset import build_master_dataset
from generate_ws23_dataset_campaign import build_campaign_workspace
from generate_ws26_tcl_campaign import build_tcl_workspace


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_master_dataset_generates_expected_families(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    state_dir = repo_root / "tar_state"
    output_dir = repo_root / "dataset_artifacts" / "tar_master_dataset_v1"
    manifests_dir = state_dir / "manifests"
    state_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)

    _write_jsonl(
        state_dir / "problem_studies.jsonl",
        [
            {
                "problem_id": "problem-1",
                "problem": "Investigate retrieval failures in NLP",
                "domain": "natural_language_processing",
                "profile_id": "natural_language_processing",
                "resolution_confidence": 0.65,
                "hypotheses": ["Retrieval hurts answer quality under weak grounding."],
                "experiments": [
                    {
                        "template_id": "prompt_retrieval_ablation",
                        "name": "Prompt Retrieval Ablation",
                        "benchmark": "retrieval_quality",
                        "benchmark_tier": "canonical",
                        "metrics": ["rouge", "hallucination_rate"],
                        "success_criteria": ["hallucination_rate drops"],
                    }
                ],
                "benchmark_tier": "canonical",
                "benchmark_ids": ["beir_fiqa_canonical"],
                "benchmark_names": ["BEIR FiQA"],
                "benchmark_truth_statuses": ["unsupported"],
                "benchmark_alignment": "refused",
                "canonical_comparable": False,
                "environment": {
                    "reproducibility_complete": False,
                    "unresolved_packages": ["evaluate"],
                    "lock_incomplete_reason": "Missing pinned versions for required packages: evaluate.",
                },
                "next_action": "Refuse canonical claim and build a validation-grade retrieval study.",
                "status": "planned",
            }
        ],
    )
    _write_jsonl(
        state_dir / "problem_executions.jsonl",
        [
            {
                "problem_id": "problem-1",
                "problem": "Investigate retrieval failures in NLP",
                "domain": "natural_language_processing",
                "status": "dependency_failure",
                "execution_mode": "local_python",
                "summary": "Dependency validation failed before experiment execution.",
                "imports_ok": ["numpy"],
                "imports_failed": [{"module": "evaluate", "error": "No module named 'evaluate'"}],
                "recommended_next_step": "Install dependencies and rerun the study.",
                "sandbox_policy": {
                    "mode": "docker_only",
                    "profile": "production",
                    "network_policy": "off",
                    "read_only_mounts": ["/workspace", "/data"],
                    "writable_mounts": ["/workspace/tar_runs/problem-1/artifacts"],
                    "artifact_dir": "/workspace/tar_runs/problem-1/artifacts",
                    "workspace_root": "/workspace",
                },
            }
        ],
    )
    _write_jsonl(
        state_dir / "verification_reports.jsonl",
        [
            {
                "trial_id": "trial-1",
                "control_score": 1.8,
                "seed_variance": {
                    "num_runs": 3,
                    "stable": False,
                    "loss_std": 0.08,
                    "dimensionality_std": 0.12,
                    "calibration_ece_mean": 0.11,
                },
                "calibration": {"ece": 0.12, "accuracy": 0.44, "mean_confidence": 0.55},
                "ablations": [{"name": "no_anchor_penalty", "delta_vs_control": 0.0}],
                "verdict": "unstable",
                "recommendations": ["Increase seed count before claiming a result."],
            }
        ],
    )
    _write_jsonl(
        state_dir / "research_decisions.jsonl",
        [
            {
                "decision_id": "decision-1",
                "prompt": "Investigate retrieval failures in NLP",
                "mode": "problem_study",
                "problem_id": "problem-1",
                "evidence_bundle": {
                    "confidence": 0.66,
                    "supporting_document_ids": ["research:paper-1"],
                    "traces": [{"document_id": "research:paper-1"}],
                },
                "hypotheses": ["Retrieval hurts answer quality under weak grounding."],
                "selected_action": "Build a validation-grade retrieval study first.",
                "confidence": 0.66,
                "notes": [],
            }
        ],
    )
    (state_dir / "research_projects.json").write_text(
        json.dumps(
            [
                {
                    "project_id": "project-1",
                    "title": "retrieval_failures_in_nlp",
                    "goal": "Investigate retrieval failures in NLP",
                    "status": "active",
                    "active_thread_id": "thread-1",
                    "latest_decision_summary": "Need a validation-grade retrieval study.",
                    "budget_ledger": {"budget_pressure_level": "low"},
                    "resume_snapshot": {
                        "current_question_id": "question-1",
                        "next_action_id": "action-1",
                        "blockers": [],
                        "budget_remaining_summary": {"experiments_remaining": 4},
                    },
                    "planned_actions": [
                        {
                            "action_id": "action-1",
                            "description": "Create a validation study.",
                            "status": "planned",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        state_dir / "falsification_plans.jsonl",
        [
            {
                "plan_id": "plan-1",
                "project_id": "project-1",
                "thread_id": "thread-1",
                "status": "planned",
                "trigger_reason": "contradiction pressure",
                "coverage": {"overall_sufficient": False},
                "tests": [
                    {
                        "kind": "contradiction_resolution",
                        "description": "Resolve contradiction with a retrieval-off ablation.",
                        "expected_falsification_value": 0.8,
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        state_dir / "project_priority_records.jsonl",
        [
            {
                "record_id": "priority-1",
                "project_id": "project-1",
                "action_id": "action-1",
                "priority_score": 0.7,
                "expected_value": 0.6,
                "evidence_debt": 0.2,
                "contradiction_pressure": 0.4,
                "staleness_penalty": 0.0,
                "budget_pressure": 0.1,
                "benchmark_readiness": 0.5,
                "recommended_state": "continue",
                "rationale": ["cheap decisive test"],
            }
        ],
    )
    _write_jsonl(
        state_dir / "claim_verdicts.jsonl",
        [
            {
                "verdict_id": "verdict-1",
                "trial_id": "trial-1",
                "status": "provisional",
                "policy": {
                    "policy_id": "frontier_claim_policy_v1",
                    "min_seed_runs": 3,
                    "max_seed_loss_std": 0.08,
                    "max_seed_dimensionality_std": 0.75,
                    "max_calibration_ece": 0.15,
                    "min_ablation_gap": 0.05,
                    "min_supporting_sources": 2,
                    "max_allowed_contradictions": 0,
                    "require_canonical_benchmark": True,
                },
                "verification_report_trial_id": "trial-1",
                "benchmark_problem_id": "problem-1",
                "benchmark_execution_mode": "problem_study",
                "supporting_benchmark_ids": ["beir_fiqa_canonical"],
                "supporting_benchmark_names": ["BEIR FiQA"],
                "canonical_comparability_source": "problem_study",
                "verdict_inputs_complete": False,
                "linkage_status": "ambiguous",
                "canonical_benchmark_required": True,
                "canonical_benchmark_satisfied": False,
                "confidence": 0.61,
            }
        ],
    )
    _write_jsonl(
        state_dir / "evidence_debt_records.jsonl",
        [
            {
                "record_id": "debt-1",
                "project_id": "project-1",
                "falsification_gap": 0.6,
                "replication_gap": 0.5,
                "benchmark_gap": 0.8,
                "claim_linkage_gap": 0.4,
                "calibration_gap": 0.2,
                "overall_debt": 0.62,
                "promotion_blocked": True,
                "rationale": ["Need stronger benchmark and linkage support."],
            }
        ],
    )
    _write_jsonl(
        state_dir / "project_staleness_records.jsonl",
        [
            {
                "record_id": "stale-1",
                "project_id": "project-1",
                "last_progress_at": "2026-04-01T00:00:00+00:00",
                "hours_since_progress": 80.0,
                "staleness_level": "stale",
                "reason": "pending operator review",
                "resume_candidate": True,
                "closure_candidate": False,
            }
        ],
    )
    _write_jsonl(
        state_dir / "portfolio_decisions.jsonl",
        [
            {
                "decision_id": "portfolio-1",
                "selected_project_id": "project-1",
                "selected_action_id": "action-1",
                "deferred_project_ids": [],
                "parked_project_ids": [],
                "escalated_project_ids": [],
                "retired_project_ids": [],
                "rationale": ["best evidence gain per unit cost"],
            }
        ],
    )
    (state_dir / "inference_endpoints.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "endpoint_name": "assistant-problem-1",
                        "checkpoint_name": "checkpoint-1",
                        "role": "assistant",
                        "host": "127.0.0.1",
                        "port": 8101,
                        "backend": "transformers",
                        "base_url": "http://127.0.0.1:8101",
                        "status": "failed",
                        "last_error": "health check timeout after startup",
                        "last_health_at": "2026-04-10T00:00:00+00:00",
                        "manifest_path": "<STATE_DIR>/endpoints/assistant-problem-1/endpoint_manifest.json",
                        "stdout_log_path": "<STATE_DIR>/endpoints/assistant-problem-1/stdout.log",
                        "stderr_log_path": "<STATE_DIR>/endpoints/assistant-problem-1/stderr.log",
                        "trust_remote_code": False,
                        "health": {
                            "endpoint_name": "assistant-problem-1",
                            "status": "failed",
                            "ok": False,
                            "http_status": 500,
                            "detail": "startup timeout; inspect stderr tail",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (manifests_dir / "run-problem-1.json").write_text(
        json.dumps(
            {
                "manifest_version": "tar.run.v1",
                "manifest_id": "run-problem-1",
                "kind": "science_bundle",
                "problem_id": "problem-1",
                "trial_id": "trial-1",
                "sandbox_policy": {
                    "mode": "docker_only",
                    "profile": "production",
                    "network_policy": "off",
                    "read_only_mounts": ["/workspace", "/data"],
                    "writable_mounts": ["/workspace/tar_runs/problem-1/artifacts"],
                    "artifact_dir": "/workspace/tar_runs/problem-1/artifacts",
                    "workspace_root": "/workspace",
                },
                "reproducibility_complete": False,
                "unresolved_packages": ["evaluate"],
                "lock_incomplete_reason": "Dependency lock incomplete.",
                "image_manifest": {"image_tag": "tar-payload:locked"},
            }
        ),
        encoding="utf-8",
    )

    manifest = build_master_dataset(state_dir, output_dir, version="tar-master-v1")

    assert manifest["records"] >= 14
    assert (output_dir / "tar_master_dataset.jsonl").exists()
    assert (output_dir / "tar_master_dataset_train.jsonl").exists()
    assert manifest["task_families"]["problem_scoping"] >= 1
    assert manifest["task_families"]["execution_diagnosis"] >= 1
    assert manifest["task_families"]["verification_judgement"] >= 1
    assert manifest["task_families"]["decision_rationale"] >= 1
    assert manifest["task_families"]["project_resume"] >= 1
    assert manifest["task_families"]["falsification_planning"] >= 1
    assert manifest["task_families"]["prioritization"] >= 1
    assert manifest["task_families"]["portfolio_governance"] >= 1
    assert manifest["task_families"]["reproducibility_refusal"] >= 1
    assert manifest["task_families"]["sandbox_policy_reasoning"] >= 1
    assert manifest["task_families"]["endpoint_observability_diagnosis"] >= 1
    assert manifest["task_families"]["portfolio_staleness_recovery"] >= 1
    assert manifest["task_families"]["claim_lineage_audit"] >= 1
    assert manifest["task_families"]["evidence_debt_judgement"] >= 1
    assert manifest["duplicate_examples_removed"] >= 0
    assert manifest["files"]["master"]["sha256"]
    assert manifest["split_integrity"]["lineage_safe"] is True
    assert manifest["source_artifacts"]


def test_build_master_dataset_sanitizes_workspace_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    state_dir = repo_root / "tar_state"
    output_dir = repo_root / "dataset_artifacts" / "tar_master_dataset_v1"
    state_dir.mkdir(parents=True)

    _write_jsonl(
        state_dir / "problem_studies.jsonl",
        [
            {
                "problem_id": "problem-2",
                "problem": "Investigate calibration drift",
                "domain": "generic_ml",
                "profile_id": "generic_ml",
                "hypotheses": ["Calibration worsens first."],
                "experiments": [],
                "environment": {
                    "reproducibility_complete": True,
                    "build_context_path": str(repo_root / "tar_state" / "science_envs" / "generic_ml"),
                },
                "next_action": f"Inspect {repo_root / 'tar_state' / 'science_envs' / 'generic_ml'}",
                "status": "planned",
            }
        ],
    )

    build_master_dataset(state_dir, output_dir, version="tar-master-v1")
    rows = (output_dir / "tar_master_dataset.jsonl").read_text(encoding="utf-8").splitlines()
    assert rows
    assert "<REPO_ROOT>" in rows[0] or "<STATE_DIR>" in rows[0]


def test_build_master_dataset_includes_tcl_native_families(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    state_dir = repo_root / "tar_state"
    output_dir = repo_root / "dataset_artifacts" / "tar_master_dataset_v1"
    trial_dir = repo_root / "tar_runs" / "trial-1" / "verification" / "control"
    trial_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    (state_dir / "recovery.json").write_text(
        json.dumps(
            {
                "trial_id": "trial-1",
                "status": "completed",
                "last_known_stable_hyperparameters": {"alpha": 0.04, "eta": 0.01},
                "last_fail_reason": None,
                "last_fail_metrics": None,
                "consecutive_fail_fast": 0,
                "last_strategy_family": "elastic_anchor",
                "last_anchor_path": "anchors/thermodynamic_anchor.safetensors",
                "max_effective_dimensionality_achieved": 7.8,
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "config.json").write_text(
        json.dumps(
            {
                "backend_id": "asc_text",
                "strategy_family": "elastic_anchor",
                "governor_thresholds": {
                    "max_quenching_loss": 0.8,
                    "min_dimensionality_ratio": 0.55,
                },
                "data_provenance": {
                    "research_grade": False,
                    "has_fallback": True,
                },
                "backend_provenance": {
                    "required_metrics": [
                        "training_loss",
                        "effective_dimensionality",
                        "entropy_sigma",
                        "drift_rho",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "payload_summary.json").write_text(
        json.dumps(
            {
                "trial_id": "trial-1-control",
                "strategy_family": "elastic_anchor",
                "governor_action": "terminate",
                "governor_reasons": ["weight_drift_limit"],
                "anchor_effective_dimensionality": 3.6,
                "last_metrics": {
                    "step": 2,
                    "entropy_sigma": 0.0024,
                    "drift_rho": 0.11,
                    "grad_norm": 0.69,
                    "effective_dimensionality": 2.92,
                    "dimensionality_ratio": 1.15,
                    "equilibrium_fraction": 0.0,
                    "equilibrium_gate": False,
                    "training_loss": 0.71,
                },
                "calibration": {"ece": 0.18, "accuracy": 0.42},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        trial_dir / "thermo_metrics.jsonl",
        [
            {
                "trial_id": "trial-1-control",
                "step": 1,
                "drift_rho": 0.07,
                "effective_dimensionality": 2.81,
                "dimensionality_ratio": 1.11,
                "equilibrium_fraction": 0.0,
                "equilibrium_gate": False,
                "training_loss": 0.67,
            },
            {
                "trial_id": "trial-1-control",
                "step": 2,
                "drift_rho": 0.11,
                "effective_dimensionality": 2.92,
                "dimensionality_ratio": 1.15,
                "equilibrium_fraction": 0.0,
                "equilibrium_gate": False,
                "training_loss": 0.71,
            },
        ],
    )

    manifest = build_master_dataset(state_dir, output_dir, version="tar-master-v1")

    assert manifest["task_families"]["tcl_regime_diagnosis"] >= 1
    assert manifest["task_families"]["tcl_failure_mode_classification"] >= 1
    assert manifest["task_families"]["tcl_anchor_policy_judgement"] >= 1
    assert manifest["task_families"]["tcl_intervention_selection"] >= 1
    assert manifest["task_families"]["tcl_trace_analysis"] >= 1
    assert manifest["task_families"]["tcl_trace_anomaly_diagnosis"] >= 1
    assert manifest["task_families"]["tcl_regime_transition_forecast"] >= 1
    assert manifest["task_families"]["tcl_recovery_planning"] >= 1
    assert manifest["task_families"]["tcl_recovery_confidence_estimation"] >= 1
    assert manifest["task_families"]["tcl_run_triage"] >= 1
    rows = (output_dir / "tar_master_dataset.jsonl").read_text(encoding="utf-8").splitlines()
    assert any('"task_family": "tcl_regime_diagnosis"' in row for row in rows)
    assert any('"task_family": "tcl_failure_mode_classification"' in row for row in rows)
    assert any('"task_family": "tcl_intervention_selection"' in row for row in rows)
    assert any('"task_family": "tcl_trace_anomaly_diagnosis"' in row for row in rows)
    assert any('"task_family": "tcl_recovery_planning"' in row for row in rows)
    assert any('"task_family": "tcl_run_triage"' in row for row in rows)


def test_build_master_dataset_merges_multiple_state_dirs(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    state_a = repo_root / "state_a"
    state_b = repo_root / "state_b"
    output_dir = repo_root / "dataset_artifacts" / "merged"
    state_a.mkdir(parents=True)
    state_b.mkdir(parents=True)

    study_record = {
        "problem_id": "shared-problem",
        "problem": "Investigate calibration drift",
        "domain": "generic_ml",
        "profile_id": "generic_ml",
        "hypotheses": ["Calibration worsens first."],
        "experiments": [],
        "next_action": "Run a calibration validation pass.",
        "status": "planned",
    }
    _write_jsonl(state_a / "problem_studies.jsonl", [study_record])
    _write_jsonl(state_b / "problem_studies.jsonl", [study_record])

    manifest = build_master_dataset([state_a, state_b], output_dir, version="tar-master-v1")

    assert manifest["records"] == 1
    assert manifest["duplicate_examples_removed"] == 1
    assert len((output_dir / "tar_master_dataset.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_build_master_dataset_keeps_project_lineage_in_single_split(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    state_dir = repo_root / "tar_state"
    output_dir = repo_root / "dataset_artifacts" / "lineage"
    state_dir.mkdir(parents=True)

    _write_jsonl(
        state_dir / "problem_studies.jsonl",
        [
            {
                "problem_id": "problem-lineage-1",
                "project_id": "project-lineage-1",
                "thread_id": "thread-lineage-1",
                "problem": "Investigate lineage-safe splitting",
                "domain": "generic_ml",
                "profile_id": "generic_ml",
                "hypotheses": ["Keep all related project examples together."],
                "experiments": [],
                "next_action": "Review split provenance.",
                "status": "planned",
            }
        ],
    )
    _write_json(
        state_dir / "research_projects.json",
        [
            {
                "project_id": "project-lineage-1",
                "title": "lineage_project",
                "goal": "Investigate lineage-safe splitting",
                "status": "active",
                "active_thread_id": "thread-lineage-1",
                "latest_decision_summary": "Keep related examples together.",
                "budget_ledger": {"budget_pressure_level": "low"},
                "resume_snapshot": {
                    "project_id": "project-lineage-1",
                    "current_question_id": "question-lineage-1",
                    "next_action_id": "action-lineage-1",
                    "blockers": [],
                    "budget_remaining_summary": {"experiments_remaining": 4},
                },
                "planned_actions": [
                    {"action_id": "action-lineage-1", "description": "Review split provenance.", "status": "planned"}
                ],
            }
        ],
    )
    _write_jsonl(
        state_dir / "evidence_debt_records.jsonl",
        [
            {
                "record_id": "debt-lineage-1",
                "project_id": "project-lineage-1",
                "falsification_gap": 0.4,
                "replication_gap": 0.2,
                "benchmark_gap": 0.3,
                "claim_linkage_gap": 0.1,
                "calibration_gap": 0.1,
                "overall_debt": 0.35,
                "promotion_blocked": False,
            }
        ],
    )

    build_master_dataset(state_dir, output_dir, version="tar-master-v1")
    rows = [
        json.loads(line)
        for line in (output_dir / "tar_master_dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lineage_rows = [row for row in rows if row["lineage_key"] == "project:project-lineage-1"]
    assert len({row["split"] for row in lineage_rows}) == 1


def test_ws23_campaign_generator_produces_private_state_bundle(tmp_path: Path) -> None:
    workspace = tmp_path / "campaign"
    summary = build_campaign_workspace(workspace, projects_per_campaign=2, prefix="unit")

    assert summary["projects"] == 12
    assert (workspace / "tar_state" / "problem_studies.jsonl").exists()
    assert (workspace / "tar_state" / "claim_verdicts.jsonl").exists()
    assert (workspace / "tar_state" / "inference_endpoints.json").exists()
    assert (workspace / "tar_state" / "manifests").exists()


def test_ws26_tcl_campaign_generator_produces_tcl_state_bundle(tmp_path: Path) -> None:
    workspace = tmp_path / "ws26_campaign"
    summary = build_tcl_workspace(workspace, trials_per_scenario=2, prefix="unit")

    assert summary["trial_count"] == 16
    assert summary["recovery_records"] == 16
    assert (workspace / "tar_state" / "recovery.json").exists()
    assert (workspace / "tar_state" / "recovery_history.jsonl").exists()
    assert (workspace / "tar_runs").exists()
    assert (workspace / "ws26_tcl_campaign_manifest.json").exists()
