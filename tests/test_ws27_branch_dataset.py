import json
from pathlib import Path

from build_ws27_branch_dataset import (
    NON_REGRESSION_FAMILIES,
    WEAK_FAMILIES,
    build_ws27_datasets,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def _seed_record(
    *,
    example_id: str,
    task_family: str,
    lineage_key: str,
    split: str,
) -> dict:
    if task_family == "project_resume":
        target = {
            "budget_pressure_level": "medium",
            "resume_snapshot": {
                "active_thread_id": "thread-1",
                "current_question_id": "question-1",
                "next_action_id": "action-1",
                "blockers": [],
                "budget_remaining_summary": {"experiments_remaining": 2},
                "latest_evidence_summary": "Use compact resume output.",
            },
            "next_action": {
                "action_kind": "run_problem_study",
                "status": "planned",
            },
        }
        input_context = {
            "title": "resume_project",
            "goal": "Check contract compaction",
            "status": "active",
            "active_thread_id": "thread-1",
            "latest_decision_summary": "Return the compact resume state.",
            "budget": {"budget_pressure_level": "medium"},
            "current_question_id": "question-1",
            "blockers": [],
            "budget_remaining_summary": {"experiments_remaining": 2},
        }
    else:
        input_context = {"family": task_family}
        target = {"family": task_family}
    return {
        "example_id": example_id,
        "dedupe_key": example_id,
        "dataset_version": "tar-master-ws26-merged-v1",
        "lineage_key": lineage_key,
        "task_family": task_family,
        "task_name": f"{task_family}_task",
        "source_kind": task_family,
        "tags": [],
        "input_context": input_context,
        "target": target,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ask"},
            {"role": "assistant", "content": json.dumps(target)},
        ],
        "provenance": {
            "state_file": "seed.jsonl",
            "source_id": example_id,
            "state_root": "tar_state",
            "observed": True,
            "content_hash": example_id,
        },
        "split": split,
    }


def _seed_source_release(root: Path) -> Path:
    dataset_dir = root / "dataset_artifacts" / "tar_master_dataset_ws26_merged_v1"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    families = [
        "tcl_intervention_selection",
        "project_resume",
        "falsification_planning",
        "benchmark_honesty",
        "verification_judgement",
        "claim_lineage_audit",
        "decision_rationale",
        "sandbox_policy_reasoning",
        "portfolio_governance",
        "problem_scoping",
    ]
    records: list[dict] = []
    for family_index, family in enumerate(families):
        for example_index in range(3):
            example_id = f"{family}-{example_index}"
            records.append(
                _seed_record(
                    example_id=example_id,
                    task_family=family,
                    lineage_key=f"lineage:{family}:{example_index}",
                    split=("train" if example_index == 0 else "validation" if example_index == 1 else "test"),
                )
            )
    manifest = {
        "dataset_version": "tar-master-ws26-merged-v1",
        "records": len(records),
        "splits": {"train": 10, "validation": 10, "test": 10},
        "task_families": {family: 3 for family in families},
    }
    _write_json(dataset_dir / "manifest.json", manifest)
    _write_jsonl(dataset_dir / "tar_master_dataset.jsonl", records)
    _write_jsonl(dataset_dir / "tar_master_dataset_train.jsonl", [row for row in records if row["split"] == "train"])
    _write_jsonl(
        dataset_dir / "tar_master_dataset_validation.jsonl",
        [row for row in records if row["split"] == "validation"],
    )
    _write_jsonl(dataset_dir / "tar_master_dataset_test.jsonl", [row for row in records if row["split"] == "test"])
    return dataset_dir


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_build_ws27_delta_release_targets_weak_families(tmp_path: Path):
    source_dataset = _seed_source_release(tmp_path)
    delta_output = tmp_path / "dataset_artifacts" / "tar_master_dataset_ws27_delta_v1"
    branch_output = tmp_path / "dataset_artifacts" / "tar_master_dataset_ws27_branch_v1"

    manifests = build_ws27_datasets(
        source_dataset_dir=source_dataset,
        delta_output_dir=delta_output,
        branch_output_dir=branch_output,
        delta_version="tar-master-ws27-delta-v1",
        branch_version="tar-master-ws27-branch-v1",
        delta_target_records=8,
        branch_target_records=12,
    )

    delta_records = _load_jsonl(delta_output / "tar_master_dataset.jsonl")
    assert manifests["delta"]["release_kind"] == "ws27_delta"
    assert len(delta_records) == 8
    assert {row["task_family"] for row in delta_records}.issubset(set(WEAK_FAMILIES))
    assert manifests["delta"]["selection_components"]["ws27_delta_focus"] == 8


def test_build_ws27_branch_release_preserves_component_mix_and_lineage_safety(tmp_path: Path):
    source_dataset = _seed_source_release(tmp_path)
    delta_output = tmp_path / "dataset_artifacts" / "tar_master_dataset_ws27_delta_v1"
    branch_output = tmp_path / "dataset_artifacts" / "tar_master_dataset_ws27_branch_v1"

    manifests = build_ws27_datasets(
        source_dataset_dir=source_dataset,
        delta_output_dir=delta_output,
        branch_output_dir=branch_output,
        delta_version="tar-master-ws27-delta-v1",
        branch_version="tar-master-ws27-branch-v1",
        delta_target_records=6,
        branch_target_records=15,
    )

    branch_manifest = _load_json(branch_output / "manifest.json")
    branch_records = _load_jsonl(branch_output / "tar_master_dataset.jsonl")
    train_records = _load_jsonl(branch_output / "tar_master_dataset_train.jsonl")
    validation_records = _load_jsonl(branch_output / "tar_master_dataset_validation.jsonl")
    test_records = _load_jsonl(branch_output / "tar_master_dataset_test.jsonl")

    assert manifests["branch"]["release_kind"] == "ws27_branch"
    assert branch_manifest["selection_policy"]["component_mix"] == {
        "delta": 7,
        "representative": 4,
        "non_regression": 4,
    }
    assert branch_manifest["selection_components"] == {
        "ws27_delta_component": 7,
        "ws27_non_regression_component": 4,
        "ws27_representative_component": 4,
    }
    assert branch_manifest["split_integrity"]["lineage_safe"] is True
    assert len(branch_records) == 15
    assert len(train_records) + len(validation_records) + len(test_records) == 15

    split_lineages = {
        "train": {row["lineage_key"] for row in train_records},
        "validation": {row["lineage_key"] for row in validation_records},
        "test": {row["lineage_key"] for row in test_records},
    }
    assert split_lineages["train"].isdisjoint(split_lineages["validation"])
    assert split_lineages["train"].isdisjoint(split_lineages["test"])
    assert split_lineages["validation"].isdisjoint(split_lineages["test"])

    assert set(branch_manifest["selection_policy"]["weak_families"]) == set(WEAK_FAMILIES)
    assert set(branch_manifest["selection_policy"]["non_regression_families"]) == set(
        NON_REGRESSION_FAMILIES
    )

    project_resume_rows = [row for row in branch_records if row["task_family"] == "project_resume"]
    assert project_resume_rows
    for row in project_resume_rows:
        assert row["target"] == {
            "budget_pressure_level": "medium",
            "active_thread_id": "thread-1",
            "current_question_id": "question-1",
            "next_action_id": "action-1",
            "next_action_kind": "run_problem_study",
            "next_action_status": "planned",
        }
        assert '"next_action_status"' in row["messages"][1]["content"]
