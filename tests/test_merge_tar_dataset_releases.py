from __future__ import annotations

import json
from pathlib import Path

from merge_tar_dataset_releases import merge_dataset_releases


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )


def _seed_release(root: Path, *, version: str, example: dict) -> Path:
    dataset_dir = root / version
    _write_jsonl(dataset_dir / "tar_master_dataset.jsonl", [example])
    _write_jsonl(
        dataset_dir / "tar_master_dataset_train.jsonl",
        [example] if example.get("split") == "train" else [],
    )
    _write_jsonl(
        dataset_dir / "tar_master_dataset_validation.jsonl",
        [example] if example.get("split") == "validation" else [],
    )
    _write_jsonl(
        dataset_dir / "tar_master_dataset_test.jsonl",
        [example] if example.get("split") == "test" else [],
    )
    _write_json(
        dataset_dir / "manifest.json",
        {
            "dataset_version": version,
            "records": 1,
            "splits": {
                "train": 1 if example.get("split") == "train" else 0,
                "validation": 1 if example.get("split") == "validation" else 0,
                "test": 1 if example.get("split") == "test" else 0,
            },
        },
    )
    return dataset_dir


def test_merge_dataset_releases_merges_and_dedupes_by_dedupe_key(tmp_path: Path) -> None:
    shared = {
        "example_id": "shared-example",
        "dedupe_key": "shared-dedupe",
        "lineage_key": "project:shared",
        "task_family": "problem_scoping",
        "task_name": "scope",
        "source_kind": "problem_study",
        "split": "train",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "ask"}],
        "input_context": {"prompt": "x"},
        "target": {"next_action": "a"},
        "provenance": {"source_id": "shared-example"},
    }
    dataset_a = _seed_release(tmp_path, version="release-a", example=shared)
    dataset_b = _seed_release(tmp_path, version="release-b", example=dict(shared))

    output_dir = tmp_path / "merged"
    manifest = merge_dataset_releases([dataset_a, dataset_b], output_dir, version="merged-v1")

    assert manifest["records"] == 1
    assert manifest["duplicate_examples_removed"] == 1
    merged_rows = (output_dir / "tar_master_dataset.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(merged_rows) == 1


def test_merge_dataset_releases_recomputes_split_from_lineage(tmp_path: Path) -> None:
    example_a = {
        "example_id": "resume-1",
        "dedupe_key": "resume-1",
        "lineage_key": "project:resume-1",
        "task_family": "project_resume",
        "task_name": "resume",
        "source_kind": "research_project",
        "split": "train",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "ask"}],
        "input_context": {"project_id": "resume-1"},
        "target": {"budget_pressure_level": "low"},
        "provenance": {"source_id": "resume-1"},
    }
    example_b = {
        "example_id": "triage-1",
        "dedupe_key": "triage-1",
        "lineage_key": "project:resume-1",
        "task_family": "tcl_run_triage",
        "task_name": "triage",
        "source_kind": "tcl_recovery_state",
        "split": "test",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "ask"}],
        "input_context": {"status": "completed"},
        "target": {"operator_decision": "resume_controlled"},
        "provenance": {"source_id": "triage-1"},
    }
    dataset_a = _seed_release(tmp_path, version="release-a", example=example_a)
    dataset_b = _seed_release(tmp_path, version="release-b", example=example_b)

    output_dir = tmp_path / "merged"
    merge_dataset_releases([dataset_a, dataset_b], output_dir, version="merged-v1")

    rows = [
        json.loads(line)
        for line in (output_dir / "tar_master_dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert len({row["split"] for row in rows}) == 1
