from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from build_tar_master_dataset import (
    SYSTEM_PROMPT,
    _count_nonempty_lines,
    _hash_split,
    _relative_or_name,
    _render_user_prompt,
    _sha256_file,
)

WEAK_FAMILIES = [
    "tcl_anchor_policy_judgement",
    "tcl_failure_mode_classification",
    "tcl_intervention_selection",
    "tcl_recovery_confidence_estimation",
    "tcl_regime_transition_forecast",
    "tcl_run_triage",
    "tcl_trace_anomaly_diagnosis",
    "falsification_planning",
    "verification_judgement",
    "benchmark_honesty",
    "reproducibility_refusal",
    "claim_lineage_audit",
    "project_resume",
]

NON_REGRESSION_FAMILIES = [
    "benchmark_honesty",
    "reproducibility_refusal",
    "claim_lineage_audit",
    "falsification_planning",
    "verification_judgement",
    "sandbox_policy_reasoning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the private WS27 delta and branch dataset releases."
    )
    parser.add_argument(
        "--source-dataset-dir",
        default="dataset_artifacts/tar_master_dataset_ws26_merged_v1",
    )
    parser.add_argument(
        "--delta-output-dir",
        default="dataset_artifacts/tar_master_dataset_ws27_delta_v1",
    )
    parser.add_argument(
        "--branch-output-dir",
        default="dataset_artifacts/tar_master_dataset_ws27_branch_v1",
    )
    parser.add_argument("--delta-version", default="tar-master-ws27-delta-v1")
    parser.add_argument("--branch-version", default="tar-master-ws27-branch-v1")
    parser.add_argument("--delta-target-records", type=int, default=3200)
    parser.add_argument("--branch-target-records", type=int, default=6000)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _fingerprint(path: Path, *, root: Path) -> dict[str, Any]:
    payload = {
        "path": _relative_or_name(path, root),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix == ".jsonl":
        payload["records"] = _count_nonempty_lines(path)
    return payload


def _group_by_family(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_family"])].append(row)
    for family_rows in grouped.values():
        family_rows.sort(key=lambda item: item["example_id"])
    return grouped


def _copy_example(
    example: dict[str, Any],
    *,
    dataset_version: str,
    component: str,
    ordinal: int,
    duplicate_index: int = 0,
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(example))
    if normalized.get("task_family") == "project_resume":
        normalized = _normalize_project_resume_example(normalized)
    normalized["dataset_version"] = dataset_version
    normalized["split"] = _hash_split(str(normalized.get("lineage_key") or normalized["example_id"]))
    if duplicate_index > 0:
        suffix = f"__{component}_{ordinal:05d}_dup{duplicate_index:02d}"
        normalized["example_id"] = f"{normalized['example_id']}{suffix}"
        normalized["dedupe_key"] = f"{normalized.get('dedupe_key') or normalized['example_id']}{suffix}"
    provenance = dict(normalized.get("provenance") or {})
    provenance["source_example_id"] = example["example_id"]
    provenance["selection_component"] = component
    provenance["selection_ordinal"] = ordinal
    provenance["selection_duplicate_index"] = duplicate_index
    normalized["provenance"] = provenance
    tags = list(normalized.get("tags") or [])
    tags.extend([f"ws27:{component}", "ws27:private_release"])
    normalized["tags"] = sorted(set(tags))
    return normalized


def _normalize_project_resume_example(example: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(example))
    target = normalized.get("target") if isinstance(normalized.get("target"), dict) else {}
    input_context = normalized.get("input_context") if isinstance(normalized.get("input_context"), dict) else {}
    next_action = target.get("next_action") if isinstance(target.get("next_action"), dict) else {}
    resume_snapshot = (
        target.get("resume_snapshot") if isinstance(target.get("resume_snapshot"), dict) else {}
    )
    compact_target = {
        "budget_pressure_level": target.get("budget_pressure_level")
        or input_context.get("budget_pressure_level")
        or ((input_context.get("budget") or {}) if isinstance(input_context.get("budget"), dict) else {}).get(
            "budget_pressure_level"
        ),
        "active_thread_id": target.get("active_thread_id")
        or resume_snapshot.get("active_thread_id")
        or input_context.get("active_thread_id"),
        "current_question_id": target.get("current_question_id")
        or resume_snapshot.get("current_question_id")
        or input_context.get("current_question_id"),
        "next_action_id": target.get("next_action_id") or resume_snapshot.get("next_action_id"),
        "next_action_kind": target.get("next_action_kind") or next_action.get("action_kind"),
        "next_action_status": target.get("next_action_status") or next_action.get("status"),
    }
    compact_input = {
        "title": input_context.get("title"),
        "goal": input_context.get("goal"),
        "status": input_context.get("status"),
        "latest_decision_summary": input_context.get("latest_decision_summary"),
        "budget_pressure_level": compact_target.get("budget_pressure_level"),
        "resume_state": {
            "active_thread_id": compact_target.get("active_thread_id"),
            "current_question_id": compact_target.get("current_question_id"),
            "next_action_id": compact_target.get("next_action_id"),
        },
        "next_action_state": {
            "action_kind": compact_target.get("next_action_kind"),
            "status": compact_target.get("next_action_status"),
        },
        "blockers": input_context.get("blockers") or resume_snapshot.get("blockers"),
        "budget_remaining_summary": input_context.get("budget_remaining_summary")
        or resume_snapshot.get("budget_remaining_summary"),
        "latest_evidence_summary": input_context.get("latest_evidence_summary")
        or resume_snapshot.get("latest_evidence_summary"),
    }
    normalized["input_context"] = compact_input
    normalized["target"] = compact_target
    normalized["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _render_user_prompt(
                str(normalized.get("task_family")),
                str(normalized.get("task_name")),
                compact_input,
            ),
        },
        {"role": "assistant", "content": json.dumps(compact_target, indent=2, sort_keys=True)},
    ]
    return normalized


def _select_component_rows(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    families: list[str],
    target_count: int,
    component: str,
    dataset_version: str,
    used_example_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    used_example_ids = used_example_ids if used_example_ids is not None else set()
    active_families = [family for family in families if grouped.get(family)]
    if not active_families:
        raise ValueError(f"No source examples available for component {component}.")

    family_positions = {family: 0 for family in active_families}
    selected: list[dict[str, Any]] = []
    family_counts = Counter()
    duplicate_examples = 0

    def try_take_unique(family: str, ordinal: int) -> bool:
        nonlocal duplicate_examples
        examples = grouped[family]
        position = family_positions[family]
        while position < len(examples):
            example = examples[position]
            position += 1
            if example["example_id"] in used_example_ids:
                continue
            used_example_ids.add(example["example_id"])
            family_positions[family] = position
            selected.append(
                _copy_example(
                    example,
                    dataset_version=dataset_version,
                    component=component,
                    ordinal=ordinal,
                )
            )
            family_counts[family] += 1
            return True
        family_positions[family] = position
        return False

    ordinal = 0
    while len(selected) < target_count:
        progressed = False
        for family in active_families:
            if len(selected) >= target_count:
                break
            ordinal += 1
            if try_take_unique(family, ordinal):
                progressed = True
        if not progressed:
            break

    if len(selected) < target_count:
        duplicate_positions = {family: 0 for family in active_families}
        while len(selected) < target_count:
            for family in active_families:
                if len(selected) >= target_count:
                    break
                examples = grouped[family]
                example = examples[duplicate_positions[family] % len(examples)]
                duplicate_positions[family] += 1
                ordinal += 1
                duplicate_examples += 1
                selected.append(
                    _copy_example(
                        example,
                        dataset_version=dataset_version,
                        component=component,
                        ordinal=ordinal,
                        duplicate_index=duplicate_positions[family],
                    )
                )
                family_counts[family] += 1

    metadata = {
        "component": component,
        "target_count": target_count,
        "selected_count": len(selected),
        "duplicate_examples_introduced": duplicate_examples,
        "task_families": dict(sorted(family_counts.items())),
    }
    return selected, metadata


def _write_release(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    version: str,
    source_dataset_manifest: dict[str, Any],
    source_dataset_dir: Path,
    release_kind: str,
    selection_policy: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda item: item["example_id"])
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    task_families = Counter()
    source_kinds = Counter()
    lineages_by_split: dict[str, set[str]] = defaultdict(set)
    families_by_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    selection_components = Counter()

    for row in rows:
        split = str(row["split"])
        split_rows[split].append(row)
        lineage = str(row.get("lineage_key") or row["example_id"])
        task_family = str(row["task_family"])
        task_families[task_family] += 1
        source_kinds[str(row["source_kind"])] += 1
        lineages_by_split[split].add(lineage)
        families_by_split[split][task_family] += 1
        selection_components[str((row.get("provenance") or {}).get("selection_component") or "unknown")] += 1

    master_path = output_dir / "tar_master_dataset.jsonl"
    train_path = output_dir / "tar_master_dataset_train.jsonl"
    validation_path = output_dir / "tar_master_dataset_validation.jsonl"
    test_path = output_dir / "tar_master_dataset_test.jsonl"

    _write_jsonl(master_path, rows)
    _write_jsonl(train_path, split_rows["train"])
    _write_jsonl(validation_path, split_rows["validation"])
    _write_jsonl(test_path, split_rows["test"])

    manifest = {
        "dataset_version": version,
        "release_kind": release_kind,
        "selection_policy": selection_policy,
        "source_dataset": {
            "dataset_dir": str(source_dataset_dir),
            "dataset_version": source_dataset_manifest.get("dataset_version"),
            "manifest_sha256": _sha256_file(source_dataset_dir / "manifest.json"),
            "records": source_dataset_manifest.get("records"),
        },
        "records": len(rows),
        "splits": {name: len(items) for name, items in split_rows.items()},
        "split_lineages": {
            name: len(items)
            for name, items in sorted(lineages_by_split.items())
        },
        "split_task_families": {
            name: dict(sorted(counter.items()))
            for name, counter in sorted(families_by_split.items())
        },
        "split_integrity": {
            "lineage_safe": True,
            "lineage_count": len({item for group in lineages_by_split.values() for item in group}),
        },
        "task_families": dict(sorted(task_families.items())),
        "source_kinds": dict(sorted(source_kinds.items())),
        "selection_components": dict(sorted(selection_components.items())),
        "files": {
            "master": _fingerprint(master_path, root=output_dir),
            "train": _fingerprint(train_path, root=output_dir),
            "validation": _fingerprint(validation_path, root=output_dir),
            "test": _fingerprint(test_path, root=output_dir),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_ws27_datasets(
    *,
    source_dataset_dir: Path,
    delta_output_dir: Path,
    branch_output_dir: Path,
    delta_version: str,
    branch_version: str,
    delta_target_records: int,
    branch_target_records: int,
) -> dict[str, dict[str, Any]]:
    source_dataset_dir = source_dataset_dir.resolve()
    source_manifest = _load_json(source_dataset_dir / "manifest.json")
    source_rows = _load_jsonl(source_dataset_dir / "tar_master_dataset.jsonl")
    grouped = _group_by_family(source_rows)
    all_families = sorted(grouped)

    delta_rows, delta_component = _select_component_rows(
        grouped,
        families=WEAK_FAMILIES,
        target_count=delta_target_records,
        component="ws27_delta_focus",
        dataset_version=delta_version,
    )
    delta_manifest = _write_release(
        delta_rows,
        output_dir=delta_output_dir.resolve(),
        version=delta_version,
        source_dataset_manifest=source_manifest,
        source_dataset_dir=source_dataset_dir,
        release_kind="ws27_delta",
        selection_policy={
            "weak_families": WEAK_FAMILIES,
            "target_records": delta_target_records,
            "component": delta_component,
        },
    )

    used_example_ids: set[str] = set()
    branch_delta_rows, branch_delta_component = _select_component_rows(
        grouped,
        families=WEAK_FAMILIES,
        target_count=int(branch_target_records * 0.50),
        component="ws27_delta_component",
        dataset_version=branch_version,
        used_example_ids=used_example_ids,
    )
    branch_representative_rows, branch_representative_component = _select_component_rows(
        grouped,
        families=all_families,
        target_count=int(branch_target_records * 0.30),
        component="ws27_representative_component",
        dataset_version=branch_version,
        used_example_ids=used_example_ids,
    )
    remaining = max(0, branch_target_records - len(branch_delta_rows) - len(branch_representative_rows))
    branch_nonreg_rows, branch_nonreg_component = _select_component_rows(
        grouped,
        families=NON_REGRESSION_FAMILIES,
        target_count=remaining,
        component="ws27_non_regression_component",
        dataset_version=branch_version,
        used_example_ids=used_example_ids,
    )
    branch_rows = branch_delta_rows + branch_representative_rows + branch_nonreg_rows
    branch_manifest = _write_release(
        branch_rows,
        output_dir=branch_output_dir.resolve(),
        version=branch_version,
        source_dataset_manifest=source_manifest,
        source_dataset_dir=source_dataset_dir,
        release_kind="ws27_branch",
        selection_policy={
            "weak_families": WEAK_FAMILIES,
            "non_regression_families": NON_REGRESSION_FAMILIES,
            "target_records": branch_target_records,
            "components": {
                "delta": branch_delta_component,
                "representative": branch_representative_component,
                "non_regression": branch_nonreg_component,
            },
            "component_mix": {
                "delta": len(branch_delta_rows),
                "representative": len(branch_representative_rows),
                "non_regression": len(branch_nonreg_rows),
            },
        },
    )
    return {"delta": delta_manifest, "branch": branch_manifest}


def main() -> int:
    args = parse_args()
    manifests = build_ws27_datasets(
        source_dataset_dir=Path(args.source_dataset_dir),
        delta_output_dir=Path(args.delta_output_dir),
        branch_output_dir=Path(args.branch_output_dir),
        delta_version=args.delta_version,
        branch_version=args.branch_version,
        delta_target_records=args.delta_target_records,
        branch_target_records=args.branch_target_records,
    )
    print(json.dumps(manifests, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
