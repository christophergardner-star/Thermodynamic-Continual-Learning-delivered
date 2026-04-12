from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from build_tar_master_dataset import _count_nonempty_lines, _hash_split, _relative_or_name, _sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge versioned TAR dataset releases.")
    parser.add_argument("--dataset-dir", action="append", dest="dataset_dirs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _fingerprint_file(path: Path, *, root: Path) -> dict[str, Any]:
    payload = {
        "path": _relative_or_name(path, root),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix == ".jsonl":
        payload["records"] = _count_nonempty_lines(path)
    return payload


def merge_dataset_releases(
    dataset_dirs: Iterable[Path],
    output_dir: Path,
    *,
    version: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_dirs = [Path(path).resolve() for path in dataset_dirs]

    deduped: dict[str, dict[str, Any]] = {}
    pre_dedup_count = 0
    source_releases: list[dict[str, Any]] = []
    for dataset_dir in normalized_dirs:
        manifest = _load_json(dataset_dir / "manifest.json")
        source_releases.append(
            {
                "dataset_dir": str(dataset_dir),
                "dataset_version": manifest.get("dataset_version"),
                "manifest_sha256": _sha256_file(dataset_dir / "manifest.json"),
                "records": manifest.get("records"),
            }
        )
        for example in _load_jsonl(dataset_dir / "tar_master_dataset.jsonl"):
            pre_dedup_count += 1
            dedupe_key = str(example.get("dedupe_key") or example["example_id"])
            normalized = dict(example)
            normalized["dataset_version"] = version
            provenance = dict(normalized.get("provenance") or {})
            provenance.setdefault("source_dataset_version", manifest.get("dataset_version"))
            provenance.setdefault("source_dataset_dir", str(dataset_dir))
            normalized["provenance"] = provenance
            deduped[dedupe_key] = normalized

    examples = sorted(deduped.values(), key=lambda item: item["example_id"])

    master_path = output_dir / "tar_master_dataset.jsonl"
    train_path = output_dir / "tar_master_dataset_train.jsonl"
    validation_path = output_dir / "tar_master_dataset_validation.jsonl"
    test_path = output_dir / "tar_master_dataset_test.jsonl"

    split_examples: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    task_families = Counter()
    source_kinds = Counter()
    lineages_by_split: dict[str, set[str]] = defaultdict(set)
    families_by_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }

    for example in examples:
        lineage_key = str(example.get("lineage_key") or example["example_id"])
        split = _hash_split(lineage_key)
        example["split"] = split
        split_examples[split].append(example)
        task_families[example["task_family"]] += 1
        source_kinds[example["source_kind"]] += 1
        families_by_split[split][example["task_family"]] += 1
        lineages_by_split[split].add(lineage_key)

    for path, rows in (
        (master_path, examples),
        (train_path, split_examples["train"]),
        (validation_path, split_examples["validation"]),
        (test_path, split_examples["test"]),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    output_files = {
        "master": _fingerprint_file(master_path, root=output_dir),
        "train": _fingerprint_file(train_path, root=output_dir),
        "validation": _fingerprint_file(validation_path, root=output_dir),
        "test": _fingerprint_file(test_path, root=output_dir),
    }

    manifest = {
        "dataset_version": version,
        "merge_strategy": "stable_dedupe_key_plus_lineage_hash_split",
        "source_releases": source_releases,
        "records": len(examples),
        "pre_dedup_records": pre_dedup_count,
        "duplicate_examples_removed": pre_dedup_count - len(examples),
        "splits": {name: len(items) for name, items in split_examples.items()},
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
        "files": output_files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    manifest = merge_dataset_releases(
        [Path(item) for item in args.dataset_dirs],
        Path(args.output_dir).resolve(),
        version=args.version,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
