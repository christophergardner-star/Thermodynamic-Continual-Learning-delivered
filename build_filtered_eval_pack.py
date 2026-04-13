from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SUITE_FILES = {
    "resume": "eval_resume.jsonl",
    "honesty": "eval_honesty.jsonl",
    "falsification": "eval_falsification.jsonl",
    "portfolio": "eval_portfolio.jsonl",
    "tcl": "eval_tcl.jsonl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=False))
            handle.write("\n")


def _fingerprint(path: Path, *, root: Path) -> dict:
    records = None
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            records = sum(1 for _ in handle if _.strip())
    return {
        "path": str(path.relative_to(root)),
        "records": records,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_slice_item_ids(slice_file: Path) -> list[str]:
    item_ids = []
    for row in _load_jsonl(slice_file):
        item_id = row.get("item_id")
        if item_id:
            item_ids.append(item_id)
    return item_ids


def build_filtered_eval_pack(
    *,
    source_eval_dirs: list[Path],
    slice_file: Path,
    output_dir: Path,
    eval_version: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_ids = _load_slice_item_ids(slice_file)
    requested_set = set(requested_ids)

    items_by_id: dict[str, dict] = {}
    source_manifests = []
    rubrics_payload = None
    for source_dir in source_eval_dirs:
        source_dir = source_dir.resolve()
        manifest_path = source_dir / "eval_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_manifests.append(
            {
                "eval_dir": str(source_dir),
                "manifest_sha256": _sha256(manifest_path),
                "eval_version": manifest.get("eval_version"),
            }
        )
        core_rows = _load_jsonl(source_dir / "eval_core.jsonl")
        for row in core_rows:
            item_id = row.get("item_id")
            if item_id in requested_set:
                items_by_id[item_id] = row
        if rubrics_payload is None:
            rubrics_payload = json.loads((source_dir / "scoring_rubrics.json").read_text(encoding="utf-8"))

    ordered_items = [items_by_id[item_id] for item_id in requested_ids if item_id in items_by_id]
    missing = [item_id for item_id in requested_ids if item_id not in items_by_id]
    if missing:
        raise KeyError(f"Missing {len(missing)} requested eval items; first missing item_id: {missing[0]}")

    core_path = output_dir / "eval_core.jsonl"
    _write_jsonl(core_path, ordered_items)

    suite_rows: dict[str, list[dict]] = defaultdict(list)
    family_counts: Counter[str] = Counter()
    suite_counts: Counter[str] = Counter()
    lineage_keys: set[str] = set()
    for row in ordered_items:
        family_counts[row["task_family"]] += 1
        lineage_key = row.get("lineage_key")
        if lineage_key:
            lineage_keys.add(lineage_key)
        for suite_name in row.get("suite_names", []):
            if suite_name == "core":
                continue
            suite_rows[suite_name].append(row)
            suite_counts[suite_name] += 1

    for suite_name, filename in SUITE_FILES.items():
        _write_jsonl(output_dir / filename, suite_rows.get(suite_name, []))

    rubric_path = output_dir / "scoring_rubrics.json"
    rubric_path.write_text(json.dumps(rubrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "eval_version": eval_version,
        "items": len(ordered_items),
        "lineage_count": len(lineage_keys),
        "task_families": dict(sorted(family_counts.items())),
        "suites": dict(sorted(suite_counts.items())),
        "source_slice": {
            "path": str(slice_file),
            "sha256": _sha256(slice_file),
            "records": len(requested_ids),
        },
        "source_eval_packs": source_manifests,
        "files": {
            "core": _fingerprint(core_path, root=output_dir),
            "resume": _fingerprint(output_dir / SUITE_FILES["resume"], root=output_dir),
            "honesty": _fingerprint(output_dir / SUITE_FILES["honesty"], root=output_dir),
            "falsification": _fingerprint(output_dir / SUITE_FILES["falsification"], root=output_dir),
            "portfolio": _fingerprint(output_dir / SUITE_FILES["portfolio"], root=output_dir),
            "tcl": _fingerprint(output_dir / SUITE_FILES["tcl"], root=output_dir),
            "rubrics": _fingerprint(rubric_path, root=output_dir),
        },
    }
    (output_dir / "eval_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a filtered eval pack from an existing eval slice.")
    parser.add_argument(
        "--source-eval-dir",
        dest="source_eval_dirs",
        action="append",
        required=True,
        help="Existing eval pack directory containing eval_core.jsonl and eval_manifest.json. Repeatable.",
    )
    parser.add_argument("--slice-file", required=True, help="JSONL slice file containing item_id rows.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the filtered eval pack.")
    parser.add_argument("--eval-version", required=True, help="Version label for the filtered pack.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_filtered_eval_pack(
        source_eval_dirs=[Path(path).resolve() for path in args.source_eval_dirs],
        slice_file=Path(args.slice_file).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        eval_version=args.eval_version,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
