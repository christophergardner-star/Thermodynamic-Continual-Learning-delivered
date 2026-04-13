from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

DEFAULT_TARGET_FAMILIES = ("falsification_planning", "project_resume")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_errors(errors_path: Path) -> Iterable[dict]:
    with errors_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_parse_regression_slice(
    eval_run_dirs: list[Path],
    output_dir: Path,
    include_families: tuple[str, ...] = DEFAULT_TARGET_FAMILIES,
    max_examples_per_family: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_path = output_dir / "ws27_parse_regression_slice.jsonl"
    manifest_path = output_dir / "manifest.json"

    written = []
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    seen_item_ids: set[str] = set()
    source_files = []

    for eval_run_dir in eval_run_dirs:
        errors_path = eval_run_dir / "errors.jsonl"
        if not errors_path.exists():
            raise FileNotFoundError(f"Missing errors.jsonl in {eval_run_dir}")
        source_files.append(
            {
                "eval_run_dir": str(eval_run_dir),
                "errors_file": str(errors_path),
                "sha256": _sha256(errors_path),
            }
        )
        for row in _load_errors(errors_path):
            if row.get("error_bucket") != "parse_error":
                continue
            task_family = row.get("task_family")
            if task_family not in include_families:
                continue
            if max_examples_per_family is not None and family_counts[task_family] >= max_examples_per_family:
                continue
            item_id = row.get("item_id")
            if not item_id or item_id in seen_item_ids:
                continue
            seen_item_ids.add(item_id)
            family_counts[task_family] += 1
            source_counts[eval_run_dir.name] += 1
            written.append(
                {
                    "item_id": item_id,
                    "example_id": row.get("example_id"),
                    "task_family": task_family,
                    "suite_names": row.get("suite_names", []),
                    "error_bucket": row.get("error_bucket"),
                    "source_eval_run": eval_run_dir.name,
                    "gold_summary": row.get("gold_summary"),
                    "predicted_summary": row.get("predicted_summary"),
                    "prediction_text": row.get("prediction_text"),
                }
            )

    with slice_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in written:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    manifest = {
        "version": "ws27-parse-regression-v1",
        "slice_file": slice_path.name,
        "count": len(written),
        "families": dict(sorted(family_counts.items())),
        "source_runs": dict(sorted(source_counts.items())),
        "include_families": list(include_families),
        "max_examples_per_family": max_examples_per_family,
        "source_files": source_files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a WS27 parse-regression slice from eval error outputs."
    )
    parser.add_argument(
        "--eval-run-dir",
        dest="eval_run_dirs",
        action="append",
        required=True,
        help="Path to an eval run directory containing errors.jsonl. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the parse-regression slice and manifest.",
    )
    parser.add_argument(
        "--include-family",
        dest="include_families",
        action="append",
        default=None,
        help="Task family to include. Defaults to falsification_planning and project_resume.",
    )
    parser.add_argument(
        "--max-examples-per-family",
        type=int,
        default=None,
        help="Optional cap per task family.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    include_families = tuple(args.include_families or DEFAULT_TARGET_FAMILIES)
    manifest = build_parse_regression_slice(
        eval_run_dirs=[Path(path).resolve() for path in args.eval_run_dirs],
        output_dir=Path(args.output_dir).resolve(),
        include_families=include_families,
        max_examples_per_family=args.max_examples_per_family,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
