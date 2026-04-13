from __future__ import annotations

import json
from pathlib import Path

from build_filtered_eval_pack import build_filtered_eval_pack


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_build_filtered_eval_pack_filters_requested_items(tmp_path: Path) -> None:
    source_a = tmp_path / "pack_a"
    source_b = tmp_path / "pack_b"
    common_manifest = {
        "eval_version": "source-pack",
    }
    for source_dir in (source_a, source_b):
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "scoring_rubrics.json").write_text("{}", encoding="utf-8")
        (source_dir / "eval_manifest.json").write_text(json.dumps(common_manifest), encoding="utf-8")
        for filename in (
            "eval_resume.jsonl",
            "eval_honesty.jsonl",
            "eval_falsification.jsonl",
            "eval_portfolio.jsonl",
            "eval_tcl.jsonl",
        ):
            (source_dir / filename).write_text("", encoding="utf-8")

    rows_a = [
        {
            "item_id": "id-a",
            "task_family": "project_resume",
            "suite_names": ["core", "resume"],
            "lineage_key": "lineage:a",
        },
        {
            "item_id": "id-b",
            "task_family": "falsification_planning",
            "suite_names": ["core", "falsification"],
            "lineage_key": "lineage:b",
        },
    ]
    rows_b = [
        {
            "item_id": "id-c",
            "task_family": "project_resume",
            "suite_names": ["core", "resume"],
            "lineage_key": "lineage:c",
        }
    ]
    _write_jsonl(source_a / "eval_core.jsonl", rows_a)
    _write_jsonl(source_b / "eval_core.jsonl", rows_b)

    slice_file = tmp_path / "slice.jsonl"
    _write_jsonl(
        slice_file,
        [
            {"item_id": "id-b"},
            {"item_id": "id-c"},
        ],
    )

    manifest = build_filtered_eval_pack(
        source_eval_dirs=[source_a, source_b],
        slice_file=slice_file,
        output_dir=tmp_path / "out",
        eval_version="filtered-pack",
    )

    assert manifest["items"] == 2
    assert manifest["task_families"] == {
        "falsification_planning": 1,
        "project_resume": 1,
    }

    core_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "eval_core.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["item_id"] for row in core_rows] == ["id-b", "id-c"]
