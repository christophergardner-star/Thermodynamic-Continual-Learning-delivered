from __future__ import annotations

import json
from pathlib import Path

from build_ws27_parse_regression_slice import build_parse_regression_slice


def _write_errors(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_build_parse_regression_slice_filters_and_counts(tmp_path: Path) -> None:
    probe_dir = tmp_path / "probe_eval"
    reg_dir = tmp_path / "regression_eval"
    _write_errors(
        probe_dir / "errors.jsonl",
        [
            {
                "item_id": "probe:a",
                "example_id": "a",
                "task_family": "falsification_planning",
                "suite_names": ["falsification"],
                "error_bucket": "parse_error",
                "gold_summary": {"x": 1},
                "predicted_summary": {"parse_error": "invalid_json"},
                "prediction_text": '{"x":',
            },
            {
                "item_id": "probe:b",
                "example_id": "b",
                "task_family": "project_resume",
                "suite_names": ["resume"],
                "error_bucket": "parse_error",
                "gold_summary": {"y": 1},
                "predicted_summary": {"parse_error": "invalid_json"},
                "prediction_text": '{"y":',
            },
            {
                "item_id": "probe:c",
                "example_id": "c",
                "task_family": "verification_judgement",
                "suite_names": ["falsification"],
                "error_bucket": "falsification_or_verification_mismatch",
                "gold_summary": {"z": 1},
                "predicted_summary": {"z": 0},
                "prediction_text": '{"z":0}',
            },
        ],
    )
    _write_errors(
        reg_dir / "errors.jsonl",
        [
            {
                "item_id": "reg:a",
                "example_id": "d",
                "task_family": "falsification_planning",
                "suite_names": ["falsification"],
                "error_bucket": "parse_error",
                "gold_summary": {"x": 2},
                "predicted_summary": {"parse_error": "invalid_json"},
                "prediction_text": '{"x":',
            },
            {
                "item_id": "reg:a",
                "example_id": "d-dup",
                "task_family": "falsification_planning",
                "suite_names": ["falsification"],
                "error_bucket": "parse_error",
                "gold_summary": {"x": 2},
                "predicted_summary": {"parse_error": "invalid_json"},
                "prediction_text": '{"x":',
            },
        ],
    )

    manifest = build_parse_regression_slice(
        eval_run_dirs=[probe_dir, reg_dir],
        output_dir=tmp_path / "slice",
    )

    assert manifest["count"] == 3
    assert manifest["families"] == {
        "falsification_planning": 2,
        "project_resume": 1,
    }
    assert manifest["source_runs"] == {
        "probe_eval": 2,
        "regression_eval": 1,
    }

    rows = [
        json.loads(line)
        for line in (tmp_path / "slice" / "ws27_parse_regression_slice.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["item_id"] for row in rows] == ["probe:a", "probe:b", "reg:a"]


def test_build_parse_regression_slice_honors_family_cap(tmp_path: Path) -> None:
    eval_dir = tmp_path / "probe_eval"
    _write_errors(
        eval_dir / "errors.jsonl",
        [
            {
                "item_id": "probe:a",
                "example_id": "a",
                "task_family": "falsification_planning",
                "suite_names": ["falsification"],
                "error_bucket": "parse_error",
                "gold_summary": {},
                "predicted_summary": {"parse_error": "invalid_json"},
                "prediction_text": "{}",
            },
            {
                "item_id": "probe:b",
                "example_id": "b",
                "task_family": "falsification_planning",
                "suite_names": ["falsification"],
                "error_bucket": "parse_error",
                "gold_summary": {},
                "predicted_summary": {"parse_error": "invalid_json"},
                "prediction_text": "{}",
            },
        ],
    )

    manifest = build_parse_regression_slice(
        eval_run_dirs=[eval_dir],
        output_dir=tmp_path / "slice",
        max_examples_per_family=1,
    )

    assert manifest["count"] == 1
    assert manifest["families"] == {"falsification_planning": 1}
