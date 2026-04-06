"""
Small coding-eval helpers for ASC coding models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples <= 0 or k <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    prod = 1.0
    for i in range(num_samples - num_correct + 1, num_samples + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def iter_predictions(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def summarize_predictions(path: Path, k: int) -> dict:
    rows = list(iter_predictions(path))
    correct = sum(1 for row in rows if row.get("passed"))
    return {
        "samples": len(rows),
        "correct": correct,
        "pass_at_k": pass_at_k(len(rows), correct, k),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize coding benchmark predictions")
    parser.add_argument("--predictions_jsonl", required=True)
    parser.add_argument("-k", type=int, default=1)
    args = parser.parse_args()
    summary = summarize_predictions(Path(args.predictions_jsonl), args.k)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
