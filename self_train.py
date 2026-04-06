"""
Export high-quality research traces for future ASC fine-tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_database import ResearchDatabase


def main() -> int:
    parser = argparse.ArgumentParser(description="Export high-quality researcher traces")
    parser.add_argument("--db", default="research_db.sqlite")
    parser.add_argument("--output", default="research_traces.jsonl")
    parser.add_argument("--min_score", type=float, default=0.5)
    args = parser.parse_args()

    db = ResearchDatabase(args.db)
    count = db.export_jsonl(args.output, min_score=args.min_score)
    db.close()
    manifest = {
        "db": args.db,
        "output": args.output,
        "min_score": args.min_score,
        "records": count,
    }
    Path(args.output).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
