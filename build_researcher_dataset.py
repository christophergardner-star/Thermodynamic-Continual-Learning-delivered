"""
Blend DB traces and note files into a JSONL corpus for coding/research tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_database import ResearchDatabase


def iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a researcher corpus")
    parser.add_argument("--db", default=None)
    parser.add_argument("--notes_jsonl", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_score", type=float, default=0.5)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    with output.open("w", encoding="utf-8") as handle:
        if args.db:
            db = ResearchDatabase(args.db)
            for entry in db.high_quality_entries(args.min_score):
                handle.write(json.dumps({"text": entry.prompt + "\n\n" + entry.response, "source": "research_db"}) + "\n")
                records += 1
            db.close()
        for raw_path in args.notes_jsonl:
            for item in iter_jsonl(Path(raw_path)):
                text = item.get("text") or item.get("content") or item.get("response")
                if text:
                    handle.write(json.dumps({"text": text, "source": raw_path}) + "\n")
                    records += 1

    manifest = {
        "output": str(output),
        "records": records,
        "db": args.db,
        "notes_jsonl": args.notes_jsonl,
        "min_score": args.min_score,
    }
    output.with_name("corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
