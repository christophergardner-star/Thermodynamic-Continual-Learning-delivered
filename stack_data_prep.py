"""
Prepare code/research text corpora for ASC coding runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None


DEFAULT_TOKENIZER_ID = "Qwen/Qwen2.5-Coder-7B"


def iter_source_texts(paths: list[str]) -> Iterable[tuple[str, str]]:
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get("text") or item.get("code") or item.get("content")
                if text:
                    yield raw, text
            continue
        if path.is_file():
            yield raw, path.read_text(encoding="utf-8")
            continue
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".py", ".md", ".txt", ".json", ".jsonl"}:
                if child.suffix.lower() == ".jsonl":
                    yield from iter_source_texts([str(child)])
                else:
                    yield str(child), child.read_text(encoding="utf-8")


def build_token_counter(tokenizer_id: Optional[str]):
    if tokenizer_id and tokenizer_id.lower() not in {"none", "whitespace", "basic"} and AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        return count_tokens

    def count_tokens(text: str) -> int:
        return len(text.split())

    return count_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare coding corpus")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--tokenizer_id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--min_tokens", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count_tokens = build_token_counter(args.tokenizer_id)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    kept = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for source, text in iter_source_texts(args.inputs):
            fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            token_count = count_tokens(text)
            if token_count < args.min_tokens or token_count > args.max_tokens:
                continue
            handle.write(
                json.dumps(
                    {
                        "text": text,
                        "source": source,
                        "token_count": token_count,
                    }
                )
                + "\n"
            )
            kept += 1

    manifest_path = Path(args.manifest) if args.manifest else output_path.with_name("corpus_manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "inputs": args.inputs,
                "output": str(output_path),
                "tokenizer_id": args.tokenizer_id,
                "min_tokens": args.min_tokens,
                "max_tokens": args.max_tokens,
                "records": kept,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"records": kept, "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
