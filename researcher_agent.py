"""
Cruxy researcher loop for this repo.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cruxy_identity import CRUXY_NAME, CRUXY_SYSTEM_PROMPT
from research_database import ResearchDatabase, ResearchEntry


CLAIM_LABELS = ("fact", "measured_result", "inference", "hypothesis")


@dataclass
class Claim:
    label: str
    text: str


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def classify_claim(sentence: str) -> str:
    lowered = sentence.lower()
    if any(token in lowered for token in ("measured", "observed", "accuracy", "loss", "perplexity", "benchmark", "%")):
        return "measured_result"
    if any(token in lowered for token in ("hypothesis", "may", "might", "could", "perhaps", "speculate", "possible")):
        return "hypothesis"
    if any(token in lowered for token in ("therefore", "thus", "suggests", "implies", "indicates")):
        return "inference"
    return "fact"


def classify_claims(text: str) -> list[Claim]:
    return [Claim(classify_claim(sentence), sentence) for sentence in split_sentences(text)]


def render_claim_report(claims: list[Claim]) -> str:
    counts = {label: 0 for label in CLAIM_LABELS}
    for claim in claims:
        counts[claim.label] += 1
    lines = ["Claim Classification:"]
    for label in CLAIM_LABELS:
        lines.append(f"- {label}: {counts[label]}")
    for claim in claims[:6]:
        lines.append(f"- {claim.label}: {claim.text}")
    return "\n".join(lines)


def run_python(code: str, timeout: int = 10) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "task.py"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "Execution timed out"
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output.strip()


def default_tasks(topics: Iterable[str]) -> list[tuple[str, str]]:
    tasks = []
    for topic in topics:
        tasks.append(("research", f"Summarize the key bottleneck in {topic} and propose a testable experiment."))
        tasks.append(("code", f"Write a small Python utility related to {topic}."))
    return tasks


def generate_stub_response(task_type: str, prompt: str) -> str:
    if task_type == "code":
        return (
            "Here is a runnable baseline.\n"
            "```python\n"
            "def main():\n"
            "    print('baseline utility ok')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
            "```\n"
            "This is a fact for the current run."
        )
    return (
        f"{CRUXY_NAME} research note. The current prompt is about {prompt}. "
        "This measured result is not available yet. Therefore the next step is to run a benchmark. "
        "A stronger hypothesis could be tested after that."
    )


def extract_code_block(text: str) -> str | None:
    match = re.search(r"```python\n(.*?)```", text, flags=re.S)
    if not match:
        return None
    return match.group(1).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Cruxy research loop")
    parser.add_argument("--db", default="research_db.sqlite")
    parser.add_argument("--session", default="default")
    parser.add_argument("--topics", default="algorithms,continual learning")
    parser.add_argument("--max_tasks", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    db = ResearchDatabase(args.db)
    topics = [topic.strip() for topic in args.topics.split(",") if topic.strip()]
    tasks = default_tasks(topics)[: args.max_tasks]

    for task_type, prompt in tasks:
        response = generate_stub_response(task_type, prompt)
        claims = classify_claims(response)
        report = render_claim_report(claims)
        success = False
        metadata = {
            "system_prompt": CRUXY_SYSTEM_PROMPT,
            "claim_report": [claim.__dict__ for claim in claims],
            "dry_run": args.dry_run,
        }

        if task_type == "code":
            code = extract_code_block(response)
            if code:
                success, output = run_python(code)
                metadata["execution_output"] = output
                metadata["confidence"] = 1.0 if success else 0.2
            else:
                metadata["confidence"] = 0.1
        else:
            metadata["confidence"] = 0.6

        full_response = response + "\n\n" + report
        score = float(metadata["confidence"])
        db.add_entry(
            ResearchEntry(
                session=args.session,
                task_type=task_type,
                prompt=prompt,
                response=full_response,
                score=score,
                success=success,
                metadata=metadata,
            )
        )
        print(json.dumps({"task_type": task_type, "score": score, "success": success}, indent=2))

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
