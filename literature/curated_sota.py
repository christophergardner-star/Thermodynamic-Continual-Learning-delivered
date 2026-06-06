"""Human-curated external SoTA loader.

PapersWithCode's public API (the ingestor's only structured SoTA source) is dead, so
external SoTA leaderboard entries can no longer be auto-ingested. This module is the
ONLY sanctioned way to give the NoveltyGate a real external bar: a human adds entries
BY HAND from real papers actually in hand, each carrying genuine provenance (a paper
title + a citation: arXiv id / DOI / URL). Entries are written with source="external"
so the NoveltyGate compares against them (best_result(exclude_source="tar_internal")).

Integrity rule: an entry WITHOUT a citation is refused. Never add a metric value you
cannot cite to a real paper — that is exactly the fabrication truth-lock exists to
prevent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from literature.schemas import SoTAEntry

_DEFAULT_PATH = Path(__file__).resolve().parent / "curated_external_sota.json"


def _citation_of(entry: dict[str, Any]) -> str:
    return str(
        entry.get("citation")
        or entry.get("url")
        or entry.get("arxiv_id")
        or entry.get("doi")
        or entry.get("paper_id")
        or ""
    ).strip()


def load_curated_external_sota(graph: Any, path: Optional[Path | str] = None) -> int:
    """Upsert human-curated external SoTA entries (source="external") into the graph.

    Returns the number of entries written. Entries missing a paper_title or a citation
    are skipped (anti-fabrication guard). Safe to call repeatedly (idempotent via a
    deterministic entry_id). Never raises on a malformed file/row.
    """
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if isinstance(data, dict):
        entries = data.get("entries", [])
    elif isinstance(data, list):
        entries = data
    else:
        entries = []

    written = 0
    for raw in entries if isinstance(entries, list) else []:
        if not isinstance(raw, dict):
            continue
        try:
            paper_title = str(raw.get("paper_title") or "").strip()
            citation = _citation_of(raw)
            benchmark_id = str(raw.get("benchmark_id") or "").strip()
            method_name = str(raw.get("method_name") or "").strip()
            metric_name = str(raw.get("metric_name") or "").strip()
            if not (paper_title and citation and benchmark_id and method_name and metric_name):
                # Anti-fabrication: every external SoTA number must be cited.
                continue
            entry = SoTAEntry(
                entry_id=f"external::{benchmark_id}::{method_name}::{metric_name}",
                benchmark_id=benchmark_id,
                method_name=method_name,
                metric_name=metric_name,
                metric_value=float(raw["metric_value"]),
                higher_is_better=bool(raw.get("higher_is_better", False)),
                paper_id=str(raw.get("paper_id") or raw.get("arxiv_id") or "") or None,
                paper_title=paper_title,
                year=int(raw["year"]) if str(raw.get("year") or "").strip().isdigit() else None,
                venue=str(raw.get("venue") or "") or None,
                venue_tier=str(raw.get("venue_tier") or "unknown"),
                code_available=bool(raw.get("code_url")),
                code_url=str(raw.get("code_url") or citation) or None,
                source="external",
            )
            graph.upsert_sota_entry(entry)
            written += 1
        except Exception:
            continue
    return written
