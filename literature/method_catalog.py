"""
Hand-curated continual-learning METHOD CATALOG — recombination material for the
solution-finding proposer (docs/solution_loop_implementation_plan.md, Ingredient 1).

Integrity (mirror of literature/curated_sota.py):
  * An entry WITHOUT a real method-defining citation (paper_title + one of
    arxiv_id/doi/url) is REFUSED by the loader. Structure is enforced here;
    CORRECTNESS of each citation is a human gate — entries carry
    citation_status ("unverified" until the lead confirms the id resolves to the
    named paper). The top-level "_verified": false means the catalog as a whole
    has not yet been signed off.
  * NO metric values live here. Cited benchmark numbers go exclusively through
    curated_external_sota.json (the NoveltyGate bar).

This module is import-safe and never raises on a malformed file/row.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CATALOG_PATH = Path(__file__).with_name("method_catalog.json")

MECHANISM_CLASSES = frozenset({
    "regularization", "replay", "distillation", "architectural",
    "parameter_isolation", "prompt_based", "subspace_projection",
    "optimizer_based", "bayesian",
})


def _has_citation(c: Any) -> bool:
    if not isinstance(c, dict):
        return False
    return bool(c.get("paper_title")) and bool(c.get("arxiv_id") or c.get("doi") or c.get("url"))


def load_method_catalog(path: Path | None = None) -> list[dict[str, Any]]:
    """Return valid catalog entries. Refuses entries missing method_key,
    full_name, a known mechanism_class, or a citation. Fail-quiet ([] on error)."""
    p = Path(path) if path else _CATALOG_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in (data.get("methods", []) if isinstance(data, dict) else []):
        if not isinstance(m, dict):
            continue
        key = str(m.get("method_key", "") or "")
        if not (key and m.get("full_name") and m.get("mechanism_class")):
            continue
        if m.get("mechanism_class") not in MECHANISM_CLASSES:
            continue
        if not _has_citation(m.get("citation")):
            continue  # anti-fabrication: every method must cite its defining paper
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def catalog_is_verified(path: Path | None = None) -> bool:
    """True only when the human has signed off the catalog (_verified: true)."""
    p = Path(path) if path else _CATALOG_PATH
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("_verified", False))
    except Exception:
        return False


def render_catalog_block(methods: list[dict] | None = None, *, max_methods: int = 80) -> str:
    """Compact prior-art block for the proposer prompt: one line per method,
    mechanism class + summary + failure modes. This is the recombination material
    that turns 'test another TCL variant' into 'compose a mechanism from the field'."""
    methods = methods if methods is not None else load_method_catalog()
    lines: list[str] = []
    for m in methods[:max_methods]:
        fails = "; ".join(str(f) for f in (m.get("failure_modes") or []))
        combos = ", ".join(str(c) for c in (m.get("combinable_with") or []))
        line = f"- {m['method_key']} [{m['mechanism_class']}] {m.get('mechanism_summary', '')}"
        if fails:
            line += f" | fails: {fails}"
        if combos:
            line += f" | combines with: {combos}"
        lines.append(line)
    return "\n".join(lines)
