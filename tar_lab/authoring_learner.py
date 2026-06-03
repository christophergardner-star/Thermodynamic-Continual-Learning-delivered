"""Phase 3.2a self-improvement: authoring style memory (advisory).

Read-only aggregator over the paper-claim review history (human_review_state.json claim
reviews + author_state.json revision signal). Learns which CLAIM STYLES survive human
review vs get cut/revised, so future drafting can bias toward survivable styles. Produces
tar_state/authoring/authoring_style_memory.json.

INTEGRITY RAIL: factual-only authoring is non-negotiable. This memory shapes STYLE (phrasing,
hedging, scope) ONLY — it NEVER invents, asserts, or alters a fact, and it NEVER edits an
existing paper or the corpus. Advisory until a human uses it. Off-switch:
tar_state/authoring.disabled.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_REL = Path("authoring") / "authoring_style_memory.json"
DISABLE_FLAG = "authoring.disabled"

_ACCEPT = {"approved", "approved_manifest_ready", "approve", "approve_claim_scope"}
_REJECT = {"rejected", "reject", "deprioritise_paper", "revision_requested"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state(workspace, *parts) -> Path:
    return Path(workspace) / "tar_state" / Path(*parts)


def is_disabled(workspace) -> bool:
    return _state(workspace, DISABLE_FLAG).exists()


def _jload(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def rebuild_style_memory(workspace) -> dict:
    """Aggregate accepted-vs-cut claim signal into an advisory style memory. Read-only."""
    accepted = 0
    rejected = 0
    accepted_notes: list[str] = []
    cut_notes: list[str] = []

    hr = _jload(_state(workspace, "human_review_state.json")) or {}
    # claim reviews (paper-level) — separate from experiment proposals
    for cr in (hr.get("claim_reviews", []) if isinstance(hr, dict) else []):
        if not isinstance(cr, dict):
            continue
        status = str(cr.get("status", "") or "")
        decision = str(cr.get("decision", "") or "")
        note = str(cr.get("human_notes", "") or "")[:160]
        if status in _ACCEPT or decision in _ACCEPT:
            accepted += 1
            if note:
                accepted_notes.append(note)
        elif status in _REJECT or decision in _REJECT:
            rejected += 1
            if note:
                cut_notes.append(note)
    # history entries of kind 'claim_review'
    for h in (hr.get("history", []) if isinstance(hr, dict) else []):
        if not isinstance(h, dict) or str(h.get("kind", "")) != "claim_review":
            continue
        d = str(h.get("decision", "") or "")
        note = str(h.get("human_notes", "") or "")[:160]
        if d in _ACCEPT:
            accepted += 1
            if note:
                accepted_notes.append(note)
        elif d in _REJECT:
            rejected += 1
            if note:
                cut_notes.append(note)

    # revision pressure from author_state
    author = _jload(_state(workspace, "author_state.json")) or {}
    revisions = 0
    papers = author.get("papers", author.get("paper_queue", [])) if isinstance(author, dict) else []
    if isinstance(papers, list):
        for p in papers:
            if isinstance(p, dict):
                revisions += len(p.get("revision_requests", []) or [])

    total = accepted + rejected
    memory = {
        "generated_at": _now(),
        "advisory_only": True,
        "style_only_never_facts": True,
        "summary": {
            "claim_decisions": total, "accepted": accepted, "cut_or_revised": rejected,
            "accept_rate": round(accepted / total, 3) if total else None,
            "author_revision_requests": revisions,
        },
        "accepted_style_notes": accepted_notes[-20:],
        "cut_style_notes": cut_notes[-20:],
        "guidance": (
            "Bias drafting toward the phrasing/scope of accepted claims; avoid the patterns in "
            "cut/revised claims. STYLE only — never assert a fact this memory cannot support."
            if total else
            "Armed — no claim review history yet. Will populate as papers are reviewed/revised."
        ),
    }
    try:
        p = _state(workspace, *REGISTRY_REL.parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(memory, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass
    return memory


def load_style_memory(workspace) -> dict:
    return _jload(_state(workspace, *REGISTRY_REL.parts)) or {}


def _main():
    from tar_storage import resolve_workspace
    ws = resolve_workspace(Path(__file__).resolve().parent.parent)
    print(json.dumps(rebuild_style_memory(ws).get("summary", {}), indent=2))


if __name__ == "__main__":
    _main()
