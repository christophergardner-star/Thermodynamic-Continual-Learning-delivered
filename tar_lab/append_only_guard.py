"""Phase 3 safety rail: append-only / additive self-improvement.

Existing algorithms and knowledge are READ-ONLY to TAR's self-improvement. TAR may CREATE
new candidate artifacts (method variants, internal knowledge entries, memories) but may
NEVER overwrite, edit, or delete an existing method, canonical algorithm, knowledge entry,
or prior result. This is the methods/knowledge analogue of RAIL 1 (append-only results).

Any future self-improvement write of a NEW method/knowledge artifact MUST go through
guarded_create_new(), which refuses if the target already exists OR is a protected
algorithm/knowledge path. The advisory Phase-3a aggregators write only their own JSON
registries and touch no algorithm/knowledge file at all; this guard is the enforced
gateway for the deferred actuation (variant code, KG entries) and is proven by
tests/test_append_only_self_improvement.py.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# Canonical ALGORITHM sources + KNOWLEDGE stores that self-improvement must never write.
_PROTECTED_BASENAMES = {
    # algorithms
    "method_registry.py", "tcl.py", "asc_model.py", "tar_optimizer_backend.py",
    "method_synthesizer.py",
    # knowledge stores
    "knowledge_graph.json", "knowledge_graph.db", "literature_graph.db",
}
# Whole directories that hold canonical algorithm code (the lab package itself) — new
# method artifacts go to additive areas, never here.
_PROTECTED_DIR_PARTS = (("tar_lab",),)  # tar_lab/*.py canonical modules

# Additive areas where NEW (non-existing) artifacts may be created.
_ADDITIVE_DIR_PARTS = (
    ("tar_state", "synthesized_method_variants"),
    ("tar_state", "method_refinement"),
    ("tar_state", "authoring"),
    ("tar_state", "knowledge_internal"),
)


class AppendOnlyViolation(Exception):
    """Raised when a self-improvement write would overwrite or touch protected state."""


def _norm(p) -> Path:
    return Path(p).resolve()


def is_protected(path) -> bool:
    """True if `path` is a canonical algorithm source or a knowledge store."""
    p = _norm(path)
    if p.name in _PROTECTED_BASENAMES:
        return True
    # Any existing .py directly under the tar_lab package is canonical algorithm code.
    try:
        rel = p.relative_to(_REPO)
    except Exception:
        rel = None
    if rel is not None:
        for parts in _PROTECTED_DIR_PARTS:
            if rel.parts[:len(parts)] == parts and p.suffix == ".py":
                return True
    return False


def is_additive_area(path) -> bool:
    p = _norm(path)
    # An additive-area directory name must appear in the path (e.g. synthesized_method_variants).
    names = {pp for parts in _ADDITIVE_DIR_PARTS for pp in parts}
    return any(seg in names for seg in p.parts)


def guarded_create_new(path, content: str, *, require_additive_area: bool = True) -> Path:
    """Create a NEW artifact. Refuses to overwrite an existing file or touch a protected
    algorithm/knowledge path. Optionally restricts writes to the additive areas.

    This is the ONLY sanctioned way for self-improvement to persist a new method/knowledge
    artifact. It cannot modify existing algorithms or knowledge.
    """
    p = _norm(path)
    if is_protected(p):
        raise AppendOnlyViolation(f"refusing to write protected algorithm/knowledge path: {p}")
    if p.exists():
        raise AppendOnlyViolation(f"refusing to OVERWRITE existing artifact (append-only): {p}")
    if require_additive_area and not is_additive_area(p):
        raise AppendOnlyViolation(f"refusing to write outside additive self-improvement areas: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def file_hash(path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return None


def snapshot(paths) -> dict:
    """Snapshot sha256 of a set of paths (for the canonical/KG-unchanged regression test)."""
    return {str(Path(p)): file_hash(p) for p in paths}


def assert_unchanged(snap: dict) -> None:
    """Raise AppendOnlyViolation if any snapshotted file changed since the snapshot."""
    changed = [path for path, h in snap.items() if file_hash(path) != h]
    if changed:
        raise AppendOnlyViolation(f"protected algorithm/knowledge files changed: {changed}")


def protected_method_and_knowledge_paths(workspace) -> list[Path]:
    """The concrete canonical algorithm + knowledge files to protect/snapshot."""
    ws = Path(workspace)
    out = [
        _REPO / "tar_lab" / "method_registry.py",
        _REPO / "tar_lab" / "method_synthesizer.py",
        _REPO / "tcl.py",
        _REPO / "asc_model.py",
        ws / "tar_state" / "knowledge_graph.json",
    ]
    return [p for p in out if p.exists()]
