"""
Canonical-index tamper-evidence anchor (truth-lock TL-6, 2026-06-04).

The audit found anchors/ empty — no tamper-evident root for the canonical results.
This adds a hash-CHAIN over canonical_results_index.jsonl: each anchor records the
SHA-256 of the index, the previous anchor's hash, the git HEAD and a timestamp, and
its own hash over those fields. Any post-hoc edit to the index (or to a historical
anchor) breaks the chain and is detectable by verify_canonical_anchor_chain().

Append-only chain:  anchors/canonical_anchor_chain.jsonl
Head pointer:       anchors/canonical_anchor_head.json

TL-6(a) note: manifest re-verification on READ is already enforced by
verify_canonical_3gate (gate-2) for every publication candidate (TL-4); this module
adds the missing index-level tamper-evidence (TL-6b).

CLI:  python -m tar_lab.canonical_anchor write     # extend the chain (snapshot now)
      python -m tar_lab.canonical_anchor verify    # check chain integrity + index drift
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ANCHOR_BODY_KEYS = ("seq", "at", "git_head", "index_sha256", "prev_anchor_sha256")


def _sha256_file(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else hashlib.sha256(b"").hexdigest()
    except Exception:
        return hashlib.sha256(b"").hexdigest()


def _git_head(repo_root: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root),
                           capture_output=True, text=True, check=False)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def _anchor_hash(body: dict) -> str:
    canonical = json.dumps({k: body.get(k) for k in _ANCHOR_BODY_KEYS}, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _paths(workspace: Path) -> tuple[Path, Path, Path, Path]:
    anchors = Path(workspace) / "anchors"
    chain = anchors / "canonical_anchor_chain.jsonl"
    head = anchors / "canonical_anchor_head.json"
    index = Path(workspace) / "tar_state" / "comparisons" / "canonical_results_index.jsonl"
    return anchors, chain, head, index


def write_canonical_anchor(workspace: Path, repo_root: "Path | None" = None) -> dict[str, Any]:
    """Append a new tamper-evident anchor over the current canonical index. Idempotent-safe
    (always appends a fresh sequence number). Returns the written anchor record."""
    anchors, chain, head, index = _paths(workspace)
    anchors.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]

    prev_hash = ""
    seq = 0
    if chain.exists():
        lines = [l for l in chain.read_text(encoding="utf-8").splitlines() if l.strip()]
        seq = len(lines)
        if lines:
            try:
                prev_hash = json.loads(lines[-1]).get("anchor_sha256", "")
            except Exception:
                prev_hash = ""

    body = {
        "seq": seq,
        "at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(repo_root),
        "index_sha256": _sha256_file(index),
        "prev_anchor_sha256": prev_hash,
    }
    rec = dict(body)
    rec["anchor_sha256"] = _anchor_hash(body)

    with open(chain, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    tmp = head.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    tmp.replace(head)

    # Seam 3 (2026-06-06): replicate the anchor chain into the code working copy when
    # the live workspace is a separate mirror (E: state vs C: checkout), so the
    # tamper-evidence is also captured under version control. Best-effort: a mirror
    # fault must never break the authoritative anchor write above.
    try:
        if Path(repo_root).resolve() != Path(workspace).resolve():
            repo_anchors = Path(repo_root) / "anchors"
            repo_anchors.mkdir(parents=True, exist_ok=True)
            for src in (chain, head):
                if src.exists():
                    (repo_anchors / src.name).write_bytes(src.read_bytes())
    except Exception:
        pass

    return rec


def verify_canonical_anchor_chain(workspace: Path) -> tuple[bool, str]:
    """Verify the anchor chain links + each anchor's own hash, and report whether the
    current index has drifted from the latest anchor. Returns (chain_ok, message)."""
    _anchors, chain, _head, index = _paths(workspace)
    if not chain.exists():
        return (True, "no_anchor_chain_yet")
    recs = []
    for line in chain.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            return (False, "unparseable anchor line")
    prev = ""
    for i, r in enumerate(recs):
        if str(r.get("prev_anchor_sha256", "")) != prev:
            return (False, f"chain break at seq {r.get('seq', i)} (prev_anchor mismatch)")
        if _anchor_hash(r) != r.get("anchor_sha256"):
            return (False, f"anchor hash mismatch at seq {r.get('seq', i)}")
        prev = r["anchor_sha256"]
    latest = recs[-1]
    index_drift = _sha256_file(index) != latest.get("index_sha256")
    return (True, f"chain_ok len={len(recs)} latest_seq={latest.get('seq')} index_drift={index_drift}")


def _main() -> None:
    import sys
    from tar_storage import resolve_workspace
    ws = resolve_workspace(Path(__file__).resolve().parents[1])
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "write":
        print(json.dumps(write_canonical_anchor(ws), indent=2))
    else:
        ok, msg = verify_canonical_anchor_chain(ws)
        print(f"anchor chain ok={ok}: {msg}")


if __name__ == "__main__":
    _main()
