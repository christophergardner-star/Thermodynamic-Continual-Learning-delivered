"""
Truth-Lock backfill DRY-RUN (TL-7, step 1) — READ-ONLY.

Runs the canonical 3-gate verification logic (provenance sibling + manifest-hash +
deterministic recompute) over every result the system currently treats as canonical,
WITHOUT writing the index or changing anything. Purpose: see exactly which existing
results would pass/fail truth-lock enforcement, so we fix the producers (not the gate)
before enforcing — per TAR_TruthLock_Implementation_Plan.md TL-7.

Populations assessed:
  (A) orchestrator results: tar_state/experiments/<run_id>/result.json
  (B) canonical index entries: tar_state/comparisons/canonical_results_index.jsonl -> result_path

Read-only: calls the gate-check functions and reads git (status/ls-files); writes nothing.
Usage: python scripts/truthlock_backfill_dryrun.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from tar_lab.canonical_registry import (  # noqa: E402
    _check_gate_1_sibling,
    _check_gate_2_manifest,
    _check_gate_3_recompute,
)

_TAR_STATE = _REPO / "tar_state"
_COMP = _TAR_STATE / "comparisons"


def _one_word(exc: Exception) -> str:
    s = str(exc).strip().replace("\n", " ")
    return (s[:140] + "…") if len(s) > 140 else s


def assess(result_path: Path) -> dict:
    """Run gates 1->2->3 read-only; stop at first failure and record the reason."""
    r = {"path": str(result_path), "g1": "-", "g2": "-", "g3": "-",
         "passed": False, "reason": ""}
    if not result_path.exists():
        r["reason"] = "result_file_missing"
        return r
    try:
        env_fields = _check_gate_1_sibling(result_path)
        r["g1"] = "PASS"
    except Exception as exc:
        r["g1"] = "FAIL"; r["reason"] = f"gate1: {_one_word(exc)}"
        return r
    try:
        _check_gate_2_manifest(env_fields, _REPO)
        r["g2"] = "PASS"
    except Exception as exc:
        r["g2"] = "FAIL"; r["reason"] = f"gate2: {_one_word(exc)}"
        return r
    try:
        _check_gate_3_recompute(result_path)
        r["g3"] = "PASS"
    except Exception as exc:
        r["g3"] = "FAIL"; r["reason"] = f"gate3: {_one_word(exc)}"
        return r
    r["passed"] = True
    return r


def main() -> None:
    targets: list[tuple[str, Path]] = []

    # (A) orchestrator experiments/<run_id>/result.json
    exp_dir = _TAR_STATE / "experiments"
    if exp_dir.is_dir():
        for d in sorted(exp_dir.iterdir()):
            rp = d / "result.json"
            if rp.exists():
                targets.append(("A:experiments", rp))

    # (B) canonical index -> result_path (comparison files)
    idx = _COMP / "canonical_results_index.jsonl"
    seen = {str(p.resolve()) for _, p in targets}
    if idx.exists():
        for line in idx.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rpt = str(rec.get("result_path", "") or "")
            if not rpt:
                continue
            p = Path(rpt)
            if not p.is_absolute():
                p = (_REPO / rpt).resolve()
            if str(p.resolve()) not in seen:
                targets.append(("B:canonical_index", p))
                seen.add(str(p.resolve()))

    results = [(src, assess(p)) for src, p in targets]
    n = len(results)
    npass = sum(1 for _, r in results if r["passed"])

    # failure breakdown
    by_gate = {"gate1": 0, "gate2": 0, "gate3": 0, "missing": 0}
    for _, r in results:
        if r["passed"]:
            continue
        if r["reason"].startswith("gate1"):
            by_gate["gate1"] += 1
        elif r["reason"].startswith("gate2"):
            by_gate["gate2"] += 1
        elif r["reason"].startswith("gate3"):
            by_gate["gate3"] += 1
        else:
            by_gate["missing"] += 1

    print("=" * 78)
    print("TRUTH-LOCK BACKFILL DRY-RUN (read-only) — what WOULD pass canonical verification")
    print("=" * 78)
    print(f"assessed: {n}   PASS all 3 gates: {npass}   FAIL: {n - npass}")
    print(f"first-failing-gate breakdown: {by_gate}")
    print("-" * 78)
    for src, r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"  [{tag}] {src:18} g1={r['g1']:4} g2={r['g2']:4} g3={r['g3']:4} "
              f"{Path(r['path']).name[:46]:46} {r['reason']}")

    # cross-reference the honest inventory's publication-allowed set
    inv = _TAR_STATE / "honest_evidence_inventory.json"
    print("-" * 78)
    if inv.exists():
        try:
            d = json.loads(inv.read_text(encoding="utf-8"))
            allowed = []
            for k, v in (d.items() if isinstance(d, dict) else []):
                if isinstance(v, dict) and v.get("publication_allowed") in (True, "PUBLICATION_ALLOWED"):
                    allowed.append(k)
            print(f"honest_evidence_inventory publication-allowed keys: {allowed or '(parse: see file)'}")
        except Exception as exc:
            print(f"honest_evidence_inventory: parse error {_one_word(exc)}")
    print("=" * 78)
    print("READ-ONLY: nothing was written. This report informs the producer-fix + enforce order.")


if __name__ == "__main__":
    main()
