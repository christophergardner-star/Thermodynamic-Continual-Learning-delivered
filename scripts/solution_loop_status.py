"""
Read-only status of the solution-finding loop (solution-loop Phase 5 observability).

Reports readiness + campaign progress WITHOUT running anything:
  - method catalog: loaded count, verified flag
  - SI anomaly: inventory record present? gap present + top-composite? prereg joint criterion?
  - kill-ledger: killed-region count + recent kills
  - candidates: director directives whose method is a novel candidate (tar_novel)
Use at/after activation to watch the campaign. Never writes.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_REPO))


def _jload(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=r"E:\TAR\Thermodynamic-Continual-Learning-delivered")
    args = ap.parse_args()
    ws = Path(args.workspace)
    ts = ws / "tar_state"
    print(f"[solution-loop status] {ws}\n")

    # 1. catalog
    try:
        from literature.method_catalog import load_method_catalog, catalog_is_verified
        cat = load_method_catalog()
        print(f"method catalog : {len(cat)} methods loaded | verified={catalog_is_verified()}")
    except Exception as exc:
        print(f"method catalog : ERROR {exc}")

    # 2. anomaly seeded?
    inv = _jload(ts / "honest_evidence_inventory.json") or {}
    has_inv = any(isinstance(r, dict) and r.get("experiment_id") == "si_stability_anomaly"
                  for r in inv.get("results", []))
    prereg = _jload(ts / "autonomous_research" / "preregistration.json") or {}
    has_crit = any(isinstance(h, dict) and h.get("name") == "si_stability_without_collapse"
                   for h in prereg.get("hypotheses", []))
    gap_present = top_gap = None
    db = ts / "literature" / "literature_graph.db"
    if db.exists():
        try:
            c = sqlite3.connect(str(db))
            row = c.execute("SELECT gap_id, composite_score FROM research_gaps "
                            "WHERE domain='continual_learning' AND status='open' "
                            "ORDER BY composite_score DESC LIMIT 1").fetchone()
            top_gap = row
            gap_present = c.execute(
                "SELECT COUNT(*) FROM research_gaps WHERE gap_id LIKE 'tar_anomaly::si_stability%'"
            ).fetchone()[0] > 0
            c.close()
        except Exception as exc:
            print(f"  (gap query error: {exc})")
    print(f"SI anomaly     : inventory={has_inv} | gap_present={gap_present} | prereg_criterion={has_crit}")
    print(f"  top open CL gap: {top_gap[0] if top_gap else None} (composite={top_gap[1] if top_gap else None})")
    print(f"  -> anomaly is the top gap the director will pick: "
          f"{bool(top_gap and str(top_gap[0]).startswith('tar_anomaly::si_stability'))}")

    # 3. kill-ledger
    try:
        from tar_lab.solution_loop import load_killed_fingerprints, render_kill_ledger_block
        killed = load_killed_fingerprints(ws)
        print(f"\nkill-ledger    : {len(killed)} killed region(s)")
        block = render_kill_ledger_block(ws, max_entries=8)
        if block:
            print(block)
    except Exception as exc:
        print(f"kill-ledger    : ERROR {exc}")

    # 4. candidate directives (novel methods proposed)
    ds = _jload(ts / "research_director_state.json") or {}
    try:
        from tar_lab.method_identity import internal_source_tag
        novel = [d for d in ds.get("experiment_directives", [])
                 if isinstance(d, dict) and internal_source_tag(str(d.get("method", "") or "tcl")) == "tar_novel"]
        print(f"\ncandidate directives (novel methods): {len(novel)}")
        for d in novel[:10]:
            print(f"  - {d.get('experiment_id')}  method={d.get('method')}  status={d.get('status')}")
    except Exception as exc:
        print(f"candidate directives: ERROR {exc}")

    print("\n(read-only; run scripts/seed_si_anomaly.py --apply at activation to seed the campaign)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
