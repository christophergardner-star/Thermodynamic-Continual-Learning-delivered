"""Phase 2.1 self-improvement: scientific calibration loop.

Read-only aggregator that asks "is TAR's confidence calibrated?" — it compares what TAR
PREDICTED/expected against what it actually OBSERVED, and turns observed effect sizes into
a power-based sample-size recommendation for FUTURE experiments. Produces a durable,
inspectable registry: tar_state/calibration/calibration_registry.json.

Two views:
  1. effect-size calibration (per result with a Cohen's d): observed |d|, achieved power,
     and seeds-needed-for-80%-power — i.e. "for adequate power on this effect, run N seeds".
  2. frontier calibration: did a frontier TAR ranked highly actually pan out? Flags
     optimistic miscalibration (e.g. fp-catastrophic-forgetting: ranked priority 10 but
     falsified with 17 null / 4 adverse).

INTEGRITY RAIL (#3): this is ADVISORY ONLY. It NEVER edits a pre-registration, never changes
a hypothesis post-hoc, and never lowers an evidence standard. Any sample-size change a human
makes from these recommendations must be pre-registered with an amendment-log entry BEFORE
running. Off-switch: tar_state/calibration.disabled.

Consumer (B3): propose_seed_amendments() finally CONSUMES the registry — it materialises each
underpowered result as a PROPOSED pre-registration seed amendment in a durable, human-gated
log (calibration/preregistration_amendments.json). It still NEVER edits a pre-registration or
a seed count: proposals sit at status "proposed_pending_human_approval" until a human acts, and
the log is append-only-safe (human decisions are preserved across cycles). The director only
calls it when tar_state/calibration_amendments.enabled exists (OFF by default).
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_REL = Path("calibration") / "calibration_registry.json"
AMENDMENTS_REL = Path("calibration") / "preregistration_amendments.json"
DISABLE_FLAG = "calibration.disabled"
_TARGET_POWER = 0.8


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


def recommended_sample_size(effect_size_d: float, current_n: int = 5, alpha: float = 0.05):
    """Seeds needed for ~80% power at the given (observed) effect size. Uses the project's
    stat_utils power model; returns None if unavailable. Advisory only."""
    try:
        from tar_lab import stat_utils
        res = stat_utils.power_analysis(abs(float(effect_size_d)), n_seeds=int(max(1, current_n)), alpha=alpha)
        return getattr(res, "seeds_needed_80pct", None)
    except Exception:
        return None


def _bk_count(entry: dict) -> int:
    v = entry.get("breakthroughs_found", entry.get("breakthrough_count", 0))
    if isinstance(v, list):
        return len(v)
    try:
        return int(v or 0)
    except Exception:
        return 0


def rebuild_calibration(workspace) -> dict:
    """Build and persist the calibration registry. Read-only on sources."""
    now = _now()

    # --- 1. Effect-size calibration: scan comparison result files carrying a Cohen's d ---
    effect_rows: list[dict] = []
    comp_dir = _state(workspace, "comparisons")
    if comp_dir.exists():
        for f in sorted(comp_dir.glob("*.json")):
            data = _jload(f)
            if not isinstance(data, dict):
                continue
            d = data.get("cohens_d")
            if d is None:
                continue
            try:
                d = float(d)
            except Exception:
                continue
            seeds_run = int(data.get("seeds_run", 0) or 0)
            rec_n = data.get("seeds_needed_80pct")
            if rec_n is None and seeds_run > 0:
                rec_n = recommended_sample_size(d, seeds_run)
            power = data.get("achieved_power")
            underpowered = bool(power is not None and float(power) < _TARGET_POWER) or bool(
                rec_n is not None and seeds_run and int(rec_n) > seeds_run
            )
            effect_rows.append({
                "result_id": str(data.get("result_id", f.stem) or f.stem),
                "observed_cohens_d": round(d, 4),
                "seeds_run": seeds_run,
                "achieved_power": round(float(power), 4) if power is not None else None,
                "seeds_needed_for_80pct_power": int(rec_n) if rec_n is not None else None,
                "verdict": str(data.get("verdict", "") or ""),
                "wilcoxon_p": data.get("wilcoxon_p"),
                "recommendation": (
                    f"underpowered (power={float(power):.2f} at n={seeds_run}); "
                    f"~{int(rec_n)} seeds needed for 80% power — "
                    f"pre-register the larger n with an amendment before running."
                    if (underpowered and rec_n is not None and power is not None)
                    else "adequately powered" if power is not None and not underpowered
                    else "insufficient data"
                ),
                "calibration_flag": "underpowered" if underpowered else "adequate",
            })

    # --- 2. Frontier calibration: ranked-high vs actually-panned-out ---
    frontier_rows: list[dict] = []
    fp = _jload(_state(workspace, "frontier_problems.json")) or {}
    for e in (fp.get("problems", []) if isinstance(fp, dict) else []):
        if not isinstance(e, dict):
            continue
        fid = str(e.get("id", "") or e.get("problem_id", "") or "")
        if not fid:
            continue
        bk = _bk_count(e)
        nulls = int(e.get("null_count", 0) or 0)
        adverse = int(e.get("adverse_count", 0) or 0)
        total = bk + nulls + adverse
        priority = e.get("priority")
        try:
            priority = int(priority)
        except Exception:
            priority = None
        truth = str(e.get("truth_status", "") or "")
        bk_rate = (bk / total) if total else None
        # Optimistic miscalibration: TAR ranked it highly (low priority number) but it
        # falsified, or accumulated many verdicts with almost no breakthroughs.
        ranked_high = priority is not None and priority <= 15
        optimistic = bool(ranked_high and (truth == "falsified" or (total >= 5 and (bk_rate or 0) < 0.1)))
        frontier_rows.append({
            "frontier_id": fid,
            "priority_rank": priority,
            "breakthroughs": bk, "nulls": nulls, "adverse": adverse,
            "total_verdicts": total,
            "breakthrough_rate": round(bk_rate, 3) if bk_rate is not None else None,
            "truth_status": truth,
            "calibration_flag": "optimistic_miscalibration" if optimistic else "ok",
            "note": (
                "Ranked highly but did not pan out — treat its prior as over-optimistic; "
                "do not headline it; falsified-retirement already halts fresh probes."
                if optimistic else ""
            ),
        })

    n_under = sum(1 for r in effect_rows if r["calibration_flag"] == "underpowered")
    n_optim = sum(1 for r in frontier_rows if r["calibration_flag"] == "optimistic_miscalibration")
    registry = {
        "generated_at": now,
        "advisory_only": True,
        "summary": {"effect_results": len(effect_rows), "underpowered": n_under,
                    "frontiers": len(frontier_rows), "optimistic_miscalibrations": n_optim},
        "effect_size_calibration": effect_rows,
        "frontier_calibration": frontier_rows,
    }
    try:
        p = _state(workspace, *REGISTRY_REL.parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass
    return registry


def load_calibration(workspace) -> dict:
    return _jload(_state(workspace, *REGISTRY_REL.parts)) or {}


def load_seed_amendments(workspace) -> dict:
    return _jload(_state(workspace, *AMENDMENTS_REL.parts)) or {}


def propose_seed_amendments(workspace, registry: dict | None = None) -> dict:
    """RAIL #3 consumer: materialise underpowered calibration rows as PROPOSED
    pre-registration seed amendments.

    Each underpowered result (achieved power < target AND power-based seeds-needed >
    seeds-run) yields one amendment recommending that a confirmatory re-run pre-register
    the larger seed count. STRICTLY advisory + human-gated:
      * writes only its own log (calibration/preregistration_amendments.json);
      * NEVER edits a pre-registration, a hypothesis, or a seed count;
      * append-only-safe + idempotent — an existing amendment_id is preserved untouched
        (so a human-set status like "approved"/"rejected" survives every cycle), only
        genuinely new proposals are added.
    Returns the amendments doc.
    """
    if registry is None:
        registry = load_calibration(workspace)
    rows = registry.get("effect_size_calibration", []) if isinstance(registry, dict) else []

    existing_doc = load_seed_amendments(workspace)
    existing = existing_doc.get("amendments", []) if isinstance(existing_doc, dict) else []
    by_id: dict[str, dict] = {
        str(a.get("amendment_id")): a
        for a in existing if isinstance(a, dict) and a.get("amendment_id")
    }

    added = 0
    for r in rows:
        if not isinstance(r, dict) or r.get("calibration_flag") != "underpowered":
            continue
        rec_n = r.get("seeds_needed_for_80pct_power")
        if rec_n is None:
            continue
        try:
            rec_n = int(rec_n)
        except Exception:
            continue
        seeds_run = int(r.get("seeds_run", 0) or 0)
        if rec_n <= seeds_run:
            continue
        result_id = str(r.get("result_id", "") or "")
        if not result_id:
            continue
        amendment_id = f"seed-amend::{result_id}::n{rec_n}"
        if amendment_id in by_id:
            continue  # preserve the existing entry (incl. any human decision)
        by_id[amendment_id] = {
            "amendment_id": amendment_id,
            "rail": 3,
            "status": "proposed_pending_human_approval",
            "result_id": result_id,
            "observed_cohens_d": r.get("observed_cohens_d"),
            "seeds_run": seeds_run,
            "recommended_seeds": rec_n,
            "rationale": str(
                r.get("recommendation", "")
                or f"underpowered at n={seeds_run}; ~{rec_n} seeds needed for 80% power"
            ),
            "proposed_at": _now(),
            "note": (
                "ADVISORY. A human must pre-register this larger n with an amendment-log "
                "entry BEFORE any confirmatory re-run. TAR does not change seeds itself."
            ),
        }
        added += 1

    pending = sum(
        1 for a in by_id.values()
        if a.get("status") == "proposed_pending_human_approval"
    )
    doc = {
        "generated_at": _now(),
        "advisory_only": True,
        "rail": 3,
        "summary": {"total": len(by_id), "added_this_cycle": added, "pending": pending},
        "amendments": sorted(by_id.values(), key=lambda a: str(a.get("amendment_id", ""))),
    }
    try:
        p = _state(workspace, *AMENDMENTS_REL.parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass
    return doc
