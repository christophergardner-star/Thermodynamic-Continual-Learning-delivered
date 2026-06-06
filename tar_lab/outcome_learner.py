"""Phase 1.3 self-improvement: outcome registry.

Read-only aggregator over experiment outcomes (experiment_archive.json + the previously
write-only findings / failure-diagnosis LLM memos in llm_cache/). Produces:
  - method x dataset reliability priors (what reliably runs/works), and
  - a per-experiment outcome digest (status, verdict, forgetting, latest finding &
    failure-diagnosis),
so the Research Director can finally CONSUME experiment outcomes instead of dropping them
(today findings/diagnoses are read only by the dashboard).

Director use (conservative, penalty-only — preserves exploration; rail #5):
  - failure_penalty(): modestly deprioritise re-proposing an experiment whose archived
    terminal state was an operational FAILURE (errored), so a broken experiment does not
    keep cluttering at high priority. Reversible once it completes. NOT applied to
    scientific NULL/ADVERSE verdicts — those are handled by frontier truth_status /
    falsified-retirement / the calibration loop, and underpowered nulls are legitimately
    worth re-running.
  - outcome_context(): surfaces the latest finding / failure-diagnosis so they reach the
    director state and the LLM follow-up proposer (closes the write-only gap).

The method x dataset priors are recorded for the operator and the Phase-2 calibration loop.

Reward (Phase 1.3b, OPT-IN, exploration-safe — see reliability_reward()):
  TAR could previously only PUNISH a priority (vetoes, failure penalty), never reinforce.
  reliability_reward() adds a small, bounded boost for an experiment whose method x dataset
  config has a RELIABLE OPERATIONAL track record (clean completions + reproducible low
  variance). Crucially it rewards measurement RELIABILITY, NOT a favourable scientific
  outcome — a config that reliably does *badly* earns the same reward — so it does NOT
  reinforce a single winning method (which would collapse exploration), and a novel config
  with no track record earns 0 rather than a penalty (exploration is never pushed down).
  It is OFF unless the operator creates tar_state/outcome_reward.enabled, and bounded well
  below the failure penalty + status boosts so it nudges, never dominates.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

PRIORS_REL = Path("outcome_learning") / "outcome_priors.json"
DISABLE_FLAG = "outcome_learning.disabled"

_FAILURE_PENALTY = 50.0
_REPRO_MIN_SAMPLES = 3
_REPRO_MAX_STD = 0.08
_MEMO_MAXLEN = 600

# Reward (opt-in) — bounded well below _FAILURE_PENALTY (50) and the status boosts (36-58)
# so a reliable config is nudged, never allowed to dominate or collapse exploration.
_RELIABILITY_REWARD_CAP = 12.0
_REWARD_MIN_COMPLETED = 3          # need a real track record, not one lucky run
_REWARD_MIN_COMPLETION_RATE = 0.5  # a config that mostly fails is not "reliable"


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


def _memo(workspace, prefix: str, exp_id: str) -> str:
    if not exp_id:
        return ""
    data = _jload(_state(workspace, "llm_cache", f"{prefix}_{exp_id}.json"))
    if isinstance(data, dict):
        return str(data.get("content", "") or "")[:_MEMO_MAXLEN]
    return ""


def _forgetting_values(entry: dict):
    prog = entry.get("progress", {}) if isinstance(entry.get("progress", {}), dict) else {}
    vals = prog.get("forgetting_so_far", []) or []
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except Exception:
            pass
    return out


def rebuild_outcome_priors(workspace) -> dict:
    """Read the archive + memos and (re)write the durable outcome prior. Returns it.

    Read-only on the source state; writes only outcome_learning/outcome_priors.json.
    """
    arch = _jload(_state(workspace, "experiment_archive.json")) or {}
    entries = arch.get("experiments", []) if isinstance(arch, dict) else (arch if isinstance(arch, list) else [])

    by_exp: dict[str, dict] = {}
    by_md: dict[str, dict] = {}

    for e in entries:
        if not isinstance(e, dict):
            continue
        exp_id = str(e.get("id", "") or "")
        if not exp_id:
            continue
        method = str(e.get("method", "") or "")
        dataset = str(e.get("dataset", "") or "")
        status = str(e.get("status", "") or "")
        stage = str(e.get("stage", "") or "")
        error = str(e.get("error", "") or "")
        verdict = str(e.get("verdict", "") or "")
        is_failure = (status == "failed") or (bool(error) and status not in {"complete", "skipped"})
        fvals = _forgetting_values(e)
        forgetting_mean = round(statistics.fmean(fvals), 4) if fvals else None

        by_exp[exp_id] = {
            "method": method, "dataset": dataset, "status": status, "stage": stage,
            "verdict": verdict, "forgetting_mean": forgetting_mean,
            "is_failure": is_failure, "error": error[:200],
            "finding": _memo(workspace, "findings", exp_id),
            "failure_diagnosis": _memo(workspace, "failure", exp_id),
            "completed_at": str(e.get("completed_at", "") or e.get("archived_at", "") or ""),
        }

        key = f"{method}::{dataset}"
        md = by_md.setdefault(key, {"method": method, "dataset": dataset, "n": 0,
                                    "completed": 0, "failed": 0, "skipped": 0,
                                    "forgetting": [], "verdicts": {}})
        md["n"] += 1
        if status == "complete":
            md["completed"] += 1
        elif is_failure:
            md["failed"] += 1
        elif status == "skipped":
            md["skipped"] += 1
        md["forgetting"].extend(fvals)
        if verdict:
            md["verdicts"][verdict] = md["verdicts"].get(verdict, 0) + 1

    # Finalise method x dataset reliability stats
    for key, md in by_md.items():
        fvals = md.pop("forgetting", [])
        md["forgetting_mean"] = round(statistics.fmean(fvals), 4) if fvals else None
        md["forgetting_std"] = round(statistics.pstdev(fvals), 4) if len(fvals) >= 2 else None
        md["n_forgetting"] = len(fvals)
        md["reproducible"] = bool(
            len(fvals) >= _REPRO_MIN_SAMPLES
            and md["forgetting_std"] is not None
            and md["forgetting_std"] < _REPRO_MAX_STD
        )

    priors = {"generated_at": _now(), "by_experiment": by_exp, "by_method_dataset": by_md}
    try:
        p = _state(workspace, *PRIORS_REL.parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(priors, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass
    return priors


def load_outcome_priors(workspace) -> dict:
    return _jload(_state(workspace, *PRIORS_REL.parts)) or {}


def failure_penalty(priors: dict, experiment_id: str) -> tuple[float, str]:
    """Penalty (>=0) to subtract: deprioritise re-proposing an experiment that previously
    FAILED operationally. Penalty-only; 0.0 otherwise."""
    if not isinstance(priors, dict) or not experiment_id:
        return 0.0, ""
    rec = (priors.get("by_experiment", {}) or {}).get(str(experiment_id))
    if isinstance(rec, dict) and rec.get("is_failure"):
        why = (rec.get("error") or "previous run failed")[:120]
        return _FAILURE_PENALTY, f"prior run failed operationally: {why}"
    return 0.0, ""


def reliability_reward(
    priors: dict,
    experiment_id: str,
    method: str = "",
    dataset: str = "",
) -> tuple[float, str]:
    """Bounded reward (>=0) to ADD: lightly prefer an experiment whose method x dataset
    config has a RELIABLE operational track record.

    Exploration-safe by construction:
      * rewards measurement RELIABILITY (clean completions + reproducible low variance),
        NOT a favourable scientific outcome — a config that reliably does *badly* earns the
        same reward, so a single winning method is never reinforced;
      * a novel config (no track record) earns 0, never a penalty, so exploration is not
        pushed down;
      * capped at _RELIABILITY_REWARD_CAP, far below the failure penalty + status boosts.

    method/dataset describe the experiment being scored (the per-experiment record may not
    exist yet for a fresh proposal); they fall back to the archived record when omitted.
    """
    if not isinstance(priors, dict) or not experiment_id:
        return 0.0, ""
    rec = (priors.get("by_experiment", {}) or {}).get(str(experiment_id)) or {}
    method = str(method or rec.get("method", "") or "")
    dataset = str(dataset or rec.get("dataset", "") or "")
    if not method or not dataset:
        return 0.0, ""
    md = (priors.get("by_method_dataset", {}) or {}).get(f"{method}::{dataset}")
    if not isinstance(md, dict):
        return 0.0, ""
    n = int(md.get("n", 0) or 0)
    completed = int(md.get("completed", 0) or 0)
    if n <= 0 or completed < _REWARD_MIN_COMPLETED:
        return 0.0, ""
    completion_rate = completed / n
    if completion_rate < _REWARD_MIN_COMPLETION_RATE:
        return 0.0, ""
    # 60% of the cap scales with how cleanly the config runs; the remaining 40% is a bonus
    # for reproducible (low-variance) measurements. Bounded to the cap.
    reward = _RELIABILITY_REWARD_CAP * 0.6 * completion_rate
    reproducible = bool(md.get("reproducible"))
    if reproducible:
        reward += _RELIABILITY_REWARD_CAP * 0.4
    reward = round(min(reward, _RELIABILITY_REWARD_CAP), 1)
    if reward <= 0.0:
        return 0.0, ""
    why = (f"reliable config {method}/{dataset}: {completed}/{n} clean"
           + (", reproducible" if reproducible else ""))
    return reward, why


def outcome_context(priors: dict, experiment_id: str) -> dict:
    """Compact per-experiment outcome digest for the director to surface (closes the
    write-only gap for findings / failure diagnoses). Empty dict if nothing known."""
    if not isinstance(priors, dict) or not experiment_id:
        return {}
    rec = (priors.get("by_experiment", {}) or {}).get(str(experiment_id))
    if not isinstance(rec, dict):
        return {}
    out = {}
    for k in ("verdict", "forgetting_mean", "is_failure", "finding", "failure_diagnosis"):
        v = rec.get(k)
        if v:
            out[k] = v
    return out
