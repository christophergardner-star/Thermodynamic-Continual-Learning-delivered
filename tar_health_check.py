"""
TAR Live Health Check — Sections Y + Z.

Read-only cross-loop consistency and bug detection.
Safe to run at any time while TAR is live: no writes, no GPU, no snapshot/restore.

Sections
--------
  Y    Director coherence   — duplicate configs, frontier ID validity, VRAM budgets, paper readiness
  Z    System coherence     — daemon/queue/archive/author cross-loop integrity, zombie PIDs, null fields

Run:  python tar_health_check.py
Exit 0 = all checks passed.
Exit 1 = one or more failures. Review output; do NOT stop TAR without diagnosis.
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

_results: list[tuple[str, str, str, str]] = []


def check(sec: str, name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    _results.append((sec, status, name, detail))
    print(f"  {status} {sec}:{name}" + (f" — {detail}" if detail else ""), flush=True)


def skip(sec: str, name: str, reason: str = "") -> None:
    _results.append((sec, SKIP, name, reason))
    print(f"  {SKIP} {sec}:{name}" + (f" — {reason}" if reason else ""), flush=True)


def section(label: str, title: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"Section {label}: {title}", flush=True)
    print("="*60, flush=True)


# ── Path constants ─────────────────────────────────────────────────────────────
_QUEUE_PATH      = WORKSPACE / "tar_state" / "experiment_queue.json"
_ARCHIVE_PATH    = WORKSPACE / "tar_state" / "experiment_archive.json"
_FRONTIER_PATH   = WORKSPACE / "tar_state" / "frontier_problems.json"
_AUTHOR_STATE    = WORKSPACE / "tar_state" / "author_state.json"
_PROC_REG_PATH   = WORKSPACE / "tar_state" / "process_registry.json"
_DAEMON_PATH     = WORKSPACE / "tar_state" / "living_research_daemon.json"
_WATCHDOG_LOCK   = WORKSPACE / "tar_state" / "watchdog.lock.json"
_DIRECTOR_STATE  = WORKSPACE / "tar_state" / "research_director_state.json"


# ── JSON helper ────────────────────────────────────────────────────────────────
def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# ── Load state (single read per file) ─────────────────────────────────────────
_q        = _load_json(_QUEUE_PATH,    {"experiments": []})
_arc      = _load_json(_ARCHIVE_PATH,  {"experiments": []})
_auth     = _load_json(_AUTHOR_STATE,  {})
_daemon   = _load_json(_DAEMON_PATH,   {})
_proc     = _load_json(_PROC_REG_PATH, {})
_director = _load_json(_DIRECTOR_STATE, {})

# Normalize author paper list: author_state uses paper_queue (list) + current_paper (dict)
# Each item has project_id (not id), experiment_ids, waiting_for_experiments, etc.
def _iter_papers(auth: dict):
    """Yield (paper_id, paper_data) for all papers in author state."""
    for _p in auth.get("paper_queue", []):
        _pid = str(_p.get("project_id") or _p.get("paper_id") or _p.get("id") or "")
        if _pid:
            yield _pid, _p
    _cp = auth.get("current_paper", {})
    if isinstance(_cp, dict):
        _pid = str(_cp.get("project_id") or _cp.get("paper_id") or _cp.get("id") or "")
        if _pid:
            yield _pid, _cp


# ══════════════════════════════════════════════════════════════════════════════
# Y: Director coherence
# ══════════════════════════════════════════════════════════════════════════════
section("Y", "Director coherence — queue integrity, frontier validity, paper readiness")

# Y1: No duplicate experiment configurations in queue
# Groups by (frontier_problem_id, dataset, method, config, seeds).
# Only flags when >=3 share the same key within the same frontier problem —
# a pair might be intentional A/B testing; 3+ is almost certainly a Director bug.
try:
    _q_live = [_r for _r in _q.get("experiments", [])
               if _r.get("status") in {"pending", "queued", "running"}]
    _seen_cfg: dict[tuple, list[str]] = {}
    for _r in _q_live:
        _fp = str(_r.get("frontier_problem_id", "") or "")
        if not _fp:
            continue  # skip experiments not assigned to a frontier problem
        _cfg_key = (
            _fp,
            str(_r.get("dataset", "") or ""),
            str(_r.get("method", "") or ""),
            json.dumps(_r.get("config_overrides") or {}, sort_keys=True),
            json.dumps(sorted(_r.get("seeds") or [])),
        )
        _seen_cfg.setdefault(_cfg_key, []).append(str(_r.get("id", "")))
    # Require >=3 in a group; pairs may be intentional (different hypotheses, same setup)
    _dup_groups = {_k: _v for _k, _v in _seen_cfg.items() if len(_v) >= 3}
    check("Y", "no_duplicate_experiment_configs",
          len(_dup_groups) == 0,
          f"{len(_dup_groups)} group(s) with 3+ identical configs: "
          + "; ".join(str(_v) for _v in _dup_groups.values()))
except Exception as _e:
    check("Y", "no_duplicate_experiment_configs", False, str(_e)[:120])

# Y2: No re-queuing of archived complete experiments
try:
    _arc_complete_ids = {
        str(_r.get("id", ""))
        for _r in _arc.get("experiments", [])
        if str(_r.get("status", "")).lower() == "complete"
    }
    _q_pending_ids = {
        str(_r.get("id", ""))
        for _r in _q.get("experiments", [])
        if _r.get("status") in {"pending", "queued"}
    }
    _requeued = _arc_complete_ids & _q_pending_ids
    check("Y", "no_requeued_complete_experiments",
          len(_requeued) == 0,
          f"IDs in archive(complete) AND queue(pending): {sorted(_requeued)}")
except Exception as _e:
    check("Y", "no_requeued_complete_experiments", False, str(_e)[:120])

# Y3: All frontier_problem_id references in queue are valid
try:
    _fp_data = _load_json(_FRONTIER_PATH, {})
    # frontier_problems.json uses "problems" key (not "frontier_problems")
    _fp_list = _fp_data.get("problems", _fp_data.get("frontier_problems", []))
    _known_fp_ids = {str(_fp.get("id", "")) for _fp in _fp_list}
    _bad_fp_refs = [
        str(_r.get("id", "")) + "→" + str(_r.get("frontier_problem_id", ""))
        for _r in _q.get("experiments", [])
        if _r.get("frontier_problem_id")
           and str(_r.get("frontier_problem_id", "")).strip()
           and str(_r.get("frontier_problem_id", "")).strip() not in _known_fp_ids
    ]
    check("Y", "all_frontier_problem_ids_valid",
          len(_bad_fp_refs) == 0,
          f"unknown frontier IDs: {_bad_fp_refs}")
except Exception as _e:
    check("Y", "all_frontier_problem_ids_valid", False, str(_e)[:120])

# Y4: Director state file is recent (< 12 hours)
try:
    import time as _time_y
    if _DIRECTOR_STATE.exists():
        _dir_age = _time_y.time() - _DIRECTOR_STATE.stat().st_mtime
        check("Y", "director_state_fresh",
              _dir_age < 43200,
              f"age={_dir_age/3600:.1f}h (threshold=12h)")
    else:
        skip("Y", "director_state_fresh", "research_director_state.json not found")
except Exception as _e:
    check("Y", "director_state_fresh", False, str(_e)[:80])

# Y5: VRAM budget entries exist for all queued datasets
try:
    _sched_src = (WORKSPACE / "tar_scheduler.py").read_text(encoding="utf-8")
    _q_datasets = {
        str(_r.get("dataset", ""))
        for _r in _q.get("experiments", [])
        if _r.get("dataset")
    }
    _missing_vram = [
        _ds for _ds in _q_datasets
        if f'"{_ds}"' not in _sched_src and f"'{_ds}'" not in _sched_src
    ]
    check("Y", "all_queued_datasets_have_vram_budget",
          len(_missing_vram) == 0,
          f"datasets not in _VRAM_BUDGET: {_missing_vram}")
except Exception as _e:
    check("Y", "all_queued_datasets_have_vram_budget", False, str(_e)[:80])

# Y6: Papers with readiness=outline_now/write_now have >=1 trusted result
try:
    from tar_lab.validation import _trusted_experiment_ids as _trusted_fn
    _trusted_set = _trusted_fn(WORKSPACE)
    _bad_readiness_papers = []
    for _pid, _pdata in _iter_papers(_auth):
        if not bool(_pdata.get("human_approved", False)):
            continue  # unapproved papers can't trigger writing; skip
        _readiness = str(_pdata.get("readiness", "") or "").lower()
        if _readiness in {"outline_now", "write_now"}:
            _exp_ids = _pdata.get("experiment_ids") or []
            _trusted_count = sum(1 for _eid in _exp_ids if _eid in _trusted_set)
            if _trusted_count == 0:
                _bad_readiness_papers.append(_pid)
    check("Y", "ready_papers_have_trusted_evidence",
          len(_bad_readiness_papers) == 0,
          f"papers with readiness=outline/write but 0 trusted results: {_bad_readiness_papers}")
except Exception as _e:
    check("Y", "ready_papers_have_trusted_evidence", False, str(_e)[:120])

# Y7: No contradictory paper state (blocked + human_approved + empty experiment_ids)
try:
    _contradictory = []
    for _pid, _pdata in _iter_papers(_auth):
        _blocked    = str(_pdata.get("status", "") or _pdata.get("paper_status", "")).lower() == "blocked"
        _approved   = bool(_pdata.get("human_approved", False))
        _no_exp_ids = len(_pdata.get("experiment_ids") or []) == 0
        if _blocked and _approved and _no_exp_ids:
            _contradictory.append(_pid)
    check("Y", "no_contradictory_paper_state",
          len(_contradictory) == 0,
          f"blocked+approved+empty: {_contradictory}")
except Exception as _e:
    check("Y", "no_contradictory_paper_state", False, str(_e)[:80])


# ══════════════════════════════════════════════════════════════════════════════
# Z: System coherence
# ══════════════════════════════════════════════════════════════════════════════
section("Z", "System coherence — daemon/queue/archive/author cross-loop integrity")

# Z1: Daemon running_ids ↔ queue running entries exact match
try:
    _daemon_running = set(_daemon.get("running_ids") or [])
    _queue_running  = {
        str(_r.get("id", ""))
        for _r in _q.get("experiments", [])
        if _r.get("status") == "running"
    }
    check("Z", "daemon_running_ids_match_queue",
          _daemon_running == _queue_running,
          f"daemon={sorted(_daemon_running)} queue={sorted(_queue_running)}")
except Exception as _e:
    check("Z", "daemon_running_ids_match_queue", False, str(_e)[:120])

# Z2: Daemon active_experiment_id is in queue running set
try:
    _active = str(_daemon.get("active_experiment_id") or "").strip()
    if _active:
        _q_running_ids = {
            str(_r.get("id", ""))
            for _r in _q.get("experiments", [])
            if _r.get("status") == "running"
        }
        check("Z", "active_experiment_id_is_running_in_queue",
              _active in _q_running_ids,
              f"active={_active!r} not in running set {sorted(_q_running_ids)}")
    else:
        skip("Z", "active_experiment_id_is_running_in_queue", "no active experiment")
except Exception as _e:
    check("Z", "active_experiment_id_is_running_in_queue", False, str(_e)[:80])

# Z3: Windows zombie PID detection via tasklist
# psutil.pid_exists() and kernel32.OpenProcess both lie for zombie PIDs on Windows;
# tasklist is the reliable ground truth.
try:
    _running_with_pid = [
        _r for _r in _q.get("experiments", [])
        if _r.get("status") == "running" and int(_r.get("pid") or 0) > 0
    ]
    _zombie_pids = []
    for _r in _running_with_pid:
        _pid = int(_r.get("pid"))
        _tl = subprocess.run(
            ["tasklist", "/FI", f"PID eq {_pid}", "/NH"],
            capture_output=True, text=True, timeout=10,
        )
        if str(_pid) not in _tl.stdout:
            _zombie_pids.append((_r.get("id"), _pid))
    check("Z", "no_zombie_running_pids_tasklist",
          len(_zombie_pids) == 0,
          f"zombie PIDs (in queue as running but not in tasklist): {_zombie_pids}")
except FileNotFoundError:
    skip("Z", "no_zombie_running_pids_tasklist", "tasklist not available (non-Windows)")
except Exception as _e:
    check("Z", "no_zombie_running_pids_tasklist", False, str(_e)[:120])

# Z4: Watchdog lock PID is live
try:
    if _WATCHDOG_LOCK.exists():
        _lock = _load_json(_WATCHDOG_LOCK, {})
        _lock_pid = int(_lock.get("pid") or 0)
        if _lock_pid:
            _tl_lock = subprocess.run(
                ["tasklist", "/FI", f"PID eq {_lock_pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            )
            _lock_live = str(_lock_pid) in _tl_lock.stdout
            check("Z", "watchdog_lock_pid_live", _lock_live,
                  f"pid={_lock_pid} {'found' if _lock_live else 'NOT FOUND in tasklist'}")
        else:
            skip("Z", "watchdog_lock_pid_live", "no PID in lock file")
    else:
        skip("Z", "watchdog_lock_pid_live", "watchdog.lock.json not found")
except FileNotFoundError:
    skip("Z", "watchdog_lock_pid_live", "tasklist not available (non-Windows)")
except Exception as _e:
    check("Z", "watchdog_lock_pid_live", False, str(_e)[:80])

# Z5: Daemon last_tick is recent (< 2 × poll_interval_s + 60s)
try:
    from datetime import datetime, timezone as _tz
    _last_tick = str(_daemon.get("last_tick") or "")
    _poll_int  = float(_daemon.get("poll_interval_s") or 30.0)
    _threshold = _poll_int * 2 + 60.0
    if _last_tick:
        _tick_dt  = datetime.fromisoformat(_last_tick.replace("Z", "+00:00"))
        _tick_age = (datetime.now(_tz.utc) - _tick_dt).total_seconds()
        if _daemon.get("status") == "running":
            check("Z", "daemon_last_tick_recent",
                  _tick_age < _threshold,
                  f"age={_tick_age:.0f}s threshold={_threshold:.0f}s")
        else:
            skip("Z", "daemon_last_tick_recent",
                 f"daemon status={_daemon.get('status')!r} (not running)")
    else:
        skip("Z", "daemon_last_tick_recent", "no last_tick in daemon state")
except Exception as _e:
    check("Z", "daemon_last_tick_recent", False, str(_e)[:80])

# Z6: No experiment in both queue (live) AND archive (complete)
try:
    _arc_complete = {
        str(_r.get("id", ""))
        for _r in _arc.get("experiments", [])
        if str(_r.get("status", "")).lower() == "complete"
    }
    _q_live_ids = {
        str(_r.get("id", ""))
        for _r in _q.get("experiments", [])
        if _r.get("status") not in {"complete", "failed", "cancelled"}
    }
    _in_both = _arc_complete & _q_live_ids
    check("Z", "no_experiment_in_archive_complete_and_queue_live",
          len(_in_both) == 0,
          f"IDs in both: {sorted(_in_both)}")
except Exception as _e:
    check("Z", "no_experiment_in_archive_complete_and_queue_live", False, str(_e)[:80])

# Z7: Director paper_directives waiting_for_experiments ⊆ {queue_pending ∪ queue_running}
# Source of truth is research_director_state.json (paper_directives), not author_state.json.
# author_state.json no longer stores waiting_for_experiments IDs to avoid drift.
try:
    _q_pending_running = {
        str(_r.get("id", ""))
        for _r in _q.get("experiments", [])
        if _r.get("status") in {"pending", "queued", "running"}
    }
    _stale_waiting: dict[str, list] = {}
    for _directive in (_director.get("paper_directives") or []):
        _dpid    = str(_directive.get("paper_id") or _directive.get("project_id") or "")
        _waiting = _directive.get("waiting_for_experiments") or _directive.get("waiting_on_experiment_ids") or []
        _stale   = [_eid for _eid in _waiting if str(_eid) not in _q_pending_running]
        if _stale and _dpid:
            _stale_waiting[_dpid] = _stale
    check("Z", "director_waiting_for_experiments_all_live",
          len(_stale_waiting) == 0,
          f"stale waiting entries in director directives (experiment finished but still listed): {_stale_waiting}")
except Exception as _e:
    check("Z", "director_waiting_for_experiments_all_live", False, str(_e)[:120])

# Z8: No orphaned experiment IDs in author papers
try:
    from tar_lab.validation import (
        _trusted_experiment_ids as _trusted_fn2,
        _superseded_experiment_ids as _superseded_fn,
    )
    _trusted_set2    = _trusted_fn2(WORKSPACE)
    _superseded_set  = _superseded_fn(WORKSPACE)
    _all_known = (
        {str(_r.get("id", "")) for _r in _q.get("experiments", [])}
        | {str(_r.get("id", "")) for _r in _arc.get("experiments", [])}
        | _trusted_set2
        | _superseded_set
    )
    _orphan_map: dict[str, list] = {}
    for _pid, _pdata in _iter_papers(_auth):
        _orphans = [
            _eid for _eid in (_pdata.get("experiment_ids") or [])
            if _eid and _eid not in _all_known
        ]
        if _orphans:
            _orphan_map[_pid] = _orphans
    check("Z", "no_orphaned_experiment_ids_in_papers",
          len(_orphan_map) == 0,
          f"orphans (not in queue/archive/trusted/superseded): {_orphan_map}")
except Exception as _e:
    check("Z", "no_orphaned_experiment_ids_in_papers", False, str(_e)[:120])

# Z9: Dashboard null field safety — no None/null in fields the endpoint slices
# Root cause of the /api/experiments TypeError: e.get("started_at", "")[:16]
# returns None (not "") when the key exists but the value is JSON null.
try:
    _null_fields = []
    for _r in _q.get("experiments", []):
        for _field in ("started_at", "completed_at", "id", "dataset", "method"):
            if _field in _r and _r[_field] is None:
                _null_fields.append(f"{_r.get('id', '?')}.{_field}=null")
    check("Z", "no_null_in_dashboard_sliced_fields",
          len(_null_fields) == 0,
          f"null fields (will crash /api/experiments): {_null_fields[:10]}")
except Exception as _e:
    check("Z", "no_null_in_dashboard_sliced_fields", False, str(_e)[:80])

# Z10: Seed count adequacy — no paper-linked complete experiment with < 5 seeds
# Only checks experiments actually linked in author papers (Gate B only blocks those).
# Excludes integration-test-* and other non-paper experiments.
try:
    from tar_lab.validation import _superseded_experiment_ids as _superseded_fn2
    _superseded_set2 = _superseded_fn2(WORKSPACE)
    _MIN_SEEDS = 5
    # Gather experiment_ids from human_approved papers only
    # (Gate B only blocks papers that are actually approved for writing)
    _paper_linked_ids = set()
    for _pid2, _pdata2 in _iter_papers(_auth):
        if not bool(_pdata2.get("human_approved", False)):
            continue
        for _eid2 in (_pdata2.get("experiment_ids") or []):
            if _eid2:
                _paper_linked_ids.add(_eid2)
    # Build lookup for archive entries
    _arc_by_id = {str(_r.get("id", "")): _r for _r in _arc.get("experiments", [])}
    _underpowered = []
    for _lid in _paper_linked_ids:
        _r = _arc_by_id.get(_lid)
        if _r is None:
            continue  # not yet archived (still pending/running)
        _status = str(_r.get("status", "")).lower()
        _seeds  = int(_r.get("seed_count") or len(_r.get("seeds") or []) or 0)
        if (
            _status == "complete"
            and _lid not in _superseded_set2
            and 0 < _seeds < _MIN_SEEDS
        ):
            _underpowered.append(f"{_lid}(seeds={_seeds})")
    check("Z", "no_underpowered_complete_archived_results",
          len(_underpowered) == 0,
          f"paper-linked results will block Gate B (seed_count<{_MIN_SEEDS}): {_underpowered}")
except Exception as _e:
    check("Z", "no_underpowered_complete_archived_results", False, str(_e)[:120])

# Z11: No missing result files for complete archived experiments
try:
    _missing_results = []
    for _r in _arc.get("experiments", []):
        if str(_r.get("status", "")).lower() == "complete":
            _rpath = str(_r.get("result_path") or "").strip()
            if _rpath and not Path(_rpath).exists():
                _missing_results.append(f"{_r.get('id', '?')}→{_rpath}")
    check("Z", "no_missing_result_files",
          len(_missing_results) == 0,
          f"missing: {_missing_results[:5]}")
except Exception as _e:
    check("Z", "no_missing_result_files", False, str(_e)[:80])

# Z12: Circular dependency detection in queue depends_on chains
try:
    _dep_map: dict[str, list[str]] = {}
    for _r in _q.get("experiments", []):
        _eid  = str(_r.get("id", ""))
        _deps = [str(_d) for _d in (_r.get("depends_on") or []) if _d]
        if _deps:
            _dep_map[_eid] = _deps

    def _has_cycle(graph: dict, start: str, visited: set, stack: set) -> bool:
        visited.add(start); stack.add(start)
        for nbr in graph.get(start, []):
            if nbr not in visited:
                if _has_cycle(graph, nbr, visited, stack):
                    return True
            elif nbr in stack:
                return True
        stack.discard(start)
        return False

    _cycle_nodes = []
    _visited_dfs: set = set()
    for _node in list(_dep_map.keys()):
        if _node not in _visited_dfs:
            if _has_cycle(_dep_map, _node, _visited_dfs, set()):
                _cycle_nodes.append(_node)
    check("Z", "no_circular_depends_on_chains",
          len(_cycle_nodes) == 0,
          f"cycle involving: {_cycle_nodes}")
except Exception as _e:
    check("Z", "no_circular_depends_on_chains", False, str(_e)[:80])

# Z13: process_registry ↔ daemon consistency
# Daemon's running_ids should all appear in process_registry with status=running.
try:
    _daemon_running2 = set(_daemon.get("running_ids") or [])
    # process_registry uses "stage" field (not "status")
    _reg_running = {
        str(_v.get("experiment_id", ""))
        for _v in (_proc.values() if isinstance(_proc, dict) else [])
        if str(_v.get("stage", _v.get("status", ""))).lower() == "running"
           and str(_v.get("experiment_id", "")).strip()
    }
    _unregistered = _daemon_running2 - _reg_running
    check("Z", "daemon_running_registered_in_process_registry",
          len(_unregistered) == 0,
          f"running per daemon but absent from process_registry: {sorted(_unregistered)}")
except Exception as _e:
    check("Z", "daemon_running_registered_in_process_registry", False, str(_e)[:80])


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print("TAR HEALTH CHECK SUMMARY", flush=True)
print("="*60, flush=True)

_by_section: dict[str, list] = {}
for _sec, _st, _nm, _dt in _results:
    _by_section.setdefault(_sec, []).append((_st, _nm, _dt))

_total_pass = _total_fail = _total_skip = 0
for _sec in sorted(_by_section):
    _sec_results = _by_section[_sec]
    _sp = sum(1 for _s, _, _ in _sec_results if _s == PASS)
    _sf = sum(1 for _s, _, _ in _sec_results if _s == FAIL)
    _ss = sum(1 for _s, _, _ in _sec_results if _s == SKIP)
    _total_pass += _sp; _total_fail += _sf; _total_skip += _ss
    _char = "✓" if _sf == 0 else "✗"
    print(f"  {_char} Section {_sec}: {_sp} pass  {_sf} fail  {_ss} skip")
    for _st, _nm, _dt in _sec_results:
        if _st == FAIL:
            print(f"      {FAIL} {_sec}:{_nm}" + (f" — {_dt}" if _dt else ""))

print(f"\n  Sections:   Y–Z ({len(_by_section)} sections)")
print(f"  Checks:     {_total_pass} passed, {_total_fail} failed, {_total_skip} skipped"
      f"  out of {len(_results)}")

_overall_pass = _total_fail == 0

if _overall_pass:
    print("\n  TAR health check PASSED.")
    print("  All cross-loop consistency checks clean. TAR may continue running.")
else:
    print(f"\n  {_total_fail} failure(s). Review output above before taking action.")
    print("  Do NOT stop TAR without diagnosing the failure first.")

sys.exit(0 if _overall_pass else 1)
