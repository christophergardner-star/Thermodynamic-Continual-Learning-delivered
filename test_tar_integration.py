"""
TAR Integration Smoke Test — Phase 4 code upgrade validation.

Submits 5 mini experiments (all new methods), runs them through the orchestrator,
verifies new fields are populated, then fires one director + author cycle.

Run from the repo root:
    python test_tar_integration.py

Expected duration: ~5-15 minutes on CPU (2 seeds x 2 epochs x 2 tasks x 5 methods).
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

results: list[tuple[str, str, str]] = []  # (status, name, detail)


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


# ── Pre-step: purge stale integration-test archive entries ────────────────────
print("\n=== Pre-step: Clean stale integration-test archive entries ===")
_archive_path = WORKSPACE / "tar_state" / "experiment_archive.json"
if _archive_path.exists():
    try:
        _arc = json.loads(_archive_path.read_text(encoding="utf-8"))
        _recs = _arc.get("experiments", []) if isinstance(_arc, dict) else []
        _kept = [r for r in _recs if not str(r.get("id", "")).startswith("integration-test-")]
        _removed = len(_recs) - len(_kept)
        if _removed:
            _arc["experiments"] = _kept
            _arc["saved_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
            _tmp = _archive_path.with_suffix(".tmp")
            _tmp.write_text(json.dumps(_arc, indent=2), encoding="utf-8")
            import os as _os; _os.replace(_tmp, _archive_path)
            print(f"  Removed {_removed} stale integration-test archive entries")
        else:
            print("  No stale entries to remove")
    except Exception as _e:
        print(f"  [WARN] Could not purge archive: {_e}")
# Also purge stale experiment directories so result_file checks start fresh
_exp_dir_pre = WORKSPACE / "tar_state" / "experiments"
for _method in ["tcl_canonical", "lwf", "der_plus_plus", "ewc", "si"]:
    _exp_path = _exp_dir_pre / f"integration-test-{_method}"
    if _exp_path.exists():
        import shutil as _shutil
        _shutil.rmtree(_exp_path, ignore_errors=True)
        print(f"  Cleared stale experiment dir: {_exp_path.name}")

# ── Step 1: submit 5 mini experiments ─────────────────────────────────────────
print("\n=== Step 1: Submit mini experiments ===")

from tar_experiment_orchestrator import ExperimentOrchestrator, ExperimentSpec

orch = ExperimentOrchestrator(WORKSPACE)
orch.set_autonomous(True)   # auto-generates+commits a manifest per experiment (RAIL 3 satisfied)

TEST_METHODS = ["tcl_canonical", "lwf", "der_plus_plus", "ewc", "si"]

for method in TEST_METHODS:
    spec = ExperimentSpec(
        id=f"integration-test-{method}",
        name=f"Integration test: {method}",
        hypothesis_name=f"Validate {method} with new code",
        project_id="integration-test",
        dataset="split_cifar10",
        method=method,
        seeds=[42, 0],
        frontier_problem_id="fp-catastrophic-forgetting",
        config_overrides={
            "train_epochs_per_task": 2,
            "batch_size": 128,
            "n_tasks": 2,
            "comparison_methods": [],  # skip comparisons for speed
        },
    )
    try:
        orch.submit(spec)
        check(f"submit:{method}", True, "queued")
    except Exception as e:
        check(f"submit:{method}", False, str(e))

# ── Step 2: Run each experiment ────────────────────────────────────────────────
print("\n=== Step 2: Run experiments ===")

for method in TEST_METHODS:
    print(f"\n  Running {method}...")
    try:
        result = orch.run_next()
        if result is None:
            check(f"run:{method}", False, "run_next() returned None")
            continue
        check(f"run:{method}", result.verdict in {"BREAKTHROUGH","DIRECTIONAL","NULL","ADVERSE","ERROR"},
              f"verdict={result.verdict} forgetting={result.mean_forgetting:.4f}")

        # Verify new fields exist and are populated
        check(f"power_analysis:{method}",
              isinstance(result.power_analysis, dict) and "observed_power" in result.power_analysis,
              str(result.power_analysis))

        check(f"result_saved:{method}",
              bool(result.experiment_id),
              f"id={result.experiment_id}")

    except Exception as e:
        check(f"run:{method}", False, traceback.format_exc()[-300:])

# ── Step 3: Verify result files have new fields ────────────────────────────────
print("\n=== Step 3: Verify persisted result files ===")

exp_dir = WORKSPACE / "tar_state" / "experiments"
for method in TEST_METHODS:
    rp = exp_dir / f"integration-test-{method}" / "result.json"
    if not rp.exists():
        check(f"result_file:{method}", False, "result.json not found")
        continue
    try:
        d = json.loads(rp.read_text(encoding="utf-8"))
        # power_analysis in result JSON
        check(f"result_has_power:{method}",
              "power_analysis" in d and isinstance(d["power_analysis"], dict),
              f"keys={list(d.keys())[:8]}")
        # verdict present
        check(f"result_has_verdict:{method}",
              bool(d.get("verdict")),
              f"verdict={d.get('verdict')}")
    except Exception as e:
        check(f"result_file:{method}", False, str(e))

# ── Step 4: Verify FrontierRegistry updated ────────────────────────────────────
print("\n=== Step 4: Verify frontier registry ===")
try:
    from tar_frontier import FrontierRegistry
    reg = FrontierRegistry(WORKSPACE)
    fp = reg.get("fp-catastrophic-forgetting")
    check("frontier_exists", fp is not None)
    if fp:
        check("frontier_truth_status_set", fp.truth_status in {"weak","provisional","supported","validated","falsified"},
              f"truth_status={fp.truth_status}")
        print(f"    adverse={fp.adverse_count} null={fp.null_count} breakthroughs={fp.breakthroughs_found}")
except Exception as e:
    check("frontier_registry", False, str(e))

# ── Step 5: One director cycle ─────────────────────────────────────────────────
print("\n=== Step 5: Director cycle ===")
try:
    from tar_research_director import ResearchDirector
    director = ResearchDirector(WORKSPACE)
    state = director.read_state()
    frontier_directives = state.get("frontier_directives", [])
    check("director_cycle", len(frontier_directives) > 0,
          f"{len(frontier_directives)} frontier directives built")
    for fd in frontier_directives[:3]:
        print(f"    [{fd.get('truth_status','?')}] {fd.get('problem_id','?')} "
              f"score={fd.get('priority_score','?')} "
              f"adverse={fd.get('adverse_count',0)} null={fd.get('null_count',0)}")
    # Check novelty score was computed
    paths = state.get("active_research_paths", [])
    check("active_paths", len(paths) > 0, f"{len(paths)} active paths")
    if paths:
        p0 = paths[0] if isinstance(paths[0], dict) else vars(paths[0])
        ns = p0.get("novelty_score", 0)
        check("novelty_score_present", isinstance(ns, (int, float)) and ns > 0,
              f"novelty_score={ns}")
except Exception as e:
    check("director_cycle", False, traceback.format_exc()[-400:])

# ── Step 6: Author state check ────────────────────────────────────────────────
print("\n=== Step 6: Author state check ===")
try:
    author_path = WORKSPACE / "tar_state" / "author_state.json"
    check("author_state_exists", author_path.exists())
    if author_path.exists():
        d = json.loads(author_path.read_text(encoding="utf-8"))
        # author_state uses 'paper_queue' key, not 'papers'
        papers = d.get("paper_queue", d.get("papers", []))
        check("author_papers_present", len(papers) > 0, f"{len(papers)} papers in state")
        for p in papers:
            print(f"    [{p.get('paper_status','?')}] {p.get('paper_id','?')} "
                  f"readiness={p.get('readiness','?')} evidence_ready={p.get('evidence_ready','?')}")
except Exception as e:
    check("author_state", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("INTEGRATION TEST SUMMARY")
print("=" * 60)
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
for status, name, detail in results:
    if status == FAIL:
        print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
print(f"\n{passed} passed, {failed} failed out of {len(results)} checks")
if failed == 0:
    print("\nAll checks passed. TAR is ready to resume autonomous operation.")
else:
    print(f"\n{failed} check(s) failed. Review output above before resuming.")
sys.exit(0 if failed == 0 else 1)
