"""Keystone #2 (K2.3a wiring) — gap -> registered frontier in the director.

ResearchDirector._register_frontiers_from_gaps() promotes an opted-in domain's top
research gap to a registered FrontierProblem (via frontier_problem_from_gap), the
missing bridge that lets a gap-derived problem reach the experiment catalog. It must
be INERT by default (empty _FRONTIER_AUTONOMY_DOMAINS) and only fire for domains a
human has explicitly opted in.
"""
from pathlib import Path

import tar_research_director as trd
from literature.knowledge_graph import LiteratureKnowledgeGraph
from literature.schemas import Benchmark, ResearchGap

_BID = "benchmark:402cf4341499b795"


def _setup_db(ws: Path) -> Path:
    db = ws / "tar_state" / "literature" / "literature_graph.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    g = LiteratureKnowledgeGraph(str(db))
    g.upsert_benchmark(Benchmark(
        benchmark_id=_BID, name="Continual Learning On Split Cifar 10",
        task="continual_learning", domain="continual_learning",
        metrics=["forgetting"], metrics_higher_better={"forgetting": False}))
    gap = ResearchGap(
        gap_id=f"tar_internal_gap::{_BID}::tcl_family", gap_type="negative_result",
        title="Alternative mechanism needed: TCL family on Split-CIFAR-10",
        description="phase18 n=5: canonical TCL worse than SI; pursue alternatives.",
        domain="continual_learning", benchmark_id=_BID,
        method_names=["tcl", "tcl_canonical", "tcl_full"],
        impact_score=0.6, tractability_score=0.5, novelty_score=0.5, status="open")
    gap.recompute_composite()
    g.upsert_gap(gap)
    g.close()
    return db


def test_register_frontiers_inert_by_default(tmp_path):
    _setup_db(tmp_path)
    d = trd.ResearchDirector(tmp_path)
    # default _FRONTIER_AUTONOMY_DOMAINS is empty -> the bridge does nothing
    assert d._register_frontiers_from_gaps() == {}


def test_register_frontiers_when_domain_opted_in(tmp_path, monkeypatch):
    _setup_db(tmp_path)
    monkeypatch.setattr(trd, "_FRONTIER_AUTONOMY_DOMAINS", frozenset({"continual_learning"}))
    d = trd.ResearchDirector(tmp_path)

    out = d._register_frontiers_from_gaps()
    fid = out.get("continual_learning", "")
    assert fid.startswith("fp-gap-"), "opted-in domain's top gap must register a frontier"

    # persisted to frontier_problems.json + satisfies the register guard fields
    from tar_frontier import FrontierRegistry
    reg = FrontierRegistry(tmp_path)
    assert fid in reg._problems
    fp = reg._problems[fid]
    assert fp.well_known_problem and fp.candidate_datasets and fp.external_baselines
    assert fp.domain == "continual_learning"
    # external baselines are real external methods, not the internal TCL family
    assert not (set(fp.external_baselines) & {"tcl", "tcl_canonical", "tcl_full"})


def test_end_to_end_gap_to_queued_experiment(tmp_path, monkeypatch):
    """Acceptance: opted-in domain + a gap -> registered frontier -> queued experiment,
    end to end through update_state (the previously-missing autonomous bridge)."""
    _setup_db(tmp_path)
    monkeypatch.setattr(trd, "_FRONTIER_AUTONOMY_DOMAINS", frozenset({"continual_learning"}))
    d = trd.ResearchDirector(tmp_path)

    state = d.update_state()
    ed = state.get("experiment_directives", [])
    gap_dirs = [e for e in ed if str(e.get("frontier_problem_id", "")).startswith("fp-gap-")]
    assert gap_dirs, "opted-in domain's gap must yield a queued experiment directive"
    e = gap_dirs[0]
    assert e.get("scheduler_intent") in {"propose_now", "queue_now", "hold_dependency"}
    assert str(e.get("dataset", "")).strip() and str(e.get("method", "")).strip()


def test_catalog_generates_experiment_for_gap_frontier(tmp_path):
    """_frontier_experiment_catalog now builds a default probe for fp-gap-* frontiers
    from their own grounding fields (the live path; the hardcoded id/domain blocks miss
    gap-derived frontiers, which is why arming produced no experiment on live state)."""
    d = trd.ResearchDirector(tmp_path)
    frontier = {
        "problem_id": "fp-gap-test-tcl",
        "title": "Alternative mechanism needed: TCL on Split-CIFAR-10",
        "domain": "continual_learning",
        "candidate_datasets": ["split_cifar10"],
        "candidate_backbones": ["resnet18"],
        "external_baselines": ["ewc", "si", "sgd_baseline"],
        "global_problem_statement": "Beat established baselines on Split-CIFAR-10.",
    }
    out = d._frontier_experiment_catalog(frontier, {}, None)
    assert len(out) == 1
    e = out[0]
    assert e["experiment_id"].startswith("director-") and "fp-gap-" in e["experiment_id"]
    assert e["dataset"] == "split_cifar10" and e["method"] == "tcl"
    assert "ewc" in e["comparison_methods"]
    assert e["frontier_problem_id"] == "fp-gap-test-tcl"


def test_catalog_unmatched_non_gap_frontier_still_empty(tmp_path):
    """Safety: the fallback is gated to fp-gap- ids, so an unmatched NON-gap frontier
    still returns [] (existing behavior unchanged)."""
    d = trd.ResearchDirector(tmp_path)
    frontier = {
        "problem_id": "fp-unknown-xyz", "title": "X", "domain": "some_unhandled_domain",
        "candidate_datasets": ["split_cifar10"], "external_baselines": ["ewc"],
    }
    assert d._frontier_experiment_catalog(frontier, {}, None) == []


def test_register_frontiers_idempotent(tmp_path, monkeypatch):
    _setup_db(tmp_path)
    monkeypatch.setattr(trd, "_FRONTIER_AUTONOMY_DOMAINS", frozenset({"continual_learning"}))
    d = trd.ResearchDirector(tmp_path)
    first = d._register_frontiers_from_gaps()
    second = d._register_frontiers_from_gaps()
    assert first == second and first.get("continual_learning")
    from tar_frontier import FrontierRegistry
    reg = FrontierRegistry(tmp_path)
    # exactly one gap-derived frontier (no duplicates across cycles)
    gap_frontiers = [pid for pid in reg._problems if pid.startswith("fp-gap-")]
    assert len(gap_frontiers) == 1
