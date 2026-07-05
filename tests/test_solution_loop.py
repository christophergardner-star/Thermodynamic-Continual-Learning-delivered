"""
Regression tests for the solution-finding loop (docs/solution_loop_implementation_plan.md).
Grouped by plan phase. Phase 0 = integrity pre-work that makes the loop sound.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from tar_lab.method_identity import (
    internal_source_tag,
    EXCLUDED_FROM_NOVELTY_BAR,
    _ESTABLISHED_BASELINES,
)
from tar_lab.method_registry import METHOD_REGISTRY


# ── Phase 0.1 — tcl registry-key collision ────────────────────────────────────

def test_registry_canonical_key_no_proxy_collision():
    # The canonical algorithm is registered under a key method_identity treats as
    # canonical, and the bare proxy key "tcl" is NOT a generic-registry method.
    assert "tcl_canonical" in METHOD_REGISTRY
    assert "tcl" not in METHOD_REGISTRY


# ── Phase 0.2 — three source classes + no circular novelty bar ────────────────

@pytest.mark.parametrize("method,expected", [
    ("tcl", "tar_internal"),
    ("tcl_full", "tar_internal"),
    ("tcl_penalty_only", "tar_internal"),
    ("ewc", "tar_internal_baseline"),
    ("si", "tar_internal_baseline"),
    ("si_generic", "tar_internal_baseline"),
    ("sgd_baseline", "tar_internal_baseline"),
    ("der_plus_plus", "tar_internal_baseline"),
    ("si_clamp_decay", "tar_novel"),      # novel: built on SI but NOT the SI baseline
    ("hybrid_ema_pathint", "tar_novel"),
    ("totally_unknown", "tar_novel"),     # fail-safe default = excluded
])
def test_source_tag_three_classes(method, expected):
    assert internal_source_tag(method) == expected


def test_novel_and_flagship_excluded_established_is_the_bar():
    # The core invariant: a method TAR invented (flagship OR novel) can never be
    # the comparison bar; only reproduced established baselines are.
    assert internal_source_tag("si_clamp_decay") in EXCLUDED_FROM_NOVELTY_BAR
    assert internal_source_tag("tcl") in EXCLUDED_FROM_NOVELTY_BAR
    assert internal_source_tag("ewc") not in EXCLUDED_FROM_NOVELTY_BAR
    assert "si" in _ESTABLISHED_BASELINES


def test_best_result_excludes_internal_and_novel():
    from literature.knowledge_graph import LiteratureKnowledgeGraph
    d = tempfile.mkdtemp()
    g = LiteratureKnowledgeGraph(os.path.join(d, "t.db"))
    g.conn.execute("PRAGMA foreign_keys=OFF")
    for name, src, val in [
        ("si", "tar_internal_baseline", 0.047),
        ("tcl", "tar_internal", 0.13),
        ("cand_x", "tar_novel", 0.01),   # would wrongly win if not excluded
    ]:
        g.conn.execute(
            "INSERT INTO sota_entries (entry_id,benchmark_id,method_name,metric_name,"
            "metric_value,higher_is_better,source,fetched_at) VALUES (?,?,?,?,?,?,?,?)",
            (f"e::{name}", "b1", name, "forgetting", val, 0, src, "2026-01-01T00:00:00Z"),
        )
    g.conn.commit()
    best = g.best_result("b1", "forgetting", higher_is_better=False,
                         exclude_source=EXCLUDED_FROM_NOVELTY_BAR)
    assert best is not None and best.method_name == "si"
    # single-string exclude still works (backwards compat)
    best2 = g.best_result("b1", "forgetting", higher_is_better=False, exclude_source="tar_internal")
    assert best2 is not None and best2.method_name == "cand_x"  # novel not excluded here
    g.close()


# ── Phase 0.3 — preregistered criteria are enforced (were write-only) ──────────

def _orch(tmp_path):
    from tar_experiment_orchestrator import ExperimentOrchestrator
    (tmp_path / "tar_state" / "autonomous_research").mkdir(parents=True, exist_ok=True)
    return ExperimentOrchestrator(tmp_path)


def _write_prereg(tmp_path, exp_id, criteria):
    import json
    p = tmp_path / "tar_state" / "autonomous_research" / "preregistration.json"
    p.write_text(json.dumps({"hypotheses": [
        {"experiment_id": exp_id, "name": exp_id, "criteria": criteria}
    ]}), encoding="utf-8")


def test_prereg_criteria_none_when_unregistered(tmp_path):
    from types import SimpleNamespace
    o = _orch(tmp_path)
    spec = SimpleNamespace(id="exp-x", name="exp-x")
    report, met = o._evaluate_prereg_criteria(
        spec, mean_delta=-0.1, p_val=0.001, cohens_d=2.0,
        std_forgetting=0.005, mean_accuracy=0.8, accuracy_list=[0.8, 0.81])
    assert report == {} and met is None


def test_prereg_joint_criteria_met_and_collapse(tmp_path):
    from types import SimpleNamespace
    o = _orch(tmp_path)
    crit = {"max_delta": -0.01, "max_p": 0.05, "min_d": 0.5,
            "max_forgetting_std": 0.0075, "min_mean_acc": 0.79, "min_seed_acc": 0.55}
    _write_prereg(tmp_path, "exp-good", crit)
    spec = SimpleNamespace(id="exp-good", name="exp-good")
    # A candidate that genuinely beats the joint bar
    report, met = o._evaluate_prereg_criteria(
        spec, mean_delta=-0.05, p_val=0.001, cohens_d=1.2,
        std_forgetting=0.004, mean_accuracy=0.80, accuracy_list=[0.79, 0.80, 0.81])
    assert met is True and not report.get("collapse_detected")

    # A "stability by not learning" cheat: tight variance but a seed at chance
    report2, met2 = o._evaluate_prereg_criteria(
        spec, mean_delta=-0.05, p_val=0.001, cohens_d=1.2,
        std_forgetting=0.001, mean_accuracy=0.65, accuracy_list=[0.80, 0.50, 0.80])
    assert report2.get("collapse_detected") is True
    assert report2["min_seed_acc"]["passed"] is False


# ── Phase 0.4 — silent-SGD guard ──────────────────────────────────────────────

def test_harness_a_rejects_unknown_method():
    from tar_lab.multimodal_payloads import run_split_cifar10_benchmark
    with pytest.raises(ValueError):
        run_split_cifar10_benchmark(None, "some_novel_method")  # guard fires before config use


# ── Phase 1 — method catalog loader + guards ──────────────────────────────────

def test_method_catalog_loads_and_is_all_cited():
    from literature.method_catalog import load_method_catalog, MECHANISM_CLASSES, _has_citation
    cat = load_method_catalog()
    assert len(cat) >= 15, "seed catalog should carry a substantial method set"
    for m in cat:
        assert m["mechanism_class"] in MECHANISM_CLASSES
        assert _has_citation(m["citation"]), f"{m['method_key']} must be cited"
    keys = {m["method_key"] for m in cat}
    assert {"ewc", "si", "der_plus_plus", "l2p"} <= keys


def test_method_catalog_refuses_uncited(tmp_path):
    import json
    from literature.method_catalog import load_method_catalog
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({"methods": [
        {"method_key": "good", "full_name": "Good", "mechanism_class": "replay",
         "citation": {"paper_title": "X", "arxiv_id": "1234.5678"}},
        {"method_key": "uncited", "full_name": "Bad", "mechanism_class": "replay",
         "citation": {"paper_title": "No id"}},          # refused: no arxiv/doi/url
        {"method_key": "badclass", "full_name": "Y", "mechanism_class": "magic",
         "citation": {"paper_title": "Z", "doi": "10.x/y"}},  # refused: bad class
    ]}), encoding="utf-8")
    cat = load_method_catalog(p)
    assert {m["method_key"] for m in cat} == {"good"}


def test_method_catalog_unverified_until_signed_off():
    from literature.method_catalog import catalog_is_verified
    # Seeded catalog ships _verified:false — citations are model-supplied, human must confirm.
    assert catalog_is_verified() is False


def test_render_catalog_block_nonempty():
    from literature.method_catalog import render_catalog_block
    block = render_catalog_block()
    assert "ewc" in block and "[regularization]" in block and len(block.splitlines()) >= 15


# ── Phase 2 — SI anomaly seeding (dry-run + schema + criteria consistency) ─────

def test_si_anomaly_gap_constructs_and_is_top_composite():
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("seed_si_anomaly", repo / "scripts" / "seed_si_anomaly.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    from literature.schemas import ResearchGap
    gap = ResearchGap(gap_id=mod._GAP_ID, gap_type="theoretical", title="t",
                      description=mod._ANOMALY_STATEMENT, domain="continual_learning",
                      method_names=["si"], impact_score=0.95, novelty_score=0.90,
                      tractability_score=0.85)
    gap.recompute_composite()
    assert gap.composite_score > 0.9  # must outrank the existing negative_result gaps


def test_si_joint_criteria_match_evaluator_keys():
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("seed_si_anomaly", repo / "scripts" / "seed_si_anomaly.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    # Every criterion the seeder writes must be one the orchestrator evaluator enforces.
    enforced = {"max_delta", "max_p", "min_d", "max_forgetting_std", "min_mean_acc", "min_seed_acc"}
    assert set(mod._JOINT_CRITERIA) <= enforced
    assert mod._JOINT_CRITERIA["min_seed_acc"] == 0.55  # collapse guard present


# ── Phase 3 — widened proposer (catalog injected, method carried, not TCL-pinned) ──

def test_proposer_widened_injects_catalog_and_carries_method(monkeypatch):
    import tar_lab.llm_bridge as lb

    captured = {}

    def fake_call(prompt, **kw):
        captured["prompt"] = prompt
        # A composed candidate from the design space, not TCL:
        return json.dumps([{
            "experiment_id": "si-clamp-decay-probe",
            "title": "SI importance + hard clamp + decay",
            "dataset": "split_cifar10", "backbone": "resnet18",
            "method": "si_clamp_decay", "mechanism_class": "regularization",
            "lineage": ["si", "ewc"],
            "estimated_runtime_h": 4.0, "config_overrides": {},
            "hypothesis": "clamped SI importance keeps stability without collapse",
            "why": "targets the SI stability/collapse cliff",
        }])

    monkeypatch.setattr(lb, "call_claude", fake_call)
    # avoid cache interference
    monkeypatch.setattr(lb, "_cache_read", lambda *a, **k: None)
    monkeypatch.setattr(lb, "_cache_write", lambda *a, **k: None)

    from pathlib import Path
    out = lb.propose_followup_experiments(
        Path(tempfile.mkdtemp()), frontier_id="fp-x", frontier_title="SI anomaly",
        global_problem_statement="stability without collapse",
        candidate_datasets=["split_cifar10"], candidate_backbones=["resnet18"],
        external_baselines=["si", "ewc"], completed_summaries=["e1: split_cifar10"],
        exclude_ids=set(), max_proposals=2,
        method_catalog_block="- si [regularization] path-integral importance\n- ewc [regularization] fisher penalty",
    )
    # catalog + widened mandate present in the prompt
    assert "recombination material" in captured["prompt"]
    assert "WHOLE continual-learning design space" in captured["prompt"]
    # the proposed composed method is carried through (NOT forced to tcl)
    assert len(out) == 1
    assert out[0]["method"] == "si_clamp_decay"
    assert out[0]["mechanism_class"] == "regularization"
    assert out[0]["lineage"] == ["si", "ewc"]


def test_proposer_legacy_mode_without_catalog(monkeypatch):
    import tar_lab.llm_bridge as lb
    cap = {}

    def fake_call(prompt, **kw):
        cap["prompt"] = prompt
        return json.dumps([{"experiment_id": "e1", "title": "t", "dataset": "split_cifar10",
                            "backbone": "resnet18", "estimated_runtime_h": 4.0,
                            "config_overrides": {}, "hypothesis": "h", "why": "w"}])

    monkeypatch.setattr(lb, "call_claude", fake_call)
    monkeypatch.setattr(lb, "_cache_read", lambda *a, **k: None)
    monkeypatch.setattr(lb, "_cache_write", lambda *a, **k: None)
    from pathlib import Path
    out = lb.propose_followup_experiments(
        Path(tempfile.mkdtemp()), frontier_id="fp-x", frontier_title="t",
        global_problem_statement="p", candidate_datasets=["split_cifar10"],
        candidate_backbones=["resnet18"], external_baselines=["ewc"],
        completed_summaries=[], exclude_ids=set(), max_proposals=1,
    )  # no catalog block -> legacy mode, no method key required
    assert "WHOLE continual-learning design space" not in cap["prompt"]
    assert out and "method" not in out[0]


# ── Phase 4 — kill-ledger + deterministic pruning ─────────────────────────────

def test_fingerprint_stable_and_config_sensitive():
    from tar_lab.solution_loop import candidate_fingerprint as fp
    # order-insensitive, value-sensitive
    assert fp("si_clamp", {"a": 1, "b": 2.0}) == fp("si_clamp", {"b": 2.0, "a": 1})
    assert fp("si_clamp", {"a": 1}) != fp("si_clamp", {"a": 2})
    assert fp("si_clamp", {"a": 1}) != fp("other", {"a": 1})


def test_kill_ledger_records_and_prunes(tmp_path):
    from tar_lab.solution_loop import record_kill, is_killed, load_killed_fingerprints, render_kill_ledger_block
    assert not is_killed(tmp_path, "si_clamp_decay", {"lam": 0.5}, "regularization")
    record_kill(tmp_path, experiment_id="e1", method="si_clamp_decay",
                config_overrides={"lam": 0.5}, mechanism_class="regularization",
                verdict="COLLAPSED", kill_reason="worst seed acc 0.50 < 0.55",
                criteria_failed=["min_seed_acc"])
    # exact region is now pruned; a different HP setting is NOT
    assert is_killed(tmp_path, "si_clamp_decay", {"lam": 0.5}, "regularization")
    assert not is_killed(tmp_path, "si_clamp_decay", {"lam": 0.9}, "regularization")
    assert len(load_killed_fingerprints(tmp_path)) == 1
    block = render_kill_ledger_block(tmp_path)
    assert "si_clamp_decay" in block and "COLLAPSED" in block


# ── Phase 0.3/4 scoping — legacy director experiments must NOT be loop-downgraded/killed ──

def _build_null_result(tmp_path, exp_id, criteria):
    """Run _build_result for a NULL-producing result under the given prereg criteria.
    Returns (verdict, kill_ledger_exists)."""
    from tar_experiment_orchestrator import ExperimentOrchestrator, ExperimentSpec
    (tmp_path / "tar_state" / "autonomous_research").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tar_state" / "autonomous_research" / "preregistration.json").write_text(
        json.dumps({"hypotheses": [{"experiment_id": exp_id, "name": exp_id, "criteria": criteria}]}),
        encoding="utf-8")
    o = ExperimentOrchestrator(tmp_path)
    spec = ExperimentSpec(name=exp_id, project_id="p", hypothesis_name="h",
                          dataset="split_cifar10", method="si_cand", seeds=[0, 1, 2],
                          config_overrides={"lam": 0.5})
    object.__setattr__(spec, "id", exp_id) if not hasattr(spec, "id") else None
    # Forgetting ~ equal to the fallback baseline -> mean_delta ~ 0 -> NULL verdict.
    base = o._load_baseline()[:3]
    res = o._build_result(spec, [{"seed": s} for s in range(3)], list(base), [0.8, 0.8, 0.8])
    led = tmp_path / "tar_state" / "solution_loop" / "kill_ledger.jsonl"
    return res.verdict, led.exists()


def test_legacy_criteria_not_kill_recorded(tmp_path):
    # Legacy director prereg (no joint keys): NULL result must NOT create a loop kill.
    verdict, killed = _build_null_result(tmp_path, "legacy-exp",
                                         {"max_p": 0.05, "min_d": 0.5, "max_delta": -0.01})
    assert verdict == "NULL"
    assert killed is False, "legacy (non-loop) experiment must not be recorded as a loop kill"


def test_forbidden_synthesis_names_block_baseline_collision():
    # Bug-hunt CONFIRMED finding: a synthesized method carrying an established-
    # baseline name would be tagged tar_internal_baseline -> bar-eligible.
    from tar_lab.method_identity import forbidden_synthesis_name
    for name in ("gem", "er", "mas", "icarl", "gdumb", "replay", "EWC", " si "):
        assert forbidden_synthesis_name(name) is True
    for name in ("si_clamp_decay", "hybrid_ema_pathint", "cand_gem_v2", ""):
        assert forbidden_synthesis_name(name) is False


def test_generic_callers_use_canonical_registry_key():
    # Bug-hunt CONFIRMED finding: the tcl->tcl_canonical rename must be
    # propagated to every offline generic-registry caller.
    import re
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    for fname in ("phase16_cifar100_rerun.py", "phase17_tinyimagenet_rerun.py",
                  "run_hyperparameter_selection.py", "analyze_dpr_forgetting_correlation.py"):
        text = (repo / fname).read_text(encoding="utf-8")
        assert "tcl_canonical" in text, f"{fname} must use the renamed registry key"
        # Strip the safe token so the checks below only see BARE 'tcl'.
        scrubbed = text.replace("'tcl_canonical'", "").replace('"tcl_canonical"', "")
        # (a) no bare "tcl": / "tcl"= config-key definition remains.
        assert not re.search(r"['\"]tcl['\"]\s*[:=]", scrubbed), \
            f"{fname} still defines the bare 'tcl' registry key"
        # (b) no bare ["tcl"] SUBSCRIPT lookup remains. This is the class of bug
        #     that wasted a multi-day run: the config key was renamed but a
        #     downstream method_results["tcl"] lookup was missed -> KeyError after
        #     training, before the artifact is written. Labels like "tcl_vs_ewc"
        #     are fine (not a bare-'tcl' subscript).
        assert not re.search(r"\[\s*['\"]tcl['\"]\s*\]", scrubbed), \
            f"{fname} still subscripts the bare 'tcl' key (renamed-key/lookup drift)"


def test_loop_candidate_is_kill_recorded(tmp_path):
    # Joint-criterion prereg (loop candidate): NULL result IS recorded to the kill-ledger.
    verdict, killed = _build_null_result(tmp_path, "loop-exp",
                                         {"max_delta": -0.01, "max_forgetting_std": 0.0075,
                                          "min_mean_acc": 0.79, "min_seed_acc": 0.55})
    assert verdict == "NULL"
    assert killed is True, "loop candidate NULL must be pruned via the kill-ledger"
