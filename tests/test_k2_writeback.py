"""Keystone #2 (K2.2) — write results back to the recalled literature DB.

Covers the integrity-critical half of K2.2: TAR's own results are written into the
SAME LiteratureKnowledgeGraph the director recalls, tagged source='tar_internal', and
are EXCLUDED from the external SoTA the NoveltyGate compares against — so TAR cannot
cite its own result as the prior art it must beat (circular self-validation).
"""
from pathlib import Path

from literature.knowledge_graph import LiteratureKnowledgeGraph
from literature.schemas import Benchmark, SoTAEntry

_BID = "benchmark:402cf4341499b795"


def _mk_graph(db_path: Path) -> LiteratureKnowledgeGraph:
    g = LiteratureKnowledgeGraph(str(db_path))
    g.upsert_benchmark(
        Benchmark(
            benchmark_id=_BID,
            name="Continual Learning On Split Cifar 10",
            task="continual_learning",
            domain="continual_learning",
            metrics=["forgetting"],
            metrics_higher_better={"forgetting": False},
        )
    )
    return g


def test_internal_source_excluded_from_external_best(tmp_path):
    g = _mk_graph(tmp_path / "lit.db")
    # External literature SoTA: forgetting 0.10 (lower is better).
    g.upsert_sota_entry(SoTAEntry(
        entry_id="ext::ewc", benchmark_id=_BID, method_name="ewc",
        metric_name="forgetting", metric_value=0.10, higher_is_better=False,
        source="external"))
    # TAR's own, better, result: forgetting 0.05 — tagged internal.
    g.upsert_sota_entry(SoTAEntry(
        entry_id="tar_internal::tcl", benchmark_id=_BID, method_name="tcl",
        metric_name="forgetting", metric_value=0.05, higher_is_better=False,
        source="tar_internal"))

    # NoveltyGate path: excluding internal, the external best is EWC 0.10 (NOT TAR's 0.05).
    ext = g.best_result(_BID, "forgetting", higher_is_better=False, exclude_source="tar_internal")
    assert ext is not None and ext.method_name == "ewc"
    assert abs(ext.metric_value - 0.10) < 1e-9 and ext.source == "external"

    # Without exclusion, the overall best IS TAR's internal 0.05 (proves it is stored + read).
    overall = g.best_result(_BID, "forgetting", higher_is_better=False)
    assert overall is not None and overall.method_name == "tcl"
    assert overall.source == "tar_internal"
    g.close()


def test_source_roundtrips_through_db(tmp_path):
    g = _mk_graph(tmp_path / "lit.db")
    g.upsert_sota_entry(SoTAEntry(
        entry_id="tar_internal::x", benchmark_id=_BID, method_name="tcl",
        metric_name="forgetting", metric_value=0.07, higher_is_better=False,
        source="tar_internal"))
    tbl = g.get_sota_table(_BID, "forgetting")
    srcs = {e.entry_id: e.source for e in tbl.entries}
    assert srcs.get("tar_internal::x") == "tar_internal"
    g.close()


def test_writeback_helper_tags_internal_and_excludes(tmp_path):
    import tar_living_research as tlr

    ws = Path(tmp_path)
    db = ws / "tar_state" / "literature" / "literature_graph.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    g = _mk_graph(db)
    # External baseline so the exclusion is observable after write-back.
    g.upsert_sota_entry(SoTAEntry(
        entry_id="ext::ewc", benchmark_id=_BID, method_name="ewc",
        metric_name="forgetting", metric_value=0.12, higher_is_better=False,
        source="external"))
    g.close()

    record = {
        "hypothesis": {"name": "deep_anchor"},
        "result": {
            "mechanism_forgetting": [0.04, 0.06, 0.05],
            "mechanism_accuracy": [0.80, 0.79, 0.81],
        },
    }
    spec = {"method": "tcl", "dataset": "split_cifar10"}

    n = tlr._write_results_to_knowledge_graph(ws, [(record, spec)])
    assert n == 1

    g2 = LiteratureKnowledgeGraph(str(db))
    # The dataset 'split_cifar10' resolved to the existing benchmark by name.
    overall = g2.best_result(_BID, "forgetting", higher_is_better=False)
    assert overall is not None and overall.source == "tar_internal"
    assert abs(overall.metric_value - 0.05) < 1e-6  # mean of [0.04, 0.06, 0.05]
    assert overall.method_name == "tcl"
    assert overall.extra_metrics.get("accuracy") is not None
    # And it is excluded from the external SoTA -> EWC 0.12.
    ext = g2.best_result(_BID, "forgetting", higher_is_better=False, exclude_source="tar_internal")
    assert ext is not None and ext.method_name == "ewc"
    g2.close()


def test_writeback_opens_negative_result_gap(tmp_path):
    import tar_living_research as tlr

    ws = Path(tmp_path)
    db = ws / "tar_state" / "literature" / "literature_graph.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    _mk_graph(db).close()

    record = {
        "hypothesis": {"name": "deep_anchor"},
        "result": {"mechanism_forgetting": [0.15, 0.16], "verdict": "NULL", "n_better": 1},
    }
    spec = {"method": "tcl", "dataset": "split_cifar10"}
    tlr._write_results_to_knowledge_graph(ws, [(record, spec)])

    g = LiteratureKnowledgeGraph(str(db))
    gaps = g.get_top_gaps(domain="continual_learning")
    target = f"tar_internal_gap::{_BID}::tcl"
    match = [gp for gp in gaps if gp.gap_id == target]
    assert match, "non-win result should open a negative_result gap"
    assert match[0].gap_type == "negative_result" and match[0].status == "open"
    g.close()


def test_winning_result_opens_no_gap(tmp_path):
    import tar_living_research as tlr

    ws = Path(tmp_path)
    db = ws / "tar_state" / "literature" / "literature_graph.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    _mk_graph(db).close()

    record = {
        "hypothesis": {"name": "win"},
        "result": {"mechanism_forgetting": [0.02, 0.03], "verdict": "BREAKTHROUGH", "n_better": 5},
    }
    spec = {"method": "tcl", "dataset": "split_cifar10"}
    tlr._write_results_to_knowledge_graph(ws, [(record, spec)])

    g = LiteratureKnowledgeGraph(str(db))
    assert g.gap_count("open") == 0, "a confirmed win should not open a negative_result gap"
    g.close()


def test_writeback_skips_when_db_absent(tmp_path):
    import tar_living_research as tlr

    ws = Path(tmp_path)  # no tar_state/literature/literature_graph.db present
    n = tlr._write_results_to_knowledge_graph(
        ws,
        [({"result": {"mechanism_forgetting": [0.1]}}, {"method": "tcl", "dataset": "split_cifar10"})],
    )
    assert n == 0  # fail-safe: refuses to write internal results without an external-grounded DB
