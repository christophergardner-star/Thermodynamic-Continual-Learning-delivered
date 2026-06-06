"""Keystone #2 (K2.3b, accepted scope) — honest handling of a missing external bar.

PapersWithCode's API is dead, so the literature DB has 0 external SoTA entries. Two
guarantees:
  (A) NoveltyGate must NOT claim a result "establishes a new state of the art" when
      there is no external SoTA to compare against (it previously returned
      sota_verdict="better", rank=1 for an empty table).
  (B) external SoTA may only enter via the curated loader, which REFUSES uncited
      entries (anti-fabrication) — the shipped template is empty.
"""
import json
from pathlib import Path

from literature.knowledge_graph import LiteratureKnowledgeGraph
from literature.novelty_gate import NoveltyGate
from literature.schemas import Benchmark, SoTAEntry

_BID = "benchmark:402cf4341499b795"


def _graph_with_benchmark(tmp_path) -> LiteratureKnowledgeGraph:
    g = LiteratureKnowledgeGraph(str(tmp_path / "lit.db"))
    g.upsert_benchmark(Benchmark(
        benchmark_id=_BID, name="Continual Learning On Split Cifar 10",
        task="continual_learning", domain="continual_learning",
        metrics=["forgetting"], metrics_higher_better={"forgetting": False}))
    return g


# ---- (A) honest verdict when there is no external bar ----------------------------

def test_no_external_baseline_does_not_claim_sota(tmp_path):
    g = _graph_with_benchmark(tmp_path)  # no SoTA entries -> no external bar
    gate = NoveltyGate(g, load_embedding_model=False)
    rep = gate.evaluate(
        method_name="tcl_canonical", method_description="thermodynamic continual learning",
        benchmark_id=_BID, metric_name="forgetting", metric_value=0.154, higher_is_better=False)
    assert rep.sota_rank is None and rep.sota_delta is None  # no external rank/delta claimed
    text = rep.contribution_statement.lower()
    assert "no external sota" in text or "provisional" in text
    # must NOT make a bare "establishes a new state of the art" claim
    assert "establishes a new state of the art" not in text
    g.close()


def test_external_baseline_restores_comparison(tmp_path):
    g = _graph_with_benchmark(tmp_path)
    g.upsert_sota_entry(SoTAEntry(
        entry_id="external::ewc", benchmark_id=_BID, method_name="ewc",
        metric_name="forgetting", metric_value=0.10, higher_is_better=False, source="external"))
    gate = NoveltyGate(g, load_embedding_model=False)
    # a worse result (0.20 forgetting vs the 0.10 external bar) -> known_result, ranked
    rep = gate.evaluate(
        method_name="x", method_description="y", benchmark_id=_BID,
        metric_name="forgetting", metric_value=0.20, higher_is_better=False)
    assert rep.sota_rank is not None
    assert rep.verdict == "known_result"
    g.close()


# ---- (B) curated external-SoTA loader (anti-fabrication) -------------------------

def test_curated_loader_requires_citation(tmp_path):
    from literature.curated_sota import load_curated_external_sota

    g = _graph_with_benchmark(tmp_path)
    f = tmp_path / "curated.json"
    f.write_text(json.dumps({"entries": [
        {  # cited -> accepted (metric_value here is an illustrative TEST fixture)
            "benchmark_id": _BID, "method_name": "ewc", "metric_name": "forgetting",
            "metric_value": 0.10, "higher_is_better": False,
            "paper_title": "Overcoming catastrophic forgetting in neural networks",
            "citation": "arXiv:1612.00796", "year": 2017, "venue": "PNAS",
        },
        {  # NO citation -> refused (anti-fabrication guard)
            "benchmark_id": _BID, "method_name": "uncited", "metric_name": "forgetting",
            "metric_value": 0.05, "higher_is_better": False, "paper_title": "Some paper",
        },
    ]}), encoding="utf-8")

    n = load_curated_external_sota(g, f)
    assert n == 1  # only the cited entry
    best = g.best_result(_BID, "forgetting", higher_is_better=False, exclude_source="tar_internal")
    assert best is not None and best.method_name == "ewc" and best.source == "external"
    g.close()


def test_shipped_template_is_empty_no_fabrication(tmp_path):
    from literature.curated_sota import load_curated_external_sota

    g = _graph_with_benchmark(tmp_path)
    # default path = the shipped literature/curated_external_sota.json (entries: [])
    assert load_curated_external_sota(g) == 0
    g.close()
