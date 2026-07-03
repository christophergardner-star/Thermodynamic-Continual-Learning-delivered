"""
Regression tests for the solution-finding loop (docs/solution_loop_implementation_plan.md).
Grouped by plan phase. Phase 0 = integrity pre-work that makes the loop sound.
"""
from __future__ import annotations

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
