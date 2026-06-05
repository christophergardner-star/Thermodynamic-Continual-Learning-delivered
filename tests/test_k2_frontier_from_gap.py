"""Keystone #2 (K2.3a, first brick) — gap -> register-ready FrontierProblem.

frontier_problem_from_gap() must produce a FrontierProblem that satisfies the
FrontierRegistry.register() guard (well_known_problem + the four required_text
fields + external_baselines/candidate_datasets/candidate_backbones + domain),
sourcing the real-world grounding from the per-domain well-known catalog. The
external baselines must be real external methods, never the internal TCL family.
Verified against the real seeded tcl_family gap when the live DB is present.
"""
from pathlib import Path

import pytest

from tar_frontier import FrontierRegistry, frontier_problem_from_gap

_ROOT = Path(__file__).resolve().parents[1]


def _register_ok(problem, tmp_path):
    (tmp_path / "tar_state").mkdir(parents=True, exist_ok=True)
    reg = FrontierRegistry(tmp_path)
    reg.register(problem)  # raises ValueError if the guard is not satisfied
    assert reg._problems.get(problem.id) is not None


def test_from_synthetic_gap_is_register_ready(tmp_path):
    gap = {
        "gap_id": "tar_internal_gap::benchmark:402cf4341499b795::tcl_family",
        "gap_type": "negative_result",
        "title": "Alternative mechanism needed: TCL family does not establish superiority on Split-CIFAR-10",
        "description": "phase18 n=5: canonical TCL 0.154 worse than SI 0.047; pursue alternatives.",
        "domain": "continual_learning",
        "method_names": ["tcl", "tcl_canonical", "tcl_full"],
    }
    p = frontier_problem_from_gap(gap)

    assert p.well_known_problem is True
    assert p.candidate_datasets and p.candidate_backbones and p.external_baselines
    for fld in (p.industry_problem_title, p.global_problem_statement, p.why_important, p.research_guidance):
        assert str(fld).strip(), "required_text field must be non-empty for the guard"
    assert p.domain == "continual_learning"
    assert p.id.startswith("fp-gap-")
    # external baselines must be EXTERNAL methods, never the internal TCL family
    assert not (set(p.external_baselines) & {"tcl", "tcl_canonical", "tcl_full"})
    # and it actually passes the registry guard
    _register_ok(p, tmp_path)


def test_unknown_domain_falls_back_but_stays_register_ready(tmp_path):
    p = frontier_problem_from_gap({"gap_id": "g1", "title": "X", "domain": "no_such_domain_xyz"})
    assert p.candidate_datasets and p.external_baselines and p.candidate_backbones
    assert p.domain == "no_such_domain_xyz"  # explicit domain preserved (guard needs it non-empty)
    _register_ok(p, tmp_path)


def test_against_real_seeded_gap(tmp_path):
    db = _ROOT / "tar_state" / "literature" / "literature_graph.db"
    if not db.exists():
        pytest.skip("live literature DB not present")
    from literature.knowledge_graph import LiteratureKnowledgeGraph

    g = LiteratureKnowledgeGraph(str(db))
    try:
        gaps = g.get_top_gaps(n=10, domain="continual_learning")
    finally:
        g.close()
    target = [x for x in gaps if str(x.gap_id).endswith("tcl_family")]
    if not target:
        pytest.skip("seeded tcl_family gap not present in live DB")

    p = frontier_problem_from_gap(target[0])
    _register_ok(p, tmp_path)  # the REAL seeded gap is register-ready
    assert "tcl" in p.research_guidance.lower()  # implicated internal methods surfaced
