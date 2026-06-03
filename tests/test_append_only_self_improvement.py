"""Phase 3 safety property: self-improvement is append-only / additive.

Proves that running TAR's self-improvement aggregators NEVER overwrites a canonical
algorithm or knowledge file, and that the append-only guard refuses to overwrite an
existing artifact or a protected algorithm/knowledge path.
"""
import json
import tempfile
from pathlib import Path

import pytest

from tar_lab import append_only_guard as g
from tar_lab import outcome_learner, method_refinement_engine, authoring_learner


def _temp_ws_with_archive() -> Path:
    ws = Path(tempfile.mkdtemp())
    sd = ws / "tar_state"
    sd.mkdir(parents=True)
    (sd / "experiment_archive.json").write_text(json.dumps({"experiments": [
        {"id": "e1", "method": "tcl", "dataset": "split_cifar10", "status": "complete",
         "progress": {"forgetting_so_far": [0.20, 0.21, 0.19]}},
    ]}), encoding="utf-8")
    return ws


def test_self_improvement_never_changes_canonical_methods_or_kg():
    ws = _temp_ws_with_archive()
    protected = g.protected_method_and_knowledge_paths(ws)
    # snapshot whatever protected files exist (repo method files always exist)
    snap = g.snapshot([str(p) for p in g.protected_method_and_knowledge_paths(Path(".").resolve())])
    assert snap, "expected canonical algorithm files to snapshot"

    outcome_learner.rebuild_outcome_priors(ws)
    method_refinement_engine.rebuild_variant_proposals(ws)
    authoring_learner.rebuild_style_memory(ws)

    # canonical algorithm + knowledge files must be byte-for-byte unchanged
    g.assert_unchanged(snap)


def test_guard_refuses_to_overwrite_canonical_method():
    import tar_lab
    canon = Path(tar_lab.__file__).parent / "method_registry.py"
    assert canon.exists()
    with pytest.raises(g.AppendOnlyViolation):
        g.guarded_create_new(canon, "tampered")


def test_guard_refuses_to_overwrite_existing_artifact():
    ws = _temp_ws_with_archive()
    existing = ws / "tar_state" / "experiment_archive.json"
    with pytest.raises(g.AppendOnlyViolation):
        g.guarded_create_new(existing, "tampered", require_additive_area=False)


def test_guard_allows_new_additive_artifact():
    ws = _temp_ws_with_archive()
    newf = ws / "tar_state" / "synthesized_method_variants" / "tcl-variant.py"
    g.guarded_create_new(newf, "# new variant")
    assert newf.exists()
