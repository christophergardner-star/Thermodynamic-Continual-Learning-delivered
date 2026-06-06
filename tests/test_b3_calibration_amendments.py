"""Workstream B3 — consume the calibration registry as RAIL-3 seed amendments.

calibration_learner rebuilt a registry every cycle but load_calibration had ZERO consumers.
propose_seed_amendments() closes that loop WITHOUT violating integrity rail #3:
  * underpowered results become PROPOSED pre-registration seed amendments (human-gated);
  * it never edits a pre-registration, a hypothesis, or a seed count;
  * the log is append-only-safe + idempotent — a human-set status survives every cycle.
"""
import json
from pathlib import Path

from tar_lab import calibration_learner as cal


def _write_registry(ws: Path, effect_rows: list[dict]) -> dict:
    reg = {"generated_at": "t", "advisory_only": True,
           "effect_size_calibration": effect_rows, "frontier_calibration": []}
    p = ws / "tar_state" / "calibration" / "calibration_registry.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg), encoding="utf-8")
    return reg


def _rows():
    return [
        # underpowered, needs more seeds -> proposal
        {"result_id": "res_under", "observed_cohens_d": 0.42, "seeds_run": 5,
         "achieved_power": 0.41, "seeds_needed_for_80pct_power": 32,
         "calibration_flag": "underpowered",
         "recommendation": "underpowered (power=0.41 at n=5); ~32 seeds needed..."},
        # adequately powered -> no proposal
        {"result_id": "res_ok", "observed_cohens_d": 1.4, "seeds_run": 5,
         "achieved_power": 0.92, "seeds_needed_for_80pct_power": 4,
         "calibration_flag": "adequate", "recommendation": "adequately powered"},
        # flagged underpowered but rec_n <= seeds_run -> no proposal (nothing to amend)
        {"result_id": "res_edge", "observed_cohens_d": 0.9, "seeds_run": 10,
         "achieved_power": 0.7, "seeds_needed_for_80pct_power": 8,
         "calibration_flag": "underpowered", "recommendation": "..."},
    ]


def test_only_actionable_underpowered_rows_become_proposals(tmp_path):
    _write_registry(tmp_path, _rows())
    doc = cal.propose_seed_amendments(tmp_path)  # registry=None -> reads the file
    ids = {a["amendment_id"] for a in doc["amendments"]}
    assert ids == {"seed-amend::res_under::n32"}  # only the actionable one
    a = doc["amendments"][0]
    assert a["status"] == "proposed_pending_human_approval"
    assert a["recommended_seeds"] == 32 and a["seeds_run"] == 5 and a["rail"] == 3
    assert doc["summary"]["pending"] == 1 and doc["summary"]["added_this_cycle"] == 1


def test_idempotent_no_duplicates(tmp_path):
    reg = _write_registry(tmp_path, _rows())
    cal.propose_seed_amendments(tmp_path, reg)
    doc2 = cal.propose_seed_amendments(tmp_path, reg)  # second cycle
    assert len(doc2["amendments"]) == 1
    assert doc2["summary"]["added_this_cycle"] == 0  # nothing new added


def test_human_decision_is_preserved_across_cycles(tmp_path):
    reg = _write_registry(tmp_path, _rows())
    cal.propose_seed_amendments(tmp_path, reg)
    # a human approves the proposal
    p = tmp_path / "tar_state" / "calibration" / "preregistration_amendments.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["amendments"][0]["status"] = "approved"
    p.write_text(json.dumps(doc), encoding="utf-8")
    # next cycle must NOT reset the human decision
    doc2 = cal.propose_seed_amendments(tmp_path, reg)
    assert doc2["amendments"][0]["status"] == "approved"
    assert doc2["summary"]["pending"] == 0


def test_never_writes_a_preregistration(tmp_path):
    """RAIL #3: it must touch only its own log, never the actual pre-registration."""
    reg = _write_registry(tmp_path, _rows())
    cal.propose_seed_amendments(tmp_path, reg)
    assert not (tmp_path / "tar_state" / "autonomous_research" / "preregistration.json").exists()
    assert (tmp_path / "tar_state" / "calibration" / "preregistration_amendments.json").exists()


def test_empty_registry_is_safe(tmp_path):
    doc = cal.propose_seed_amendments(tmp_path, {"effect_size_calibration": []})
    assert doc["amendments"] == [] and doc["summary"]["total"] == 0
