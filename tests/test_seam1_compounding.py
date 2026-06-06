"""Seam 1 (2026-06-06) — compounding store unification regression lock.

The recall store (VectorVault) never indexed finalized results, and finalize only
keyed `ar-*` plans, so director-*-probe experiments never compounded (no literature
write-back, no recall index). The write-back store and the recall store never touched,
so the autonomous loop could not close. These tests lock:
  (1) index_experiment_result round-trips through vault.search -> recall returns TAR's
      own just-finished trial;
  (2) _result_from_spec_record builds a usable plan-less record for a director-probe
      (the shape the write-back + vault-index paths consume).

VectorVault (via sentence-transformers) and tar_living_research both transitively import
torch, so imports are kept function-local to dodge the Windows torch-DLL pytest flake
while the live daemon holds torch; these run green in a daemon-down window.
"""
from __future__ import annotations

import json
from pathlib import Path


def _record(name: str = "seam1-hyp", verdict: str = "NULL") -> dict:
    return {
        "hypothesis": {"name": name, "mechanism_description": "energy-based importance"},
        "result": {
            "hypothesis_name": name,
            "verdict": verdict,
            "mean_delta": -0.03,
            "p_val": 0.12,
            "cohens_d": 0.4,
            "n_better": 3,
            "mechanism_forgetting": [0.05, 0.06],
            "mechanism_accuracy": [0.8, 0.79],
            "notes": "split_cifar10 probe",
        },
    }


def test_index_experiment_result_roundtrip(tmp_path):
    from tar_lab.memory import VectorVault

    vault = VectorVault(str(tmp_path))
    vault.index_experiment_result(
        _record("catastrophic-forgetting-carryover"),
        method="tcl",
        dataset="split_cifar10",
        experiment_id="director-catastrophic-forgetting-carryover-probe",
    )
    hits = vault.search("catastrophic forgetting tcl split_cifar10 result", n_results=3)
    ids = [str(getattr(h, "document_id", "") or "") for h in hits]
    assert any(i.startswith("experiment_result:") for i in ids), f"own result not recalled: {ids}"


def test_result_from_spec_record_planless(tmp_path):
    import tar_living_research as tlr

    rp = tmp_path / "result.json"
    rp.write_text(
        json.dumps(
            {
                "seed_results": [{"forgetting": 0.05, "accuracy": 0.8}],
                "mean_delta": -0.02,
                "p_val": 0.2,
                "cohens_d": 0.3,
                "n_better": 2,
                "verdict": "NULL",
                "notes": "x",
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "id": "director-foo-probe",
        "method": "tcl",
        "dataset": "split_cifar10",
        "hypothesis_name": "foo",
        "result_path": str(rp),
        "status": "complete",
    }
    rec = tlr._result_from_spec_record(spec)
    assert rec is not None
    assert rec["hypothesis"]["name"] == "foo"
    assert rec["result"]["mechanism_forgetting"] == [0.05]
    assert rec["result"]["verdict"] == "NULL"
    # missing result file -> None (fail-safe)
    assert tlr._result_from_spec_record({"id": "director-bar-probe", "result_path": ""}) is None
