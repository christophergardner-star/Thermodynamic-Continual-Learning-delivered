"""Tier-3 (2026-06-09): index RunPod-bridge confirmatory results into the recall vault
at the HONEST trust tier, so the Research Director can REASON OVER their numbers without
overclaiming canonical / SoTA status.

These results CANNOT pass verify_canonical_3gate -- empty pod env sibling (gate 1),
non-git pod working copy (gate 2), and hardware-sensitive numbers that will not
deterministically recompute (gate 3). They are therefore deliberately kept OUT of
canonical_results_index.jsonl and the SoTA table. The vault is a RECALL aid, not a
verified claim -- this closes the "think with, not just count" gap while the SoTA bar
stays correctly gated.

Every indexed record carries trust_tier="trusted_rerun", publication_allowed=False, and
provenance="runpod_bridge_unverified" (structured metadata) AND an explicit caveat in the
recall TEXT, so a recall can never imply a verified result. SoTA / canonical_results_index
are NEVER written here. See docs/TAR_Tier3_Result_Ingestion_Scope.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

_CAVEAT = (
    "trust_tier=trusted_rerun (RunPod bridge); publication_allowed=False; "
    "pending canonical 3-gate verification (hardware-sensitive; NOT SoTA-eligible)"
)

_TRUST_META = {
    "trust_tier": "trusted_rerun",
    "publication_allowed": False,
    "provenance": "runpod_bridge_unverified",
}


def _mean(xs: Any) -> float:
    vals = [x for x in (xs or []) if isinstance(x, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def build_recall_record(runner_key: str, result_data: dict) -> Optional[dict]:
    """Map a RunPod comparison result -> a vault recall payload. Pure + deterministic +
    torch-free (testable without the vault). Returns a dict with keys
    {record, method, dataset, experiment_id, extra_metadata}, or None if the runner_key
    is not a known confirmatory type."""
    rk = str(runner_key or "")
    exp_id = f"runpod:{rk}"
    rd = result_data if isinstance(result_data, dict) else {}

    if rk == "hpc_replication_phase2":
        per = [r for r in (rd.get("per_seed_results") or []) if isinstance(r, dict)]
        hpc_f = [r.get("hpc_forgetting") for r in per]
        base_f = [r.get("baseline_forgetting") for r in per]
        deltas = [r.get("delta") for r in per if isinstance(r.get("delta"), (int, float))]
        method, dataset = "high_penalty_conservative", "split_cifar10"
        mech = ("HPC high-penalty-conservative (lambda=0.05) reduces forgetting vs the "
                "baseline config (confirmatory replication).")
        notes = (f"{_CAVEAT}. n={rd.get('seeds_run', len(per))} "
                 f"SPRT={rd.get('sprt_final_decision', '')}; "
                 f"hpc_mean={_mean(hpc_f):.4f}, baseline_mean={_mean(base_f):.4f}.")
        result = {
            "hypothesis_name": method,
            "mechanism_forgetting": hpc_f,
            "baseline_forgetting": base_f,
            "mean_delta": rd.get("mean_delta", _mean(deltas)),
            "p_val": rd.get("wilcoxon_p", 1.0),
            "cohens_d": rd.get("cohens_d", 0.0),
            "n_better": sum(1 for d in deltas if d < 0),
            "verdict": str(rd.get("verdict", "") or ""),
            "notes": notes,
        }
    elif rk == "hpc_lambda_momentum_abl":
        prim = (rd.get("comparisons") or {}).get("primary_hpc_vs_high_lambda") or {}
        method, dataset = "high_penalty_conservative", "split_cifar10"
        mech = ("Ablation: the HPC forgetting benefit is isolated to lambda=0.05, NOT to "
                "conservative LR scaling (hpc vs tcl_high_lambda not significant).")
        notes = f"{_CAVEAT}. {str(rd.get('decision_rationale', ''))[:200]}"
        result = {
            "hypothesis_name": "hpc_lambda_momentum_ablation",
            "mechanism_forgetting": [],
            "mean_delta": 0.0,
            "p_val": prim.get("wilcoxon_p", 1.0) if isinstance(prim, dict) else 1.0,
            "cohens_d": prim.get("cohens_d", 0.0) if isinstance(prim, dict) else 0.0,
            "n_better": 0,
            "verdict": str(rd.get("decision", "") or ""),
            "notes": notes,
        }
    elif rk == "phase16_cifar100_rerun":
        tcl = (rd.get("method_results") or {}).get("tcl") or {}
        seed_f = [r.get("forgetting") for r in (tcl.get("seed_results") or []) if isinstance(r, dict)]
        method, dataset = "tcl", str(rd.get("dataset", "split_cifar100") or "split_cifar100")
        mech = ("Full-protocol TCL vs 6 baselines on CIFAR-100; TCL did not establish "
                "superiority (EWC nominally better, no Bonferroni-significant win).")
        notes = f"{_CAVEAT}. {str(rd.get('honest_verdict', ''))[:400]}"
        result = {
            "hypothesis_name": "phase16_tcl_cifar100",
            "mechanism_forgetting": seed_f,
            "mean_delta": 0.0,
            "p_val": 1.0,
            "cohens_d": 0.0,
            "n_better": 0,
            "verdict": "NULL",  # honest: not superior (full comparison in notes)
            "notes": notes,
        }
    else:
        return None

    record = {
        "hypothesis": {"name": result["hypothesis_name"], "mechanism_description": mech},
        "result": result,
    }
    return {
        "record": record,
        "method": method,
        "dataset": dataset,
        "experiment_id": exp_id,
        "extra_metadata": dict(_TRUST_META),
    }


def ingest_runpod_result_into_recall(workspace, runner_key: str, result_path) -> bool:
    """Open the recall vault and index one RunPod result at the trusted_rerun tier.
    Fail-safe: never raises into the caller; retries on transient vault/SQLite lock.
    NEVER writes the SoTA table or canonical_results_index.jsonl."""
    try:
        result_data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    except Exception:
        return False
    payload = build_recall_record(runner_key, result_data)
    if payload is None:
        return False
    for _ in range(4):
        try:
            from tar_lab.memory import VectorVault
            vault = VectorVault(str(workspace))
            vault.index_experiment_result(
                payload["record"],
                method=payload["method"],
                dataset=payload["dataset"],
                experiment_id=payload["experiment_id"],
                extra_metadata=payload["extra_metadata"],
            )
            return True
        except Exception:
            time.sleep(1.5)
    return False
