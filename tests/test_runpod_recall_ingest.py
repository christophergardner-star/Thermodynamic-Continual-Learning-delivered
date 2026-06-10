"""Tier-3 recall-ingest tests: the pure RunPod-result -> vault-record mapping +
the truth-lock guardrails. Torch-free / vault-free (build_recall_record is pure)."""
from tar_lab.runpod_recall_ingest import build_recall_record, _CAVEAT


def _trust_ok(p):
    em = p["extra_metadata"]
    assert em["trust_tier"] == "trusted_rerun"
    assert em["publication_allowed"] is False
    assert em["provenance"] == "runpod_bridge_unverified"
    # the caveat must be in the recall TEXT too, so a recall can never imply "verified"
    assert _CAVEAT in p["record"]["result"]["notes"]


def test_hpc_replication_maps_with_real_numbers():
    rd = {
        "verdict": "REPLICATION_SUCCESS", "wilcoxon_p": 0.0017, "cohens_d": -0.975,
        "mean_delta": -0.0695, "seeds_run": 12, "sprt_final_decision": "accept_H1",
        "per_seed_results": [
            {"seed": 9, "hpc_forgetting": 0.208, "baseline_forgetting": 0.458, "delta": -0.25},
            {"seed": 10, "hpc_forgetting": 0.241, "baseline_forgetting": 0.319, "delta": -0.078},
        ],
    }
    p = build_recall_record("hpc_replication_phase2", rd)
    assert p["method"] == "high_penalty_conservative" and p["dataset"] == "split_cifar10"
    assert p["experiment_id"] == "runpod:hpc_replication_phase2"
    res = p["record"]["result"]
    assert res["verdict"] == "REPLICATION_SUCCESS"
    assert res["p_val"] == 0.0017 and res["cohens_d"] == -0.975
    assert res["mechanism_forgetting"] == [0.208, 0.241]
    assert res["n_better"] == 2  # both deltas < 0
    _trust_ok(p)


def test_hpc_lambda_maps_decision_and_primary():
    rd = {
        "decision": "LAMBDA_IS_MECHANISM",
        "decision_rationale": "No significant difference between HPC and tcl_high_lambda (p=1.0).",
        "comparisons": {"primary_hpc_vs_high_lambda": {"wilcoxon_p": 1.0, "cohens_d": 1.223}},
    }
    p = build_recall_record("hpc_lambda_momentum_abl", rd)
    assert p["record"]["result"]["verdict"] == "LAMBDA_IS_MECHANISM"
    assert p["record"]["result"]["p_val"] == 1.0 and p["record"]["result"]["cohens_d"] == 1.223
    _trust_ok(p)


def test_phase16_maps_tcl_and_honest_null():
    rd = {
        "dataset": "split_cifar100",
        "method_results": {"tcl": {"mean_forgetting": 0.1419,
                                   "seed_results": [{"forgetting": 0.1358}, {"forgetting": 0.15}]}},
        "honest_verdict": "TCL: mean_forgetting=0.1419 ... tcl_vs_ewc: direction=other_better",
    }
    p = build_recall_record("phase16_cifar100_rerun", rd)
    assert p["method"] == "tcl" and p["dataset"] == "split_cifar100"
    assert p["record"]["result"]["verdict"] == "NULL"  # honest: not superior
    assert p["record"]["result"]["mechanism_forgetting"] == [0.1358, 0.15]
    assert "tcl_vs_ewc" in p["record"]["result"]["notes"]  # full comparison preserved
    _trust_ok(p)


def test_unknown_runner_key_returns_none():
    assert build_recall_record("some_other_runner", {"x": 1}) is None


def test_every_known_type_is_trusted_rerun_never_published():
    for rk, rd in [
        ("hpc_replication_phase2", {"per_seed_results": []}),
        ("hpc_lambda_momentum_abl", {"comparisons": {}}),
        ("phase16_cifar100_rerun", {"method_results": {}}),
    ]:
        p = build_recall_record(rk, rd)
        assert p is not None
        assert p["extra_metadata"]["publication_allowed"] is False
        assert p["extra_metadata"]["trust_tier"] == "trusted_rerun"
