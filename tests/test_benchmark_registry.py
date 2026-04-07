from __future__ import annotations

from pathlib import Path

from tar_lab.science_profiles import ScienceProfileRegistry


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_profile_declares_named_benchmark_suites():
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    assert registry.profiles
    for profile in registry.profiles:
        assert profile.benchmark_suites, f"{profile.profile_id} is missing benchmark suites"
        benchmark_ids: set[str] = set()
        tiers: set[str] = set()
        for suite in profile.benchmark_suites:
            assert suite.benchmark_id not in benchmark_ids
            assert suite.name
            assert suite.family
            assert suite.dataset_or_env
            assert suite.metric_protocol
            assert suite.truth_status in {"canonical_ready", "validation_only", "smoke_only", "unsupported"}
            if suite.tier == "smoke":
                assert suite.truth_status == "smoke_only"
            elif suite.tier == "validation":
                assert suite.truth_status == "validation_only"
            else:
                assert suite.truth_status in {"canonical_ready", "unsupported"}
            benchmark_ids.add(suite.benchmark_id)
            tiers.add(suite.tier)
        assert "smoke" in tiers
        assert "validation" in tiers or "canonical" in tiers


def test_canonical_suite_truth_status_is_explicit_and_non_proxy():
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    for profile in registry.profiles:
        canonical = [suite for suite in profile.benchmark_suites if suite.tier == "canonical"]
        assert canonical, f"{profile.profile_id} has no canonical benchmark suites"
        for suite in canonical:
            assert not suite.proxy_allowed
            if suite.truth_status == "canonical_ready":
                assert suite.canonical_comparable
            else:
                assert suite.truth_status == "unsupported"
                assert not suite.canonical_comparable


def test_registry_lists_benchmarks_by_tier():
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    suites = registry.list_benchmarks(profile_id="quantum_ml", tier="canonical")
    assert suites
    assert all(suite.tier == "canonical" for suite in suites)
    assert {suite.family for suite in suites} >= {"depth_trainability_curve", "initialization_trainability", "noise_trainability_ablation"}
    ready = {suite.benchmark_id for suite in suites if suite.truth_status == "canonical_ready"}
    assert ready >= {
        "pennylane_barren_plateau_canonical",
        "pennylane_init_canonical",
        "qml_noise_canonical",
    }
