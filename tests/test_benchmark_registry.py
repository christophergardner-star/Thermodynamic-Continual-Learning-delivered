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
            benchmark_ids.add(suite.benchmark_id)
            tiers.add(suite.tier)
        assert "smoke" in tiers
        assert "validation" in tiers or "canonical" in tiers


def test_canonical_suites_are_explicit_and_non_proxy():
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    for profile in registry.profiles:
        canonical = [suite for suite in profile.benchmark_suites if suite.tier == "canonical"]
        assert canonical, f"{profile.profile_id} has no canonical benchmark suites"
        for suite in canonical:
            assert suite.canonical_comparable
            assert not suite.proxy_allowed


def test_registry_lists_benchmarks_by_tier():
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    suites = registry.list_benchmarks(profile_id="quantum_ml", tier="canonical")
    assert suites
    assert all(suite.tier == "canonical" for suite in suites)
    assert {suite.family for suite in suites} >= {"depth_trainability_curve", "initialization_trainability"}
