"""Keystone #2 (K2.1) — science_exec bridge emission.

Tests the director helper that builds the Stack-B payload (domain + experiment plan +
primary metric) for a science_exec directive's config_overrides. The full emit path
(supported_profiles -> directive -> _specs_from_directives passthrough) is inert until a
domain is opted into _FRONTIER_AUTONOMY_DOMAINS; this verifies the payload builder, which
is the K2.1-specific new logic.
"""
from tar_research_director import _science_exec_overrides


def test_science_exec_overrides_loads_real_profile():
    ov = _science_exec_overrides({"science_domain": "quantum_ml"})
    assert ov["domain"] == "quantum_ml"
    assert ov["benchmark_tier"] == "validation"
    assert isinstance(ov["experiments"], list) and ov["experiments"], "experiments must be non-empty"
    e0 = ov["experiments"][0]
    assert "template_id" in e0 and "benchmark" in e0 and "metrics" in e0
    assert ov.get("primary_metric")  # derived from the template's first metric


def test_science_exec_overrides_fallback_for_unknown_domain():
    ov = _science_exec_overrides({"science_domain": "no_such_domain_xyz"})
    assert ov["domain"] == "no_such_domain_xyz"
    assert ov["experiments"], "must emit a minimal validation probe even without a profile"
    assert ov["experiments"][0]["benchmark"] == "validation"
    assert ov["primary_metric"] == "accuracy"


def test_science_exec_overrides_keys_survive_metadata_filter():
    # domain/experiments/primary_metric/benchmark_tier must NOT be in the director's
    # runtime-metadata strip set (else they'd be dropped before reaching the runner).
    from tar_living_research import _DIRECTOR_RUNTIME_METADATA_KEYS
    ov = _science_exec_overrides({"science_domain": "generic_ml"})
    for k in ("domain", "experiments", "primary_metric", "benchmark_tier"):
        assert k in ov
        assert k not in _DIRECTOR_RUNTIME_METADATA_KEYS, f"{k} would be stripped"
