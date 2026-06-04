"""Acceptance tests for truth-lock (keystone #1).

Covers:
  - method_identity utility (canonical vs uniform-L2 proxy)
  - verify_canonical_3gate (read-only 3-gate) against real artifacts (tolerant)
  - classify_trust_tier TL-4 composition: proxy-only blocked; canonical + family-wise
    + verify-pass allowed; missing family-wise blocked; verify-fail blocks.

No experiments are run; no state is persisted (classify_trust_tier is called directly,
verify is monkeypatched for the composition tests).
"""
from pathlib import Path

import pytest

from tar_lab.method_identity import (
    method_identity, is_proxy_claiming_canonical, result_method_identities,
)


# ── method_identity ────────────────────────────────────────────────────────────
def test_method_identity_proxy_vs_canonical():
    assert method_identity("tcl")["uses_uniform_l2_proxy"] is True
    assert method_identity("tcl")["is_canonical_tcl"] is False
    assert method_identity("tcl_full")["is_canonical_tcl"] is True
    assert method_identity("tcl_full")["uses_regime_observer"] is True
    assert method_identity("tcl_full")["uses_uniform_l2_proxy"] is False
    assert method_identity("tcl_canonical")["is_canonical_tcl"] is True
    assert method_identity("tcl_canonical")["uses_regime_observer"] is False
    assert method_identity("ewc")["in_tcl_family"] is False


def test_is_proxy_claiming_canonical():
    assert is_proxy_claiming_canonical("tcl", claims_canonical_tcl=True) is True
    assert is_proxy_claiming_canonical("tcl_full", claims_canonical_tcl=True) is False
    assert is_proxy_claiming_canonical("tcl", claims_canonical_tcl=False) is False


def test_result_method_identities_derivation():
    # explicit recorded identity wins
    rec = {"method_identity": {"method": "tcl_full", "is_canonical_tcl": True}}
    assert result_method_identities(rec)[0]["method"] == "tcl_full"
    # derive from sweep aggregate keys
    sweep = {"aggregate": {"tcl_canonical": {}, "ewc": {}}}
    fams = {i["method"] for i in result_method_identities(sweep)}
    assert fams == {"tcl_canonical", "ewc"}
    # derive from methods list (verdict-separation wrapped)
    wrapped = {"statistics": {"methods": ["tcl", "sgd_baseline"]}}
    assert {i["method"] for i in result_method_identities(wrapped)} == {"tcl", "sgd_baseline"}


# ── verify_canonical_3gate against real artifacts (tolerant) ────────────────────
def test_verify_canonical_3gate_real(tmp_path):
    from tar_lab.canonical_registry import verify_canonical_3gate
    # nonexistent -> not verified
    ok, reason = verify_canonical_3gate(tmp_path / "nope.json")
    assert ok is False and "missing" in reason
    # a known-passing sweep result, if present in this checkout
    comp = Path("tar_state/comparisons")
    def _results(pat):  # exclude the *_env.json siblings the glob would also match
        return sorted(p for p in comp.glob(pat) if not p.name.endswith("_env.json"))
    phase11 = _results("phase11_ablation__*.json")
    if phase11:
        ok, reason = verify_canonical_3gate(phase11[-1])
        assert ok is True, f"phase11 should verify post-P2, got {reason}"
    # phase10 controlled rerun lacks git.head -> must NOT verify
    p10 = _results("phase10_controlled_rerun_*.json")
    if p10:
        ok, reason = verify_canonical_3gate(p10[-1])
        assert ok is False and reason.startswith("gate1")


# ── TL-4 composition in classify_trust_tier ─────────────────────────────────────
def _record(tmp_path, methods, *, family_wise=True, seed_count=5):
    env = tmp_path / "r_env.json"
    env.write_text("{}", encoding="utf-8")   # presence -> trusted_rerun_with_env tier
    res = tmp_path / "r.json"
    res.write_text("{}", encoding="utf-8")
    record = {"logical_name": "tl4-test", "result_path": str(res), "env_path": str(env)}
    payload = {"seed_count": seed_count, "methods": methods,
               "family_wise_significant": family_wise}
    return record, payload


def _patch_verify(monkeypatch, ok, reason="x"):
    import tar_lab.canonical_registry as cr
    monkeypatch.setattr(cr, "verify_canonical_3gate", lambda p, repo_root=None: (ok, reason))


def test_tl4_proxy_only_blocked(tmp_path, monkeypatch):
    _patch_verify(monkeypatch, True)
    from tar_lab.validation import classify_trust_tier
    rec, payload = _record(tmp_path, ["tcl", "ewc"])  # proxy TCL, no canonical
    out = classify_trust_tier(tmp_path, record=rec, result_payload=payload)
    assert out["method_identity_ok"] is False
    assert out["publication_allowed"] is False


def test_tl4_canonical_familywise_verify_allows(tmp_path, monkeypatch):
    _patch_verify(monkeypatch, True, "verified_3gate")
    from tar_lab.validation import classify_trust_tier
    rec, payload = _record(tmp_path, ["tcl_canonical", "tcl_full", "ewc"], family_wise=True)
    out = classify_trust_tier(tmp_path, record=rec, result_payload=payload)
    assert out["method_identity_ok"] is True
    assert out["family_wise_significant"] is True
    assert out["canonical_verified"] is True
    assert out["publication_allowed"] is True


def test_tl4_missing_familywise_blocked(tmp_path, monkeypatch):
    _patch_verify(monkeypatch, True)
    from tar_lab.validation import classify_trust_tier
    rec, payload = _record(tmp_path, ["tcl_canonical", "ewc"], family_wise=False)
    out = classify_trust_tier(tmp_path, record=rec, result_payload=payload)
    assert out["family_wise_significant"] is False
    assert out["publication_allowed"] is False
    assert out["canonical_verify_reason"] == "not_checked"  # short-circuited, verify not run


def test_tl4_verify_failure_blocks(tmp_path, monkeypatch):
    _patch_verify(monkeypatch, False, "gate3:recompute mismatch")
    from tar_lab.validation import classify_trust_tier
    rec, payload = _record(tmp_path, ["tcl_full", "ewc"], family_wise=True)
    out = classify_trust_tier(tmp_path, record=rec, result_payload=payload)
    assert out["canonical_verified"] is False
    assert out["publication_allowed"] is False
