"""
Method-identity — single source of truth for distinguishing the canonical TCL
algorithm from the published uniform-L2 PROXY (truth-lock TL-3, 2026-06-04).

Background: every previously published "TCL" number used method="tcl", which is a
D_PR-scaled uniform L2 anchor to the last task (+ a regime-observer LR scaler) — NOT
the per-element gradient-energy importance method (tcl.py) described in the paper.
method="tcl_canonical"/"tcl_full" run the real algorithm. Nothing recorded which one
produced a result, so a proxy number could be (and in the paper, was) presented as the
canonical algorithm. This utility lets the validation gate (TL-4) and the authoring
guard (TL-3c) refuse to count/render a proxy result under a canonical-algorithm claim.

Mirrors the families declared in tar_lab/multimodal_payloads.py:
    _TCL_OBS   = {"tcl", "tcl_full"}            # use the thermodynamic regime observer / LR control
    _TCL_CANON = {"tcl_canonical", "tcl_full"}  # use the canonical (tcl.py) per-element importance penalty
"""
from __future__ import annotations

from typing import Any

# Keep in sync with multimodal_payloads.py:897-904
_TCL_OBS = frozenset({"tcl", "tcl_full"})
_TCL_CANON = frozenset({"tcl_canonical", "tcl_full"})
_TCL_FAMILY = frozenset({"tcl", "tcl_penalty_only", "tcl_canonical", "tcl_full"})
# Variants that are (entirely or partly) the uniform-L2 proxy, NOT the canonical importance method.
_TCL_PROXY = frozenset({"tcl", "tcl_penalty_only"})

# Established external CL baselines that TAR REPRODUCES under its own protocol.
# These (and ONLY these) are allowed to serve as the NoveltyGate comparison bar.
# Everything else TAR produces — its TCL flagship AND any novel/composed/synthesized
# candidate — is EXCLUDED from the bar so TAR can never cite its own work as the prior
# art it must beat (circular self-validation). Matching is exact + case-insensitive:
# a novel method like "si_clamp_decay" is NOT the "si" baseline.
_ESTABLISHED_BASELINES = frozenset({
    "sgd", "sgd_baseline", "sgd_generic", "naive", "finetune",
    "ewc", "ewc_generic", "si", "si_generic", "mas", "mas_generic",
    "lwf", "lwf_generic", "der", "der_plus_plus", "derpp", "der++",
    "agem", "a-gem", "agem_generic", "gem", "icarl",
    "er", "experience_replay", "replay", "gdumb",
})

# Result-source tags that must NEVER count as the external/novelty comparison bar.
# Consumed by NoveltyGate's best_result(exclude_source=...).
EXCLUDED_FROM_NOVELTY_BAR = frozenset({"tar_internal", "tar_novel"})


def method_identity(method: str) -> dict[str, Any]:
    """Return the identity fingerprint of a benchmark method name.

    Keys:
      method                 - the raw method name
      family                 - 'tcl' for any TCL variant, else the method name
      in_tcl_family          - bool
      is_canonical_tcl       - True only for tcl_canonical / tcl_full (real algorithm)
      uses_uniform_l2_proxy  - True for tcl / tcl_penalty_only (the published proxy)
      uses_regime_observer   - True for tcl / tcl_full
    """
    m = str(method or "").strip()
    return {
        "method": m,
        "family": "tcl" if m in _TCL_FAMILY else m,
        "in_tcl_family": m in _TCL_FAMILY,
        "is_canonical_tcl": m in _TCL_CANON,
        "uses_uniform_l2_proxy": m in _TCL_PROXY,
        "uses_regime_observer": m in _TCL_OBS,
    }


def internal_source_tag(method: str) -> str:
    """Source tag for a TAR-produced result, by method provenance. THREE classes:

    - TCL flagship family -> 'tar_internal'. EXCLUDED from the NoveltyGate bar.
    - ESTABLISHED baseline reproduced by TAR (ewc/si/sgd/der++/agem/...) ->
      'tar_internal_baseline'. This IS the protocol-matched comparison bar (NOT excluded).
      Beating it is a capability comparison under TAR's protocol, not an external-SoTA claim.
    - NOVEL / composed / synthesized / unknown method -> 'tar_novel'. EXCLUDED from the bar
      (fail-safe default): a method TAR invented must never become the prior art it must beat.
      This closes the circular-self-validation hole for the solution-finding loop, where the
      proposer mints new method keys that are neither TCL nor established baselines.
    """
    if method_identity(method)["in_tcl_family"]:
        return "tar_internal"
    if str(method or "").strip().lower() in _ESTABLISHED_BASELINES:
        return "tar_internal_baseline"
    return "tar_novel"


def is_proxy_claiming_canonical(method: str, *, claims_canonical_tcl: bool) -> bool:
    """True iff a uniform-L2 PROXY result is being presented as the canonical TCL algorithm.

    This is the specific overclaim truth-lock forbids: a method="tcl" (proxy) result
    backing a claim about the canonical gradient-energy importance algorithm.
    `claims_canonical_tcl` is supplied by the caller (e.g. the result is labeled/cited as
    "TCL" in a context that derives the canonical mechanism).
    """
    ident = method_identity(method)
    return bool(claims_canonical_tcl) and ident["uses_uniform_l2_proxy"] and not ident["is_canonical_tcl"]


def result_method_identities(result_payload: dict | None) -> list[dict[str, Any]]:
    """Best-effort: derive method identities for a result payload.

    Prefers an explicitly recorded `method_identity` (single-method results); otherwise
    derives from the method name / the sweep `methods` list / `aggregate` keys, so the gate
    works on existing and sweep-schema results that predate explicit recording.
    """
    if not isinstance(result_payload, dict):
        return []
    # unwrap verdict-separation
    stats = result_payload.get("statistics") if isinstance(result_payload.get("statistics"), dict) else result_payload
    recorded = stats.get("method_identity") or result_payload.get("method_identity")
    if isinstance(recorded, dict) and recorded.get("method"):
        return [recorded]
    methods: list[str] = []
    if isinstance(stats.get("methods"), list):
        methods = [str(m) for m in stats["methods"]]
    elif isinstance(stats.get("aggregate"), dict):
        methods = [str(m) for m in stats["aggregate"].keys()]
    elif stats.get("method"):
        methods = [str(stats["method"])]
    return [method_identity(m) for m in methods]
