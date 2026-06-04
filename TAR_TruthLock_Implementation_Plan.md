# TAR — Truth-Lock Implementation Plan (Keystone #1: enforcement before autonomy)

**Authored:** 2026-06-04 · **Status:** build-ready · **Branch:** phase0-stackbridge
**Companion:** `TAR_Autonomy_Unification_Roadmap.md` (move #1), `TAR_Phase0_and_StackBridge_Implementation_Plan.md`
**Derivation:** 10-agent read-only audit; all line numbers verified against current source.
Code root: `C:\Users\cgard\TAR\Thermodynamic-Continual-Learning-delivered`. State: `E:\TAR\...\tar_state`.

---

## 0. Why this is keystone #1, scope, guardrails

**The problem (verified):** TAR's truthfulness rests on a hand-maintained file, not its rails. The 3-gate verifier
`register_canonical_result` (provenance + manifest-hash + deterministic recompute) **is never called — 0 of 51 indexed
results passed it**; the live writer defaults `publication_allowed=True` with no checks; `validation_state.json`
reports **11** publication-allowed while the honest inventory says **1**, and the system acts on the 11; the
human-veto re-check **fails OPEN**; and nothing checks **method identity**, so the uniform-L2 proxy is counted (and
published) as the canonical TCL algorithm.

**Truth-lock makes verified-truth the only path to "counts as a result."** It must land *before* any autonomy
increase (frontier minting, synthesis, full-autonomy ramp), because an autonomous system that cannot verify its own
results will generate false claims faster.

### Hard guardrails
- 🚫 Do not touch the running experiments (HP Selection / HPC resume), the GPU, `execution_enabled.flag`,
  `daemon_paused.flag`, or any checkpoint. All tasks here are CPU/code-only.
- These edits change the **daemon** (`tar_experiment_orchestrator.py`, `tar_lab/*`) and the **dashboard**
  (`validation` read paths). They take effect on the **next daemon/dashboard restart** (debug=False, no reloader) —
  the running processes hold old code. State this when handing off.
- Roll out behind a **backfill + dry-run** (TL-7) so we see exactly which existing results pass/fail before enforcing.
- Keep all existing append-only writes intact — truth-lock adds a *verification + labeling* layer; it never deletes
  results. A result that fails verification is **quarantined / marked non-canonical**, not destroyed.

### Verified preconditions
- `_save_result` writes to `self._exp_dir / spec.id / "result.json"` where
  `self._exp_dir = workspace/"tar_state"/"experiments"` (`tar_experiment_orchestrator.py:406`). This **exactly matches**
  `register_canonical_result`'s required layout `workspace/tar_state/experiments/<run_id>/result.json`
  (`canonical_registry.py:447-456`) — so mandatory verification is drop-in compatible.
- Method identity is derivable from `multimodal_payloads.py:897-904` (`_TCL_CANON = {"tcl_canonical","tcl_full"}`;
  `method=="tcl"` is the documented uniform-L2 **proxy**).

---

## TL-1 — Make canonical verification MANDATORY (the core change)

**Where:** `tar_experiment_orchestrator.py:3346-3362`, end of `_save_result`. Current code:
```python
        # Auto-register in canonical JSONL index for immediate publication eligibility.
        try:
            from tar_lab.result_artifacts import write_canonical_comparison_result
            from tar_lab.phase_catalog import phase_catalog_by_logical_name
            _cat = phase_catalog_by_logical_name().get(spec.id)
            write_canonical_comparison_result(... )   # NO GATES; publication_allowed defaults True
        except Exception:
            pass
        return path
```
This is the bug: a no-gate writer wrapped in `except: pass`, so any result (incl. a crashed/proxy one) becomes
"canonical, publication_allowed=True".

**Change:** after the append-only write, run the 3-gate verifier and record the outcome; only a *passing* result is
indexed as canonical. A failing result is recorded as **quarantined** (kept append-only, excluded from publication).
```python
        # Canonical verification (3-gate) is MANDATORY before a result counts.
        from tar_lab.canonical_registry import (
            register_canonical_result, CanonicalGateError,  # add CanonicalGateError export if absent
        )
        try:
            audit = register_canonical_result(path)        # gate1 sibling + gate2 manifest-hash + gate3 recompute
            self._log(f"[canonical] VERIFIED {spec.id}: env_hash={audit.get('env_snapshot_hash','')[:12]} "
                      f"recompute_ok=True")
        except Exception as exc:
            # Do NOT index as canonical. Mark quarantined so the queue/paper-gate cannot treat it as validated.
            self._mark_result_quarantined(path, reason=f"canonical_gate_failed: {exc}")
            self._log(f"[canonical] QUARANTINED {spec.id}: {exc}")
        return path
```
- Add a small `_mark_result_quarantined(path, reason)` helper that writes a sibling `result_quarantine.json`
  `{quarantined: true, reason, at}` and (optionally) moves/links the result under `comparisons/_untrusted/` so the
  existing `classify_trust_tier` quarantine path (`validation.py:263`) catches it.
- `register_canonical_result` already writes the audit record with `recomputed_aggregates` + `env_snapshot_hash`
  (`canonical_registry.py:471-473`) — these become the proof-of-verification fields the rest of truth-lock keys on.
- **Keep** `write_canonical_comparison_result` for the human-readable comparison artifact, but it must no longer be
  the thing that confers publication eligibility (see TL-2).

**Risk/mitigation:** if many legitimate autonomous results don't pass gate-2 (manifest committed) or gate-3
(recompute), enforcement could over-quarantine. → TL-7 backfills first to measure; gate failures are logged, not
crashes (the experiment result is still saved).

---

## TL-2 — Flip `publication_allowed` default to False

**Where:** `tar_lab/result_artifacts.py:273`:
```python
"publication_allowed": True if publication_allowed is None else bool(publication_allowed),
```
**Change to:**
```python
# Publication eligibility is NOT granted by this writer. It is conferred only by the
# canonical 3-gate (register_canonical_result) + classify_trust_tier (TL-4). Default deny.
"publication_allowed": False if publication_allowed is None else bool(publication_allowed),
```
Also audit `result_artifacts.py:357` (`"publication_allowed": env_is_real`) — env-present must NOT alone imply
publishable; downgrade it to `False` (let TL-4 decide). Net: nothing is publication-eligible by mere existence.

---

## TL-3 — Method-identity gate (proxy can never be labeled canonical)

**Goal:** a result produced by the uniform-L2 **proxy** (`method=="tcl"`) must never be counted or rendered as the
**canonical** gradient-energy algorithm. Today nothing records or checks which algorithm actually ran.

**3a — Record method identity at the source.** In `multimodal_payloads.py` `run_split_cifar10_benchmark`, add to the
returned result a `method_identity` dict:
```python
method_identity = {
    "method": method,
    "family": "tcl" if method in {"tcl","tcl_canonical","tcl_full","tcl_penalty_only"} else method,
    "is_canonical_tcl": method in _TCL_CANON,          # {"tcl_canonical","tcl_full"} only
    "uses_uniform_l2_proxy": method in {"tcl","tcl_penalty_only"},
    "uses_regime_observer": method in _TCL_OBS,
}
```
Propagate it into the `ExperimentResult` (add a `method_identity: dict` field, default `{}`) and persist it in
`result.json` and the env snapshot (`result_artifacts.collect_environment_snapshot` `extra=`).

**3b — Gate at the verifier.** Add a 4th check inside/after `register_canonical_result` (or in `classify_trust_tier`,
TL-4): if a result's `method` label is in the canonical TCL family BUT `method_identity.uses_uniform_l2_proxy` is
True, refuse canonical/publication tier with basis `"method_identity_proxy_mismatch"`. Concretely: a result tagged
`method="tcl"` may be reported as **"TCL (uniform-L2 proxy)"** but is **not** allowed to back a claim about the
canonical algorithm.

**3c — Gate at authoring.** In `tar_author.py` `_paper_gate_context` / the renderer, refuse to render a result whose
`method_identity.uses_uniform_l2_proxy` is True under canonical-algorithm prose/equations; require an explicit
"(uniform-L2 proxy, not canonical importance accumulator)" disclosure, or require `is_canonical_tcl`. This closes the
one place the compiled paper currently overclaims.

---

## TL-4 — Make `validation_state` honest (collapse 11 → 1)

**Where:** `tar_lab/validation.py:287-290`, `classify_trust_tier`:
```python
publication_allowed = (
    trust_tier in TRUST_PUBLICATION_ALLOWED
    and seed_count >= MIN_SEEDS_FOR_PUBLICATION
)
```
This grants publication on (tier + ≥5 seeds) alone — no canonical verification, no multiple-comparison correction,
no method-identity. **Change to require all of:**
```python
canonical_verified = bool(record.get("env_snapshot_hash")) and bool(record.get("recomputed_aggregates"))
method_ok = not (result_payload or {}).get("method_identity", {}).get("uses_uniform_l2_proxy", False) \
            or not _claims_canonical_tcl(record)          # proxy may stand only for non-canonical claims
family_corrected_sig = bool((result_payload or {}).get("family_wise_significant", False))  # Bonferroni/Holm at the claim level
publication_allowed = (
    trust_tier in TRUST_PUBLICATION_ALLOWED
    and seed_count >= MIN_SEEDS_FOR_PUBLICATION
    and canonical_verified            # TL-1 proof-of-verification present
    and method_ok                     # TL-3 method identity
    and family_corrected_sig          # multiple-comparison correction at the claim level
)
```
- `canonical_verified` keys on the audit fields TL-1 writes — so only results that passed the 3 gates qualify.
- `family_wise_significant` must be set by the result/claim builder using the existing `bonferroni_correct`/Holm in
  `stat_utils.py` over the claim's comparison family (the honest inventory already does this by hand). Until a result
  carries it, it is not publication-grade — which is the correct, conservative default.
- Expected effect: the machine `trusted_publication_allowed` count drops from 11 toward **1**, matching
  `honest_evidence_inventory.json`. Reconcile by spot-checking the two agree post-change.
- Also fix the contradiction where `publication_allowed: True` coexists with unresolved `preregistration_missing`
  issues — make an open prereg issue force `publication_allowed=False`.

---

## TL-5 — Fail the human-veto re-check CLOSED

**Where:** `tar_experiment_orchestrator.py:1778-1784`:
```python
        if self._autonomous:
            try:
                from tar_lab.human_review import approved_experiment_ids
                _is_approved = str(spec.id) in approved_experiment_ids(self.workspace)
            except Exception:
                _is_approved = True          # FAILS OPEN
```
**Change to fail closed (refuse + auditable note, not deadlock):**
```python
            except Exception as exc:
                _is_approved = False          # FAIL CLOSED: an approval-subsystem error must not auto-approve
                self._log(f"[veto_gate] approval lookup failed; failing closed: {exc}")
```
The existing `if not _is_approved:` branch already writes a refuse-note and blocks — so failing closed yields an
auditable refusal, not a silent hang. (Trade availability for integrity: the correct choice for truth-lock.)

---

## TL-6 — Provenance hardening (manifest re-verify + anchors)

- **Re-verify manifest hash on canonical read**, not only at write: call `_check_gate_2_manifest`
  (`canonical_registry.py:239`) when loading a result for publication, so a manifest changed after the snapshot is
  caught.
- **Anchors:** `anchors/` is empty on both drives. Either populate a tamper-evident root (hash-chain the canonical
  index) or remove anchor claims from docs so provenance promises match reality. (Lower priority; do after TL-1..5.)

---

## TL-7 — Rollout: backfill dry-run first, then enforce

1. **Dry-run/backfill (read-only):** run `register_canonical_result` over the 51 existing
   `canonical_results_index.jsonl` entries in a report-only mode; record pass/fail + reason per result. Expectation
   from the audit: ~1 passes (matching the honest inventory). This tells us exactly what enforcement will quarantine
   before it does so.
2. **Reconcile:** confirm the dry-run's "pass" set ≈ `honest_evidence_inventory.json`'s publication-allowed set. If
   legitimate results fail gate-2/3 for fixable reasons (e.g. missing env sibling), fix the producer, not the gate.
3. **Enforce:** land TL-1..TL-5; restart daemon + dashboard so the new code is live.
4. **Verify post-enforce:** `build_validation_state` `trusted_publication_allowed` ≈ 1; the proxy result is labeled
   "(uniform-L2 proxy)" and cannot back a canonical claim; a deliberately-corrupted result is quarantined; veto-lookup
   error blocks (not approves).

---

## Acceptance tests (new `tests/test_truth_lock.py`)
- **TL-1:** a result with a valid env sibling + committed manifest + recomputable aggregates → indexed canonical
  (audit has `env_snapshot_hash`+`recomputed_aggregates`); a result missing the env sibling → **quarantined**, not indexed.
- **TL-1 recompute:** a result whose stored aggregate ≠ recompute-from-seed_results (tamper) → gate-3 fails → quarantined.
- **TL-2:** `write_canonical_comparison_result(..., publication_allowed=None)` → record has `publication_allowed=False`.
- **TL-3:** `classify_trust_tier` on a `method="tcl"` (proxy) result claiming canonical → `publication_allowed=False`,
  basis mentions method identity; same result framed as "proxy" → allowed (subject to other gates).
- **TL-4:** a result that is trusted+≥5 seeds but lacks `family_wise_significant`/`canonical_verified` →
  `publication_allowed=False`. Synthetic set reproduces the honest "1 of N" count.
- **TL-5:** monkeypatch `approved_experiment_ids` to raise → autonomous `_execute` refuses (not approves).

---

## Task board (Lab ownership; all CPU; effective on next restart)

| ID | Task | Owner | File:line | Window |
|---|---|---|---|---|
| TL-1 | Mandatory `register_canonical_result` + quarantine helper | Integrity + Infra | orchestrator.py:3346-3362, +`_mark_result_quarantined`; canonical_registry export | NOW-SAFE |
| TL-2 | `publication_allowed` default False | Integrity | result_artifacts.py:273, :357 | NOW-SAFE |
| TL-3 | Method-identity record + gate + authoring guard | Integrity + CL/Thermo Sci + Authoring | multimodal_payloads.py:897-904 (emit), orchestrator ExperimentResult, tar_author.py gate | NOW-SAFE |
| TL-4 | `classify_trust_tier` honest criteria | Methodologist + Integrity | validation.py:287-290 | NOW-SAFE |
| TL-5 | Veto fail-closed | Alignment | orchestrator.py:1782-1783 | NOW-SAFE |
| TL-6 | Manifest re-verify on read + anchors | Provenance | canonical_registry.py:239; anchors/ | NOW-SAFE (after 1-5) |
| TL-7 | Backfill dry-run + reconcile + enforce + restart | PI + Integrity | new report script; restart daemon+dashboard | DRY-RUN now; ENFORCE on go |
| — | `tests/test_truth_lock.py` | Integrity | new | NOW-SAFE |

**Definition of done:** no result is labeled canonical / publication-allowed unless it passed the 3 gates, is
method-identity-correct (not a proxy masquerading as canonical), and is family-wise significant; the machine
publication count equals the honest inventory; the veto fails closed; and the compiled paper cannot present the proxy
as the canonical algorithm. Only then is it safe to begin turning on autonomy (moves #2–#5 of the roadmap).
