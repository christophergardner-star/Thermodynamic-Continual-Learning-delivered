# TAR Tier-3 — RunPod Result Ingestion Scope (2026-06-09)

**Goal:** make TAR *reason over* the confirmatory RunPod results (hpc_replication, hpc_lambda,
phase16), not merely count them toward the autonomy gate.
**Non-goal (hard truth-lock line):** do NOT confer canonical / `publication_allowed` / SoTA status
on results that have not genuinely passed verification. The point is to let the director *think with*
the numbers at their *honest* trust tier — not to fake a verified claim.

---

## 1. Current state — where the results are vs. who consumes them

| Store | Has the 3 results? | Consumer | Verdict |
|---|---|---|---|
| `comparisons/*.json` + `_env.json` | ✅ raw | — | on disk |
| `experiment_queue.json` | ✅ registered `complete` | autonomy ramp gate (`_phase2_status`) | **used → gate 4/6** |
| `literature_graph.db` research_gaps | ✅ correcting gap appended | director frontier/recall | **used (direction)** |
| **Vault recall** (`index_experiment_result`) | ❌ | director evidence recall | **MISSING — the target** |
| **SoTA table** (`_ingest_tar_results`) | ❌ | NoveltyGate / SoTA bar | correctly blocked (see §3) |

Result schemas differ from the canonical comparison schema:
- **phase16**: `method_results{method→{seed_results, mean_forgetting, std_forgetting, ci95, mean_accuracy, …}}`,
  `comparisons`, `gate_a_passed/gate_b_passed=True`, `trust_tier="trusted_rerun"`, `honest_verdict`. *(no `aggregate`, no `publication_allowed`)*
- **hpc_replication**: `per_seed_results`, `hpc_forgetting_mean`, `wilcoxon_p`, `cohens_d`, `verdict=REPLICATION_SUCCESS`, `sprt_final_decision`.
- **hpc_lambda**: per-condition means, `decision=LAMBDA_IS_MECHANISM`, `comparisons.primary_hpc_vs_high_lambda`.

---

## 2. The verification gate (why SoTA can't just be "fed")

SoTA ingest (`tar_evidence_ingest._ingest_tar_results`:1912) requires a `canonical_results_index.jsonl`
record with `publication_allowed=True`. That flag is **default-deny** (`write_canonical_comparison_result`:277,
TL-2) and is only conferred by `validation.classify_trust_tier` (TL-4) **after** `verify_canonical_3gate`
(`canonical_registry.py`:505):
- **Gate 1** — env sibling has `git.head` + `manifest_hash`.
- **Gate 2** — manifest committed in the code repo + hash match.
- **Gate 3** — deterministic seed/sweep **recompute** matches.

## 3. Why the RunPod results are *correctly* blocked (measured, not assumed)
Running `verify_canonical_3gate` on phase16 returns **`False | gate1: result_env.json is not valid JSON`**. Three real blockers, in order:
1. **Gate 1 — env sibling is empty/invalid.** The retrieved `_env.json` is 0-byte. The phase-rerun scripts write
   their own minimal env; the bridge's `_retrieve` sibling-fetch produced an empty file. → no `git.head`/`manifest_hash`.
2. **Gate 2 — pod provenance is not git.** The pod runs a synced `rglob` working copy (871 files), not a git clone,
   so there is no committed-manifest hash to match.
3. **Gate 3 — hardware-sensitivity defeats deterministic recompute.** Measured this session: the HPC effect was
   d=-0.44 on the 1650 vs d=-0.975 on the L40 — *same seeds, different hardware*. A local recompute will NOT match
   the pod numbers, so Gate 3 fails by design.

**Conclusion:** SoTA/`publication_allowed` for these results is not a missing-adapter problem — it is the
truth-lock gate working as intended. Forcing it would require bypassing all three gates. **Out of scope.**

---

## 4. The achievable, honest target — VAULT RECALL at `trusted_rerun` tier
The vault (`tar_lab/memory/vault.index_experiment_result`, Seam 1) is a **reasoning/recall aid, not a verified
claim**. Indexing the results there — tagged `trust_tier="trusted_rerun"`, `publication_allowed=False` — lets the
director recall and reason over the numbers *without overclaiming*. This is the thing that makes TAR "think with" them.

**C1 — result-type adapters** (`scripts/ingest_runpod_results.py`, new):
| Result | → record(s) for the vault | Notes |
|---|---|---|
| phase16 | one record **per method** (`method`, `dataset=split_cifar100`, `result.mechanism_forgetting`=seed_results, mean/std/ci, verdict from `honest_verdict`) | maps `method_results` → the `mechanism_forgetting` shape `index_experiment_result` expects |
| hpc_replication | one record (`method=high_penalty_conservative`, `dataset=split_cifar10`, deltas, `verdict=REPLICATION_SUCCESS`, p/d/SPRT) | a *replication*, not method-vs-method |
| hpc_lambda | one record (`method=hpc_vs_tcl_high_lambda`, ablation conditions, `decision=LAMBDA_IS_MECHANISM`) | mechanism finding |

Each record carries explicit `trust_tier="trusted_rerun"`, `publication_allowed=False`, `provenance="runpod_bridge_unverified"`.

**C2 — trigger:** a one-off backfill over the 3 registered `result_path`s, then wire `register_phase2_result`
(scripts/run_phase2_on_runpod.py) to also index into the vault at registration time (so future bridge results
self-feed recall). The SoTA path is intentionally untouched.

**C3 — verify:** after backfill, the daemon's `vault recall-index` count moves 6 → 9; director recall returns the
HPC/λ numbers. No SoTA/`publication_allowed` change (assert it stays 6 entries).

---

## 5. Provenance fixes (raise FUTURE pod results toward verifiability)
- **Fix the env-sibling capture/retrieval** so `_env.json` is non-empty with `git.head` + `manifest_hash`
  (bridge `_retrieve` validates the sibling; capture git.head/manifest at dispatch and inject into the pod env). → unblocks Gate 1/2.
- This does NOT solve Gate 3.

## 6. The deep open question (decision needed, NOT a quick build)
**Gate 3 + hardware-sensitivity.** To ever make pod results *canonical*, one of:
- (a) **Remote canonical recompute** — run Gate-3's deterministic recompute on a *matching* pod (the bridge already
  exists); compare within a tolerance band. Significant new machinery.
- (b) **A hardware-pinned canonical tier** — `canonical_on(<gpu>)` that records the hardware and verifies only on it.
  Weaker than hardware-invariant canonical; needs a truth-lock policy decision.
- (c) **Accept `trusted_rerun` as the ceiling** for bridge results — they inform recall + gaps, never the SoTA bar.
  Simplest + most honest given hardware-sensitivity is a real scientific caveat.

Recommendation: **(c)** for now — it matches reality (the effect *is* hardware-sensitive, so "canonical" would be
overclaiming). Revisit (a) only if a hardware-invariant claim becomes necessary for the paper.

---

## 7. Effort + sequencing
- **C1+C2+C3 (vault recall + backfill + wire):** ~1 day. **Delivers the goal** (TAR reasons over the results, honestly).
- **§5 provenance fixes:** ~½ day. Improves future runs; independent.
- **§6 gate-3:** design decision first; (a) is multi-day, (c) is zero-build.

## 8. Truth-lock guardrails (binding)
1. Vault records MUST carry `trust_tier="trusted_rerun"`, `publication_allowed=False`. Never imply verified.
2. NEVER write these to `canonical_results_index.jsonl` with `publication_allowed=True` (would enter the SoTA bar unverified).
3. The KG representation stays the **gap** (research direction) already appended — not a SoTA entry.
4. Recall surfacing in any paper draft must label them "trusted_rerun, pending canonical verification".
