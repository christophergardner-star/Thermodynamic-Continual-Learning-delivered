# TAR — Autonomy Unification Roadmap (one complete autonomous research platform)

**Authored:** 2026-06-04 · **Source:** independent 10-agent READ-ONLY capability audit + this session's findings.
**Branch:** phase0-stackbridge. **Companion specs:** `TAR_TruthLock_Implementation_Plan.md` (keystone #1),
`TAR_Phase0_and_StackBridge_Implementation_Plan.md` (keystone #2). **Provenance:** derived bottom-up from code,
runtime state, and DB row counts — not from aspirational plan docs.

---

## 1. Verdict (capability vs actuation)

**TAR is an exceptionally well-engineered scaffold in which almost every advanced capability exists as real,
working code — but the loops that would make those capabilities compound and self-drive are unwired or dormant.**
Today it is a powerful research *harness* cycling a fixed menu, not yet an autonomous researcher. The good news:
~80% of the work to make it real is **connecting existing components and turning on enforcement**, not greenfield.

Direct answers (ground truth):
- **Find real novel gaps?** Machinery is real and arXiv-grounded, but the director's decision tables
  (`research_gaps`, `sota_entries`, `method_benchmark_coverage`) are **0 rows**; 91 gap scans → **1 gap**
  (finance, auto-rejected); `frontier_problems.json` = the **5 hardcoded defaults only**. Net: finds ~nothing usable.
- **Synthesize novel algorithms?** `method_synthesizer.py` is a real LLM→code→AST→sandbox→minibench pipeline that
  **has never produced one artifact**. Every method ever run is a built-in or a **hyperparameter variant**.
- **Run novel + accurate experiments that count?** Experiments are **genuinely real** (real ResNet/CIFAR, real
  PennyLane, real graphs) and **refuse rather than fabricate** when unsupported — but "counting" is broken: the
  3-gate `register_canonical_result` is **never called (0/51 verified)**; the writer defaults `publication_allowed=True`.
- **Everything wired correctly?** No — the loops are open (Section 3).
- **How truthful?** Honest in human-written narration (`honest_evidence_inventory.json` says **1** publication-grade
  result and self-corrects), but the **machine state says 11 and the system acts on the 11**. The one compiled paper
  presents the **uniform-L2 proxy** as the canonical algorithm. **Truthfulness rests on a hand-maintained file, not
  the rails.**

---

## 2. What IS real and good (build on these)
- Real multi-domain executors (CL/CIFAR, **quantum/PennyLane**, graph/torch_geometric, tabular/sklearn-OpenML) that
  **refuse** (`status="failed"`, empty metrics) rather than fabricate.
- A rigorous statistics library (t-CIs, bootstrap, Wilcoxon, Bonferroni/Holm, NCT power, corrected SPRT).
- A real, tested **`science_exec` bridge** (just added) that runs any domain through Stack-A's rail stack.
- A real **3-gate canonical verifier** (`register_canonical_result`: provenance + manifest-hash + deterministic recompute).
- A real **algorithm synthesizer** and **gap detector**.
- A real authoring engine (LaTeX→PDF) with citation allow-lists and number-verification, and a hard validation gate.
- Real provenance capture (env snapshots, manifests) and a fully-built (disabled) **RunPod** offload path.

The pattern: **excellent components, dormant wiring/enforcement.**

---

## 3. The 7 broken seams

| # | Seam | Status | Root cause |
|---|---|---|---|
| 1 | Two loops not unified | bridge never triggered | no code emits `runner_key="science_exec"`; idea-engine `scan_frontier_gaps` in unsupervised Stack-B |
| 2 | Gap discovery finds nothing | grounded but empty | decision tables 0 rows; lexical-only (no embedder); `_FRONTIER_AUTONOMY_DOMAINS` empty; gap→registry contract mismatch |
| 3 | Synthesis never fires | real but dormant | director only emits `tcl` λ-tweaks; synthesizer is unknown-key-fallback; refinement proposals never actuated |
| 4 | Memory doesn't compound | write-only/orphaned | results never written back; VectorVault orphaned from live loop; director reads only the empty literature tables |
| 5 | Self-improvement doesn't compound | penalty-only | 2 of 6 learners wired, both penalty-only; operator-LoRA non-functional (GAP-1, synthetic data, missing adapter) |
| 6 | Truthfulness enforcement bypassed | honest-by-human | 3-gate verifier never called; `publication_allowed` defaults True; veto fails OPEN; method-identity unchecked; proxy-tcl in paper |
| 7 | Compute under-utilized | local-serial | RunPod built but `enabled:false`; no CPU parallelism (16 threads/64GB idle); seeds sequential |

---

## 4. Target architecture — one closed, truthful, self-driving loop

```
 literature ingest + INTERNAL RESULTS (compounding memory: write results back)
        ▼
 gap-scan → mint frontier (grounded, guarded, per-domain opt-in) → propose method (synthesize if novel, else sweep)
        ▼
 pre-register → run experiment (ANY domain via science_exec bridge)
        ▼
 VERIFY (canonical 3-gate: recompute + manifest + METHOD-IDENTITY)   ← only verified results "count"
        ▼
 write back to memory → learn (priors that REWARD good directions) → author (validated-only, proxy-aware) → next cycle
        ─────────────── all under armed rails, ONE supervised entrypoint ───────────────
```

---

## 5. The five unification moves (dependency order)

1. **Truth-lock FIRST (enforcement before autonomy).** Make `register_canonical_result` mandatory; flip
   `publication_allowed` default to False; add a **method-identity gate**; make `validation_state` use the same
   criteria as `honest_evidence_inventory` (collapses 11→1); fail the veto **closed**. *Detailed in
   `TAR_TruthLock_Implementation_Plan.md`. Rationale: an autonomous system that cannot verify its own results will
   generate false claims faster.*
2. **Close the science loop (keystone).** Director emits `runner_key="science_exec"` directives for non-CL domains
   (bridge already built+tested) **and** writes internal results back into the recalled memory store. Unifies the two
   stacks; makes memory compound.
3. **Turn on real discovery.** Repopulate literature decision tables (retry/backoff for 429s); load the embedder;
   seed per-domain benchmark tables; bridge gap-candidates to the registry contract; opt **one** domain into
   `_FRONTIER_AUTONOMY_DOMAINS` behind the veto window.
4. **Activate synthesis + compounding.** Wire `method_refinement_engine` → `method_synthesizer` → benchmark →
   canonical-verify → (human-approved) adopt, routed through `append_only_guard`; wire the 4 dormant learners to
   **reward** good directions, not only penalize bad ones.
5. **One supervised entrypoint + truthful authoring.** A single `--platform` launcher (watchdog→daemon→queue→dashboard)
   and an authoring guard that forbids rendering a proxy result under canonical-algorithm prose.

---

## 6. The 2-month plan (4 GB GPU + 64 GB / 16-thread CPU)

- **Tier 0 — Truth-lock (CPU, now):** move #1. Nothing else's output is trustworthy until this is on.
- **Tier 1 — Make the science real (GPU, sequential, after live runs):** finish **HPC n=20**; run **phase18
  `tcl_full`** (first time the *named* algorithm is measured); then re-derive/relabel every "tcl" number.
- **Tier 2 — Unify + breadth (mostly CPU):** moves #2–#3 (emit the bridge, write results back, turn on discovery for
  continual_learning → quantum_ml → thermodynamic_ml); **parallelize CPU seed loops (~5–8× on idle threads)**.
- **Tier 3 — Compounding (CPU):** wire the dormant learners to reward good directions; close result→memory→gap/method.
- **Defer to RunPod/3090:** operator-LoRA (Qwen-7B, 14 GB), ViT/large backbones, CIFAR-diffusion; finance (needs
  FI-2010 adapter + leakage-controlled splits); medical (needs profile + human-cleared PhysioNet DUA).

## 7. The RTX 3090 unlock (~2 months)
Enables the operator-LoRA loop (passes the 14 GB gate automatically), larger backbones/batches, full-speed suites.
**Mostly config:** nothing hardcodes 4 GB — the scheduler reads `vram_total_gb` live and auto-scales. Edits needed:
raise the per-dataset VRAM budget map (`tar_scheduler.py:27-32`), optionally lift the single-GPU-slot cap, and
implement **GAP-1** (`evaluate_eval_pack` inside `run1()`) + **replace the 29 synthetic LoRA signals with signals
mined from TAR's own results**. RunPod stays the >24 GB / many-parallel overflow valve (`enable` + API key).

## 8. Operating model & alignment ordering
Keep the 10-agent Lab as the development team; the **Integrity Officer is now the most load-bearing role**.
**Non-negotiable:** *enforcement before autonomy* — every autonomy increase (frontier minting, synthesis, full-autonomy
ramp) is preceded by its truth-lock and gated per-domain behind the veto window. Kill-switches
(`daemon_paused.flag`, `execution_enabled.flag`) and the manifest gate stay armed throughout.

## 9. Bottom line + first three actions
TAR is **not** a fully functional autonomous researcher today, but can become one unusually cheaply — the hard
infrastructure exists; the work is **connect + enforce + run a few honest experiments**. First three (CPU, none touch
running experiments):
1. **Truth-lock** (move #1) — `TAR_TruthLock_Implementation_Plan.md`.
2. **Write results back to memory + emit the science_exec bridge** (move #2).
3. **Parallelize CPU seeds + wire the dormant learners.**
