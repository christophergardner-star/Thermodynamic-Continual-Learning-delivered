All claims verified against HEAD d2f3f9a (branch phase0-stackbridge; note: both harness files carry fresh 2026-07-03 guards â€” Harness A fail-loud allowlist in commit 565d6e9, registry key-collision fix in 33ad4a6 â€” already committed, tree clean).

# SCOUT H â€” HARNESS A/B EQUIVALENCE REPORT

## 1. Concrete protocol diff (same nominal method, A vs B)

Harness A = `tar_lab/multimodal_payloads.py::run_split_cifar10_benchmark` (line 743). Harness B = `tar_lab/generic_cl_runner.py::run_generic_benchmark` (line 575) + `tar_lab/method_registry.py`.

### 1a. Shared-protocol differences (affect every method)

| Axis | Harness A | Harness B | Matchable? |
|---|---|---|---|
| Task split | Fixed `class_order=[[0,1],[2,3],[4,5],[6,7],[8,9]]` from config, identical for every seed (schemas.py:256-258; multimodal_payloads.py:835-854) | Class order **shuffled per seed** via `random.Random(seed).shuffle` (generic_cl_runner.py:173-179) | Yes â€” pass `prebuilt_task_train/test` into B (generic_cl_runner.py:586-587, 632-634) |
| Setting | `task_incremental` default (schemas.py:253): per-task 2-way heads + label remap (multimodal_payloads.py:847-848, 897-898); `class_incremental` optional (shared 10-way head, 891-894) | **Always** shared-head class-incremental: single `nn.Linear(feat_dim, 10)` global labels (generic_cl_runner.py:85-94) | Only by forcing A to `setting="class_incremental"`. Task-IL is IMPOSSIBLE in B â€” but task-IL is the phase10-13 protocol of record |
| Backbone "tiny" | `_CLTrunk`: conv 32/64/64, **no BatchNorm**, fcâ†’128 feat (multimodal_payloads.py:856-871) | `_TinyCNN`: conv 64/128/256 **with BatchNorm**, AdaptiveAvgPoolâ†’256 feat (generic_cl_runner.py:51-69) | No â€” different networks. Use `resnet18`, identical on both sides (multimodal_payloads.py:873-882 vs generic_cl_runner.py:72-82) |
| LR | `base_lr = 0.01` **hardcoded**, not in config (multimodal_payloads.py:941) | default 0.05, overridable via `config_overrides["lr"]` (generic_cl_runner.py:484) | Yes â€” force B `lr=0.01` |
| Optimizer | pluggable `build_optimizer` (sgd/adamw/cruxy), momentum 0.9, wd 1e-4 (multimodal_payloads.py:1019-1028; tar_optimizer_backend.py:65-87); **recreated per task â†’ momentum resets at boundary** (line 1019 inside task loop) | hardcoded `torch.optim.SGD`, momentum 0.9, wd default 1e-4; **created once â†’ momentum persists across task boundary** (generic_cl_runner.py:486) | Partially â€” sgd/momentum/wd match; per-task-vs-persistent momentum lifecycle NOT matchable by config on either side |
| Batch size | config default 64 (schemas.py:260) | default 128 (generic_cl_runner.py:623) | Yes â€” force B `batch_size=64` |
| Epochs | `train_epochs_per_task` (config; schemas.py:259, loop at multimodal_payloads.py:1035); phase10 record used 40 (phase10_baseline.py:37) | `epochs` param, also per-task (generic_cl_runner.py:498) | Yes â€” same semantics |
| Augmentation | default `flip_normalize` = RandomHorizontalFlip only (multimodal_payloads.py:788-795); norm std (0.247, 0.243, 0.261) | flip + **RandomCrop(32, padding=4)** hardcoded (generic_cl_runner.py:137); norm std (0.2470, 0.2435, 0.2616) | Only via prebuilt loaders built with A's transforms; RandomCrop cannot be disabled in B by config |
| DataLoader | num_workers=0, no pin_memory (multimodal_payloads.py:1029-1033) | num_workers=2, pin_memory=True (generic_cl_runner.py:188-195) | Yes via prebuilt loaders |
| Eval protocol | evaluates ALL tasks (incl. unseen) after each task, batch 256 (multimodal_payloads.py:1321-1340) | evaluates seen tasks only, batch 256 (generic_cl_runner.py:519-521, 194) | No metric impact â€” A's unseen-task cells are unused in forgetting |
| **Forgetting formula** | mean of (peakâˆ’final) over **all T tasks including the last (structurally 0)** â†’ divides by T=5 (multimodal_payloads.py:1344-1364) | mean over tasks 0..Tâˆ’2 only â†’ divides by Tâˆ’1=4 (generic_cl_runner.py:216-233) | **Deterministic Ã—1.25 offset** (5-task): `F_B_comparable = F_A Ã— T/(Tâˆ’1)`. Correct analytically |
| Accuracy | mean final per-task acc (multimodal_payloads.py:1365) | same definition (generic_cl_runner.py:540-542) | Identical |
| Seeding | torch/random/np (multimodal_payloads.py:775-780) | same (generic_cl_runner.py:119-126) but seed ALSO drives class order | Matched once loaders are prebuilt |
| Unknown methods | fail-loud allowlist raise (multimodal_payloads.py:751-766) | falls back to LLM-synthesized loading (generic_cl_runner.py:601-612) | n/a |

### 1b. Per-method algorithmic differences (NOT matchable by config â€” measure offset instead)

**EWC** (A: multimodal_payloads.py:1069-1077, 1246-1283; B: method_registry.py:110-163):
- Fisher estimator: A = per-sample gradÂ² of true-class log-prob, capped at ~100 samples/task (1251-1276); B = batch-CE gradÂ² over the **entire** task loader (131-139)
- Normalization: A `/n_samples`; B `/n_batches`
- Parameter scope: A **trunk only** (`trunk.named_parameters`, 1071); B **all params incl. head** (125-129)
- Penalty scale: A `(Î»/2)Â·Î£` (1077); B `Î»Â·Î£` (163) â€” factor-2 divergence at equal Î»
- Î» default: both 1000.0 now (schemas.py:262; method_registry.py:118) â€” but the phase10 record ran Î»=100 (phase10_baseline.py:42), and the plan doc notes phase18 also used 100 (docs/solution_loop_implementation_plan.md:158)

**SI** (A: multimodal_payloads.py:1079-1083, 1155-1160, 1285-1291; B: method_registry.py:168-230):
- Path integral: A accumulates `(-gÂ·(Î¸âˆ’Î¸_task_start)).abs()` per batch (1158-1160) â€” **abs() is non-canonical**; B accumulates signed `âˆ’gÂ·(Î¸âˆ’Î¸_task_start)` (197-199) then `relu()` at task end (213)
- Clamp point: A clamps the running Ï‰ sum (`clamp(min=0)`, 1289); B relu's each task's increment (213) â€” differ whenever a task's increment is negative
- Scope: A trunk only; B all params
- Defaults match: si_c=0.01, si_xi=0.001 (schemas.py:263-264; method_registry.py:177-178)

**tcl_canonical** (A: multimodal_payloads.py:959-971, 995-998, 1101-1104, 1293-1295; B: method_registry.py:586-673):
- Same tcl.py core (ThermalImportance/ThermalMemory; ema_beta 0.99 both â€” tcl.py:122, method_registry.py:622)
- **Î»: A default 0.01 via `tcl_canonical_lambda` (schemas.py:276, multimodal_payloads.py:965-967); B default 1.0 via `tcl_penalty_lambda` (method_registry.py:621) â€” 100Ã— apart, and they read DIFFERENT config keys.** Must be pinned explicitly on both sides in any calibration.

**der_plus_plus**: hyperparams identical (mem 200/Î± 0.2/Î² 0.5 â€” schemas.py:283-285; method_registry.py:241-243); A stores the training-forward logits (1063), B does a fresh `no_grad` forward for reservoir logits (272-274) and a second backward (generic_cl_runner.py:510-513) vs A's single combined backward (1110-1119). Gradient-equivalent, BN-stat non-equivalent.

**lwf**: A weight `lwf_lambda=1.0` (schemas.py:286); B `lwf_alpha=0.5` (method_registry.py:338) â€” different key and value.

**Impossible in B, ever (without code change):** the thermodynamic regime observer + LR governor (methods `tcl`, `tcl_full`) â€” A only (multimodal_payloads.py:919-938, 1169-1191); B's `CLMethod` hooks have no optimizer access (generic_cl_runner.py:486; acknowledged at docs/solution_loop_implementation_plan.md:148-149). **Impossible in A:** `agem`, `tcl_second_order*`, any synthesized method (allowlist, multimodal_payloads.py:756-766); task-incremental is impossible in B. So harness-equivalence can only ever be established for the {sgd, ewc, si, der++, lwf, tcl_canonical} Ã— class-incremental Ã— resnet18 intersection.

## 2. Calibration-run spec (concrete)

**Bridgeable intersection**: dataset split_cifar10, backbone resnet18, setting class_incremental, epochs matched, prebuilt loaders on B.

- A side: `ContinualLearningBenchmarkConfig(setting="class_incremental", seed=s, train_epochs_per_task=E, batch_size=64, ewc_lambda=1000.0, si_c=0.01, si_xi=0.001)`, `method âˆˆ {sgd_baseline, ewc, si}`, `backbone="resnet18"`.
- B side: `run_generic_benchmark(dataset_name="split_cifar10", backbone_name="resnet18", method_name âˆˆ {sgd_generic, ewc_generic, si_generic}, seeds=[s], epochs=E, config_overrides={"lr":0.01,"weight_decay":1e-4,"batch_size":64,"ewc_lambda":1000.0,"si_c":0.01,"si_xi":0.001}, prebuilt_task_train/test=<loaders built with A's fixed class order [[0,1]..[8,9]], A's transforms (flip+normalize, A's constants), num_workers=0>)`.
- Optional 4th pair: `tcl_canonical` on both with Î» pinned (`tcl_canonical_lambda=0.01` in A **and** `tcl_penalty_lambda=0.01` in B).
- Seeds: n=10 â€” `[42, 0, 1, 2, 3, 123, 456, 789, 1337, 7]` (superset of phase10 seeds phase10_baseline.py:35 and BaselineComparisonPlan defaults schemas.py:315).
- Epoch grid: main grid E=5 (schema default); spot-check E=40 (protocol-of-record regime) for {sgd, ewc} Ã— seeds {42, 0, 1}.
- Post-hoc correction applied before comparison: `F_A_corrected = F_A Ã— 5/4`.
- Prereg must record the residual non-matchables: momentum lifecycle, EWC estimator/scope/Î»-halving, SI abs-vs-relu/scope. The caveat requirement already exists at docs/solution_loop_implementation_plan.md:155-158.

**GPU-hours on the GTX 1650 (4GB)** â€” anchored to the system's own estimate `estimated_runtime_h=8.0` for 5 seeds Ã— 40 epochs/task Ã— resnet18 (tar_experiment_orchestrator.py:236-238) â†’ ~1.6 h/seed at E=40, ~0.2 h/seed at E=5:
- Main grid: 3 methods Ã— 10 seeds Ã— 2 harnesses Ã— ~0.2 h â‰ˆ **12 h** (+~10% for B-side full-loader Fisher)
- E=40 spot-check: 2 Ã— 3 Ã— 2 Ã— ~1.6 h â‰ˆ **19 h**
- Total â‰ˆ **13-32 GPU-hours**; VRAM ~1.5 GB fits the 4GB card, but must NOT run concurrently with the live HPC training run (pid 12560).

## 3. Result persistence / can calibration use existing writers?

**Yes**, if both sides run through the orchestrator:
- Harness A via `_run_cifar10` (tar_experiment_orchestrator.py:2389-2505) and Harness B via `_run_generic_cl` (:2772-2861, dispatched by `runner_key=="generic_cl"` :2178) both flow into `_build_result` (:3006) â†’ `_save_result` (:3421-3499) â†’ `tar_state/experiments/<spec.id>/result.json` + `spec.json` + env sibling + TL-1 3-gate verify/quarantine. `_exp_dir = workspace/tar_state/experiments` (:406); 25 experiment dirs exist at E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state\experiments.
- Both carry per-seed `{"seed","forgetting","accuracy"}`; B adds `{bwt, fwt, intransigence_index, forgetting_per_task, ece_trajectory, bn_reset_at_boundaries, dpr_per_task}` (generic_cl_runner.py:666-681); A adds `{"comparisons": {...}}` (orchestrator:2434-2439). A comparison script reads paired result.json files, disambiguates harness by `runner_key` in spec.json, applies the Ã—5/4 correction to A-side forgetting.
- Phase scripts of record (Harness A) instead write `tar_state/comparisons/<name>__<stamp>.json` + `_env.json` + `canonical_results_index.jsonl` (tar_lab/result_artifacts.py:235-291; 111 files exist in E:\...\tar_state\comparisons). Whether standalone B scripts (run_hyperparameter_selection.py, run_mechanistic_ablation.py) all route through the canonical writer: UNKNOWN (not verified line-by-line).
- Caveat: `_build_result` computes delta/p/verdict vs a TCL baseline (:3018-3051) â€” irrelevant noise for calibration; use the raw seed_results.

## 4. Proposed acceptance criterion for "harness-equivalent"

Per method m, on n=10 paired seeds, d_i = F_B,i âˆ’ (5/4)Â·F_A,i:
1. **Primary (equivalence)**: TOST with margin Îµ = 0.02 absolute forgetting (system-native meaningful-effect size â€” the ADVERSE threshold at orchestrator:3048), both one-sided p < 0.05. Same test on accuracy with Îµ = 0.02.
2. **Ordering invariance**: method ranking on mean forgetting {sgd, ewc, si} identical across harnesses (Kendall Ï„ = 1.0).
3. **Fallback (calibrated offset)**: if TOST fails but sd(d) < 0.01, declare harnesses "offset-equivalent" with recorded b_m = mean(d); all cross-harness claims must apply b_m and carry the protocol-equivalence caveat (per docs/solution_loop_implementation_plan.md:155-158). If sd(d) â‰¥ 0.01 the harnesses are NOT equivalent for method m and cross-harness numeric comparison is prohibited.
The Îµ=0.02 margin should be sanity-checked against actual phase10 per-method forgetting std before prereg (values live in tar_state comparisons artifacts; not read in this pass â€” UNKNOWN).

**Key files**: C:\Users\cgard\TAR\Thermodynamic-Continual-Learning-delivered\tar_lab\multimodal_payloads.py, tar_lab\generic_cl_runner.py, tar_lab\method_registry.py, tar_lab\schemas.py (config defaults 251-290), tar_lab\method_identity.py, tar_optimizer_backend.py, tar_experiment_orchestrator.py, phase10_baseline.py, docs\solution_loop_implementation_plan.md.
