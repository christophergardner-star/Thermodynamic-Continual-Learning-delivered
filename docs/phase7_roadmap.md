# Phase 7 Roadmap — External Scientific Credibility

This document is the authoritative plan for making TAR externally reviewable
on one real scientific domain.

## Core Reframe

Phase 5 closes the autonomous loop.
Phase 6 makes TAR frontier-capable and self-positioning.

Neither phase proves that TAR produces results that matter to a domain expert
outside this project.

The proxy benchmarks (digits, karate) were necessary to validate the machinery.
They are not sufficient to justify scientific claims. An external reviewer has
no baseline for "digits_optimizer_validation loss 1.31" — it is not in the
literature, it has no comparison, and it cannot be independently checked.

Phase 7 closes that gap. The objective is one reviewer-grade, reproducible
research package in a real scientific domain, generated and managed by TAR
end-to-end.

## Core Scientific Claim

**Phase 7 is not about proving TAR is best. It is about proving TAR can produce
honest, reproducible, externally reviewable results on a standard continual-
learning benchmark.**

If TCL wins: the thermodynamic governor's regime-awareness reduces catastrophic
forgetting on a benchmark that external reviewers already understand, with
statistical grounding they can verify independently.

If TCL does not win: TAR produced an honest, well-characterised null result on
a real benchmark — which is itself scientifically valuable and far more credible
than an internally validated proxy score.

Either outcome closes the credibility gap. A cherry-picked result does not.

## Domain: Continual Learning on Split-CIFAR-10

**Why this domain:**

The thermodynamic continual learning (TCL) framework claims that activation
geometry — measured by the thermoobserver's participation ratio and effective
dimensionality — reveals the learning regime, and that regime-aware training
reduces catastrophic forgetting.

Split-CIFAR-10 is the standard benchmark for testing exactly that claim.
It has published baselines (EWC, SI, standard SGD), established metrics
(backward transfer, forgetting measure, average accuracy), and a well-defined
comparison protocol. An external reviewer from the continual learning literature
will immediately know what the numbers mean.

The scientific question TAR will investigate: does thermodynamic regime-awareness
produce measurably lower forgetting or higher accuracy than existing
regularization-based approaches on a standard benchmark?

**What this is not:**

- Not a seven-domain survey. One domain, done properly.
- Not "TAR beats EWC" as a goal. The goal is an honest, reproducible comparison
  with statistical grounding and explicit competing theories.
- Not GPU-serving infrastructure. Serving is support for reviewers, not the
  deliverable.

## Incremental Learning Setting

Phase 7 uses **task-incremental** Split-CIFAR-10.

In the task-incremental setting, the task identity is provided at test time —
the model knows which task it is being evaluated on. This is the easier and
cleaner setting, and it is the correct anchor for Phase 7:

- forgetting is still real and measurable
- baselines (EWC, SI) were originally validated in this setting
- results are directly comparable to the published literature
- failure modes are interpretable without ambiguity about which head is active

**Class-incremental Split-CIFAR-10** (task identity not provided at test time)
is harder, more reviewer-relevant for the current state of the literature, and
the correct target for a Phase 8 extension — but only after Phase 7 produces
a clean task-incremental result. Overreaching to class-incremental before the
task-incremental machinery is validated produces results that are harder to
interpret and harder to compare honestly.

The roadmap explicitly states this boundary so it is not accidentally crossed
during implementation.

## Dependency

Phase 7 must follow closed Phase 6 + WS38 pod slice:

- `WS44` complete (contribution positioning)
- `WS38 run2` deployed (operator with 0.0 external parse error rate)
- Main at `d247f42` or later

## Active Forward Roadmap

### `WS45` — Real Domain Anchor

Purpose:

- add Split-CIFAR-10 as a first-class benchmark family
- wire standard continual learning metrics into the measurement stack
- register the three baselines TAR must compare against

Pod policy: not required.

---

#### Fixed experimental contract

The following values are frozen at WS45 and must not change in WS46 or WS48
without explicit versioning and a corresponding update to the dataset manifest.
Changing these values after results are collected invalidates the comparison.

| Parameter | Value | Reason |
|-----------|-------|--------|
| Dataset | CIFAR-10 via torchvision | canonical, reproducible download |
| Setting | task-incremental | see Incremental Learning Setting above |
| n_tasks | 5 | standard for Split-CIFAR-10 |
| Class order | {0,1}, {2,3}, {4,5}, {6,7}, {8,9} | fixed, not randomised |
| Classes per task | 2 | |
| Train epochs per task | 5 | same for all methods |
| Batch size | 64 | same for all methods |
| Architecture | 3-conv + 2-FC (see below) | same for all methods |
| Parameter budget | ~85K trainable params | same for all methods; do not tune per-method |
| Seeds | [42, 123, 456, 789, 1337] | fixed 5-seed set |
| Evaluation | after every task on all seen tasks | not just after final task |
| Test split | torchvision default (10K total, 1K per class) | |
| Augmentation | random horizontal flip + normalize only | same for all methods |

Architecture (fixed for all methods):

```
Conv2d(3, 32, 3, padding=1) → ReLU → MaxPool2d(2)
Conv2d(32, 64, 3, padding=1) → ReLU → MaxPool2d(2)
Conv2d(64, 64, 3, padding=1) → ReLU → MaxPool2d(2)
Flatten → Linear(1024, 128) → ReLU → Linear(128, n_classes_for_task)
```

Each task has its own output head (task-incremental). The shared trunk (conv
layers + first FC) is where forgetting occurs and where thermoobserver hooks.

What counts as a result worth reporting even if TCL does not win:

- TCL reduces `mean_forgetting` by ≥ 5 percentage points vs SGD baseline
  → scientifically meaningful, even below EWC
- TCL shows anomalous behaviour on specific tasks (anomaly elevation triggers)
  → scientifically interesting, generates competing theories
- TCL forgetting is comparable to EWC with no additional hyperparameter tuning
  → demonstrates thermodynamic regularization as a principled alternative
- TCL is significantly worse → also honest and worth reporting

---

#### Schema changes

File: `tar_lab/schemas.py`

1. New schema `ContinualLearningMetrics`:

```python
class ContinualLearningMetrics(BaseModel):
    task_id: int
    task_accuracy: float                # accuracy on this task after training on all tasks
    accuracy_right_after_training: float  # R_{j,j}: accuracy immediately after task j trained
    backward_transfer: float            # R_{T,j} - R_{j,j}
    forgetting_measure: float           # max over k<T of (R_{k,j} - R_{T,j})
    forward_transfer: float             # R_{i,j} - b_j; set 0.0 for first task
    stability_plasticity_gap: float     # abs(plasticity_loss - stability_loss) proxy
```

2. New schema `ContinualLearningBenchmarkResult`:

```python
class ContinualLearningBenchmarkResult(BaseModel):
    benchmark_id: str                       # uuid4 hex
    method: str                             # "tcl" | "ewc" | "si" | "sgd_baseline"
    seed: int
    n_tasks: int = 5
    per_task_metrics: list[ContinualLearningMetrics]
    mean_backward_transfer: float
    mean_forgetting: float
    final_mean_accuracy: float
    last_task_accuracy: float               # accuracy on task 4 only, after all tasks
    thermodynamic_trace_path: Optional[str] = None  # TCL only
```

3. New schema `ContinualLearningBenchmarkConfig`:

```python
class ContinualLearningBenchmarkConfig(BaseModel):
    dataset: str = "split_cifar10"
    setting: str = "task_incremental"       # locked; "class_incremental" is Phase 8
    n_tasks: int = 5
    classes_per_task: int = 2
    class_order: list[list[int]] = [        # frozen
        [0, 1], [2, 3], [4, 5], [6, 7], [8, 9]
    ]
    train_epochs_per_task: int = 5
    batch_size: int = 64
    seed: int = 42
    ewc_lambda: float = 5000.0
    si_c: float = 0.1
    si_xi: float = 0.001
    tcl_governor_enabled: bool = True
    augmentation: str = "flip_normalize"    # locked
```

---

#### Benchmark implementation

File: `tar_lab/multimodal_payloads.py`

Add function `run_split_cifar10_benchmark`:

```python
def run_split_cifar10_benchmark(
    config: ContinualLearningBenchmarkConfig,
    method: str,
    observer: Optional[ActivationThermoObserver] = None,
) -> ContinualLearningBenchmarkResult:
```

Implementation notes:

- Download CIFAR-10 via `torchvision.datasets.CIFAR10`; cache under
  `dataset_artifacts/split_cifar10/`
- Split into 5 tasks: task 0 = classes {0,1}, task 1 = {2,3}, ..., task 4 = {8,9}
- Architecture: simple ConvNet (3 conv layers + 2 FC) — not the benchmark
  itself, just enough to produce meaningful forgetting
- `method="tcl"`: attach `observer`, call `observer.step()` after each batch,
  let the governor modulate learning rate via `_tcl_lr_adjustment()` below
- `method="ewc"`: after each task, compute Fisher matrix on task data and add
  EWC regularization term to loss
- `method="si"`: maintain cumulative importance weights via SI path-integral
  method
- `method="sgd_baseline"`: plain SGD, no regularization
- After training all tasks: evaluate on each task's held-out test set, compute
  `ContinualLearningMetrics` for each task
- If `observer` is provided (TCL only): write trace to
  `tar_state/cl_traces/{benchmark_id}_{seed}.json`

Add helper `_tcl_lr_adjustment(observer: ActivationThermoObserver, base_lr: float) -> float`:

- if `observer.current_regime` is `"critical"` or `"ordered"`: reduce lr by
  factor 0.5 (thermodynamic braking)
- if `observer.current_regime` is `"disordered"`: increase lr by factor 1.2
  (thermodynamic acceleration)
- if regime is unknown or observer has no data: return `base_lr` unchanged

---

#### Problem family registration

File: `tar_lab/multimodal_payloads.py` or `tar_lab/problem_runner.py`
(whichever file registers benchmark IDs)

Register the following benchmark IDs as a known family:

```
"split_cifar10_task0_validation"
"split_cifar10_task1_validation"
"split_cifar10_task2_validation"
"split_cifar10_task3_validation"
"split_cifar10_task4_validation"
```

Each maps to `run_split_cifar10_benchmark(config, method="tcl")` with the
appropriate task held out for final evaluation.

Also register `"split_cifar10_comparison"` as the combined multi-method
benchmark that runs all three methods (TCL, EWC, SGD) and returns all results.

---

#### Thermoobserver integration

File: `tar_lab/thermoobserver.py`

Add `current_regime` property to `ActivationThermoObserver`:

```python
@property
def current_regime(self) -> str:
    # returns "ordered" | "critical" | "disordered" | "unknown"
    # based on most recent SmoothedStatMetrics.rho and effective_dimensionality
    # ordered: rho > 1.1 and effective_dimensionality < 0.3 * anchor
    # disordered: rho < 0.9 and effective_dimensionality > 1.5 * anchor
    # critical: otherwise (transition zone)
    # unknown: if not statistically_ready
```

---

#### Test requirements

File: `tests/test_real_domain_anchor.py`

- `test_split_cifar10_sgd_runs`: SGD baseline completes 5 tasks without error,
  returns valid `ContinualLearningBenchmarkResult`
- `test_split_cifar10_forgetting_positive_sgd`: SGD shows positive forgetting
  measure (mean_forgetting > 0.0) — validates benchmark produces real forgetting
- `test_split_cifar10_tcl_runs`: TCL method with live observer completes 5 tasks
- `test_thermoobserver_regime_property`: observer in known state returns correct
  regime string
- `test_cl_metrics_schema_valid`: `ContinualLearningMetrics` instantiates and
  rejects unknown fields
- `test_benchmark_result_schema_valid`: `ContinualLearningBenchmarkResult`
  instantiates with per_task_metrics list

Notes: these tests run on CPU with a minimal dataset sample (first 200 examples
per task). Mark `@pytest.mark.slow` for full runs. Fast tests use n_tasks=2,
train_epochs_per_task=1, and a 200-sample subset.

---

#### Acceptance criteria

- Split-CIFAR-10 benchmark family registered and runnable
- SGD baseline produces positive forgetting — confirms the benchmark is
  measuring real catastrophic forgetting, not a flat baseline
- TCL path runs with thermoobserver attached and regime modulation active
- EWC and SI baselines run and produce results in the same schema
- `ContinualLearningMetrics` per task computed correctly for all methods
- Thermoobserver exposes `current_regime` property

---

### `WS46` — Baseline Comparison Protocol

Purpose:

- define the exact experimental protocol TAR uses to compare TCL against
  the three baselines
- require statistical significance — not just mean comparison
- register this protocol as the measurement instrument for WS48

Pod policy: pod required for full multi-seed runs.

---

#### Schema changes

File: `tar_lab/schemas.py`

1. New schema `BaselineComparisonPlan`:

```python
class BaselineComparisonPlan(BaseModel):
    plan_id: str
    project_id: str
    benchmark: str = "split_cifar10"
    methods: list[str]                          # ["tcl", "ewc", "si", "sgd_baseline"]
    seeds: list[int]                            # at least 5: [42, 123, 456, 789, 1337]
    primary_metric: str = "mean_forgetting"     # lower is better
    secondary_metrics: list[str] = [
        "final_mean_accuracy",
        "mean_backward_transfer",
    ]
    significance_test: str = "mann_whitney_u"   # non-parametric, no normality assumption
    effect_size_metric: str = "cohens_d"
    significance_threshold: float = 0.05
    created_at: str
    status: str = "proposed"                    # "proposed" | "running" | "complete"
```

2. New schema `BaselineComparisonResult`:

```python
class BaselineComparisonResult(BaseModel):
    result_id: str
    plan_id: str
    project_id: str
    completed_at: str
    method_results: dict[str, list[ContinualLearningBenchmarkResult]]
    # key: method name, value: list of results across seeds
    method_means: dict[str, dict[str, float]]
    # key: method, inner key: metric name, value: mean
    method_stds: dict[str, dict[str, float]]
    pairwise_pvalues: dict[str, dict[str, float]]
    # key: "tcl_vs_ewc" etc., inner key: metric name, value: p-value
    pairwise_effect_sizes: dict[str, dict[str, float]]
    conclusion: str                             # operator or rule-generated
    tcl_is_significantly_better: bool           # True if p < 0.05 on primary metric vs all baselines
    tcl_is_significantly_worse: bool
    honest_assessment: str                      # plain-language honest summary
```

3. New schema `StatisticalTestRecord`:

```python
class StatisticalTestRecord(BaseModel):
    test_id: str
    test_type: str                              # "mann_whitney_u" | "cohens_d"
    metric: str
    group_a: str                                # method name
    group_b: str
    group_a_values: list[float]
    group_b_values: list[float]
    statistic: float
    p_value: Optional[float]
    effect_size: Optional[float]
    significant: bool
```

---

#### Orchestrator changes

File: `tar_lab/orchestrator.py`

1. Add `plan_baseline_comparison(project_id: str, seeds: list[int]) -> BaselineComparisonPlan`:

   - creates a `BaselineComparisonPlan` with all four methods and the provided seeds
   - persists to `tar_state/comparisons/{plan_id}.json`
   - returns the plan

2. Add `run_baseline_comparison(plan: BaselineComparisonPlan) -> BaselineComparisonResult`:

   Pre-run controls (same for all methods — locked, not tuned per-method):
   - same architecture (3-conv + 2-FC, ~85K params)
   - same class order from `ContinualLearningBenchmarkConfig.class_order`
   - same seed set from plan
   - same augmentation policy
   - same train_epochs_per_task and batch_size

   For each method × seed:
   - call `run_split_cifar10_benchmark(config._replace(seed=seed), method)`
   - collect into `method_results`

   Compute means and stds per method per metric.

   For each TCL-vs-baseline pair (TCL vs EWC, TCL vs SI, TCL vs SGD):
   - run Mann-Whitney U on `mean_forgetting` values across 5 seeds
   - compute Cohen's d on same values
   - record `StatisticalTestRecord`

   Set `tcl_is_significantly_better = True` only if p < 0.05 on `mean_forgetting`
   for ALL THREE pairwise comparisons. Mixed results do not qualify.

   Set `tcl_is_significantly_worse = True` if p < 0.05 and TCL mean_forgetting
   is higher (worse) than any baseline.

   Generate `honest_assessment` string:
   - template: "TCL vs {method}: mean_forgetting {tcl:.3f} vs {baseline:.3f}
     (p={p:.3f}, d={d:.3f}). [Significantly better/worse/not significant]."
   - one sentence per comparison
   - final sentence: "Overall: [TCL significantly reduces forgetting across all
     baselines / TCL does not significantly differ from baselines / Mixed result:
     see per-method breakdown]."
   - this string is what a reviewer reads first — it must be accurate

   Call `position_contribution(project_id)` after results are computed.
   Persist result to `tar_state/comparisons/{result_id}.json`.
   Persist each `StatisticalTestRecord` to `tar_state/comparisons/stats/{test_id}.json`.
   Return result.

3. Add `get_comparison_result(project_id: str) -> Optional[BaselineComparisonResult]`:

   - loads most recent comparison result for the project

---

#### Operator inspection

- CLI: `tar comparison plan PROJECT_ID` → create and display a plan
- CLI: `tar comparison run PROJECT_ID` → run the full comparison (pod-scale)
- CLI: `tar comparison show PROJECT_ID` → display most recent result
- CLI: `tar comparison stats PROJECT_ID` → show all statistical test records

---

#### Test requirements

File: `tests/test_baseline_comparison_protocol.py`

- `test_comparison_plan_schema_valid`: `BaselineComparisonPlan` instantiates
  correctly, defaults are correct
- `test_comparison_result_schema_valid`: `BaselineComparisonResult` with empty
  dicts instantiates without error
- `test_statistical_test_record_schema_valid`: `StatisticalTestRecord`
  instantiates and rejects unknown fields
- `test_plan_persisted`: `plan_baseline_comparison()` writes to
  `tar_state/comparisons/`
- `test_honest_assessment_not_significant`: when TCL p-value > 0.05,
  `honest_assessment` contains "not significantly" (not a victory claim)
- `test_honest_assessment_significant`: when TCL p-value < 0.05 on all
  comparisons, `tcl_is_significantly_better = True` and assessment reflects it
- `test_tcl_worse_flagged`: when TCL mean > EWC mean on forgetting (worse),
  `tcl_is_significantly_worse` flag is set when p < 0.05

---

#### Acceptance criteria

- `BaselineComparisonPlan` and `BaselineComparisonResult` schemas present and validated
- `plan_baseline_comparison()` persists plan correctly
- `run_baseline_comparison()` collects results for all methods × seeds
- Mann-Whitney U and Cohen's d computed correctly for each pairwise comparison
- `honest_assessment` never claims significance unless p < 0.05
- `tcl_is_significantly_worse` is checked and set — this must not be hidden
- `contribution_positioning` automatically triggered after comparison completes

---

### `WS47` — Reproducibility Packaging

Purpose:

- make every WS48 output rerunnable by an external reviewer with a single
  command
- sealed datasets, exact environment, and a manifest that an external party
  can verify independently

Pod policy: not required.

---

#### Schema changes

File: `tar_lab/schemas.py`

1. New schema `EnvironmentManifest`:

```python
class EnvironmentManifest(BaseModel):
    manifest_id: str
    captured_at: str
    python_version: str
    torch_version: str
    cuda_version: Optional[str]
    platform: str
    package_hashes: dict[str, str]      # package name -> sha256 of wheel or dist-info RECORD
    dataset_checksums: dict[str, str]   # dataset name -> sha256 of archive
    repo_commit: str                    # git HEAD at time of capture
    repo_dirty: bool                    # True if working tree has uncommitted changes
```

2. New schema `SealedDatasetManifest`:

```python
class SealedDatasetManifest(BaseModel):
    dataset_id: str
    name: str                           # "split_cifar10"
    source_url: str                     # canonical download URL
    archive_sha256: str
    split_config: dict                  # task assignments, seed, class mapping
    sealed_at: str
    n_train_per_task: int
    n_test_per_task: int
```

3. New schema `ReproducibilityPackage`:

```python
class ReproducibilityPackage(BaseModel):
    package_id: str
    project_id: str
    created_at: str
    environment_manifest: EnvironmentManifest
    dataset_manifests: list[SealedDatasetManifest]
    comparison_result_id: str
    positioning_report_id: str
    anchor_pack_manifest_id: str
    rerun_script_path: str              # relative path to rerun.sh or rerun.py
    artifact_paths: list[str]           # all output files included in this package
    package_sha256: str                 # sha256 of the entire package directory tree
```

---

#### Orchestrator changes

File: `tar_lab/orchestrator.py`

1. Add `capture_environment_manifest() -> EnvironmentManifest`:

   - reads `sys.version`, `torch.__version__`, `torch.version.cuda`
   - runs `pip list --format=json` and hashes each installed package's
     `dist-info/RECORD` file if available
   - reads `git rev-parse HEAD` and `git status --porcelain`
   - computes SHA256 of dataset archives in `dataset_artifacts/`
   - persists to `tar_state/reproducibility/environment_{manifest_id}.json`

2. Add `seal_dataset_manifest(dataset_name: str) -> SealedDatasetManifest`:

   - computes SHA256 of the dataset archive under `dataset_artifacts/`
   - records the split config (task-to-class mapping, seed used)
   - persists to `tar_state/reproducibility/dataset_{dataset_id}.json`

3. Add `create_reproducibility_package(project_id: str,
       comparison_result_id: str) -> ReproducibilityPackage`:

   - calls `capture_environment_manifest()`
   - calls `seal_dataset_manifest("split_cifar10")`
   - writes `task_order_manifest.json` — the frozen class order:
     ```json
     {"task_order": [[0,1],[2,3],[4,5],[6,7],[8,9]], "setting": "task_incremental"}
     ```
   - writes `seed_manifest.json`:
     ```json
     {"seeds": [42, 123, 456, 789, 1337], "primary_metric": "mean_forgetting"}
     ```
   - generates `rerun.py` under
     `tar_state/reproducibility/packages/{package_id}/rerun.py`:
     ```python
     # Generated by TAR — reproducibility package {package_id}
     # Requires: see environment_manifest.json for exact package versions
     # One-command rerun: python rerun.py
     import json
     from tar_lab.orchestrator import TAROrchestrator
     o = TAROrchestrator(".")
     plan = o.plan_baseline_comparison("{project_id}", seeds=[42, 123, 456, 789, 1337])
     result = o.run_baseline_comparison(plan)
     print(result.honest_assessment)
     with open("rerun_result.json", "w") as f:
         json.dump(result.model_dump(), f, indent=2)
     print("Result written to rerun_result.json")
     ```
   - writes `reviewer_summary.md` — a human-readable summary:
     ```markdown
     # TAR Phase 7 Reviewer Summary

     ## Benchmark
     Split-CIFAR-10, task-incremental, 5 tasks

     ## Methods compared
     TCL (thermodynamic governor), EWC, SI, SGD baseline

     ## Primary metric
     mean_forgetting (lower is better)

     ## Result
     {honest_assessment from BaselineComparisonResult}

     ## Statistical tests
     {one line per pairwise test: method, p-value, Cohen's d}

     ## Anomalies elevated
     {count and brief description}

     ## Competing theories
     {count open / count invalidated}

     ## Novelty vs literature
     {novelty_vs_literature score from ContributionPositioningReport}

     ## To reproduce
     python rerun.py
     ```
   - collects all artifact paths
   - computes SHA256 over the package directory tree
   - persists `ReproducibilityPackage` to
     `tar_state/reproducibility/packages/{package_id}/manifest.json`

4. Wire `create_reproducibility_package()` into `publish_handoff_package()`:

   - `publish_handoff_package()` must call this and include `package_id` in
     the returned dict

---

#### Test requirements

File: `tests/test_reproducibility_packaging.py`

- `test_environment_manifest_schema_valid`: `EnvironmentManifest` instantiates
  correctly
- `test_sealed_dataset_manifest_schema_valid`: `SealedDatasetManifest`
  instantiates and rejects unknown fields
- `test_reproducibility_package_schema_valid`: `ReproducibilityPackage`
  instantiates correctly
- `test_capture_environment_manifest`: `capture_environment_manifest()` returns
  a manifest with non-empty `python_version`, `torch_version`, `repo_commit`
- `test_rerun_script_generated`: `create_reproducibility_package()` writes a
  `rerun.py` file under the package directory
- `test_package_sha256_non_empty`: created package has non-empty `package_sha256`

---

#### Acceptance criteria

- `EnvironmentManifest` captures Python, torch, CUDA versions, and repo commit
- `SealedDatasetManifest` records dataset SHA256 and split config
- `create_reproducibility_package()` generates a `rerun.py` and SHA256s the
  package tree
- `publish_handoff_package()` includes `package_id`
- An external reviewer can run `python rerun.py` with the correct environment
  and reproduce the comparison result

---

### `WS48` — Reviewer Package Generation

Purpose:

- execute the full comparison on a real pod
- let TAR manage the run end-to-end: literature ingest, gap scan, project
  promotion, baseline comparison, anomaly elevation, competing theory
  generation, contribution positioning, reproducibility packaging
- produce one reviewer-grade artifact directory

Pod policy: **required** — full multi-seed Split-CIFAR-10 comparison across
four methods is GPU-bound.

---

#### Pre-run checklist (laptop, before pod start)

All of the following must be true before the pod run begins:

1. WS45 accepted: `run_split_cifar10_benchmark()` passes all fast tests
2. WS46 accepted: `plan_baseline_comparison()` and `run_baseline_comparison()`
   pass all tests
3. WS47 accepted: `create_reproducibility_package()` passes all tests
4. Suite clean: `pytest tests -q --tb=short` passes with no failures
5. Commit pushed: main is clean

---

#### Pod run sequence

On the pod, execute `ws48_reviewer_package.py`:

```python
from tar_lab.orchestrator import TAROrchestrator

workspace = "/workspace/Thermodynamic-Continual-Learning-delivered"
o = TAROrchestrator(workspace)

# Step 1 — ingest continual learning literature
print("=== Step 1: Literature ingest ===")
o.ingest_literature(domains=["cs.LG", "q-bio.NC"], max_papers=30)

# Step 2 — gap scan seeded toward continual learning
print("=== Step 2: Frontier gap scan ===")
scan = o.run_frontier_gap_scan(topic="catastrophic forgetting thermodynamic regularization")
print(scan)

# Step 3 — promote gap to project
print("=== Step 3: Agenda review and commit ===")
from tar_lab.agenda import AgendaEngine
agenda = AgendaEngine(workspace, o)
review = agenda.run_agenda_review()
agenda.commit_pending_decisions()

# Step 4 — get active project
projects = [p for p in o.list_projects() if p.get("status") == "active"]
assert projects, "No active project after gap promotion"
project_id = projects[0]["project_id"]
print(f"Active project: {project_id}")

# Step 5 — plan and run baseline comparison
print("=== Step 5: Baseline comparison (TCL vs EWC vs SI vs SGD) ===")
plan = o.plan_baseline_comparison(project_id, seeds=[42, 123, 456, 789, 1337])
result = o.run_baseline_comparison(plan)
print(f"TCL significantly better: {result.tcl_is_significantly_better}")
print(f"TCL significantly worse: {result.tcl_is_significantly_worse}")
print(f"Honest assessment: {result.honest_assessment}")

# Step 6 — anomaly elevation (automatic via heartbeat, force here)
print("=== Step 6: Anomaly elevation ===")
anomalies = o.elevate_anomalies()
print(f"Anomalies elevated: {len(anomalies)}")

# Step 7 — competing theories (automatic after claims accepted)
print("=== Step 7: Competing theories ===")
theories = o.get_competing_theories()
print(f"Competing theories generated: {len(theories)}")

# Step 8 — contribution positioning
print("=== Step 8: Contribution positioning ===")
positioning = o.position_contribution(project_id,
    trial_id=result.result_id,
    result_description=result.honest_assessment)
print(f"Novelty vs literature: {positioning.novelty_vs_literature:.3f}")

# Step 9 — reproducibility package
print("=== Step 9: Reproducibility package ===")
pkg = o.create_reproducibility_package(project_id, result.result_id)
print(f"Package ID: {pkg.package_id}")
print(f"Package SHA256: {pkg.package_sha256}")
print(f"Rerun script: {pkg.rerun_script_path}")

# Step 10 — publish handoff
print("=== Step 10: Handoff package ===")
handoff = o.publish_handoff_package(project_id, result.result_id,
    description=result.honest_assessment)
print(f"Handoff: {handoff}")

print("=== WS48 COMPLETE ===")
print(f"Honest assessment: {result.honest_assessment}")
print(f"Novelty vs literature: {positioning.novelty_vs_literature:.3f}")
print(f"Reproducibility package: tar_state/reproducibility/packages/{pkg.package_id}/")
```

---

#### Reviewer artifact contents

The final artifact directory at
`tar_state/reproducibility/packages/{package_id}/` must contain:

| File | Contents |
|------|----------|
| `manifest.json` | `ReproducibilityPackage` with all IDs and SHA256 |
| `environment_manifest.json` | exact Python, torch, CUDA, package hashes |
| `dataset_manifest.json` | Split-CIFAR-10 SHA256 and split config |
| `task_order_manifest.json` | frozen class order and setting declaration |
| `seed_manifest.json` | frozen seed set and primary metric declaration |
| `comparison_result.json` | `BaselineComparisonResult` with all per-seed data |
| `statistical_tests.json` | all `StatisticalTestRecord` objects |
| `positioning_report.json` | `ContributionPositioningReport` |
| `anomaly_elevations.json` | any `AnomalyElevationRecord` objects |
| `competing_theories.json` | `CompetingTheory` objects for the project |
| `rerun.py` | one-command rerun script |
| `reviewer_summary.md` | human-readable summary for external reviewer |
| `thermoobserver_traces/` | TCL regime traces for each seed |

---

#### WS48 acceptance criteria

- full comparison completes: 4 methods × 5 seeds = 20 benchmark runs
- `honest_assessment` reflects the actual statistical outcome; it does not
  claim significance unless p < 0.05 on primary metric vs all baselines
- `tcl_is_significantly_worse` is checked and reported honestly
- `AnomalyElevationRecord` generated if any TCL seed result is anomalous
- at least 1 `CompetingTheory` generated for the primary claim
- `ContributionPositioningReport` contains `novelty_vs_literature` derived from
  vault similarity, not asserted
- `ReproducibilityPackage` with SHA256 exists and `rerun.py` runs without error
- all artifact files listed in the table above are present in the package
  directory

---

## Execution Dependency Map

```
WS45 (real domain anchor — laptop)
  -> WS46 (baseline comparison protocol — laptop + pod for full runs)
    -> WS47 (reproducibility packaging — laptop)
      -> WS48 (reviewer package generation — pod)
```

All four workstreams are sequential. No parallelism is safe here because each
depends on the previous workstream's schema and methods being stable.

## New Files Summary

| File | WS | Purpose |
|------|----|---------|
| `tests/test_real_domain_anchor.py` | WS45 | Split-CIFAR-10 benchmark and CL metrics |
| `tests/test_baseline_comparison_protocol.py` | WS46 | comparison plan, stats, honest assessment |
| `tests/test_reproducibility_packaging.py` | WS47 | environment manifest, package generation |
| `ws48_reviewer_package.py` | WS48 | pod run script |
| `tar_state/cl_traces/` | WS45 | thermoobserver trace files from TCL runs |
| `tar_state/comparisons/` | WS46 | comparison plans and results |
| `tar_state/comparisons/stats/` | WS46 | statistical test records |
| `tar_state/reproducibility/` | WS47 | environment and dataset manifests |
| `tar_state/reproducibility/packages/` | WS47/WS48 | sealed reviewer packages |

## Schema Change Summary

| Schema | File | WS | Change |
|--------|------|----|--------|
| `ContinualLearningMetrics` | schemas.py | WS45 | new: per-task BWT, forgetting, accuracy |
| `ContinualLearningBenchmarkResult` | schemas.py | WS45 | new: multi-seed benchmark output |
| `ContinualLearningBenchmarkConfig` | schemas.py | WS45 | new: benchmark configuration |
| `BaselineComparisonPlan` | schemas.py | WS46 | new: multi-method comparison plan |
| `BaselineComparisonResult` | schemas.py | WS46 | new: means, stds, p-values, effect sizes |
| `StatisticalTestRecord` | schemas.py | WS46 | new: per-test statistical record |
| `EnvironmentManifest` | schemas.py | WS47 | new: exact environment capture |
| `SealedDatasetManifest` | schemas.py | WS47 | new: dataset SHA256 and split config |
| `ReproducibilityPackage` | schemas.py | WS47 | new: full reviewer package manifest |

## Completion Picture — Phase 7

When Phase 7 is complete, TAR will have produced one externally reviewable
scientific artifact:

- a reproducible comparison of TCL against EWC, SI, and SGD on
  task-incremental Split-CIFAR-10
- statistically grounded with Mann-Whitney U and Cohen's d across 5 fixed seeds
- using the same architecture, parameter budget, class order, augmentation,
  and epoch budget for all methods — no per-method tuning
- honestly reported: the `honest_assessment` field reflects the actual statistical
  outcome whether TCL wins, loses, or shows mixed results
- anomalous results elevated and competing theories generated automatically
- contribution positioned against the vault of ingested continual learning
  literature
- sealed in a package with frozen task-order, seed, and dataset manifests
  that a domain expert can download, run `python rerun.py`, and reproduce

**What comes after Phase 7 (not in scope here):**

- class-incremental Split-CIFAR-10 (harder setting, Phase 8)
- larger backbone or Split-CIFAR-100
- multi-domain comparison (Split-CUB, Split-miniImageNet)
- publication write-up using Phase 7 artifacts as the evidence base

None of those require new machinery. They require Phase 7 to be done right first.

That is what "externally believable" looks like in practice.
