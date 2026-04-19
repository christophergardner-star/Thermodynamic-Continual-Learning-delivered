from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GovernorMetrics(StrictModel):
    timestamp: str = Field(default_factory=utc_now_iso)
    trial_id: str
    step: int = Field(ge=0)
    energy_e: float = Field(ge=0.0)
    entropy_sigma: float = Field(ge=0.0)
    drift_l2: float = Field(ge=0.0)
    drift_rho: float = Field(ge=0.0)
    grad_norm: float = Field(ge=0.0)
    regime_rho: float = Field(default=0.0, ge=0.0)
    effective_dimensionality: float = Field(default=0.0, ge=0.0)
    effective_dimensionality_std_err: float = Field(default=0.0, ge=0.0)
    dimensionality_ratio: float = Field(default=0.0, ge=0.0)
    entropy_sigma_std_err: float = Field(default=0.0, ge=0.0)
    regime_rho_std_err: float = Field(default=0.0, ge=0.0)
    stat_window_size: int = Field(default=0, ge=0)
    stat_sample_count: int = Field(default=0, ge=0)
    statistically_ready: bool = False
    equilibrium_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    equilibrium_gate: bool = False
    training_loss: Optional[float] = Field(default=None, ge=0.0)
    gpu_temperature_c: Optional[float] = None
    gpu_memory_temperature_c: Optional[float] = None
    gpu_power_w: Optional[float] = None


class QuantitativeJustification(StrictModel):
    energy_e: float = Field(ge=0.0)
    entropy_sigma: float = Field(ge=0.0)
    drift_rho: float = Field(ge=0.0)
    grad_norm: float = Field(ge=0.0)
    regime_rho: float = Field(default=0.0, ge=0.0)
    effective_dimensionality: float = Field(default=0.0, ge=0.0)
    effective_dimensionality_std_err: float = Field(default=0.0, ge=0.0)
    equilibrium_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    energy_slope: float
    entropy_slope: float
    drift_slope: float
    dimensionality_slope: float = 0.0


class GovernorThresholds(StrictModel):
    max_drift_l2: float = Field(default=0.40, gt=0.0)
    max_drift_rho: float = Field(default=0.08, gt=0.0)
    max_entropy_sigma: float = Field(default=0.12, gt=0.0)
    max_grad_norm: float = Field(default=2.5, gt=0.0)
    max_gpu_temperature_c: float = Field(default=78.0, gt=0.0)
    min_dimensionality_ratio: float = Field(default=0.35, gt=0.0)
    max_quenching_loss: float = Field(default=1.20, gt=0.0)


class RuntimeSpec(StrictModel):
    image: str = "pytorch/pytorch:latest"
    memory_limit_gb: int = Field(default=40, ge=1)
    cpu_limit: int = Field(default=12, ge=1)
    gpu_index: int = Field(default=0, ge=0)
    power_limit_w: int = Field(default=300, ge=1)
    gpu_target_temp_c: int = Field(default=70, ge=1, le=95)
    working_dir: str = "/workspace"
    env: Dict[str, str] = Field(default_factory=dict)
    volumes: Dict[str, str] = Field(default_factory=dict)
    read_only_volumes: Dict[str, str] = Field(default_factory=dict)
    sandbox_profile: Literal["production", "dev_override"] = "production"
    allowed_writable_mounts: List[str] = Field(
        default_factory=lambda: ["/workspace/tar_runs", "/workspace/logs", "/workspace/anchors"]
    )
    network_policy: Literal["default", "none", "restricted"] = "none"
    image_locked: bool = False
    image_manifest_path: Optional[str] = None
    run_manifest_path: Optional[str] = None


class LocalLLMConfig(StrictModel):
    base_url: str
    api_key: str = "local"
    model: str
    temperature: float = 0.0
    timeout_s: float = Field(default=120.0, gt=0.0)
    max_retries: int = Field(default=3, ge=1, le=10)
    model_tier: Literal["efficient", "frontier"] = "efficient"
    cost_per_token_input: float = Field(default=0.0, ge=0.0)
    cost_per_token_output: float = Field(default=0.0, ge=0.0)
    context_window: int = Field(default=8192, ge=1)
    supports_tool_use: bool = True


class FrontierModelConfig(StrictModel):
    frontier_role_config: LocalLLMConfig
    efficient_role_config: LocalLLMConfig
    routing_policy: Literal["stakes_aware", "always_frontier", "always_efficient"] = "stakes_aware"
    max_frontier_budget_usd: float = Field(default=0.0, ge=0.0)
    frontier_decisions: List[str] = Field(
        default_factory=lambda: [
            "director_propose",
            "breakthrough_review",
            "falsification_plan",
            "generative_director_proposal",
        ]
    )


class ModelRoutingRecord(StrictModel):
    record_id: str
    decision_type: str
    tier_selected: Literal["efficient", "frontier"]
    model_id: str
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    budget_remaining_usd: Optional[float] = None
    timestamp_utc: str = Field(default_factory=utc_now_iso)


class RoutingSummary(StrictModel):
    frontier_calls: int = Field(default=0, ge=0)
    efficient_calls: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    budget_remaining_usd: Optional[float] = None
    budget_exhausted: bool = False


class CrossDomainBridgeRecord(StrictModel):
    bridge_id: str
    timestamp: str
    source_domain: str
    target_domain: str
    source_paper_id: str
    target_paper_id: str
    bridge_type: Literal["analogy", "method_transfer", "shared_formalism", "empirical_parallel"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    vault_indexed: bool = False


class AnomalyElevationRecord(StrictModel):
    elevation_id: str
    timestamp: str
    breakthrough_id: str
    project_id: str
    surprise_score: float = Field(default=0.0, ge=0.0, le=1.0)
    prior_contradiction_score: float = Field(default=0.0, ge=0.0, le=1.0)
    vault_score_mean: float
    vault_score_std: float = Field(default=0.0, ge=0.0)
    elevation_reason: str
    replication_priority: Literal["immediate", "high", "normal"] = "normal"
    replicated: bool = False


class CompetingTheory(StrictModel):
    theory_id: str
    timestamp: str
    trial_id: str
    project_id: str
    description: str
    predicted_outcome: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["operator", "heuristic"]
    status: Literal["open", "invalidated", "confirmed", "inconclusive"] = "open"
    invalidated_by_trial_id: str = ""
    vault_indexed: bool = False


class HeadToHeadExperimentPlan(StrictModel):
    plan_id: str
    timestamp: str
    trial_id: str
    project_id: str
    primary_theory_description: str
    competing_theory_id: str
    discriminating_variable: str
    expected_primary_outcome: str
    expected_competing_outcome: str
    status: Literal["proposed", "scheduled", "completed", "abandoned"] = "proposed"


class TheoryInvalidationRecord(StrictModel):
    invalidation_id: str
    timestamp: str
    theory_id: str
    trial_id: str
    project_id: str
    evidence_summary: str
    confidence: float = Field(ge=0.0, le=1.0)


class SoTAComparison(StrictModel):
    comparison_id: str
    timestamp: str
    paper_id: str
    paper_title: str
    domain: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    outperforms: bool
    delta_description: str


class ContributionPositioningReport(StrictModel):
    report_id: str
    timestamp: str
    project_id: str
    trial_id: str
    novelty_vs_literature: float = Field(ge=0.0, le=1.0)
    surprise_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sota_comparisons: List[SoTAComparison] = Field(default_factory=list)
    competing_theories_open: int = Field(default=0, ge=0)
    competing_theories_invalidated: int = Field(default=0, ge=0)
    positioning_summary: str
    vault_indexed: bool = False


class ContinualLearningMetrics(StrictModel):
    task_id: int
    task_accuracy: float
    accuracy_right_after_training: float
    backward_transfer: float
    forgetting_measure: float
    forward_transfer: float
    stability_plasticity_gap: float


class ContinualLearningBenchmarkResult(StrictModel):
    benchmark_id: str
    method: str
    seed: int
    n_tasks: int = 5
    per_task_metrics: List[ContinualLearningMetrics]
    mean_backward_transfer: float
    mean_forgetting: float
    final_mean_accuracy: float
    last_task_accuracy: float
    thermodynamic_trace_path: str = ""


class ContinualLearningBenchmarkConfig(StrictModel):
    dataset: str = "split_cifar10"
    setting: str = "task_incremental"
    n_tasks: int = 5
    classes_per_task: int = 2
    class_order: List[List[int]] = Field(
        default_factory=lambda: [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    )
    train_epochs_per_task: int = 5
    batch_size: int = 64
    seed: int = 42
    ewc_lambda: float = 100.0
    si_c: float = 0.1
    si_xi: float = 0.001
    tcl_governor_enabled: bool = True
    # Dimensionality-weighted L2 penalty: after each task TCL anchors the
    # trunk weights and penalises drift scaled by the task's anchor D_PR.
    # Higher D_PR = more structured representation = stronger penalty.
    # Set to 0.0 to disable (governor-only TCL, no weight consolidation).
    tcl_penalty_lambda: float = 0.01
    augmentation: str = "flip_normalize"


class StatisticalTestRecord(StrictModel):
    test_id: str
    timestamp: str
    test_type: str
    metric: str
    group_a: str
    group_b: str
    group_a_values: List[float]
    group_b_values: List[float]
    statistic: float
    p_value: float = -1.0
    effect_size: float = 0.0
    significant: bool


class BaselineComparisonPlan(StrictModel):
    plan_id: str
    timestamp: str
    project_id: str
    benchmark: str = "split_cifar10"
    setting: str = "task_incremental"
    methods: List[str] = Field(default_factory=lambda: ["tcl", "ewc", "si", "sgd_baseline"])
    seeds: List[int] = Field(default_factory=lambda: [42, 123, 456, 789, 1337])
    primary_metric: str = "mean_forgetting"
    secondary_metrics: List[str] = Field(
        default_factory=lambda: ["final_mean_accuracy", "mean_backward_transfer"]
    )
    significance_threshold: float = 0.05
    status: str = "proposed"


class BaselineComparisonResult(StrictModel):
    result_id: str
    plan_id: str
    project_id: str
    completed_at: str
    method_means: Dict[str, Dict[str, float]]
    method_stds: Dict[str, Dict[str, float]]
    pairwise_pvalues: Dict[str, float]
    pairwise_effect_sizes: Dict[str, float]
    tcl_is_significantly_better: bool
    tcl_is_significantly_worse: bool
    honest_assessment: str
    statistical_test_ids: List[str]


class EnvironmentManifest(StrictModel):
    manifest_id: str
    captured_at: str
    python_version: str
    torch_version: str
    cuda_version: str = ""
    platform: str
    package_hashes: Dict[str, str]
    dataset_checksums: Dict[str, str]
    repo_commit: str
    repo_dirty: bool


class SealedDatasetManifest(StrictModel):
    dataset_id: str
    name: str
    archive_sha256: str
    split_config: Dict[str, Any]
    sealed_at: str
    n_train_total: int = 0
    n_test_total: int = 0


class ReproducibilityPackage(StrictModel):
    package_id: str
    project_id: str
    created_at: str
    environment_manifest_id: str
    dataset_manifest_id: str
    comparison_result_id: str
    positioning_report_id: str = ""
    anchor_pack_manifest_id: str = ""
    rerun_script_path: str
    reviewer_summary_path: str
    artifact_paths: List[str]
    package_sha256: str


DataAccessMode = Literal["OFFLINE_FALLBACK", "CACHED_REAL", "DOWNLOAD_REAL"]
DataPurity = Literal["fallback", "cached_real", "download_real", "local_real", "mixed"]
RunIntent = Literal["control", "plumbing", "research"]
BackendStatus = Literal["executable", "scaffold"]
ExperimentBackendRunStatus = Literal["planned", "running", "completed", "failed", "interrupted"]
BenchmarkTier = Literal["smoke", "validation", "canonical"]
BenchmarkTruthStatus = Literal["canonical_ready", "validation_only", "smoke_only", "unsupported"]
BenchmarkAlignment = Literal["aligned", "downgraded", "refused", "mixed"]
ScheduleStatus = Literal[
    "scheduled",
    "leased",
    "running",
    "retry_wait",
    "recoverable_crash",
    "completed",
    "failed_terminal",
    "cancelled",
]
AlertSeverity = Literal["info", "warning", "error", "critical"]
SandboxExecutionMode = Literal["docker_only"]
SandboxNetworkPolicy = Literal["off", "restricted", "profile_required"]
SandboxProfile = Literal["production", "dev_override"]
MemoryStoreHealth = Literal["healthy", "rebuild_required", "rebuilding", "degraded"]
DependencyResolutionStatus = Literal["pinned", "missing_version", "optional_missing"]
ResearchProjectStatus = Literal[
    "active",
    "paused",
    "blocked",
    "awaiting_human_review",
    "proposed",
    "parked",
    "completed",
    "abandoned",
]
ResearchThreadStatus = Literal["open", "testing", "falsifying", "supported", "contradicted", "parked", "closed"]
ResearchActionStatus = Literal["planned", "queued", "running", "completed", "failed", "skipped", "invalidated"]
ResearchConfidenceState = Literal["unknown", "exploratory", "provisional", "supported", "contradicted"]
ResearchQuestionStatus = Literal["open", "resolved", "parked"]
ResearchBudgetPressure = Literal["low", "medium", "high", "exhausted"]
ResearchPortfolioRecommendation = Literal["continue", "defer", "park", "resume", "escalate", "retire", "complete", "block"]
ResearchStalenessLevel = Literal["fresh", "watch", "stale", "critical"]
ResearchActionKind = Literal[
    "create_problem_study",
    "run_problem_study",
    "review_execution_result",
    "verify_claim",
    "mechanism_ablation",
    "replication_check",
    "seed_variance_check",
    "contradiction_resolution",
    "benchmark_stress_probe",
    "calibration_check",
    "environment_reproduction_check",
    "claim_linkage_sanity_check",
    "await_dependency",
    "human_review",
    "custom",
]
PrioritizationPolicyMode = Literal["balanced", "falsification_first"]
FalsificationPlanStatus = Literal["active", "satisfied", "completed", "obsolete", "cancelled"]
FalsificationTestStatus = Literal["planned", "attached", "running", "completed", "failed", "skipped"]
FalsificationTriggerType = Literal[
    "confidence_rising",
    "contradiction_pressure",
    "low_replication",
    "benchmark_pressure",
    "calibration_weakness",
    "claim_linkage_gap",
    "environment_reproduction_risk",
]
FalsificationSeverity = Literal["low", "medium", "high", "critical"]
FalsificationTestKind = Literal[
    "mechanism_ablation",
    "replication_check",
    "seed_variance_check",
    "contradiction_resolution",
    "benchmark_stress_probe",
    "calibration_check",
    "environment_reproduction_check",
    "claim_linkage_sanity_check",
]
ResearchStopReason = Literal[
    "budget_exhausted",
    "evidence_saturated",
    "contradiction_detected",
    "dependency_missing",
    "benchmark_unavailable",
    "awaiting_human_review",
    "superseded_by_better_thread",
    "runtime_failure",
    "goal_completed",
    "operator_paused",
]
ResearchResumeReason = Literal[
    "new_budget_allocated",
    "dependency_restored",
    "new_evidence_arrived",
    "scheduled_followup_due",
    "contradiction_requires_resolution",
    "human_requested_resume",
]
PublicationPackageStatus = Literal["not_ready", "provisional", "ready"]
PublicationClaimDisposition = Literal["accepted", "provisional", "rejected", "contradicted", "insufficient_evidence"]
PublicationAlternativeSource = Literal["claim_verdict", "hypothesis", "contradiction_review"]
PublicationLineageEventType = Literal[
    "problem_study",
    "problem_execution",
    "research_decision",
    "verification",
    "claim_verdict",
    "falsification_plan",
    "portfolio_decision",
]
FamilyProposalStatus = Literal[
    "pending",
    "feasibility_running",
    "feasibility_failed",
    "approved",
    "rejected",
]
SelfImprovementCycleStatus = Literal[
    "idle",
    "curating",
    "probing",
    "gate_failed",
    "training",
    "deploying",
    "paused_consecutive_failures",
    "paused_cycle_limit",
    "completed",
]
TrainingSignalKind = Literal[
    "research_decision",
    "falsification_plan",
    "claim_verdict",
    "portfolio_governance",
    "problem_study",
]
AgendaDecisionKind = Literal[
    "promote_gap_project",
    "park_stale_project",
    "defer_gap",
    "cap_enforced",
    "no_action",
]
AgendaDecisionStatus = Literal[
    "pending_veto",
    "committed",
    "vetoed",
]


class DatasetSourceConfig(StrictModel):
    name: str
    split: str = "train"
    subset: Optional[str] = None
    text_fields: List[str] = Field(default_factory=lambda: ["text", "content", "abstract", "code"])
    max_samples: Optional[int] = Field(default=None, ge=1)
    streaming: bool = False
    mode: DataAccessMode = "OFFLINE_FALLBACK"
    sampling_strategy: str = "deterministic_sharding"


class TokenizerProvenance(StrictModel):
    stream_name: Literal["anchor", "research"]
    tokenizer_id: str = "hash-tokenizer"
    tokenizer_class: str = "HashTokenizer"
    tokenizer_hash: str = ""
    tokenizer_vocab_size: int = Field(default=0, ge=0)
    tokenizer_version: Optional[str] = None
    integrity_check: bool = False
    is_fallback: bool = False


class DataProvenance(StrictModel):
    stream_name: Literal["anchor", "research"]
    dataset_name: str
    dataset_subset: Optional[str] = None
    dataset_split: str = "train"
    data_mode: DataAccessMode = "OFFLINE_FALLBACK"
    data_purity: DataPurity = "fallback"
    source_kind: Literal["synthetic", "local_file", "huggingface", "sklearn", "environment", "simulator"] = "huggingface"
    dataset_identifier: str = ""
    local_path: Optional[str] = None
    sampling_strategy: str = "deterministic_sharding"
    dataset_fingerprint: str = ""
    tokenizer_id: str = "hash-tokenizer"
    tokenizer_class: str = "HashTokenizer"
    tokenizer_hash: str = ""
    tokenizer_vocab_size: int = Field(default=0, ge=0)
    tokenizer_version: Optional[str] = None
    integrity_check: bool = False
    is_real_data: bool = False
    is_fallback: bool = False
    provenance_complete: bool = False
    research_safe: bool = False
    tokenizer_provenance: Optional[TokenizerProvenance] = None
    notes: List[str] = Field(default_factory=list)


class DataBundleProvenance(StrictModel):
    anchor: DataProvenance
    research: DataProvenance
    run_intent: RunIntent = "control"
    data_purity: DataPurity = "fallback"
    integrity_check: bool = False
    tokenizer_provenance: Dict[str, TokenizerProvenance] = Field(default_factory=dict)
    provenance_complete: bool = False
    research_grade: bool = False
    has_fallback: bool = False


class DatasetShard(StrictModel):
    shard_index: int = Field(ge=0)
    path: str
    container_path: Optional[str] = None
    records: int = Field(ge=0)


class DatasetManifest(StrictModel):
    stream_name: Literal["anchor", "research"]
    tokenizer_id: Optional[str] = None
    records: int = Field(ge=0)
    shards: List[DatasetShard] = Field(default_factory=list)
    source: DatasetSourceConfig
    provenance: Optional[DataProvenance] = None
    run_intent: RunIntent = "control"
    provenance_complete: bool = False
    research_grade: bool = False


class PreparedDataBundle(StrictModel):
    anchor_manifest_path: str
    research_manifest_path: str
    anchor_manifest: DatasetManifest
    research_manifest: DatasetManifest
    data_provenance: Optional[DataBundleProvenance] = None
    run_intent: RunIntent = "control"
    provenance_complete: bool = False
    research_grade: bool = False


class LiteraturePolicySignal(StrictModel):
    objective_slug: str
    evidence_count: int = Field(default=0, ge=0)
    recommended_family: Optional[Literal["elastic_anchor", "ou_drift_jitter", "layer_freeze"]] = None
    dominant_polarity: Literal["positive", "negative", "mixed", "neutral"] = "neutral"
    contradiction_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    topic_terms: List[str] = Field(default_factory=list)
    cited_document_ids: List[str] = Field(default_factory=list)
    rationale: List[str] = Field(default_factory=list)


class DirectorPolicy(StrictModel):
    version: Literal["tar.v1"] = "tar.v1"
    role: Literal["director"] = "director"
    trial_id: str
    objective_slug: str
    anchor_path: str
    experiment_family: Literal["elastic_anchor", "ou_drift_jitter", "layer_freeze"]
    pivot_required: bool = False
    failure_streak: int = Field(default=0, ge=0)
    quantitative_justification: QuantitativeJustification
    data_anchor: List[GovernorMetrics]
    literature_signal: Optional[LiteraturePolicySignal] = None

    @field_validator("objective_slug")
    @classmethod
    def validate_objective_slug(cls, value: str) -> str:
        if not value or any(ch.isspace() for ch in value):
            raise ValueError("objective_slug must be a non-empty slug")
        return value

    @field_validator("data_anchor")
    @classmethod
    def validate_data_anchor(cls, value: List[GovernorMetrics]) -> List[GovernorMetrics]:
        if len(value) != 3:
            raise ValueError("data_anchor must contain exactly the last three metric points")
        return value


class StrategistPlan(StrictModel):
    version: Literal["tar.v1"] = "tar.v1"
    role: Literal["strategist"] = "strategist"
    trial_id: str
    strategy_family: Literal["elastic_anchor", "ou_drift_jitter", "layer_freeze"]
    anchor_path: str
    fim_lambda: float = Field(gt=0.0)
    bregman_budget: float = Field(gt=0.0)
    drift_budget: float = Field(gt=0.0)
    protected_layers: List[str]
    mutable_layers: List[str]
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    governor_thresholds: GovernorThresholds
    quantitative_justification: QuantitativeJustification
    data_anchor: List[GovernorMetrics]
    retrieved_memories: List["MemorySearchHit"] = Field(default_factory=list)

    @field_validator("data_anchor")
    @classmethod
    def validate_data_anchor(cls, value: List[GovernorMetrics]) -> List[GovernorMetrics]:
        if len(value) != 3:
            raise ValueError("data_anchor must contain exactly the last three metric points")
        return value


class ScoutTask(StrictModel):
    version: Literal["tar.v1"] = "tar.v1"
    role: Literal["scout"] = "scout"
    trial_id: str
    training_entrypoint: str
    command: List[str]
    runtime: RuntimeSpec
    governor_thresholds: GovernorThresholds
    payload_config_path: Optional[str] = None
    run_manifest_path: Optional[str] = None
    dry_run: bool = False


class ProposedExperimentFamily(StrictModel):
    family_id: str
    name: str
    description: str
    config_delta: Dict[str, Any] = Field(default_factory=dict)
    rationale: str
    proposed_by: Literal["operator", "rule_heuristic"] = "rule_heuristic"
    status: FamilyProposalStatus = "pending"
    feasibility_note: Optional[str] = None
    feasibility_trial_id: Optional[str] = None
    approved_at: Optional[str] = None
    rejected_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class GenerativeDirectorProposal(StrictModel):
    proposal_id: str
    objective_slug: str
    trigger_reason: str
    proposed_family: ProposedExperimentFamily
    operator_available: bool = False
    operator_prompt_used: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)


class RegisteredFamilyState(StrictModel):
    entries: List[ProposedExperimentFamily] = Field(default_factory=list)


class FrozenAnchorPackManifest(StrictModel):
    manifest_id: str
    pack_path: str
    run_manifest_hash_sha256: str
    item_ids: List[str] = Field(default_factory=list)
    item_count: int = Field(ge=0)
    baseline_mean_score: float = Field(ge=0.0, le=1.0)
    baseline_overclaim_rate: float = Field(ge=0.0, le=1.0)
    sealed_at: str = Field(default_factory=utc_now_iso)
    sealed_by: str = "eval_validation_pass"


class TrainingSignalRecord(StrictModel):
    signal_id: str
    kind: TrainingSignalKind
    source_id: str
    project_id: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    gold_response: str
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overclaim_present: bool = False
    anchor_pack_overlap: bool = False
    created_at: str = Field(default_factory=utc_now_iso)


class CuratedDeltaRecord(StrictModel):
    delta_id: str
    cycle_id: str
    signal_ids: List[str] = Field(default_factory=list)
    signal_count: int = Field(default=0, ge=0)
    anchor_overlaps_excluded: int = Field(default=0, ge=0)
    overclaim_excluded: int = Field(default=0, ge=0)
    diversity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    kind_distribution: Dict[str, int] = Field(default_factory=dict)
    ready: bool = False
    created_at: str = Field(default_factory=utc_now_iso)


class RetrainRecord(StrictModel):
    retrain_id: str
    cycle_id: str
    delta_id: str
    run_kind: Literal["probe", "run1"]
    adapter_output_path: Optional[str] = None
    probe_mean_score: Optional[float] = None
    probe_overclaim_rate: Optional[float] = None
    anchor_hash_verified: bool = False
    gate_passed: bool = False
    gate_failure_reason: Optional[str] = None
    started_at: str = Field(default_factory=utc_now_iso)
    completed_at: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class SelfImprovementCycleRecord(StrictModel):
    cycle_id: str
    status: SelfImprovementCycleStatus = "idle"
    cycle_number: int = Field(default=1, ge=1)
    delta_id: Optional[str] = None
    probe_retrain_id: Optional[str] = None
    run1_retrain_id: Optional[str] = None
    deployed_adapter_path: Optional[str] = None
    consecutive_gate_failures: int = Field(default=0, ge=0)
    total_cycles_completed: int = Field(default=0, ge=0)
    paused_reason: Optional[str] = None
    human_resume_required: bool = False
    started_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class AgendaReviewConfig(StrictModel):
    max_active_projects: int = Field(default=5, ge=1)
    veto_window_hours: float = Field(default=24.0, ge=0.0)
    min_gap_novelty_to_promote: float = Field(default=0.55, ge=0.0, le=1.0)
    stale_project_hours: float = Field(default=72.0, ge=1.0)
    max_promotions_per_review: int = Field(default=2, ge=1)
    recycle_decisions_to_training_signal: bool = True


class AgendaDecisionRecord(StrictModel):
    decision_id: str
    review_id: str
    kind: AgendaDecisionKind
    subject_id: str
    subject_title: str
    rationale: str
    status: AgendaDecisionStatus = "pending_veto"
    veto_deadline: str
    vetoed_at: Optional[str] = None
    veto_reason: Optional[str] = None
    committed_at: Optional[str] = None
    recycled_to_signal_id: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)


class AgendaReviewRecord(StrictModel):
    review_id: str
    started_at: str = Field(default_factory=utc_now_iso)
    completed_at: Optional[str] = None
    active_project_count: int = Field(default=0, ge=0)
    gap_candidates_reviewed: int = Field(default=0, ge=0)
    decisions: List[AgendaDecisionRecord] = Field(default_factory=list)
    cap_enforced: bool = False
    notes: List[str] = Field(default_factory=list)


class AgendaSnapshot(StrictModel):
    captured_at: str = Field(default_factory=utc_now_iso)
    active_project_count: int = Field(default=0, ge=0)
    pending_veto_count: int = Field(default=0, ge=0)
    committed_this_session: int = Field(default=0, ge=0)
    vetoed_this_session: int = Field(default=0, ge=0)
    latest_review_id: Optional[str] = None
    config: AgendaReviewConfig = Field(default_factory=AgendaReviewConfig)


class BackendProvenance(StrictModel):
    backend_id: str
    summary: str
    domain: str
    entrypoint: str
    status: BackendStatus = "scaffold"
    control_only: bool = False
    research_grade_capable: bool = False
    expected_data_type: str = "text"
    requires_tokenizer: bool = False
    supports_resume: bool = False
    supports_distributed: bool = False
    requires_gpu: bool = False
    required_deps: List[str] = Field(default_factory=list)
    governor_observables: List[str] = Field(default_factory=lambda: ["D_PR", "sigma", "rho"])
    required_metrics: List[str] = Field(default_factory=list)
    required_artifacts: List[str] = Field(default_factory=list)
    valid_input_contract: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class TrainingPayloadConfig(StrictModel):
    version: Literal["tar.v1"] = "tar.v1"
    trial_id: str
    backend_id: Literal["asc_text", "toy_anchor", "asc_cv", "asc_rl", "asc_qml", "coding_asc"] = "asc_text"
    run_intent: RunIntent = "control"
    strategy_family: Literal["elastic_anchor", "ou_drift_jitter", "layer_freeze"]
    anchor_path: str
    alpha: float = Field(gt=0.0)
    eta: float = Field(gt=0.0)
    fim_lambda: float = Field(gt=0.0)
    bregman_budget: float = Field(gt=0.0)
    drift_budget: float = Field(gt=0.0)
    batch_size: int = Field(default=8, ge=1)
    steps: int = Field(default=20, ge=1)
    feature_dim: int = Field(default=16, ge=2)
    seed: int = Field(default=7, ge=0)
    train_split: float = Field(default=0.70, gt=0.0, lt=1.0)
    val_split: float = Field(default=0.15, ge=0.0, lt=1.0)
    test_split: float = Field(default=0.15, ge=0.0, lt=1.0)
    stat_window_size: int = Field(default=5, ge=1)
    min_stat_steps: int = Field(default=5, ge=1)
    anchor_batches: int = Field(default=5, ge=1)
    calibration_batches: int = Field(default=5, ge=1)
    resume_from_checkpoint: Optional[str] = None
    adapter_mode: Literal["full", "lora"] = "lora"
    lora_r: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    dry_run_backbone: Optional[str] = "__tiny_gpt2__"
    log_path: str
    output_dir: str
    anchor_manifest_path: Optional[str] = None
    research_manifest_path: Optional[str] = None
    governor_thresholds: GovernorThresholds
    protected_layers: List[str]
    mutable_layers: List[str]
    backend_provenance: Optional[BackendProvenance] = None
    data_provenance: Optional[DataBundleProvenance] = None
    tokenizer_provenance: Dict[str, TokenizerProvenance] = Field(default_factory=dict)
    provenance_complete: bool = False
    research_grade: bool = False
    notes: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("test_split")
    @classmethod
    def validate_split_sum(cls, value: float, info: Any) -> float:
        data = info.data or {}
        train = float(data.get("train_split", 0.0))
        val = float(data.get("val_split", 0.0))
        total = train + val + value
        if total > 1.000001:
            raise ValueError("train_split + val_split + test_split must be <= 1.0")
        return value


class MemorySearchHit(StrictModel):
    document_id: str
    score: float
    document: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    contradiction_surfaced: bool = False


class ResearchDocument(StrictModel):
    document_id: str
    source_kind: Literal["arxiv", "rss", "manual"]
    source_name: str
    domain: str = ""
    title: str
    summary: str
    url: str
    published_at: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    problem_statements: List[str] = Field(default_factory=list)


class ResearchIngestReport(StrictModel):
    topic: str
    fetched: int = Field(ge=0)
    indexed: int = Field(ge=0)
    sources: List[str] = Field(default_factory=list)
    documents: List[ResearchDocument] = Field(default_factory=list)


class FrontierGapRecord(StrictModel):
    gap_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    scan_id: Optional[str] = None
    content_hash: Optional[str] = None
    description: str
    domain_profile: Optional[str] = None
    evidence_count: int = Field(default=0, ge=0)
    source_document_ids: List[str] = Field(default_factory=list)
    novelty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    similarity_to_existing: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: Literal["identified", "proposed", "rejected", "promoted"] = "identified"
    rejection_reason: Optional[str] = None
    proposed_project_id: Optional[str] = None
    review_note: Optional[str] = None
    reviewed_at: Optional[str] = None


class FrontierGapScanReport(StrictModel):
    scan_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    topic: str
    gaps_identified: int = Field(default=0, ge=0)
    gaps_proposed: int = Field(default=0, ge=0)
    gaps_rejected: int = Field(default=0, ge=0)
    gaps_skipped_cross_scan: int = Field(default=0, ge=0)
    gaps: List[FrontierGapRecord] = Field(default_factory=list)
    existing_project_count: int = Field(default=0, ge=0)
    retrieval_mode: Literal["semantic", "lexical_fallback"] = "lexical_fallback"


class SciencePackageSpec(StrictModel):
    name: str
    version: Optional[str] = None
    extras: List[str] = Field(default_factory=list)
    markers: Optional[str] = None

    def requirement_line(self) -> str:
        extra = f"[{','.join(self.extras)}]" if self.extras else ""
        version = self.version or ""
        marker = f" ; {self.markers}" if self.markers else ""
        return f"{self.name}{extra}{version}{marker}"


class ScienceExperimentTemplate(StrictModel):
    template_id: str
    name: str
    hypothesis: str
    benchmark: str
    benchmark_family: Optional[str] = None
    metrics: List[str] = Field(default_factory=list)
    parameter_grid: Dict[str, List[Any]] = Field(default_factory=dict)
    success_criteria: List[str] = Field(default_factory=list)


class BenchmarkSpec(StrictModel):
    benchmark_id: str
    family: str
    name: str
    tier: BenchmarkTier = "validation"
    truth_status: BenchmarkTruthStatus = "unsupported"
    dataset_or_env: str
    metric_protocol: List[str] = Field(default_factory=list)
    canonical_comparable: bool = False
    required_imports: List[str] = Field(default_factory=list)
    requires_download: bool = False
    proxy_allowed: bool = True
    recommended_seed_runs: int = Field(default=1, ge=1)
    statistical_validation_required: bool = False
    notes: List[str] = Field(default_factory=list)


class BenchmarkAvailability(StrictModel):
    benchmark_id: str
    tier: BenchmarkTier = "validation"
    truth_status: BenchmarkTruthStatus = "unsupported"
    imports_ready: bool = False
    dataset_ready: bool = False
    canonical_ready: bool = False
    reason: Optional[str] = None
    missing_imports: List[str] = Field(default_factory=list)


class BenchmarkMetricStatistic(StrictModel):
    metric_name: str
    mean: float
    std_dev: Optional[float] = None
    ci95_low: Optional[float] = None
    ci95_high: Optional[float] = None
    sample_count: int = Field(default=1, ge=1)


class BenchmarkStatisticalSummary(StrictModel):
    statistically_ready: bool = False
    sample_count: int = Field(default=1, ge=1)
    recommended_seed_runs: int = Field(default=1, ge=1)
    significance_level: float = Field(default=0.05, gt=0.0, lt=1.0)
    primary_metric: Optional[str] = None
    metrics: List[BenchmarkMetricStatistic] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class BenchmarkExecutionStatisticalSummary(StrictModel):
    experiment_count: int = Field(default=0, ge=0)
    completed_experiment_count: int = Field(default=0, ge=0)
    statistically_ready_experiment_count: int = Field(default=0, ge=0)
    canonical_ready_completed_count: int = Field(default=0, ge=0)
    statistically_ready: bool = False
    notes: List[str] = Field(default_factory=list)


class ScienceProfile(StrictModel):
    profile_id: str
    domain: str
    summary: str
    keywords: List[str] = Field(default_factory=list)
    base_image: str = "pytorch/pytorch:latest"
    python_version: str = "3.10"
    cuda_support: Literal["required", "optional", "none"] = "optional"
    pip_packages: List[SciencePackageSpec] = Field(default_factory=list)
    apt_packages: List[str] = Field(default_factory=list)
    validation_imports: List[str] = Field(default_factory=list)
    benchmark_targets: List[str] = Field(default_factory=list)
    benchmark_suites: List[BenchmarkSpec] = Field(default_factory=list)
    metric_hooks: List[str] = Field(default_factory=list)
    experiment_templates: List[ScienceExperimentTemplate] = Field(default_factory=list)
    safety_rules: List[str] = Field(default_factory=lambda: ["locked_profile_only"])


class ProblemResolutionReport(StrictModel):
    problem: str
    profile_id: str
    domain: str
    confidence: float = Field(ge=0.0, le=1.0)
    matched_keywords: List[str] = Field(default_factory=list)
    summary: str
    benchmark_targets: List[str] = Field(default_factory=list)
    benchmark_catalog: List[BenchmarkSpec] = Field(default_factory=list)
    benchmark_tier: BenchmarkTier = "validation"
    requested_benchmark: Optional[str] = None
    metric_hooks: List[str] = Field(default_factory=list)
    validation_imports: List[str] = Field(default_factory=list)


class ProblemExperimentPlan(StrictModel):
    template_id: str
    name: str
    hypothesis: str
    benchmark: str
    benchmark_tier: BenchmarkTier = "validation"
    requested_benchmark_tier: BenchmarkTier = "validation"
    executed_benchmark_tier: BenchmarkTier = "validation"
    benchmark_truth_status: BenchmarkTruthStatus = "unsupported"
    benchmark_alignment: BenchmarkAlignment = "aligned"
    benchmark_spec: Optional[BenchmarkSpec] = None
    benchmark_availability: Optional[BenchmarkAvailability] = None
    canonical_comparable: bool = False
    metrics: List[str] = Field(default_factory=list)
    parameter_grid: Dict[str, List[Any]] = Field(default_factory=dict)
    success_criteria: List[str] = Field(default_factory=list)


class ScienceEnvironmentBundle(StrictModel):
    problem_id: str
    problem: str
    profile_id: str
    domain: str
    profile_hash: str
    docker_image_tag: str
    build_context_path: str
    dockerfile_path: str
    requirements_path: str
    study_plan_path: str
    execution_report_path: str
    build_command: List[str] = Field(default_factory=list)
    run_command: List[str] = Field(default_factory=list)
    pip_packages: List[str] = Field(default_factory=list)
    apt_packages: List[str] = Field(default_factory=list)
    validation_imports: List[str] = Field(default_factory=list)
    benchmark_tier: BenchmarkTier = "validation"
    requested_benchmark: Optional[str] = None
    canonical_only: bool = False
    no_proxy_benchmarks: bool = False
    image_manifest_path: Optional[str] = None
    run_manifest_path: Optional[str] = None
    image_manifest: Optional[ImageManifest] = None
    run_manifest: Optional[RunManifest] = None
    build_attestation_path: Optional[str] = None
    build_attestation: Optional[BuildAttestation] = None
    reproducibility_complete: bool = False
    locked_packages: List[str] = Field(default_factory=list)
    unresolved_packages: List[str] = Field(default_factory=list)
    lock_incomplete_reason: Optional[str] = None
    sandbox_policy: Optional[SandboxPolicy] = None
    install_policy: Literal["profile_locked_only"] = "profile_locked_only"
    build_status: Literal["not_requested", "dry_run", "built", "failed"] = "not_requested"
    build_stdout: Optional[str] = None
    build_stderr: Optional[str] = None


class ProblemStudyReport(StrictModel):
    problem_id: str
    project_id: Optional[str] = None
    thread_id: Optional[str] = None
    open_question_id: Optional[str] = None
    next_action_id: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)
    problem: str
    profile_id: str
    domain: str
    resolution_confidence: float = Field(ge=0.0, le=1.0)
    hypotheses: List[str] = Field(default_factory=list)
    benchmark_targets: List[str] = Field(default_factory=list)
    benchmark_tier: BenchmarkTier = "validation"
    requested_benchmark: Optional[str] = None
    canonical_only: bool = False
    no_proxy_benchmarks: bool = False
    canonical_comparable: bool = False
    benchmark_ids: List[str] = Field(default_factory=list)
    benchmark_names: List[str] = Field(default_factory=list)
    actual_benchmark_tiers: List[BenchmarkTier] = Field(default_factory=list)
    benchmark_truth_statuses: List[BenchmarkTruthStatus] = Field(default_factory=list)
    benchmark_alignment: BenchmarkAlignment = "aligned"
    benchmark_availability: List[BenchmarkAvailability] = Field(default_factory=list)
    metric_hooks: List[str] = Field(default_factory=list)
    cited_research_ids: List[str] = Field(default_factory=list)
    retrieved_memory_ids: List[str] = Field(default_factory=list)
    retrieval_mode: Literal["semantic", "lexical_fallback"] = "lexical_fallback"
    retrieval_conflict_count: int = Field(default=0, ge=0)
    evidence_bundle: Optional[EvidenceBundle] = None
    hypotheses: List[HypothesisRecord] = Field(default_factory=list)
    contradiction_review: Optional[ContradictionReview] = None
    experiments: List[ProblemExperimentPlan] = Field(default_factory=list)
    environment: ScienceEnvironmentBundle
    next_action: str
    notes: List[str] = Field(default_factory=list)
    status: Literal["planned", "environment_ready", "build_failed", "retrieval_degraded"] = "planned"


class ProblemExperimentResult(StrictModel):
    template_id: str
    name: str
    benchmark: str
    benchmark_id: Optional[str] = None
    benchmark_name: Optional[str] = None
    benchmark_tier: BenchmarkTier = "validation"
    requested_benchmark_tier: BenchmarkTier = "validation"
    executed_benchmark_tier: BenchmarkTier = "validation"
    benchmark_truth_status: BenchmarkTruthStatus = "unsupported"
    benchmark_alignment: BenchmarkAlignment = "aligned"
    dataset_or_env: Optional[str] = None
    canonical_comparable: bool = False
    provenance_complete: bool = False
    proxy_benchmark_used: bool = False
    execution_mode: str
    status: Literal["completed", "failed", "skipped"]
    metrics: Dict[str, float] = Field(default_factory=dict)
    statistical_summary: Optional[BenchmarkStatisticalSummary] = None
    artifact_paths: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ProblemExecutionReport(StrictModel):
    problem_id: str
    project_id: Optional[str] = None
    thread_id: Optional[str] = None
    action_id: Optional[str] = None
    executed_at: str = Field(default_factory=utc_now_iso)
    problem: str
    profile_id: str
    domain: str
    benchmark_tier: BenchmarkTier = "validation"
    requested_benchmark: Optional[str] = None
    canonical_comparable: bool = False
    proxy_benchmarks_used: bool = False
    benchmark_ids: List[str] = Field(default_factory=list)
    benchmark_names: List[str] = Field(default_factory=list)
    actual_benchmark_tiers: List[BenchmarkTier] = Field(default_factory=list)
    benchmark_truth_statuses: List[BenchmarkTruthStatus] = Field(default_factory=list)
    benchmark_alignment: BenchmarkAlignment = "aligned"
    execution_mode: Literal["local_python", "docker_bundle"]
    imports_ok: List[str] = Field(default_factory=list)
    imports_failed: List[Dict[str, str]] = Field(default_factory=list)
    benchmark_availability: List[BenchmarkAvailability] = Field(default_factory=list)
    experiments: List[ProblemExperimentResult] = Field(default_factory=list)
    benchmark_statistical_summary: Optional[BenchmarkExecutionStatisticalSummary] = None
    image_tag: Optional[str] = None
    manifest_path: Optional[str] = None
    manifest_hash: Optional[str] = None
    dependency_hash: Optional[str] = None
    build_attestation_path: Optional[str] = None
    build_attestation_id: Optional[str] = None
    image_digest: Optional[str] = None
    reproducibility_complete: bool = False
    sandbox_policy: Optional[SandboxPolicy] = None
    artifacts: List[ExecutionArtifact] = Field(default_factory=list)
    alert_ids: List[str] = Field(default_factory=list)
    claim_verdict: Optional[ClaimVerdict] = None
    summary: str
    recommended_next_step: str
    artifact_path: str
    status: Literal["completed", "dependency_failure", "partial_failure", "failed"]


class ProblemScheduleEntry(StrictModel):
    schedule_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    problem_id: str
    project_id: Optional[str] = None
    thread_id: Optional[str] = None
    action_id: Optional[str] = None
    problem: str
    profile_id: str
    domain: str
    benchmark_tier: BenchmarkTier = "validation"
    requested_benchmark: Optional[str] = None
    use_docker: bool = False
    build_env: bool = False
    next_run_at: str
    repeat_interval_s: Optional[int] = Field(default=None, ge=1)
    max_runs: int = Field(default=1, ge=1)
    run_count: int = Field(default=0, ge=0)
    attempt_count: int = Field(default=0, ge=0)
    priority: int = Field(default=0, ge=0)
    priority_score: Optional[float] = None
    priority_source: Optional[str] = None
    status: ScheduleStatus = "scheduled"
    retry_policy: RetryPolicy = Field(default_factory=lambda: RetryPolicy())
    retry_after: Optional[str] = None
    lease: Optional[RuntimeLease] = None
    last_execution_at: Optional[str] = None
    last_report_path: Optional[str] = None
    last_report_status: Optional[str] = None
    last_summary: Optional[str] = None
    last_error: Optional[str] = None
    last_manifest_path: Optional[str] = None
    terminal_failure_reason: Optional[str] = None
    crash_provenance: Optional[str] = None
    crash_at: Optional[str] = None
    recovery_required: bool = False
    recovery_confirmed_at: Optional[str] = None
    alert_ids: List[str] = Field(default_factory=list)


class ProblemScheduleState(StrictModel):
    entries: List[ProblemScheduleEntry] = Field(default_factory=list)


class SchedulerCycleReport(StrictModel):
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: str = Field(default_factory=utc_now_iso)
    due_count: int = Field(default=0, ge=0)
    executed_count: int = Field(default=0, ge=0)
    leased_count: int = Field(default=0, ge=0)
    retry_wait_count: int = Field(default=0, ge=0)
    completed_schedule_ids: List[str] = Field(default_factory=list)
    rescheduled_schedule_ids: List[str] = Field(default_factory=list)
    failed_schedule_ids: List[str] = Field(default_factory=list)
    alert_ids: List[str] = Field(default_factory=list)
    updated_entries: List[ProblemScheduleEntry] = Field(default_factory=list)


class CalibrationBin(StrictModel):
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)


class CalibrationReport(StrictModel):
    ece: float = Field(ge=0.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    bins: List[CalibrationBin] = Field(default_factory=list)


class SeedRunResult(StrictModel):
    seed: int
    training_loss: float = Field(ge=0.0)
    effective_dimensionality: float = Field(ge=0.0)
    equilibrium_fraction: float = Field(ge=0.0, le=1.0)
    calibration_ece: float = Field(ge=0.0)


class SeedVarianceReport(StrictModel):
    num_runs: int = Field(ge=0)
    loss_mean: float = Field(ge=0.0)
    loss_std: float = Field(ge=0.0)
    dimensionality_mean: float = Field(ge=0.0)
    dimensionality_std: float = Field(ge=0.0)
    calibration_ece_mean: float = Field(ge=0.0)
    stable: bool = False
    runs: List[SeedRunResult] = Field(default_factory=list)


class AblationResult(StrictModel):
    name: str
    training_loss: float = Field(ge=0.0)
    effective_dimensionality: float = Field(ge=0.0)
    equilibrium_fraction: float = Field(ge=0.0, le=1.0)
    calibration_ece: float = Field(ge=0.0)
    score: float
    delta_vs_control: float


class VerificationReport(StrictModel):
    trial_id: str
    verified_at: str = Field(default_factory=utc_now_iso)
    control_score: float
    seed_variance: SeedVarianceReport
    calibration: CalibrationReport
    ablations: List[AblationResult] = Field(default_factory=list)
    verdict: Literal["verified", "unstable", "inconclusive"]
    recommendations: List[str] = Field(default_factory=list)


class ClaimAcceptancePolicy(StrictModel):
    policy_id: str = "frontier_claim_policy_v1"
    min_seed_runs: int = Field(default=3, ge=1)
    max_seed_loss_std: float = Field(default=0.08, ge=0.0)
    max_seed_dimensionality_std: float = Field(default=0.75, ge=0.0)
    max_calibration_ece: float = Field(default=0.15, ge=0.0)
    min_ablation_gap: float = Field(default=0.05, ge=0.0)
    min_supporting_sources: int = Field(default=2, ge=0)
    max_allowed_contradictions: int = Field(default=0, ge=0)
    require_canonical_benchmark: bool = False


class ContradictionReview(StrictModel):
    review_id: str
    query: str
    conflicting_document_ids: List[str] = Field(default_factory=list)
    conflicting_claim_ids: List[str] = Field(default_factory=list)
    contradiction_count: int = Field(default=0, ge=0)
    summary: str
    recommended_resolution: str
    severity: Literal["none", "low", "medium", "high"] = "none"


class EvidenceBundle(StrictModel):
    bundle_id: str
    query: str
    traces: List["EvidenceTrace"] = Field(default_factory=list)
    supporting_document_ids: List[str] = Field(default_factory=list)
    supporting_claim_ids: List[str] = Field(default_factory=list)
    contradiction_review: Optional[ContradictionReview] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: List[str] = Field(default_factory=list)


class HypothesisRecord(StrictModel):
    hypothesis_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    problem: str
    hypothesis: str
    rationale: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_bundle_id: str
    supporting_document_ids: List[str] = Field(default_factory=list)
    supporting_claim_ids: List[str] = Field(default_factory=list)
    contradiction_review_id: Optional[str] = None
    proposed_benchmark_ids: List[str] = Field(default_factory=list)
    unresolved_assumptions: List[str] = Field(default_factory=list)


class ResearchDecisionRecord(StrictModel):
    decision_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    prompt: str
    mode: Literal["research_chat", "problem_study", "claim_review"] = "research_chat"
    trial_id: Optional[str] = None
    problem_id: Optional[str] = None
    thread_id: Optional[str] = None
    action_id: Optional[str] = None
    evidence_bundle: EvidenceBundle
    hypotheses: List[HypothesisRecord] = Field(default_factory=list)
    selected_action: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_review: Optional[ContradictionReview] = None
    claim_verdict_id: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class ClaimVerdict(StrictModel):
    verdict_id: str
    trial_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    decision_scope: Literal["trial_local"] = "trial_local"
    status: Literal["accepted", "provisional", "rejected", "insufficient_evidence", "contradicted"]
    rationale: List[str] = Field(default_factory=list)
    policy: ClaimAcceptancePolicy
    supporting_research_ids: List[str] = Field(default_factory=list)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    verification_report_trial_id: str
    benchmark_problem_id: Optional[str] = None
    benchmark_execution_created_at: Optional[str] = None
    benchmark_execution_mode: Optional[str] = None
    supporting_benchmark_ids: List[str] = Field(default_factory=list)
    supporting_benchmark_names: List[str] = Field(default_factory=list)
    evidence_bundle_id: Optional[str] = None
    canonical_comparability_source: Literal["problem_execution", "problem_study", "none", "ambiguous"] = "none"
    verdict_inputs_complete: bool = False
    linkage_status: Literal["exact", "none", "ambiguous"] = "none"
    linkage_note: Optional[str] = None
    contradiction_review: Optional[ContradictionReview] = None
    canonical_benchmark_required: bool = False
    canonical_benchmark_satisfied: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    review_required_before: Optional[str] = None
    escalated_at: Optional[str] = None
    escalation_reason: Optional[str] = None
    lifecycle_status: Literal["active", "aging", "escalated", "resolved"] = "active"

    @model_validator(mode="after")
    def normalize_lifecycle(self) -> "ClaimVerdict":
        if self.status in {"accepted", "rejected", "contradicted"} and self.lifecycle_status == "active":
            self.lifecycle_status = "resolved"
        if self.escalated_at and self.lifecycle_status in {"active", "aging"}:
            self.lifecycle_status = "escalated"
        return self


class ResearchBudgetLedger(StrictModel):
    wall_clock_minutes_budget: float = Field(default=180.0, ge=0.0)
    wall_clock_minutes_spent: float = Field(default=0.0, ge=0.0)
    gpu_hours_budget: float = Field(default=4.0, ge=0.0)
    gpu_hours_spent: float = Field(default=0.0, ge=0.0)
    experiment_budget: int = Field(default=6, ge=0)
    experiments_spent: int = Field(default=0, ge=0)
    replication_budget: int = Field(default=2, ge=0)
    replications_spent: int = Field(default=0, ge=0)
    budget_exhausted: bool = False
    budget_pressure_level: ResearchBudgetPressure = "low"


class PrioritizationPolicy(StrictModel):
    mode: PrioritizationPolicyMode = "balanced"
    evidence_gain_weight: float = Field(default=0.35, ge=0.0)
    falsification_weight: float = Field(default=0.2, ge=0.0)
    uncertainty_reduction_weight: float = Field(default=0.15, ge=0.0)
    benchmark_value_weight: float = Field(default=0.1, ge=0.0)
    replication_value_weight: float = Field(default=0.05, ge=0.0)
    contradiction_urgency_weight: float = Field(default=0.1, ge=0.0)
    strategic_priority_weight: float = Field(default=0.1, ge=0.0)
    dependency_readiness_weight: float = Field(default=0.15, ge=0.0)
    cost_penalty_weight: float = Field(default=0.25, ge=0.0)
    budget_pressure_penalty_weight: float = Field(default=0.15, ge=0.0)


class ActionScoreBreakdown(StrictModel):
    expected_evidence_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    falsification_value: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty_reduction: float = Field(default=0.0, ge=0.0, le=1.0)
    benchmark_value: float = Field(default=0.0, ge=0.0, le=1.0)
    replication_value: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_urgency: float = Field(default=0.0, ge=0.0, le=1.0)
    strategic_priority: float = Field(default=0.0, ge=0.0, le=1.0)
    dependency_readiness: float = Field(default=0.0, ge=0.0, le=1.0)
    cost_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    budget_pressure_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    total_score: float = 0.0


class PrioritizedActionCandidate(StrictModel):
    candidate_id: str
    project_id: str
    thread_id: str
    action_id: str
    project_title: str
    domain_profile: str
    project_status: ResearchProjectStatus
    action_kind: ResearchActionKind = "custom"
    action_status: ResearchActionStatus = "planned"
    action_description: str
    current_question: Optional[str] = None
    budget_pressure_level: ResearchBudgetPressure = "low"
    blocked: bool = False
    recommended: bool = False
    score: float = 0.0
    score_breakdown: ActionScoreBreakdown = Field(default_factory=ActionScoreBreakdown)
    rationale: List[str] = Field(default_factory=list)


class PortfolioPrioritySnapshot(StrictModel):
    snapshot_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    project_id: Optional[str] = None
    policy: PrioritizationPolicy = Field(default_factory=PrioritizationPolicy)
    candidate_count: int = Field(default=0, ge=0)
    active_project_count: int = Field(default=0, ge=0)
    blocked_project_count: int = Field(default=0, ge=0)
    selected_project_id: Optional[str] = None
    selected_action_id: Optional[str] = None
    candidates: List[PrioritizedActionCandidate] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class BudgetAllocationDecision(StrictModel):
    decision_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    policy: PrioritizationPolicy = Field(default_factory=PrioritizationPolicy)
    selected_candidate: Optional[PrioritizedActionCandidate] = None
    scheduled_job_id: Optional[str] = None
    schedule_created: bool = False
    rationale: List[str] = Field(default_factory=list)
    considered_candidates: List[PrioritizedActionCandidate] = Field(default_factory=list)


class ResearchOpenQuestion(StrictModel):
    question_id: str
    project_id: str
    thread_id: str
    question: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertainty_type: str = "unknown"
    blocking: bool = False
    status: ResearchQuestionStatus = "open"
    created_at: str = Field(default_factory=utc_now_iso)
    resolved_at: Optional[str] = None


class ResearchPlannedAction(StrictModel):
    action_id: str
    project_id: str
    thread_id: str
    action_kind: ResearchActionKind = "custom"
    description: str
    estimated_cost: float = Field(default=0.0, ge=0.0)
    expected_evidence_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    depends_on: List[str] = Field(default_factory=list)
    status: ResearchActionStatus = "planned"
    falsification_plan_id: Optional[str] = None
    falsification_test_id: Optional[str] = None
    scheduled_job_id: Optional[str] = None
    result_refs: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class FalsificationTrigger(StrictModel):
    trigger_type: FalsificationTriggerType
    reason: str
    severity: FalsificationSeverity = "medium"
    evidence_refs: List[str] = Field(default_factory=list)


class FalsificationCoverage(StrictModel):
    ablation_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    replication_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    benchmark_pressure_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    calibration_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_sufficient: bool = False


class FalsificationTest(StrictModel):
    test_id: str
    plan_id: str
    project_id: str
    thread_id: str
    kind: FalsificationTestKind
    description: str
    estimated_cost: float = Field(default=0.0, ge=0.0)
    expected_falsification_value: float = Field(default=0.0, ge=0.0, le=1.0)
    depends_on: List[str] = Field(default_factory=list)
    status: FalsificationTestStatus = "planned"
    result_summary: Optional[str] = None
    linked_action_id: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class FalsificationPlan(StrictModel):
    plan_id: str
    project_id: str
    thread_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    status: FalsificationPlanStatus = "active"
    trigger_reason: str
    triggers: List[FalsificationTrigger] = Field(default_factory=list)
    tests: List[FalsificationTest] = Field(default_factory=list)
    coverage: FalsificationCoverage = Field(default_factory=FalsificationCoverage)
    notes: List[str] = Field(default_factory=list)


class EvidenceDebtRecord(StrictModel):
    record_id: str
    project_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    falsification_gap: float = Field(default=0.0, ge=0.0, le=1.0)
    replication_gap: float = Field(default=0.0, ge=0.0, le=1.0)
    benchmark_gap: float = Field(default=0.0, ge=0.0, le=1.0)
    claim_linkage_gap: float = Field(default=0.0, ge=0.0, le=1.0)
    calibration_gap: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_debt: float = Field(default=0.0, ge=0.0, le=1.0)
    promotion_blocked: bool = False
    rationale: List[str] = Field(default_factory=list)


class ProjectStalenessRecord(StrictModel):
    record_id: str
    project_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    last_progress_at: Optional[str] = None
    hours_since_progress: float = Field(default=0.0, ge=0.0)
    staleness_level: ResearchStalenessLevel = "fresh"
    reason: str = ""
    resume_candidate: bool = False
    closure_candidate: bool = False


class ProjectPriorityRecord(StrictModel):
    record_id: str
    project_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    action_id: Optional[str] = None
    priority_score: float = 0.0
    strategic_priority: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_value: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_debt: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    staleness_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    budget_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    benchmark_readiness: float = Field(default=0.0, ge=0.0, le=1.0)
    recommended_state: ResearchPortfolioRecommendation = "defer"
    rationale: List[str] = Field(default_factory=list)


class PortfolioHealthSnapshot(StrictModel):
    created_at: str = Field(default_factory=utc_now_iso)
    total_projects: int = Field(default=0, ge=0)
    active_projects: int = Field(default=0, ge=0)
    paused_projects: int = Field(default=0, ge=0)
    blocked_projects: int = Field(default=0, ge=0)
    stale_projects: int = Field(default=0, ge=0)
    parked_projects: int = Field(default=0, ge=0)
    completed_projects: int = Field(default=0, ge=0)
    abandoned_projects: int = Field(default=0, ge=0)
    resume_candidates: int = Field(default=0, ge=0)
    promotion_blocked_projects: int = Field(default=0, ge=0)
    selected_project_id: Optional[str] = None


class ResearchPortfolio(StrictModel):
    portfolio_id: str = "portfolio-main"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    active_project_ids: List[str] = Field(default_factory=list)
    paused_project_ids: List[str] = Field(default_factory=list)
    blocked_project_ids: List[str] = Field(default_factory=list)
    stale_project_ids: List[str] = Field(default_factory=list)
    parked_project_ids: List[str] = Field(default_factory=list)
    completed_project_ids: List[str] = Field(default_factory=list)
    abandoned_project_ids: List[str] = Field(default_factory=list)
    latest_decision_id: Optional[str] = None
    latest_selected_project_id: Optional[str] = None
    health_snapshot: PortfolioHealthSnapshot = Field(default_factory=PortfolioHealthSnapshot)


class PortfolioDecision(StrictModel):
    decision_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    selected_project_id: Optional[str] = None
    selected_action_id: Optional[str] = None
    deferred_project_ids: List[str] = Field(default_factory=list)
    parked_project_ids: List[str] = Field(default_factory=list)
    resumed_project_ids: List[str] = Field(default_factory=list)
    escalated_project_ids: List[str] = Field(default_factory=list)
    retired_project_ids: List[str] = Field(default_factory=list)
    rationale: List[str] = Field(default_factory=list)
    policy_snapshot: Optional[PrioritizationPolicy] = None
    project_priority_records: List[ProjectPriorityRecord] = Field(default_factory=list)


class ResearchResumeSnapshot(StrictModel):
    project_id: str
    active_thread_id: Optional[str] = None
    current_question_id: Optional[str] = None
    next_action_id: Optional[str] = None
    latest_evidence_summary: str
    blockers: List[str] = Field(default_factory=list)
    budget_remaining_summary: Dict[str, float] = Field(default_factory=dict)
    captured_at: str = Field(default_factory=utc_now_iso)


class ResearchHypothesisThread(StrictModel):
    thread_id: str
    project_id: str
    hypothesis: str
    status: ResearchThreadStatus = "open"
    confidence_state: ResearchConfidenceState = "unknown"
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    open_question_ids: List[str] = Field(default_factory=list)
    next_action_id: Optional[str] = None
    stop_reason: Optional[ResearchStopReason] = None
    resume_reason: Optional[ResearchResumeReason] = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class ResearchProject(StrictModel):
    project_id: str
    title: str
    goal: str
    domain_profile: str
    status: ResearchProjectStatus = "active"
    priority: int = Field(default=0, ge=0)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    active_thread_id: Optional[str] = None
    budget_ledger: ResearchBudgetLedger = Field(default_factory=ResearchBudgetLedger)
    resume_snapshot: Optional[ResearchResumeSnapshot] = None
    latest_decision_summary: str = ""
    hypothesis_threads: List[ResearchHypothesisThread] = Field(default_factory=list)
    open_questions: List[ResearchOpenQuestion] = Field(default_factory=list)
    planned_actions: List[ResearchPlannedAction] = Field(default_factory=list)


class ResearchProjectState(StrictModel):
    entries: List[ResearchProject] = Field(default_factory=list)


class PublicationClaimBundle(StrictModel):
    verdict_id: str
    disposition: PublicationClaimDisposition
    summary: str
    rationale: List[str] = Field(default_factory=list)
    trial_id: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_research_ids: List[str] = Field(default_factory=list)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    benchmark_names: List[str] = Field(default_factory=list)
    canonical_benchmark_required: bool = False
    canonical_benchmark_satisfied: bool = False
    linkage_status: Literal["exact", "none", "ambiguous"] = "none"


class PublicationAlternativeBundle(StrictModel):
    alternative_id: str
    source: PublicationAlternativeSource
    status: PublicationClaimDisposition
    summary: str
    why_rejected: List[str] = Field(default_factory=list)
    related_verdict_id: Optional[str] = None


class PublicationBenchmarkAttachment(StrictModel):
    source_id: str
    source_kind: Literal["problem_study", "problem_execution", "claim_verdict"]
    benchmark_ids: List[str] = Field(default_factory=list)
    benchmark_names: List[str] = Field(default_factory=list)
    benchmark_truth_statuses: List[BenchmarkTruthStatus] = Field(default_factory=list)
    benchmark_alignment: BenchmarkAlignment = "aligned"
    canonical_comparable: bool = False
    requested_tier: Optional[BenchmarkTier] = None
    actual_tiers: List[BenchmarkTier] = Field(default_factory=list)


class PublicationLineageEntry(StrictModel):
    event_id: str
    timestamp: str = Field(default_factory=utc_now_iso)
    event_type: PublicationLineageEventType
    summary: str
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PublicationHandoffPackage(StrictModel):
    package_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    project_id: str
    project_title: str
    domain_profile: str
    package_status: PublicationPackageStatus = "not_ready"
    project_status: ResearchProjectStatus
    latest_evidence_summary: str
    claim_readiness_summary: str
    accepted_claims: List[PublicationClaimBundle] = Field(default_factory=list)
    provisional_claims: List[PublicationClaimBundle] = Field(default_factory=list)
    rejected_alternatives: List[PublicationAlternativeBundle] = Field(default_factory=list)
    experiment_lineage: List[PublicationLineageEntry] = Field(default_factory=list)
    benchmark_truth_attachments: List[PublicationBenchmarkAttachment] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    evidence_gaps: List[str] = Field(default_factory=list)
    writer_cautions: List[str] = Field(default_factory=list)
    positioning_report_id: Optional[str] = None
    artifact_path: Optional[str] = None


class BreakthroughReport(StrictModel):
    trial_id: str
    generated_at: str = Field(default_factory=utc_now_iso)
    status: Literal["breakthrough", "candidate", "rejected"]
    summary: str
    novelty_score: float = Field(ge=0.0)
    stability_score: float = Field(ge=0.0)
    calibration_score: float = Field(ge=0.0)
    supporting_research_ids: List[str] = Field(default_factory=list)
    rationale: List[str] = Field(default_factory=list)
    verification: VerificationReport
    claim_verdict: Optional[ClaimVerdict] = None
    surprise_score: float = Field(default=0.0, ge=0.0, le=1.0)
    prior_contradiction_score: float = Field(default=0.0, ge=0.0, le=1.0)


class DatasetChangeProposal(StrictModel):
    stream_name: Literal["anchor", "research"]
    action: Literal["keep", "refresh", "append", "replace"]
    dataset_name: Optional[str] = None
    subset: Optional[str] = None
    split: Optional[str] = None
    rationale: str


class LabChatResponse(StrictModel):
    mode: Literal["director_chat"] = "director_chat"
    response_text: str
    state_summary: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cited_trial_ids: List[str] = Field(default_factory=list)
    retrieved_memories: List[MemorySearchHit] = Field(default_factory=list)
    evidence_traces: List["EvidenceTrace"] = Field(default_factory=list)
    evidence_bundle: Optional[EvidenceBundle] = None
    hypotheses: List[HypothesisRecord] = Field(default_factory=list)
    contradiction_review: Optional[ContradictionReview] = None
    claim_verdict: Optional[ClaimVerdict] = None
    dataset_change: Optional[DatasetChangeProposal] = None


class SelfCorrectionNote(StrictModel):
    trial_id: str
    outcome: Literal["completed", "fail_fast", "pivoted", "dry_run"]
    energy_e: float = Field(ge=0.0)
    entropy_sigma: float = Field(ge=0.0)
    drift_rho: float = Field(ge=0.0)
    effective_dimensionality: float = Field(default=0.0, ge=0.0)
    equilibrium_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str
    corrective_action: str
    similar_trials: List[str] = Field(default_factory=list)


class GovernorDecision(StrictModel):
    action: Literal["continue", "terminate"]
    reasons: List[str] = Field(default_factory=list)
    metrics: GovernorMetrics


class RecoveryState(StrictModel):
    trial_id: Optional[str] = None
    status: Literal["idle", "planning", "running", "fail_fast", "completed", "pivoted"] = "idle"
    updated_at: str = Field(default_factory=utc_now_iso)
    last_known_stable_hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    last_fail_reason: Optional[str] = None
    last_fail_metrics: Optional[GovernorMetrics] = None
    consecutive_fail_fast: int = Field(default=0, ge=0)
    last_strategy_family: Optional[str] = None
    last_anchor_path: Optional[str] = None
    max_effective_dimensionality_achieved: float = Field(default=0.0, ge=0.0)


class KnowledgeGraphEntry(StrictModel):
    trial_id: str
    started_at: str = Field(default_factory=utc_now_iso)
    ended_at: Optional[str] = None
    strategy_family: str
    outcome: Literal["running", "completed", "fail_fast", "pivoted", "dry_run"]
    fail_reason: Optional[str] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphState(StrictModel):
    entries: List[KnowledgeGraphEntry] = Field(default_factory=list)


class FailureAutopsy(StrictModel):
    trial_id: Optional[str] = None
    status: Literal["no_failure", "fail_fast"]
    reason_of_death: Optional[str] = None
    strategy_family: Optional[str] = None
    metrics: Optional[GovernorMetrics] = None
    consecutive_fail_fast: int = Field(default=0, ge=0)


class DryRunReport(StrictModel):
    trial_id: str
    json_schema_ok: bool
    docker_command_ok: bool
    memory_ok: bool
    pivot_force_ready: bool
    governor_action: Literal["continue", "terminate"]
    recovery_status: str
    composed_command: List[str]


class LiveDockerTestReport(StrictModel):
    launched: bool
    image: str
    mode: str
    command: List[str]
    returncode: Optional[int] = None
    payload_config_path: Optional[str] = None
    gpu_visible: Optional[bool] = None
    gpu_probe_output: Optional[str] = None
    error: Optional[str] = None


class ExperimentBackendSpec(StrictModel):
    backend_id: str
    summary: str
    domain: str
    entrypoint: str
    status: BackendStatus = "scaffold"
    control_only: bool = False
    research_grade_capable: bool = False
    expected_data_type: str = "text"
    requires_tokenizer: bool = False
    supports_resume: bool = False
    supports_distributed: bool = False
    requires_gpu: bool = False
    required_deps: List[str] = Field(default_factory=list)
    governor_observables: List[str] = Field(default_factory=lambda: ["D_PR", "sigma", "rho"])
    required_metrics: List[str] = Field(default_factory=list)
    required_artifacts: List[str] = Field(default_factory=list)
    valid_input_contract: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ExperimentResumeInfo(StrictModel):
    supported: bool = False
    requested: bool = False
    mode: Literal["fresh_start", "checkpoint_resume"] = "fresh_start"
    resume_from_checkpoint: Optional[str] = None
    latest_checkpoint_path: Optional[str] = None
    checkpoint_exists: bool = False
    reason: Optional[str] = None


class ExperimentArtifactLineage(StrictModel):
    training_log_path: Optional[str] = None
    latest_checkpoint_path: Optional[str] = None
    final_checkpoint_path: Optional[str] = None
    summary_path: Optional[str] = None


class ExperimentBackendRuntimeRecord(StrictModel):
    trial_name: str
    backend_id: str
    status: ExperimentBackendRunStatus = "planned"
    output_dir: str
    manifest_path: Optional[str] = None
    backend_state_path: Optional[str] = None
    supports_resume: bool = False
    resume: ExperimentResumeInfo = Field(default_factory=ExperimentResumeInfo)
    artifact_lineage: ExperimentArtifactLineage = Field(default_factory=ExperimentArtifactLineage)
    completed_steps: int = Field(default=0, ge=0)
    current_epoch: int = Field(default=0, ge=0)
    epoch_step: int = Field(default=0, ge=0)
    launch_count: int = Field(default=0, ge=0)
    last_error: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    last_started_at: Optional[str] = None
    last_completed_at: Optional[str] = None
    last_heartbeat_at: Optional[str] = None


class ExperimentLaunchPlan(StrictModel):
    backend: ExperimentBackendSpec
    trial_name: str
    command: List[str]
    runtime: RuntimeSpec
    output_dir: str
    manifest_path: str
    backend_state_path: Optional[str] = None
    resume: ExperimentResumeInfo = Field(default_factory=ExperimentResumeInfo)
    artifact_lineage: ExperimentArtifactLineage = Field(default_factory=ExperimentArtifactLineage)
    config: Dict[str, Any] = Field(default_factory=dict)


class PaperSourceFingerprint(StrictModel):
    source_hash_sha256: str
    source_size_bytes: int = Field(default=0, ge=0)
    source_kind: Literal["pdf", "text", "other"] = "other"
    normalized_path: Optional[str] = None
    dedupe_key: Optional[str] = None


class PaperSection(StrictModel):
    section_id: str
    heading: str
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text_hash: Optional[str] = None
    word_count: int = Field(default=0, ge=0)


class BibliographyEntry(StrictModel):
    entry_id: str
    paper_id: str
    citation_key: str
    raw_text: str
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    page_number: Optional[int] = None
    source_excerpt: Optional[str] = None


class ResearchClaim(StrictModel):
    claim_id: str
    paper_id: str
    section_id: str
    label: Literal["fact", "measured_result", "inference", "hypothesis"]
    text: str
    citations: List[str] = Field(default_factory=list)
    citation_entry_ids: List[str] = Field(default_factory=list)
    polarity: Literal["positive", "negative", "neutral"] = "neutral"
    page_number: Optional[int] = None
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    citation_span_start: Optional[int] = None
    citation_span_end: Optional[int] = None
    evidence_kind: Literal["claim_sentence", "claim_clause", "table_caption", "figure_caption"] = "claim_sentence"
    citation_count: int = Field(default=0, ge=0)
    quality_flags: List[str] = Field(default_factory=list)
    source_excerpt: Optional[str] = None


class CitationEdge(StrictModel):
    source_paper_id: str
    citation_key: str
    raw_text: str
    section_id: Optional[str] = None
    page_number: Optional[int] = None
    bibliography_entry_id: Optional[str] = None
    citation_style: Literal["numeric", "author_year", "unknown"] = "unknown"
    source_excerpt: Optional[str] = None


class TableMetricHint(StrictModel):
    metric_name: str
    value: Optional[float] = None
    row_index: int = Field(default=0, ge=0)
    column_index: int = Field(default=0, ge=0)
    row_label: Optional[str] = None
    column_label: Optional[str] = None


class PaperTable(StrictModel):
    table_id: str
    section_id: Optional[str] = None
    caption: str
    raw_text: str
    rows: List[List[str]] = Field(default_factory=list)
    header: List[str] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)
    column_count: int = Field(default=0, ge=0)
    numeric_cell_count: int = Field(default=0, ge=0)
    metric_hints: List[TableMetricHint] = Field(default_factory=list)
    related_claim_ids: List[str] = Field(default_factory=list)
    page_number: Optional[int] = None
    context_excerpt: Optional[str] = None


class PaperFigure(StrictModel):
    figure_id: str
    section_id: Optional[str] = None
    caption: str
    raw_text: str
    source: Literal["text", "pdf_image", "ocr"] = "text"
    figure_label: Optional[str] = None
    caption_hash: Optional[str] = None
    ocr_text_present: bool = False
    related_claim_ids: List[str] = Field(default_factory=list)
    page_number: Optional[int] = None
    context_excerpt: Optional[str] = None


class ClaimCluster(StrictModel):
    cluster_id: str
    claim_ids: List[str] = Field(default_factory=list)
    topic_terms: List[str] = Field(default_factory=list)
    contradiction_pairs: List[List[str]] = Field(default_factory=list)
    evidence_count: int = Field(default=0, ge=0)
    paper_count: int = Field(default=0, ge=0)
    cross_paper: bool = False
    polarity_distribution: Dict[str, int] = Field(default_factory=dict)


class ClaimConflict(StrictModel):
    left_claim_id: str
    right_claim_id: str
    reason: str
    score: float = Field(ge=0.0, le=1.0)
    conflict_kind: Literal["cluster_polarity", "cross_paper_topic_polarity"] = "cross_paper_topic_polarity"
    shared_token_count: int = Field(default=0, ge=0)
    left_paper_id: Optional[str] = None
    right_paper_id: Optional[str] = None
    left_page_number: Optional[int] = None
    right_page_number: Optional[int] = None
    topic_terms: List[str] = Field(default_factory=list)


class LiteratureCapabilityReport(StrictModel):
    parser_chain: List[str] = Field(default_factory=list)
    parser_used: Optional[str] = None
    semantic_model: str = "BAAI/bge-small-en-v1.5"
    semantic_ready: bool = False
    reranker: str = "scientific-hybrid-reranker"
    reranker_ready: bool = False
    ocr_engine: Optional[str] = None
    ocr_ready: bool = False
    page_render_ready: bool = False
    notes: List[str] = Field(default_factory=list)


class EvidenceTrace(StrictModel):
    document_id: str
    kind: str
    paper_id: Optional[str] = None
    paper_title: Optional[str] = None
    claim_id: Optional[str] = None
    section_id: Optional[str] = None
    page_number: Optional[int] = None
    score: Optional[float] = None
    source_path: Optional[str] = None
    source_hash_sha256: Optional[str] = None
    excerpt: str = ""
    bibliography_entry_ids: List[str] = Field(default_factory=list)
    contradiction_count: int = Field(default=0, ge=0)
    contradiction_summary: List[str] = Field(default_factory=list)


class LiteratureIngestManifest(StrictModel):
    manifest_id: str
    requested_paths: List[str] = Field(default_factory=list)
    resolved_paths: List[str] = Field(default_factory=list)
    artifact_ids: List[str] = Field(default_factory=list)
    artifact_count: int = Field(default=0, ge=0)
    deduplicated_existing: int = Field(default=0, ge=0)
    stored_total: int = Field(default=0, ge=0)
    conflict_count: int = Field(default=0, ge=0)
    failed: List[Dict[str, str]] = Field(default_factory=list)
    parser_chain: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    manifest_path: Optional[str] = None


class PaperArtifact(StrictModel):
    paper_id: str
    source_path: str
    canonical_source_path: Optional[str] = None
    title: str
    abstract: str = ""
    sections: List[PaperSection] = Field(default_factory=list)
    claims: List[ResearchClaim] = Field(default_factory=list)
    citations: List[CitationEdge] = Field(default_factory=list)
    bibliography: List[BibliographyEntry] = Field(default_factory=list)
    tables: List[PaperTable] = Field(default_factory=list)
    figures: List[PaperFigure] = Field(default_factory=list)
    claim_clusters: List[ClaimCluster] = Field(default_factory=list)
    ocr_used: bool = False
    parser_used: Optional[str] = None
    page_count: int = Field(default=0, ge=0)
    source_fingerprint: Optional[PaperSourceFingerprint] = None
    ingest_manifest_id: Optional[str] = None
    capability_report: Optional[LiteratureCapabilityReport] = None
    extraction_notes: List[str] = Field(default_factory=list)
    extracted_at: str = Field(default_factory=utc_now_iso)
    stored_at: str = Field(default_factory=utc_now_iso)


class PaperIngestReport(StrictModel):
    requested_paths: List[str] = Field(default_factory=list)
    ingested: int = Field(default=0, ge=0)
    deduplicated_existing: int = Field(default=0, ge=0)
    stored_total: int = Field(default=0, ge=0)
    failed: List[Dict[str, str]] = Field(default_factory=list)
    artifacts: List[PaperArtifact] = Field(default_factory=list)
    conflicts: List[ClaimConflict] = Field(default_factory=list)
    capability_report: Optional[LiteratureCapabilityReport] = None
    manifest_id: Optional[str] = None
    manifest_path: Optional[str] = None
    latest_manifest: Optional[LiteratureIngestManifest] = None


class DependencyPackageRecord(StrictModel):
    requested_spec: str
    normalized_name: str
    resolved_spec: Optional[str] = None
    version: Optional[str] = None
    required: bool = True
    resolution_status: DependencyResolutionStatus = "pinned"


class DependencyLockManifest(StrictModel):
    manifest_version: Literal["tar.repro.v1"] = "tar.repro.v1"
    lock_id: str
    requirements_path: str
    packages: List[str] = Field(default_factory=list)
    package_records: List[DependencyPackageRecord] = Field(default_factory=list)
    unresolved_packages: List[str] = Field(default_factory=list)
    fully_pinned: bool = False
    lock_incomplete_reason: Optional[str] = None
    hash_sha256: str


class EnvironmentFingerprint(StrictModel):
    manifest_version: Literal["tar.repro.v1"] = "tar.repro.v1"
    fingerprint_id: str
    workspace_root: str
    source_hash_sha256: str
    dockerfile_hash_sha256: str
    requirements_hash_sha256: str
    python_version: str


class BuildAttestation(StrictModel):
    manifest_version: Literal["tar.build.v1"] = "tar.build.v1"
    attestation_id: str
    scope_kind: Literal["payload_environment", "science_bundle"]
    image_tag: str
    build_command: List[str] = Field(default_factory=list)
    builder_backend: Literal["dry_run", "subprocess", "docker_sdk"]
    build_status: Literal["not_requested", "dry_run", "built", "failed"] = "not_requested"
    returncode: Optional[int] = None
    built_at: str = Field(default_factory=utc_now_iso)
    image_manifest_hash: str
    dependency_lock_hash: str
    environment_fingerprint_id: str
    run_manifest_hash: Optional[str] = None
    image_digest: Optional[str] = None
    image_id: Optional[str] = None
    digest_source: Literal["docker_inspect", "docker_sdk", "unavailable"] = "unavailable"
    trial_id: Optional[str] = None
    problem_id: Optional[str] = None


class ImageManifest(StrictModel):
    manifest_version: Literal["tar.repro.v1"] = "tar.repro.v1"
    image_tag: str
    base_image: str
    dockerfile_path: str
    build_context_path: str
    build_command: List[str] = Field(default_factory=list)
    dependency_lock: DependencyLockManifest
    environment_fingerprint: EnvironmentFingerprint
    hash_sha256: str
    locked: bool = False
    build_status: Literal["not_requested", "dry_run", "built", "failed"] = "not_requested"


class SandboxPolicy(StrictModel):
    mode: SandboxExecutionMode = "docker_only"
    profile: SandboxProfile = "production"
    network_policy: SandboxNetworkPolicy = "off"
    allowed_mounts: List[str] = Field(default_factory=list)
    read_only_mounts: List[str] = Field(default_factory=list)
    writable_mounts: List[str] = Field(default_factory=list)
    seccomp_profile_path: Optional[str] = None
    capability_drop: List[str] = Field(default_factory=lambda: ["ALL"])
    workspace_read_only: bool = True
    cpu_limit: int = Field(default=1, ge=1)
    memory_limit_gb: int = Field(default=1, ge=1)
    timeout_s: int = Field(default=30, ge=1)
    artifact_dir: Optional[str] = None
    workspace_root: Optional[str] = None


class ExecutionArtifact(StrictModel):
    path: str
    kind: str
    sha256: Optional[str] = None
    size_bytes: Optional[int] = Field(default=None, ge=0)


class SandboxExecutionReport(StrictModel):
    ok: bool
    mode: SandboxExecutionMode = "docker_only"
    output: str = ""
    error: Optional[str] = None
    image: str
    command: List[str] = Field(default_factory=list)
    sandbox_policy: SandboxPolicy
    artifacts: List[ExecutionArtifact] = Field(default_factory=list)
    sandbox_audit_log: List[str] = Field(default_factory=list)


class RunManifest(StrictModel):
    manifest_version: Literal["tar.run.v1"] = "tar.run.v1"
    manifest_id: str
    kind: Literal["payload", "science_bundle", "sandbox_exec"]
    trial_id: Optional[str] = None
    problem_id: Optional[str] = None
    command: List[str] = Field(default_factory=list)
    config_path: Optional[str] = None
    image_manifest: ImageManifest
    sandbox_policy: SandboxPolicy
    created_at: str = Field(default_factory=utc_now_iso)
    hash_sha256: str
    reproducibility_complete: bool = False
    unresolved_packages: List[str] = Field(default_factory=list)
    lock_incomplete_reason: Optional[str] = None


class MemoryStoreManifest(StrictModel):
    manifest_version: Literal["tar.memory.v1"] = "tar.memory.v1"
    schema_version: int = Field(default=1, ge=1)
    fingerprint: str
    collection_name: str
    embedder_name: str
    embedding_dim: int = Field(ge=1)
    semantic_research_ready: bool = False
    state: MemoryStoreHealth = "rebuild_required"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    last_rebuild_at: Optional[str] = None
    last_error: Optional[str] = None
    retired_collection_names: List[str] = Field(default_factory=list)


class PayloadEnvironmentReport(StrictModel):
    image_tag: str
    dockerfile_path: str
    requirements_path: str
    manifest_path: str
    build_command: List[str] = Field(default_factory=list)
    build_status: Literal["not_requested", "dry_run", "built", "failed"] = "not_requested"
    packages: List[str] = Field(default_factory=list)
    package_records: List[DependencyPackageRecord] = Field(default_factory=list)
    unresolved_packages: List[str] = Field(default_factory=list)
    lock_incomplete_reason: Optional[str] = None
    image_manifest: Optional[ImageManifest] = None
    run_manifest: Optional[RunManifest] = None
    build_attestation_path: Optional[str] = None
    build_attestation: Optional[BuildAttestation] = None
    reproducibility_complete: bool = False


class RuntimeHeartbeat(StrictModel):
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: Optional[str] = None
    status: Literal["idle", "running", "completed", "failed"] = "idle"
    executed_jobs: int = Field(default=0, ge=0)
    stale_cleanups: int = Field(default=0, ge=0)
    failed_jobs: int = Field(default=0, ge=0)
    active_leases: int = Field(default=0, ge=0)
    retry_waiting: int = Field(default=0, ge=0)
    alert_count: int = Field(default=0, ge=0)
    escalated_verdicts: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class RetryPolicy(StrictModel):
    max_attempts: int = Field(default=3, ge=1)
    base_delay_s: int = Field(default=30, ge=1)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    max_delay_s: int = Field(default=900, ge=1)


class TARRuntimePolicy(StrictModel):
    policy_version: str = "ws34_pre.v1"
    verdict_aging_days: int = Field(default=14, ge=1)


class SelfImprovementPolicy(StrictModel):
    max_consecutive_gate_failures: int = Field(default=3, ge=1)
    max_auto_cycles: int = Field(default=5, ge=1)
    min_delta_signals: int = Field(default=20, ge=1)
    min_diversity_score: float = Field(default=0.4, ge=0.0, le=1.0)
    overclaim_hard_limit: float = 0.0
    min_mean_score_floor: float = Field(default=0.40, ge=0.0)


class TARExecutionPolicy(StrictModel):
    require_sandbox_for_generated_code: bool = True
    require_sandbox_for_external_code: bool = True
    allowed_unsandboxed_paths: List[str] = Field(default_factory=list)
    policy_version: str = "ws35.v1"


class RuntimeLease(StrictModel):
    owner_id: str
    acquired_at: str = Field(default_factory=utc_now_iso)
    heartbeat_at: str = Field(default_factory=utc_now_iso)
    expires_at: str
    attempt: int = Field(default=1, ge=1)
    heartbeat_interval_s: int = Field(default=30, ge=1)


class AlertRecord(StrictModel):
    alert_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    severity: AlertSeverity = "warning"
    source: str
    message: str
    related_schedule_id: Optional[str] = None
    related_manifest_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CheckpointRecord(StrictModel):
    name: str
    model_path: str
    backend: Literal["transformers", "vllm"]
    role: Literal["assistant", "director", "strategist", "scout"] = "assistant"
    checkpoint_kind: Literal["base", "adapter"] = "base"
    base_model_id: Optional[str] = None
    adapter_path: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CheckpointRegistryState(StrictModel):
    entries: List[CheckpointRecord] = Field(default_factory=list)


class EndpointHealth(StrictModel):
    endpoint_name: str
    checked_at: str = Field(default_factory=utc_now_iso)
    status: Literal["unknown", "starting", "healthy", "unhealthy", "stopped", "failed"] = "unknown"
    ok: bool = False
    http_status: Optional[int] = None
    latency_ms: Optional[float] = Field(default=None, ge=0.0)
    detail: Optional[str] = None
    model_id: Optional[str] = None
    backend: Optional[str] = None
    role: Optional[Literal["assistant", "director", "strategist", "scout"]] = None
    trust_remote_code: Optional[bool] = None


class EndpointRecord(StrictModel):
    endpoint_name: str
    checkpoint_name: str
    role: Literal["assistant", "director", "strategist", "scout"]
    host: str
    port: int = Field(ge=1, le=65535)
    backend: Literal["transformers", "vllm"]
    base_url: str
    command: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    status: Literal["registered", "starting", "running", "stopped", "failed"] = "registered"
    process_pid: Optional[int] = None
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    last_error: Optional[str] = None
    last_health_at: Optional[str] = None
    manifest_path: Optional[str] = None
    stdout_log_path: Optional[str] = None
    stderr_log_path: Optional[str] = None
    trust_remote_code: bool = False
    health: Optional[EndpointHealth] = None


class EndpointRegistryState(StrictModel):
    entries: List[EndpointRecord] = Field(default_factory=list)


class RoleAssignment(StrictModel):
    role: Literal["assistant", "director", "strategist", "scout"]
    checkpoint_name: str
    endpoint_name: Optional[str] = None
    assigned_at: str = Field(default_factory=utc_now_iso)
    status: Literal["assigned", "unassigned"] = "assigned"


class RoleAssignmentState(StrictModel):
    entries: List[RoleAssignment] = Field(default_factory=list)


class OperatorServingState(StrictModel):
    active_checkpoint_name: Optional[str] = None
    mode: Literal["prompt_only", "tuned_local"] = "tuned_local"
    role: Literal["assistant", "director", "strategist", "scout"] = "assistant"
    endpoint_name: Optional[str] = None
    selected_at: str = Field(default_factory=utc_now_iso)


class OperatorServingStatus(StrictModel):
    state: OperatorServingState = Field(default_factory=OperatorServingState)
    checkpoint: Optional[CheckpointRecord] = None
    endpoint: Optional[EndpointRecord] = None
    role_assignment: Optional[RoleAssignment] = None


class InferenceEndpointPlan(StrictModel):
    checkpoint: CheckpointRecord
    host: str
    port: int = Field(ge=1, le=65535)
    base_url: str
    command: List[str]
    env: Dict[str, str] = Field(default_factory=dict)
    endpoint_name: Optional[str] = None
    role: Optional[Literal["assistant", "director", "strategist", "scout"]] = None
    trust_remote_code: bool = False
    manifest_path: Optional[str] = None


class FrontierStatus(StrictModel):
    experiment_backends: List[ExperimentBackendSpec] = Field(default_factory=list)
    experiment_backend_runtime_records: List[ExperimentBackendRuntimeRecord] = Field(default_factory=list)
    payload_environment: Optional[PayloadEnvironmentReport] = None
    literature_artifacts: int = Field(default=0, ge=0)
    literature_conflicts: int = Field(default=0, ge=0)
    literature_manifests: int = Field(default=0, ge=0)
    latest_literature_manifest: Optional[LiteratureIngestManifest] = None
    embedder: str = "BAAI/bge-small-en-v1.5"
    semantic_research_ready: bool = False
    reranker: str = "scientific-hybrid-reranker"
    reranker_ready: bool = False
    literature_capabilities: Optional[LiteratureCapabilityReport] = None
    runtime_heartbeat: Optional[RuntimeHeartbeat] = None
    registered_checkpoints: List[CheckpointRecord] = Field(default_factory=list)
    managed_endpoints: List[EndpointRecord] = Field(default_factory=list)
    role_assignments: List[RoleAssignment] = Field(default_factory=list)
    operator_serving: Optional[OperatorServingStatus] = None
    claim_policy: Optional[ClaimAcceptancePolicy] = None
    recent_claim_verdicts: List[ClaimVerdict] = Field(default_factory=list)
    benchmark_profiles: Dict[str, int] = Field(default_factory=dict)
    frontier_gap_counts: Dict[str, int] = Field(default_factory=dict)
    frontier_gap_scans: int = Field(default=0, ge=0)
    latest_frontier_gap_scan: Optional[FrontierGapScanReport] = None
    recent_frontier_gaps: List[FrontierGapRecord] = Field(default_factory=list)
    safe_execution_mode: str
    active_leases: int = Field(default=0, ge=0)
    alert_count: int = Field(default=0, ge=0)
    reproducibility_ready: bool = False


class ControlRequest(StrictModel):
    command: Literal[
        "status",
        "frontier_status",
        "runtime_status",
        "queue_health",
        "dry_run",
        "pivot",
        "explain_last_fail",
        "panic",
        "live_docker_test",
        "chat",
        "check_regime",
        "ingest_research",
        "scan_frontier_gaps",
        "list_frontier_gaps",
        "list_frontier_gap_scans",
        "propose_projects_from_gaps",
        "promote_gap_project",
        "reject_gap_project",
        "propose_experiment_family",
        "list_family_proposals",
        "approve_family_proposal",
        "reject_family_proposal",
        "run_family_feasibility",
        "list_registered_families",
        "initialize_anchor_pack",
        "curate_training_signal",
        "list_training_signals",
        "assemble_curated_delta",
        "run_self_improvement_probe",
        "run_self_improvement_run1",
        "deploy_improved_adapter",
        "self_improvement_status",
        "resume_self_improvement",
        "routing_summary",
        "routing_log",
        "load_frontier_config",
        "get_anomaly_elevations",
        "get_competing_theories",
        "get_head_to_head_plans",
        "get_theory_invalidations",
        "get_positioning_reports",
        "get_positioning_report",
        "plan_baseline_comparison",
        "get_comparison_plans",
        "get_comparison_result",
        "get_reproducibility_packages",
        "create_reproducibility_package",
        "run_agenda_review",
        "agenda_status",
        "list_agenda_decisions",
        "veto_agenda_decision",
        "commit_agenda_decisions",
        "agenda_config",
        "verify_last_trial",
        "breakthrough_report",
        "resolve_problem",
        "prepare_science_env",
        "study_problem",
        "run_problem_study",
        "schedule_problem_study",
        "create_project",
        "list_projects",
        "project_status",
        "pause_project",
        "resume_project",
        "next_action",
        "operator_view",
        "project_timeline",
        "evidence_map",
        "claim_lineage",
        "resume_dashboard",
        "portfolio_status",
        "portfolio_review",
        "portfolio_decide",
        "rank_actions",
        "allocate_budget",
        "prioritization_log",
        "generate_falsification_plan",
        "falsification_status",
        "falsification_log",
        "stale_projects",
        "evidence_debt",
        "resume_candidates",
        "publication_handoff",
        "publication_log",
        "scheduler_status",
        "run_scheduler_once",
        "list_benchmarks",
        "benchmark_status",
        "prepare_payload_env",
        "rebuild_locked_image",
        "show_manifest",
        "ingest_papers",
        "literature_status",
        "list_paper_artifacts",
        "paper_artifact",
        "literature_conflicts",
        "list_experiment_backends",
        "experiment_backend_runtime_status",
        "run_runtime_cycle",
        "list_alerts",
        "retry_failed_job",
        "confirm_recovery",
        "cancel_job",
        "sandbox_policy",
        "register_checkpoint",
        "list_checkpoints",
        "build_inference_endpoint",
        "list_endpoints",
        "start_endpoint",
        "stop_endpoint",
        "restart_endpoint",
        "endpoint_health",
        "assign_role",
        "select_operator_checkpoint",
        "operator_serving_status",
        "claim_policy",
        "claim_verdict",
        "research_decision_log",
    ]
    payload: Dict[str, Any] = Field(default_factory=dict)


class ControlResponse(StrictModel):
    ok: bool
    command: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
