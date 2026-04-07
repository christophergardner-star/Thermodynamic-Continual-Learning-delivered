from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


DataAccessMode = Literal["OFFLINE_FALLBACK", "CACHED_REAL", "DOWNLOAD_REAL"]
DataPurity = Literal["fallback", "cached_real", "download_real", "local_real", "mixed"]
RunIntent = Literal["control", "plumbing", "research"]
BackendStatus = Literal["executable", "scaffold"]
BenchmarkTier = Literal["smoke", "validation", "canonical"]
BenchmarkTruthStatus = Literal["canonical_ready", "validation_only", "smoke_only", "unsupported"]
BenchmarkAlignment = Literal["aligned", "downgraded", "refused", "mixed"]
ScheduleStatus = Literal["scheduled", "leased", "running", "retry_wait", "completed", "failed_terminal", "cancelled"]
AlertSeverity = Literal["info", "warning", "error", "critical"]
SandboxExecutionMode = Literal["docker_only"]
SandboxNetworkPolicy = Literal["off", "restricted", "profile_required"]
SandboxProfile = Literal["production", "dev_override"]
MemoryStoreHealth = Literal["healthy", "rebuild_required", "rebuilding", "degraded"]
DependencyResolutionStatus = Literal["pinned", "missing_version", "optional_missing"]


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
        data = info.data
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


class ResearchDocument(StrictModel):
    document_id: str
    source_kind: Literal["arxiv", "rss", "manual"]
    source_name: str
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
    evidence_bundle: Optional[EvidenceBundle] = None
    hypotheses: List[HypothesisRecord] = Field(default_factory=list)
    contradiction_review: Optional[ContradictionReview] = None
    experiments: List[ProblemExperimentPlan] = Field(default_factory=list)
    environment: ScienceEnvironmentBundle
    next_action: str
    status: Literal["planned", "environment_ready", "build_failed"] = "planned"


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
    artifact_paths: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ProblemExecutionReport(StrictModel):
    problem_id: str
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
    image_tag: Optional[str] = None
    manifest_path: Optional[str] = None
    manifest_hash: Optional[str] = None
    dependency_hash: Optional[str] = None
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


class ExperimentLaunchPlan(StrictModel):
    backend: ExperimentBackendSpec
    trial_name: str
    command: List[str]
    runtime: RuntimeSpec
    output_dir: str
    manifest_path: str
    config: Dict[str, Any] = Field(default_factory=dict)


class PaperSection(StrictModel):
    section_id: str
    heading: str
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


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
    source_excerpt: Optional[str] = None


class CitationEdge(StrictModel):
    source_paper_id: str
    citation_key: str
    raw_text: str
    page_number: Optional[int] = None
    bibliography_entry_id: Optional[str] = None
    citation_style: Literal["numeric", "author_year", "unknown"] = "unknown"
    source_excerpt: Optional[str] = None


class PaperTable(StrictModel):
    table_id: str
    section_id: Optional[str] = None
    caption: str
    raw_text: str
    rows: List[List[str]] = Field(default_factory=list)
    page_number: Optional[int] = None
    context_excerpt: Optional[str] = None


class PaperFigure(StrictModel):
    figure_id: str
    section_id: Optional[str] = None
    caption: str
    raw_text: str
    source: Literal["text", "pdf_image", "ocr"] = "text"
    page_number: Optional[int] = None
    context_excerpt: Optional[str] = None


class ClaimCluster(StrictModel):
    cluster_id: str
    claim_ids: List[str] = Field(default_factory=list)
    topic_terms: List[str] = Field(default_factory=list)
    contradiction_pairs: List[List[str]] = Field(default_factory=list)
    evidence_count: int = Field(default=0, ge=0)


class ClaimConflict(StrictModel):
    left_claim_id: str
    right_claim_id: str
    reason: str
    score: float = Field(ge=0.0, le=1.0)
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
    excerpt: str = ""
    bibliography_entry_ids: List[str] = Field(default_factory=list)
    contradiction_count: int = Field(default=0, ge=0)
    contradiction_summary: List[str] = Field(default_factory=list)


class PaperArtifact(StrictModel):
    paper_id: str
    source_path: str
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
    capability_report: Optional[LiteratureCapabilityReport] = None
    extraction_notes: List[str] = Field(default_factory=list)
    extracted_at: str = Field(default_factory=utc_now_iso)


class PaperIngestReport(StrictModel):
    requested_paths: List[str] = Field(default_factory=list)
    ingested: int = Field(default=0, ge=0)
    failed: List[Dict[str, str]] = Field(default_factory=list)
    artifacts: List[PaperArtifact] = Field(default_factory=list)
    conflicts: List[ClaimConflict] = Field(default_factory=list)
    capability_report: Optional[LiteratureCapabilityReport] = None


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
    notes: List[str] = Field(default_factory=list)


class RetryPolicy(StrictModel):
    max_attempts: int = Field(default=3, ge=1)
    base_delay_s: int = Field(default=30, ge=1)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    max_delay_s: int = Field(default=900, ge=1)


class RuntimeLease(StrictModel):
    owner_id: str
    acquired_at: str = Field(default_factory=utc_now_iso)
    heartbeat_at: str = Field(default_factory=utc_now_iso)
    expires_at: str
    attempt: int = Field(default=1, ge=1)


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
    payload_environment: Optional[PayloadEnvironmentReport] = None
    literature_artifacts: int = Field(default=0, ge=0)
    literature_conflicts: int = Field(default=0, ge=0)
    embedder: str = "BAAI/bge-small-en-v1.5"
    semantic_research_ready: bool = False
    reranker: str = "scientific-hybrid-reranker"
    reranker_ready: bool = False
    literature_capabilities: Optional[LiteratureCapabilityReport] = None
    runtime_heartbeat: Optional[RuntimeHeartbeat] = None
    registered_checkpoints: List[CheckpointRecord] = Field(default_factory=list)
    managed_endpoints: List[EndpointRecord] = Field(default_factory=list)
    role_assignments: List[RoleAssignment] = Field(default_factory=list)
    claim_policy: Optional[ClaimAcceptancePolicy] = None
    recent_claim_verdicts: List[ClaimVerdict] = Field(default_factory=list)
    benchmark_profiles: Dict[str, int] = Field(default_factory=dict)
    safe_execution_mode: str
    active_leases: int = Field(default=0, ge=0)
    alert_count: int = Field(default=0, ge=0)
    reproducibility_ready: bool = False


class ControlRequest(StrictModel):
    command: Literal[
        "status",
        "frontier_status",
        "runtime_status",
        "dry_run",
        "pivot",
        "explain_last_fail",
        "panic",
        "live_docker_test",
        "chat",
        "check_regime",
        "ingest_research",
        "verify_last_trial",
        "breakthrough_report",
        "resolve_problem",
        "prepare_science_env",
        "study_problem",
        "run_problem_study",
        "schedule_problem_study",
        "scheduler_status",
        "run_scheduler_once",
        "list_benchmarks",
        "benchmark_status",
        "prepare_payload_env",
        "rebuild_locked_image",
        "show_manifest",
        "ingest_papers",
        "list_experiment_backends",
        "run_runtime_cycle",
        "list_alerts",
        "retry_failed_job",
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
