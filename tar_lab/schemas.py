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
    effective_dimensionality: float = Field(default=0.0, ge=0.0)
    dimensionality_ratio: float = Field(default=0.0, ge=0.0)
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
    effective_dimensionality: float = Field(default=0.0, ge=0.0)
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


class LocalLLMConfig(StrictModel):
    base_url: str
    api_key: str = "local"
    model: str
    temperature: float = 0.0
    timeout_s: float = Field(default=120.0, gt=0.0)
    max_retries: int = Field(default=3, ge=1, le=10)


class DatasetSourceConfig(StrictModel):
    name: str
    split: str = "train"
    subset: Optional[str] = None
    text_fields: List[str] = Field(default_factory=lambda: ["text", "content", "abstract", "code"])
    max_samples: Optional[int] = Field(default=None, ge=1)
    streaming: bool = False


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


class PreparedDataBundle(StrictModel):
    anchor_manifest_path: str
    research_manifest_path: str
    anchor_manifest: DatasetManifest
    research_manifest: DatasetManifest


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
    dry_run: bool = False


class TrainingPayloadConfig(StrictModel):
    version: Literal["tar.v1"] = "tar.v1"
    trial_id: str
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
    log_path: str
    output_dir: str
    anchor_manifest_path: Optional[str] = None
    research_manifest_path: Optional[str] = None
    governor_thresholds: GovernorThresholds
    protected_layers: List[str]
    mutable_layers: List[str]
    notes: Dict[str, Any] = Field(default_factory=dict)


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
    cited_trial_ids: List[str] = Field(default_factory=list)
    retrieved_memories: List[MemorySearchHit] = Field(default_factory=list)
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


class ControlRequest(StrictModel):
    command: Literal[
        "status",
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
    ]
    payload: Dict[str, Any] = Field(default_factory=dict)


class ControlResponse(StrictModel):
    ok: bool
    command: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
