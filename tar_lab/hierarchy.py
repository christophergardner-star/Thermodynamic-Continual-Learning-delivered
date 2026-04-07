from __future__ import annotations

import json
import os
import re
import hashlib
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tar_lab.schemas import (
    DatasetChangeProposal,
    DirectorPolicy,
    EvidenceTrace,
    EvidenceBundle,
    GovernorMetrics,
    GovernorThresholds,
    HypothesisRecord,
    LabChatResponse,
    LocalLLMConfig,
    MemorySearchHit,
    QuantitativeJustification,
    RuntimeSpec,
    ScoutTask,
    StrategistPlan,
    SelfCorrectionNote,
    ContradictionReview,
    ClaimVerdict,
)
from tar_lab.state import TARStateStore

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


class _DraftModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DirectorDraft(_DraftModel):
    experiment_family: str
    anchor_path: str
    pivot_required: bool = False
    objective_slug: str = "thermodynamic-anchor"


class StrategistDraft(_DraftModel):
    strategy_family: str
    fim_lambda: float = Field(gt=0.0)
    bregman_budget: float = Field(gt=0.0)
    drift_budget: float = Field(gt=0.0)
    protected_layers: List[str]
    mutable_layers: List[str]
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class ScoutDraft(_DraftModel):
    training_entrypoint: str = "tar_lab/train_template.py"
    image: str = "pytorch/pytorch:latest"
    steps: int = Field(default=20, ge=1, le=5000)
    batch_size: int = Field(default=8, ge=1, le=1024)
    power_limit_w: int = Field(default=300, ge=1)
    gpu_target_temp_c: int = Field(default=70, ge=1, le=95)


class DirectorChatDraft(_DraftModel):
    response_text: str
    state_summary: str
    cited_trial_ids: List[str] = Field(default_factory=list)
    dataset_change: Optional[DatasetChangeProposal] = None


class SelfCorrectionDraft(_DraftModel):
    outcome: str
    energy_e: float = Field(ge=0.0)
    entropy_sigma: float = Field(ge=0.0)
    drift_rho: float = Field(ge=0.0)
    effective_dimensionality: float = Field(default=0.0, ge=0.0)
    equilibrium_fraction: float = Field(default=0.0, ge=0.0)
    explanation: str
    corrective_action: str
    similar_trials: List[str] = Field(default_factory=list)


def _memory_aware_hyperparameters(
    hits: List[MemorySearchHit],
    default_alpha: float,
    default_eta: float,
) -> Dict[str, float]:
    alpha_candidates: list[float] = []
    eta_candidates: list[float] = []
    for hit in hits:
        metadata = hit.metadata or {}
        outcome = str(metadata.get("outcome", ""))
        if outcome not in {"completed", "dry_run", ""}:
            continue
        alpha = metadata.get("alpha")
        eta = metadata.get("eta")
        if isinstance(alpha, (int, float)) and alpha > 0:
            alpha_candidates.append(float(alpha))
        if isinstance(eta, (int, float)) and eta > 0:
            eta_candidates.append(float(eta))
    if alpha_candidates:
        default_alpha = sum(alpha_candidates) / len(alpha_candidates)
    if eta_candidates:
        default_eta = sum(eta_candidates) / len(eta_candidates)
    return {"alpha": round(default_alpha, 5), "eta": round(default_eta, 5)}


def _slopes(points: List[GovernorMetrics]) -> tuple[float, float, float, float]:
    if len(points) < 2:
        return 0.0, 0.0, 0.0, 0.0
    return (
        points[-1].energy_e - points[0].energy_e,
        points[-1].entropy_sigma - points[0].entropy_sigma,
        points[-1].drift_rho - points[0].drift_rho,
        points[-1].effective_dimensionality - points[0].effective_dimensionality,
    )


def _quantitative_justification(points: List[GovernorMetrics]) -> QuantitativeJustification:
    latest = points[-1]
    energy_slope, entropy_slope, drift_slope, dimensionality_slope = _slopes(points)
    return QuantitativeJustification(
        energy_e=latest.energy_e,
        entropy_sigma=latest.entropy_sigma,
        drift_rho=latest.drift_rho,
        grad_norm=latest.grad_norm,
        regime_rho=latest.regime_rho,
        effective_dimensionality=latest.effective_dimensionality,
        effective_dimensionality_std_err=latest.effective_dimensionality_std_err,
        equilibrium_fraction=latest.equilibrium_fraction,
        energy_slope=energy_slope,
        entropy_slope=entropy_slope,
        drift_slope=drift_slope,
        dimensionality_slope=dimensionality_slope,
    )


def _evidence_traces_from_hits(hits: List[MemorySearchHit]) -> List[EvidenceTrace]:
    traces: list[EvidenceTrace] = []
    for hit in hits[:3]:
        metadata = hit.metadata or {}
        raw_trace = metadata.get("evidence_trace")
        if isinstance(raw_trace, dict):
            try:
                traces.append(EvidenceTrace.model_validate(raw_trace))
                continue
            except Exception:
                pass
        contradictions = metadata.get("contradictory_claims") or []
        contradiction_summary = [
            f"{item.get('left_claim_id')} vs {item.get('right_claim_id')}: {item.get('reason')}"
            for item in contradictions[:3]
            if isinstance(item, dict)
        ]
        citation_entry_ids = metadata.get("citation_entry_ids") or []
        bibliography_entry_id = metadata.get("bibliography_entry_id")
        if bibliography_entry_id and bibliography_entry_id not in citation_entry_ids:
            citation_entry_ids = [bibliography_entry_id, *citation_entry_ids]
        traces.append(
            EvidenceTrace(
                document_id=hit.document_id,
                kind=str(metadata.get("kind", "memory")),
                paper_id=metadata.get("paper_id"),
                paper_title=metadata.get("paper_title"),
                claim_id=metadata.get("claim_id"),
                section_id=metadata.get("section_id"),
                page_number=metadata.get("page_number") if isinstance(metadata.get("page_number"), int) and metadata.get("page_number") != -1 else None,
                score=hit.score,
                source_path=metadata.get("source_path"),
                excerpt=str(metadata.get("source_excerpt") or hit.document[:320]),
                bibliography_entry_ids=[str(item) for item in citation_entry_ids if item],
                contradiction_count=len(contradictions),
                contradiction_summary=contradiction_summary,
            )
        )
    return traces


def _stable_id(prefix: str, payload: str) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _build_contradiction_review(prompt: str, traces: List[EvidenceTrace]) -> Optional[ContradictionReview]:
    conflicting_document_ids: list[str] = []
    conflicting_claim_ids: list[str] = []
    summaries: list[str] = []
    for trace in traces:
        if trace.contradiction_count <= 0:
            continue
        conflicting_document_ids.append(trace.document_id)
        if trace.claim_id:
            conflicting_claim_ids.append(trace.claim_id)
        summaries.extend(trace.contradiction_summary[:2])
    if not conflicting_document_ids:
        return None
    contradiction_count = len(summaries) if summaries else len(conflicting_document_ids)
    if contradiction_count >= 3:
        severity = "high"
    elif contradiction_count == 2:
        severity = "medium"
    else:
        severity = "low"
    return ContradictionReview(
        review_id=_stable_id("contradiction", prompt + "|" + "|".join(sorted(conflicting_document_ids))),
        query=prompt,
        conflicting_document_ids=sorted(set(conflicting_document_ids)),
        conflicting_claim_ids=sorted(set(conflicting_claim_ids)),
        contradiction_count=contradiction_count,
        summary=" ; ".join(summaries[:4]) or "Retrieved evidence contains materially conflicting claims.",
        recommended_resolution=(
            "Run an experiment that discriminates between the conflicting claims and treat any current conclusion as provisional."
        ),
        severity=severity,  # type: ignore[arg-type]
    )


def build_evidence_bundle(query: str, hits: List[MemorySearchHit]) -> EvidenceBundle:
    traces = _evidence_traces_from_hits(hits)
    contradiction_review = _build_contradiction_review(query, traces)
    supporting_document_ids = [trace.document_id for trace in traces]
    supporting_claim_ids = [trace.claim_id for trace in traces if trace.claim_id]
    confidence = min(1.0, 0.2 + (0.15 * len(traces)) - (0.1 if contradiction_review else 0.0))
    notes = []
    if not traces:
        notes.append("No evidence traces were retrieved.")
    if contradiction_review is not None:
        notes.append("Contradictory evidence is present and must be resolved before claiming success.")
    return EvidenceBundle(
        bundle_id=_stable_id("evidence", query + "|" + "|".join(supporting_document_ids)),
        query=query,
        traces=traces,
        supporting_document_ids=supporting_document_ids,
        supporting_claim_ids=supporting_claim_ids,
        contradiction_review=contradiction_review,
        confidence=max(0.0, round(confidence, 6)),
        notes=notes,
    )


def build_hypotheses(
    problem: str,
    evidence_bundle: EvidenceBundle,
    *,
    benchmark_ids: Optional[List[str]] = None,
) -> List[HypothesisRecord]:
    benchmark_ids = benchmark_ids or []
    traces = evidence_bundle.traces
    if not traces:
        return [
            HypothesisRecord(
                hypothesis_id=_stable_id("hypothesis", problem + "|fallback"),
                problem=problem,
                hypothesis="Evidence is insufficient; the next step is to ingest higher-quality literature and benchmark evidence.",
                rationale="No traceable evidence was available in memory.",
                confidence=0.15,
                evidence_bundle_id=evidence_bundle.bundle_id,
                proposed_benchmark_ids=benchmark_ids[:2],
                unresolved_assumptions=["The current evidence base is too weak to support a scientific claim."],
            )
        ]
    first = traces[0]
    hypothesis_text = (
        f"The dominant mechanism in '{problem}' is linked to evidence from "
        f"{first.paper_title or first.document_id}, and should be tested against the current benchmark plan."
    )
    unresolved = []
    if evidence_bundle.contradiction_review is not None:
        unresolved.append("Contradictory literature remains unresolved.")
    if first.page_number is None:
        unresolved.append("Some evidence lacks page-level provenance.")
    return [
        HypothesisRecord(
            hypothesis_id=_stable_id("hypothesis", problem + "|" + first.document_id),
            problem=problem,
            hypothesis=hypothesis_text,
            rationale=(
                f"Primary evidence trace: {first.paper_title or first.document_id}"
                f"{f' p.{first.page_number}' if first.page_number is not None else ''}."
            ),
            confidence=max(0.0, min(1.0, evidence_bundle.confidence)),
            evidence_bundle_id=evidence_bundle.bundle_id,
            supporting_document_ids=evidence_bundle.supporting_document_ids[:4],
            supporting_claim_ids=evidence_bundle.supporting_claim_ids[:4],
            contradiction_review_id=evidence_bundle.contradiction_review.review_id if evidence_bundle.contradiction_review else None,
            proposed_benchmark_ids=benchmark_ids[:3],
            unresolved_assumptions=unresolved,
        )
    ]


def _role_config_from_env(role: str) -> Optional[LocalLLMConfig]:
    role_key = role.upper()
    base_url = os.environ.get(f"TAR_{role_key}_BASE_URL") or os.environ.get("TAR_LLM_BASE_URL")
    model = os.environ.get(f"TAR_{role_key}_MODEL")
    if not base_url or not model:
        return None
    return LocalLLMConfig(
        base_url=base_url,
        api_key=os.environ.get(f"TAR_{role_key}_API_KEY") or os.environ.get("TAR_LLM_API_KEY", "local"),
        model=model,
        temperature=float(os.environ.get(f"TAR_{role_key}_TEMPERATURE") or os.environ.get("TAR_LLM_TEMPERATURE", "0.0")),
        timeout_s=float(os.environ.get(f"TAR_{role_key}_TIMEOUT_S") or os.environ.get("TAR_LLM_TIMEOUT_S", "120")),
        max_retries=int(os.environ.get(f"TAR_{role_key}_MAX_RETRIES") or os.environ.get("TAR_LLM_MAX_RETRIES", "3")),
    )


def _extract_json(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.S)
    if fenced:
        return fenced.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _message_content(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "\n".join(part for part in parts if part)
    return str(raw)


class LocalOpenAIRole:
    def __init__(
        self,
        role_name: str,
        config: LocalLLMConfig,
        draft_model: type[_DraftModel],
        client_factory: Optional[Callable[..., Any]] = None,
    ):
        self.role_name = role_name
        self.config = config
        self.draft_model = draft_model
        self.client_factory = client_factory

    def _client(self) -> Any:
        if self.client_factory is not None:
            return self.client_factory(self.config)
        if OpenAI is None:
            raise RuntimeError("openai is not installed; install it to enable live TAR hierarchy inference")
        return OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout_s,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> _DraftModel:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        client = self._client()
        schema = json.dumps(self.draft_model.model_json_schema(), indent=2)

        for attempt in range(self.config.max_retries):
            raw = self._chat(client, messages)
            try:
                parsed = json.loads(_extract_json(raw))
                return self.draft_model.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(
                        f"{self.role_name} failed schema validation after {self.config.max_retries} attempts: {exc}"
                    ) from exc
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Invalid Schema, fix it. Return JSON only that matches this schema exactly.\n"
                                f"{schema}\n"
                                f"Validation error: {exc}"
                            ),
                        },
                    ]
                )
        raise RuntimeError(f"{self.role_name} did not produce valid JSON")

    def _chat(self, client: Any, messages: List[Dict[str, str]]) -> str:
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )
        return _message_content(response.choices[0].message.content)


class RuleDirector:
    families = ["elastic_anchor", "ou_drift_jitter", "layer_freeze"]

    def propose(self, store: TARStateStore, trial_id: str, objective_slug: str) -> DirectorPolicy:
        recent = store.tail_metrics(3)
        if len(recent) != 3:
            raise ValueError("Director requires exactly three recent log points from logs/")

        recovery = store.load_recovery()
        last_family = recovery.last_strategy_family or self.families[0]
        failure_streak = recovery.consecutive_fail_fast
        pivot_required = failure_streak >= 3
        experiment_family = last_family
        if pivot_required:
            current_idx = self.families.index(last_family) if last_family in self.families else 0
            experiment_family = self.families[(current_idx + 1) % len(self.families)]

        anchor_path = recovery.last_anchor_path or "anchors/thermodynamic_anchor.safetensors"
        return DirectorPolicy(
            trial_id=trial_id,
            objective_slug=objective_slug,
            anchor_path=anchor_path,
            experiment_family=experiment_family,
            pivot_required=pivot_required,
            failure_streak=failure_streak,
            quantitative_justification=_quantitative_justification(recent),
            data_anchor=recent,
        )

    def chat(
        self,
        prompt: str,
        store: TARStateStore,
        memory_hits: Optional[List[MemorySearchHit]] = None,
    ) -> LabChatResponse:
        recent = store.tail_metrics(3)
        recovery = store.load_recovery()
        latest = recent[-1] if recent else None
        memory_hits = memory_hits or []
        if latest is None:
            state_summary = "No thermodynamic data is available yet."
        else:
            state_summary = (
                f"Current stability uses E={latest.energy_e:.6f}, "
                f"sigma={latest.entropy_sigma:.6f}, rho={latest.drift_rho:.6f}, "
                f"grad={latest.grad_norm:.6f}, D_PR={latest.effective_dimensionality:.4f}, "
                f"eq={latest.equilibrium_fraction:.2%}, status={recovery.status}."
            )
        cited_trial_ids = [
            str(hit.metadata.get("trial_id") or hit.metadata.get("document_id") or hit.document_id)
            for hit in memory_hits
            if hit.metadata.get("trial_id") or hit.metadata.get("document_id") or hit.document_id
        ][:3]
        evidence_bundle = build_evidence_bundle(prompt, memory_hits)
        evidence_traces = evidence_bundle.traces
        hypotheses = build_hypotheses(prompt, evidence_bundle)
        response_text = state_summary
        prompt_lower = prompt.lower()
        if any(term in prompt_lower for term in ("current ai", "ai problems", "research landscape", "frontier")):
            research_hits = [hit for hit in memory_hits if str(hit.metadata.get("kind", "")) == "research"]
            paper_hits = [hit for hit in memory_hits if str(hit.metadata.get("kind", "")).startswith("paper_")]
            if research_hits:
                problem_lines = []
                for hit in research_hits[:3]:
                    problem_lines.append(hit.document.split(".")[0].strip())
                response_text = (
                    "Current AI problem scan: "
                    + " | ".join(problem_lines)
                    + ". "
                    + state_summary
                )
                if evidence_traces:
                    trace = evidence_traces[0]
                    if trace.paper_title or trace.page_number:
                        response_text += (
                            f" Evidence anchor: {(trace.paper_title or trace.document_id)}"
                            f"{f' p.{trace.page_number}' if trace.page_number is not None else ''}."
                        )
            elif paper_hits:
                response_text = (
                    "Paper-grounded research scan: "
                    + " | ".join(hit.document.split(".")[0].strip() for hit in paper_hits[:3])
                    + ". "
                    + state_summary
                )
                if evidence_traces:
                    response_text += " Evidence trace(s): " + " ; ".join(
                        f"{trace.paper_title or trace.document_id}{f' p.{trace.page_number}' if trace.page_number is not None else ''}"
                        for trace in evidence_traces[:2]
                    ) + "."
            else:
                response_text = f"No research memory is indexed yet. {state_summary}"
        elif "stability" in prompt_lower:
            if (
                latest is not None
                and latest.dimensionality_ratio > 0.0
                and latest.training_loss is not None
                and latest.training_loss <= 1.20
                and latest.dimensionality_ratio < 0.35
            ):
                response_text += " Regime warning: thermodynamic quenching is likely because loss remains low while D_PR has collapsed."
            if memory_hits:
                response_text += (
                    " Similar trials indicate prior recovery patterns for "
                    + ", ".join(cited_trial_ids or ["uncited"])
                    + "."
                )
        else:
            response_text = f"Director received intent: {prompt.strip()}. {state_summary}"
        return LabChatResponse(
            response_text=response_text,
            state_summary=state_summary,
            confidence=evidence_bundle.confidence,
            cited_trial_ids=cited_trial_ids,
            retrieved_memories=memory_hits[:3],
            evidence_traces=evidence_traces,
            evidence_bundle=evidence_bundle,
            hypotheses=hypotheses,
            contradiction_review=evidence_bundle.contradiction_review,
        )

    def self_correction(
        self,
        trial_id: str,
        outcome: str,
        metrics: GovernorMetrics,
        memory_hits: Optional[List[MemorySearchHit]] = None,
    ) -> SelfCorrectionNote:
        memory_hits = memory_hits or []
        if outcome == "fail_fast":
            explanation = (
                f"Run failed because E={metrics.energy_e:.6f}, sigma={metrics.entropy_sigma:.6f}, "
                f"rho={metrics.drift_rho:.6f}, D_PR={metrics.effective_dimensionality:.4f}, "
                f"eq={metrics.equilibrium_fraction:.2%} breached the protected regime envelope."
            )
            corrective_action = "Lower eta, raise anchor regularization, and pivot strategy if failures persist."
        else:
            explanation = (
                f"Run remained stable with E={metrics.energy_e:.6f}, sigma={metrics.entropy_sigma:.6f}, "
                f"rho={metrics.drift_rho:.6f}, D_PR={metrics.effective_dimensionality:.4f}, "
                f"eq={metrics.equilibrium_fraction:.2%} inside the protected envelope."
            )
            corrective_action = "Reuse the stable alpha and eta neighborhood for the next comparable trial."
        similar_trials = [
            str(hit.metadata.get("trial_id"))
            for hit in memory_hits
            if hit.metadata.get("trial_id")
        ][:3]
        return SelfCorrectionNote(
            trial_id=trial_id,
            outcome=outcome,  # type: ignore[arg-type]
            energy_e=metrics.energy_e,
            entropy_sigma=metrics.entropy_sigma,
            drift_rho=metrics.drift_rho,
            effective_dimensionality=metrics.effective_dimensionality,
            equilibrium_fraction=metrics.equilibrium_fraction,
            explanation=explanation,
            corrective_action=corrective_action,
            similar_trials=similar_trials,
        )


class RuleStrategist:
    def propose(
        self,
        policy: DirectorPolicy,
        memory_hits: Optional[List[MemorySearchHit]] = None,
    ) -> StrategistPlan:
        latest = policy.data_anchor[-1]
        thresholds = GovernorThresholds(
            max_drift_l2=min(0.25, max(0.20, latest.drift_l2 * 1.5 + 0.03)),
            max_drift_rho=min(0.08, max(0.03, latest.drift_rho * 1.5 + 0.01)),
            max_entropy_sigma=min(0.12, max(0.03, latest.entropy_sigma * 1.5 + 0.02)),
            max_grad_norm=min(2.5, max(1.0, latest.grad_norm * 1.5 + 0.25)),
            max_gpu_temperature_c=78.0,
            min_dimensionality_ratio=max(0.25, min(0.55, latest.dimensionality_ratio * 0.70 if latest.dimensionality_ratio > 0 else 0.35)),
            max_quenching_loss=max(0.80, (latest.training_loss or 0.80) * 1.40),
        )
        memory_hits = memory_hits or []
        recalled = _memory_aware_hyperparameters(
            memory_hits,
            default_alpha=max(0.02, 0.08 - latest.drift_rho),
            default_eta=0.01,
        )

        base = {
            "protected_layers": ["transformer.h.0-7", "transformer.ln_f", "lm_head"],
            "mutable_layers": ["transformer.h.8-31"],
            "hyperparameters": {
                "batch_size": 8,
                "steps": 20,
                "alpha": recalled["alpha"],
                "eta": recalled["eta"],
            },
        }
        if policy.experiment_family == "elastic_anchor":
            draft = {
                "fim_lambda": 1.5,
                "bregman_budget": 0.50,
                "drift_budget": thresholds.max_drift_rho,
                **base,
            }
        elif policy.experiment_family == "ou_drift_jitter":
            draft = {
                "fim_lambda": 0.9,
                "bregman_budget": 0.35,
                "drift_budget": thresholds.max_drift_rho * 0.9,
                "protected_layers": ["transformer.h.0-3", "transformer.ln_f"],
                "mutable_layers": ["transformer.h.4-31"],
                "hyperparameters": {
                    **base["hyperparameters"],
                    "eta": 0.045,
                    "ou_sigma": 0.010,
                },
            }
        else:
            draft = {
                "fim_lambda": 2.1,
                "bregman_budget": 0.25,
                "drift_budget": thresholds.max_drift_rho * 0.75,
                "protected_layers": ["transformer.wte", "transformer.h.0-15", "lm_head"],
                "mutable_layers": ["transformer.h.16-31"],
                "hyperparameters": {
                    **base["hyperparameters"],
                    "freeze_embeddings": True,
                },
            }
        return StrategistPlan(
            trial_id=policy.trial_id,
            strategy_family=policy.experiment_family,
            anchor_path=policy.anchor_path,
            governor_thresholds=thresholds,
            quantitative_justification=policy.quantitative_justification,
            data_anchor=policy.data_anchor,
            retrieved_memories=memory_hits,
            **draft,
        )


class RuleScout:
    def propose(self, plan: StrategistPlan, workspace: str, dry_run: bool = False) -> ScoutTask:
        store = TARStateStore(workspace)
        payload_config_path = str(store.workspace / "tar_runs" / plan.trial_id / "config.json")
        container_config_path = f"/workspace/tar_runs/{plan.trial_id}/config.json"
        runtime = RuntimeSpec(
            image=os.environ.get("TAR_DOCKER_IMAGE", "pytorch/pytorch:latest"),
            gpu_index=int(os.environ.get("TAR_GPU_INDEX", "0")),
            env={
                "TAR_TRIAL_ID": plan.trial_id,
                "TAR_STRATEGY_FAMILY": plan.strategy_family,
            },
            volumes={
                workspace: "/workspace",
                str(store.data_dir): "/data",
            },
        )
        command = [
            "python",
            "-m",
            "tar_lab.train_template",
            "--config",
            container_config_path,
        ]
        if dry_run:
            command.append("--dry_run")
        return ScoutTask(
            trial_id=plan.trial_id,
            training_entrypoint="tar_lab/train_template.py",
            command=command,
            runtime=runtime,
            governor_thresholds=plan.governor_thresholds,
            payload_config_path=payload_config_path,
            dry_run=dry_run,
        )


class TriModelHierarchy:
    def __init__(
        self,
        director_config: Optional[LocalLLMConfig] = None,
        strategist_config: Optional[LocalLLMConfig] = None,
        scout_config: Optional[LocalLLMConfig] = None,
        client_factory: Optional[Callable[..., Any]] = None,
        allow_rule_fallback: bool = True,
    ):
        self.director_config = director_config or _role_config_from_env("director")
        self.strategist_config = strategist_config or _role_config_from_env("strategist")
        self.scout_config = scout_config or _role_config_from_env("scout")
        self.client_factory = client_factory
        self.allow_rule_fallback = allow_rule_fallback

        self.rule_director = RuleDirector()
        self.rule_strategist = RuleStrategist()
        self.rule_scout = RuleScout()

    @property
    def live_enabled(self) -> bool:
        return all((self.director_config, self.strategist_config, self.scout_config))

    def produce_bundle(
        self,
        store: TARStateStore,
        trial_id: str,
        workspace: str,
        objective_slug: str = "thermodynamic-anchor",
        dry_run: bool = False,
        memory_hits: Optional[List[MemorySearchHit]] = None,
    ) -> tuple[DirectorPolicy, StrategistPlan, ScoutTask]:
        if self.live_enabled:
            return self._produce_live_bundle(
                store=store,
                trial_id=trial_id,
                workspace=workspace,
                objective_slug=objective_slug,
                dry_run=dry_run,
                memory_hits=memory_hits,
            )
        if self.allow_rule_fallback and dry_run:
            policy = self.rule_director.propose(store, trial_id=trial_id, objective_slug=objective_slug)
            plan = self.rule_strategist.propose(policy, memory_hits=memory_hits)
            task = self.rule_scout.propose(plan, workspace=workspace, dry_run=dry_run)
            return policy, plan, task
        raise RuntimeError(
            "Local TAR hierarchy is not configured. Set TAR_LLM_BASE_URL plus "
            "TAR_DIRECTOR_MODEL, TAR_STRATEGIST_MODEL, and TAR_SCOUT_MODEL."
        )

    def _produce_live_bundle(
        self,
        store: TARStateStore,
        trial_id: str,
        workspace: str,
        objective_slug: str,
        dry_run: bool,
        memory_hits: Optional[List[MemorySearchHit]],
    ) -> tuple[DirectorPolicy, StrategistPlan, ScoutTask]:
        recent = store.tail_metrics(3)
        if len(recent) != 3:
            raise ValueError("Director requires exactly three recent log points from logs/")
        recovery = store.load_recovery()
        expected_q = _quantitative_justification(recent)
        memory_hits = memory_hits or []

        director = LocalOpenAIRole("director", self.director_config, DirectorDraft, self.client_factory)
        strategist = LocalOpenAIRole("strategist", self.strategist_config, StrategistDraft, self.client_factory)
        scout = LocalOpenAIRole("scout", self.scout_config, ScoutDraft, self.client_factory)

        director_prompt = self._director_prompt(
            trial_id=trial_id,
            objective_slug=objective_slug,
            recovery=recovery,
            recent=recent,
            expected_q=expected_q,
        )
        director_draft = director.generate(
            system_prompt=(
                "You are the TAR Director. Return JSON only. Choose the high-level experiment family and anchor path. "
                "Do not write prose outside JSON. Respect pivot logic after three fail-fast events."
            ),
            user_prompt=director_prompt,
        )

        policy = DirectorPolicy(
            trial_id=trial_id,
            objective_slug=director_draft.objective_slug or objective_slug,
            anchor_path=director_draft.anchor_path,
            experiment_family=self._normalize_family(director_draft.experiment_family),
            pivot_required=bool(director_draft.pivot_required or recovery.consecutive_fail_fast >= 3),
            failure_streak=recovery.consecutive_fail_fast,
            quantitative_justification=expected_q,
            data_anchor=recent,
        )

        strategist_prompt = self._strategist_prompt(policy)
        if memory_hits:
            strategist_prompt += (
                "\nRetrieved memory hits:\n"
                + json.dumps([item.model_dump(mode="json") for item in memory_hits], indent=2)
            )
        strategist_draft = strategist.generate(
            system_prompt=(
                "You are the TAR Strategist. Return JSON only. Map the Director policy to FIM, Bregman, drift budgets, "
                "layer sets, and numeric hyperparameters."
            ),
            user_prompt=strategist_prompt,
        )

        thresholds = GovernorThresholds(
            max_drift_l2=min(0.25, max(0.20, recent[-1].drift_l2 * 1.5 + 0.03)),
            max_drift_rho=min(0.08, max(0.03, recent[-1].drift_rho * 1.5 + 0.01)),
            max_entropy_sigma=min(0.12, max(0.03, recent[-1].entropy_sigma * 1.5 + 0.02)),
            max_grad_norm=min(2.5, max(1.0, recent[-1].grad_norm * 1.5 + 0.25)),
            max_gpu_temperature_c=78.0,
            min_dimensionality_ratio=max(0.25, min(0.55, recent[-1].dimensionality_ratio * 0.70 if recent[-1].dimensionality_ratio > 0 else 0.35)),
            max_quenching_loss=max(0.80, (recent[-1].training_loss or 0.80) * 1.40),
        )
        plan = StrategistPlan(
            trial_id=trial_id,
            strategy_family=self._normalize_family(strategist_draft.strategy_family),
            anchor_path=policy.anchor_path,
            fim_lambda=strategist_draft.fim_lambda,
            bregman_budget=strategist_draft.bregman_budget,
            drift_budget=strategist_draft.drift_budget,
            protected_layers=strategist_draft.protected_layers,
            mutable_layers=strategist_draft.mutable_layers,
            hyperparameters=strategist_draft.hyperparameters,
            governor_thresholds=thresholds,
            quantitative_justification=expected_q,
            data_anchor=recent,
            retrieved_memories=memory_hits,
        )

        payload_config_path = str((TARStateStore(workspace).workspace / "tar_runs" / trial_id / "config.json"))
        container_config_path = f"/workspace/tar_runs/{trial_id}/config.json"
        scout_prompt = self._scout_prompt(
            plan=plan,
            payload_config_path=payload_config_path,
            container_config_path=container_config_path,
            dry_run=dry_run,
        )
        scout_draft = scout.generate(
            system_prompt=(
                "You are the TAR Scout. Return JSON only. Select the training image and runtime parameters for the payload template."
            ),
            user_prompt=scout_prompt,
        )

        runtime = RuntimeSpec(
            image=scout_draft.image,
            gpu_index=int(os.environ.get("TAR_GPU_INDEX", "0")),
            power_limit_w=scout_draft.power_limit_w,
            gpu_target_temp_c=scout_draft.gpu_target_temp_c,
            env={
                "TAR_TRIAL_ID": trial_id,
                "TAR_STRATEGY_FAMILY": plan.strategy_family,
            },
            volumes={
                workspace: "/workspace",
                str(TARStateStore(workspace).data_dir): "/data",
            },
        )
        command = [
            "python",
            "-m",
            "tar_lab.train_template",
            "--config",
            container_config_path,
        ]
        if dry_run:
            command.append("--dry_run")
        task = ScoutTask(
            trial_id=trial_id,
            training_entrypoint=scout_draft.training_entrypoint,
            command=command,
            runtime=runtime,
            governor_thresholds=plan.governor_thresholds,
            payload_config_path=payload_config_path,
            dry_run=dry_run,
        )
        plan.hyperparameters.setdefault("steps", scout_draft.steps)
        plan.hyperparameters.setdefault("batch_size", scout_draft.batch_size)
        return policy, plan, task

    def director_chat(
        self,
        store: TARStateStore,
        prompt: str,
        memory_hits: Optional[List[MemorySearchHit]] = None,
    ) -> LabChatResponse:
        memory_hits = memory_hits or []
        evidence_bundle = build_evidence_bundle(prompt, memory_hits)
        hypotheses = build_hypotheses(prompt, evidence_bundle)
        if self.director_config is None:
            return self.rule_director.chat(prompt, store=store, memory_hits=memory_hits)

        recent = store.tail_metrics(3)
        recovery = store.load_recovery()
        director = LocalOpenAIRole("director_chat", self.director_config, DirectorChatDraft, self.client_factory)
        raw = director.generate(
            system_prompt=(
                "You are the TAR Director. Return JSON only. Summarize lab state using quantitative language. "
                "Ground the answer in E, sigma, rho, grad_norm, D_PR, equilibrium state, recovery state, and retrieved memories. "
                "Acknowledge contradictions explicitly and avoid overclaiming."
            ),
            user_prompt=(
                f"User intent: {prompt}\n"
                f"Recovery: {json.dumps(recovery.model_dump(mode='json'), indent=2)}\n"
                f"Recent metrics: {json.dumps([item.model_dump(mode='json') for item in recent], indent=2)}\n"
                f"Retrieved memories: {json.dumps([item.model_dump(mode='json') for item in memory_hits], indent=2)}\n"
                f"Evidence bundle: {json.dumps(evidence_bundle.model_dump(mode='json'), indent=2)}"
            ),
        )
        return LabChatResponse(
            response_text=raw.response_text,
            state_summary=raw.state_summary,
            confidence=evidence_bundle.confidence,
            cited_trial_ids=raw.cited_trial_ids,
            retrieved_memories=memory_hits[:3],
            evidence_traces=evidence_bundle.traces,
            evidence_bundle=evidence_bundle,
            hypotheses=hypotheses,
            contradiction_review=evidence_bundle.contradiction_review,
            dataset_change=raw.dataset_change,
        )

    def build_self_correction(
        self,
        trial_id: str,
        outcome: str,
        metrics: GovernorMetrics,
        memory_hits: Optional[List[MemorySearchHit]] = None,
    ) -> SelfCorrectionNote:
        memory_hits = memory_hits or []
        if self.director_config is None:
            return self.rule_director.self_correction(
                trial_id=trial_id,
                outcome=outcome,
                metrics=metrics,
                memory_hits=memory_hits,
            )

        director = LocalOpenAIRole("director_self_correction", self.director_config, SelfCorrectionDraft, self.client_factory)
        draft = director.generate(
            system_prompt=(
                "You are the TAR Director. Return JSON only. Explain run outcome using E, sigma, rho, D_PR, and equilibrium state. "
                "No vague language. Produce a corrective action."
            ),
            user_prompt=(
                f"Trial ID: {trial_id}\n"
                f"Outcome: {outcome}\n"
                f"Metrics: {json.dumps(metrics.model_dump(mode='json'), indent=2)}\n"
                f"Retrieved memories: {json.dumps([item.model_dump(mode='json') for item in memory_hits], indent=2)}"
            ),
        )
        return SelfCorrectionNote(
            trial_id=trial_id,
            outcome=draft.outcome,  # type: ignore[arg-type]
            energy_e=draft.energy_e,
            entropy_sigma=draft.entropy_sigma,
            drift_rho=draft.drift_rho,
            effective_dimensionality=draft.effective_dimensionality,
            equilibrium_fraction=draft.equilibrium_fraction,
            explanation=draft.explanation,
            corrective_action=draft.corrective_action,
            similar_trials=draft.similar_trials,
        )

    @staticmethod
    def _normalize_family(name: str) -> str:
        family = name.strip().lower()
        if family not in {"elastic_anchor", "ou_drift_jitter", "layer_freeze"}:
            raise ValueError(f"Unsupported strategy family: {name}")
        return family

    @staticmethod
    def _director_prompt(
        trial_id: str,
        objective_slug: str,
        recovery: Any,
        recent: List[GovernorMetrics],
        expected_q: QuantitativeJustification,
    ) -> str:
        return (
            f"Trial ID: {trial_id}\n"
            f"Objective: {objective_slug}\n"
            f"Recovery: {json.dumps(recovery.model_dump(mode='json'), indent=2)}\n"
            f"Last three metrics: {json.dumps([item.model_dump(mode='json') for item in recent], indent=2)}\n"
            f"Quantitative justification: {json.dumps(expected_q.model_dump(mode='json'), indent=2)}\n"
            "Choose one experiment_family from elastic_anchor, ou_drift_jitter, layer_freeze. "
            "If consecutive_fail_fast >= 3, pivot_required must be true."
        )

    @staticmethod
    def _strategist_prompt(policy: DirectorPolicy) -> str:
        return (
            f"Director policy:\n{json.dumps(policy.model_dump(mode='json'), indent=2)}\n"
            "Return FIM, Bregman, drift budgets, protected_layers, mutable_layers, and hyperparameters. "
            "Use quantitative justification based on E, sigma, rho, grad_norm, D_PR, and equilibrium."
        )

    @staticmethod
    def _scout_prompt(
        plan: StrategistPlan,
        payload_config_path: str,
        container_config_path: str,
        dry_run: bool,
    ) -> str:
        return (
            f"Strategist plan:\n{json.dumps(plan.model_dump(mode='json'), indent=2)}\n"
            f"Payload config host path: {payload_config_path}\n"
            f"Payload config container path: {container_config_path}\n"
            "Return the runtime image and the expected steps/batch_size for the training payload template. "
            f"Dry run: {dry_run}"
        )


Director = RuleDirector
Strategist = RuleStrategist
Scout = RuleScout
