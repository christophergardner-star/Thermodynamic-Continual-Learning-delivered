from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from tar_lab.data_manager import DataManager
from tar_lab.docker_runner import DockerRunner
from tar_lab.governor import ThermodynamicGovernor
from tar_lab.hardware import NvidiaSMI
from tar_lab.hierarchy import TriModelHierarchy
from tar_lab.memory import MemoryIndexer, VectorVault
from tar_lab.research_ingest import ResearchIngestor
from tar_lab.schemas import (
    BreakthroughReport,
    DryRunReport,
    FailureAutopsy,
    GovernorDecision,
    GovernorMetrics,
    KnowledgeGraphEntry,
    LabChatResponse,
    LiveDockerTestReport,
    PreparedDataBundle,
    ResearchIngestReport,
    RecoveryState,
    SelfCorrectionNote,
    TrainingPayloadConfig,
    VerificationReport,
)
from tar_lab.state import TARStateStore
from tar_lab.verification import VerificationRunner


class TAROrchestrator:
    def __init__(
        self,
        workspace: str = ".",
        hierarchy: Optional[TriModelHierarchy] = None,
        governor: Optional[ThermodynamicGovernor] = None,
        docker_runner: Optional[DockerRunner] = None,
        hardware: Optional[NvidiaSMI] = None,
        start_memory_indexer: bool = False,
    ):
        self.store = TARStateStore(workspace)
        self.workspace = str(self.store.workspace)
        self.data_manager = DataManager(workspace)
        self.hierarchy = hierarchy or TriModelHierarchy()
        self.governor = governor or ThermodynamicGovernor()
        self.docker_runner = docker_runner or DockerRunner()
        self.hardware = hardware or NvidiaSMI()
        self.research_ingestor = ResearchIngestor(workspace)
        self.verification_runner = VerificationRunner(workspace)
        self.vault: Optional[VectorVault] = None
        self.memory_indexer: Optional[MemoryIndexer] = None
        self.memory_error: Optional[str] = None
        try:
            self.vault = VectorVault(workspace)
            self.memory_indexer = MemoryIndexer(self.store, self.vault)
            if start_memory_indexer:
                self.memory_indexer.start()
        except Exception as exc:
            self.memory_error = str(exc)

    def seed_mock_metrics(self) -> list[GovernorMetrics]:
        trial_id = "seed"
        metrics = [
            GovernorMetrics(trial_id=trial_id, step=1, energy_e=0.010, entropy_sigma=0.020, drift_l2=0.030, drift_rho=0.010, grad_norm=0.40, effective_dimensionality=6.8, dimensionality_ratio=0.92, equilibrium_fraction=0.66, equilibrium_gate=False, training_loss=0.62),
            GovernorMetrics(trial_id=trial_id, step=2, energy_e=0.014, entropy_sigma=0.028, drift_l2=0.036, drift_rho=0.014, grad_norm=0.48, effective_dimensionality=7.1, dimensionality_ratio=0.96, equilibrium_fraction=0.82, equilibrium_gate=True, training_loss=0.54),
            GovernorMetrics(trial_id=trial_id, step=3, energy_e=0.019, entropy_sigma=0.034, drift_l2=0.041, drift_rho=0.018, grad_norm=0.55, effective_dimensionality=7.4, dimensionality_ratio=0.99, equilibrium_fraction=0.84, equilibrium_gate=True, training_loss=0.48),
        ]
        for item in metrics:
            self.store.append_metric(item)
        return metrics

    def _ensure_recent_metrics(self) -> None:
        if len(self.store.tail_metrics(3)) < 3:
            self.seed_mock_metrics()

    def _ensure_data_prepared(self, force: bool = False) -> PreparedDataBundle:
        bundle = self.data_manager.prepare_dual_stream(force=force)
        return bundle

    def _sync_memory(self) -> None:
        if self.memory_indexer is not None:
            self.memory_indexer.sync_once()

    def _retrieve_strategy_hits(self, metrics: Optional[list[GovernorMetrics]] = None) -> list[Any]:
        self._sync_memory()
        if self.vault is None:
            return []
        return self.vault.search_similar_trials(metrics or self.store.tail_metrics(3))

    def plan_trial(self, dry_run: bool = False) -> tuple[Any, Any, Any]:
        self._ensure_recent_metrics()
        data_bundle = self._ensure_data_prepared()
        trial_id = self.store.next_trial_id()
        recovery = self.store.load_recovery()
        self.store.save_recovery(recovery.model_copy(update={"trial_id": trial_id, "status": "planning"}))
        memory_hits = self._retrieve_strategy_hits()
        policy, plan, task = self.hierarchy.produce_bundle(
            self.store,
            trial_id=trial_id,
            workspace=self.workspace,
            dry_run=dry_run,
            memory_hits=memory_hits,
        )
        payload = self._build_payload_config(plan, data_bundle)
        payload_path = self.store.write_payload_config(payload)
        if task.payload_config_path != str(payload_path):
            task = task.model_copy(update={"payload_config_path": str(payload_path)})
        self.store.write_policy_bundle(policy, plan, task)
        self.store.append_knowledge_entry(
            KnowledgeGraphEntry(
                trial_id=trial_id,
                strategy_family=plan.strategy_family,
                outcome="running",
                hyperparameters=plan.hyperparameters,
            )
        )
        self.store.append_audit_event("cli", "plan_trial", {"trial_id": trial_id, "dry_run": dry_run})
        self.store.save_recovery(
            RecoveryState(
                trial_id=trial_id,
                status="running",
                last_known_stable_hyperparameters=plan.hyperparameters,
                consecutive_fail_fast=recovery.consecutive_fail_fast,
                last_strategy_family=plan.strategy_family,
                last_anchor_path=plan.anchor_path,
            )
        )
        self._sync_memory()
        return policy, plan, task

    def ingest_research(self, topic: str = "frontier ai", max_results: int = 6) -> ResearchIngestReport:
        report = self.research_ingestor.ingest(topic=topic, max_results=max_results)
        for document in report.documents:
            self.store.append_research_document(document)
            if self.vault is not None:
                self.vault.index_research_document(document)
        self.store.append_audit_event(
            "research",
            "ingest",
            {
                "topic": topic,
                "fetched": report.fetched,
                "indexed": report.indexed,
                "sources": report.sources,
            },
        )
        self._sync_memory()
        return report

    def _build_payload_config(self, plan: Any, data_bundle: PreparedDataBundle) -> TrainingPayloadConfig:
        hyper = dict(plan.hyperparameters)
        alpha = float(hyper.pop("alpha", hyper.pop("anchor_alpha", 0.05)))
        eta = float(hyper.pop("eta", hyper.pop("lr", 0.01)))
        steps = int(hyper.pop("steps", 20))
        batch_size = int(hyper.pop("batch_size", 8))
        anchor_path = self._resolve_container_path(plan.anchor_path)
        return TrainingPayloadConfig(
            trial_id=plan.trial_id,
            strategy_family=plan.strategy_family,
            anchor_path=str(anchor_path),
            alpha=alpha,
            eta=eta,
            fim_lambda=plan.fim_lambda,
            bregman_budget=plan.bregman_budget,
            drift_budget=plan.drift_budget,
            batch_size=batch_size,
            steps=steps,
            log_path="/workspace/logs/thermo_metrics.jsonl",
            output_dir=f"/workspace/tar_runs/{plan.trial_id}/output",
            anchor_manifest_path=data_bundle.anchor_manifest_path,
            research_manifest_path=data_bundle.research_manifest_path,
            governor_thresholds=plan.governor_thresholds,
            protected_layers=plan.protected_layers,
            mutable_layers=plan.mutable_layers,
            notes=hyper,
        )

    def _resolve_container_path(self, anchor_path: str) -> str:
        raw = Path(anchor_path)
        if raw.is_absolute():
            try:
                rel = raw.resolve().relative_to(self.store.workspace)
                return f"/workspace/{rel.as_posix()}"
            except ValueError:
                return f"/workspace/anchors/{raw.name}"
        return f"/workspace/{raw.as_posix()}"

    def _mock_anchor_state(self) -> dict[str, torch.Tensor]:
        return {
            "layer.weight": torch.tensor([0.0, 0.1, 0.2], dtype=torch.float32),
            "layer.bias": torch.tensor([0.0], dtype=torch.float32),
        }

    def _mock_current_state(self, force_fail_fast: bool) -> dict[str, torch.Tensor]:
        magnitude = 0.6 if force_fail_fast else 0.004
        return {
            "layer.weight": torch.tensor([magnitude, 0.1 + magnitude, 0.2 + magnitude], dtype=torch.float32),
            "layer.bias": torch.tensor([magnitude], dtype=torch.float32),
        }

    def _mock_gradients(self, force_fail_fast: bool) -> dict[str, torch.Tensor]:
        magnitude = 3.2 if force_fail_fast else 0.1
        return {
            "layer.weight": torch.full((3,), magnitude, dtype=torch.float32),
            "layer.bias": torch.tensor([magnitude], dtype=torch.float32),
        }

    def _mock_regime(self, force_fail_fast: bool) -> dict[str, Any]:
        if force_fail_fast:
            return {
                "effective_dimensionality": 1.2,
                "dimensionality_ratio": 0.18,
                "equilibrium_fraction": 0.0,
                "equilibrium_gate": False,
                "training_loss": 0.58,
            }
        return {
            "effective_dimensionality": 7.8,
            "dimensionality_ratio": 0.97,
            "equilibrium_fraction": 0.85,
            "equilibrium_gate": True,
            "training_loss": 0.41,
        }

    def _record_fail_fast(self, decision: GovernorDecision, strategy_family: str) -> None:
        recovery = self.store.load_recovery()
        fail_reason = decision.reasons[0] if decision.reasons else "unknown"
        self.store.save_recovery(
            recovery.model_copy(
                update={
                    "status": "fail_fast",
                    "last_fail_reason": fail_reason,
                    "last_fail_metrics": decision.metrics,
                    "consecutive_fail_fast": recovery.consecutive_fail_fast + 1,
                    "last_strategy_family": strategy_family,
                    "max_effective_dimensionality_achieved": max(
                        recovery.max_effective_dimensionality_achieved,
                        decision.metrics.effective_dimensionality,
                    ),
                }
            )
        )
        self.store.update_knowledge_entry(
            decision.metrics.trial_id,
            outcome="fail_fast",
            fail_reason=fail_reason,
            ended_at=_utc_now(),
        )
        self.store.append_audit_event(
            "governor",
            "sigterm",
            {
                "trial_id": decision.metrics.trial_id,
                "reason_of_death": fail_reason,
                "reasons": decision.reasons,
            },
        )

    def _record_success(
        self,
        trial_id: str,
        strategy_family: str,
        metrics: Optional[GovernorMetrics] = None,
    ) -> None:
        recovery = self.store.load_recovery()
        self.store.save_recovery(
            recovery.model_copy(
                update={
                    "status": "completed",
                    "last_fail_reason": None,
                    "last_fail_metrics": None,
                    "consecutive_fail_fast": 0,
                    "last_strategy_family": strategy_family,
                    "max_effective_dimensionality_achieved": max(
                        recovery.max_effective_dimensionality_achieved,
                        metrics.effective_dimensionality if metrics is not None else 0.0,
                    ),
                }
            )
        )
        self.store.update_knowledge_entry(trial_id, outcome="completed", ended_at=_utc_now())
        self.store.append_audit_event("orchestrator", "trial_completed", {"trial_id": trial_id})
        self._sync_memory()

    def _latest_metric(self, trial_id: Optional[str] = None) -> Optional[GovernorMetrics]:
        rows = list(self.store.iter_metrics())
        if trial_id is None:
            return rows[-1] if rows else None
        for metric in reversed(rows):
            if metric.trial_id == trial_id:
                return metric
        return None

    def recursive_analysis(
        self,
        trial_id: str,
        outcome: str,
        metrics: Optional[GovernorMetrics] = None,
    ) -> Optional[SelfCorrectionNote]:
        current = metrics or self._latest_metric(trial_id)
        if current is None:
            return None
        memory_hits = self._retrieve_strategy_hits([current])
        note = self.hierarchy.build_self_correction(
            trial_id=trial_id,
            outcome=outcome,
            metrics=current,
            memory_hits=memory_hits,
        )
        if self.vault is not None:
            self.vault.index_self_correction(note)
        self.store.append_audit_event(
            "director",
            "self_correction",
            note.model_dump(mode="json"),
        )
        self._sync_memory()
        return note

    def verify_last_trial(self, trial_id: Optional[str] = None) -> VerificationReport:
        resolved_trial_id = trial_id or self.store.load_recovery().trial_id
        if not resolved_trial_id:
            raise RuntimeError("No trial is available to verify.")
        config_path = self.store.workspace / "tar_runs" / resolved_trial_id / "config.json"
        if not config_path.exists():
            raise RuntimeError(f"Verification config not found for trial {resolved_trial_id}")
        config = TrainingPayloadConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        report = self.verification_runner.run(config)
        self.store.append_verification_report(report)
        if self.vault is not None:
            self.vault.index_verification_report(report)
        self.store.append_audit_event(
            "verification",
            "verify_last_trial",
            report.model_dump(mode="json"),
        )
        self._sync_memory()
        return report

    def breakthrough_report(self, trial_id: Optional[str] = None) -> BreakthroughReport:
        resolved_trial_id = trial_id or self.store.load_recovery().trial_id
        if not resolved_trial_id:
            raise RuntimeError("No trial is available for breakthrough analysis.")
        verification = self.store.latest_verification_report(resolved_trial_id)
        if verification is None:
            verification = self.verify_last_trial(resolved_trial_id)
        latest_metric = self._latest_metric(resolved_trial_id)
        query = (
            "current ai problems thermodynamic learning calibration robustness "
            f"trial {resolved_trial_id}"
        )
        if latest_metric is not None:
            query += (
                f" D_PR={latest_metric.effective_dimensionality:.4f}"
                f" sigma={latest_metric.entropy_sigma:.4f}"
            )
        research_hits = self.vault.search(query, n_results=5, kind="research") if self.vault is not None else []
        report = self.verification_runner.build_breakthrough_report(
            verification,
            supporting_research_ids=[hit.document_id for hit in research_hits],
        )
        self.store.append_breakthrough_report(report)
        if self.vault is not None:
            self.vault.index_breakthrough_report(report)
        self.store.append_audit_event(
            "director",
            "breakthrough_report",
            report.model_dump(mode="json"),
        )
        self._sync_memory()
        return report

    def run_dry_run(self, force_fail_fast: bool = False) -> DryRunReport:
        _, plan, task = self.plan_trial(dry_run=True)
        self.store.append_audit_event("cli", "dry_run", {"trial_id": task.trial_id})
        self.hardware.prepare_run(task.runtime.gpu_index, task.runtime.power_limit_w, task.runtime.gpu_target_temp_c, apply=False)
        launch = self.docker_runner.launch(task, dry_run=True)
        telemetry = self.hardware.query_telemetry(task.runtime.gpu_index)
        regime = self._mock_regime(force_fail_fast=force_fail_fast)
        metrics = self.governor.compute_metrics(
            trial_id=task.trial_id,
            step=1,
            anchor=self._mock_anchor_state(),
            current=self._mock_current_state(force_fail_fast=force_fail_fast),
            gradients=self._mock_gradients(force_fail_fast=force_fail_fast),
            gpu_temperature_c=telemetry.temperature_c,
            gpu_memory_temperature_c=telemetry.memory_temperature_c,
            gpu_power_w=telemetry.power_w,
            effective_dimensionality=regime["effective_dimensionality"],
            dimensionality_ratio=regime["dimensionality_ratio"],
            equilibrium_fraction=regime["equilibrium_fraction"],
            equilibrium_gate=regime["equilibrium_gate"],
            training_loss=regime["training_loss"],
        )
        self.store.append_metric(metrics)
        decision = self.governor.evaluate(metrics, plan.governor_thresholds)

        if decision.action == "terminate":
            self._record_fail_fast(decision, strategy_family=plan.strategy_family)
            self.recursive_analysis(task.trial_id, outcome="fail_fast", metrics=metrics)
        else:
            self._record_success(task.trial_id, strategy_family=plan.strategy_family, metrics=metrics)
            self.recursive_analysis(task.trial_id, outcome="dry_run", metrics=metrics)

        recovery = self.store.load_recovery()
        return DryRunReport(
            trial_id=task.trial_id,
            json_schema_ok=True,
            docker_command_ok=bool(launch.command),
            memory_ok=self.vault is not None and len(self.store.tail_metrics(3)) == 3,
            pivot_force_ready=recovery.consecutive_fail_fast >= 3 or recovery.status == "completed",
            governor_action=decision.action,
            recovery_status=recovery.status,
            composed_command=launch.command,
        )

    def live_docker_test(self) -> LiveDockerTestReport:
        _, plan, task = self.plan_trial(dry_run=True)
        try:
            visible_gpus = self.hardware.list_gpus()
            if visible_gpus and task.runtime.gpu_index >= len(visible_gpus):
                raise RuntimeError(
                    f"Requested GPU index {task.runtime.gpu_index} is unavailable. "
                    f"Visible devices: {visible_gpus}"
                )
            self.hardware.prepare_run(
                task.runtime.gpu_index,
                task.runtime.power_limit_w,
                task.runtime.gpu_target_temp_c,
                apply=True,
            )
            launch = self.docker_runner.live_test(task)
            launched = launch.returncode in (None, 0)
            if launched:
                self._record_success(
                    task.trial_id,
                    strategy_family=plan.strategy_family,
                    metrics=self._latest_metric(task.trial_id),
                )
                self.recursive_analysis(task.trial_id, outcome="completed")
            else:
                self.recursive_analysis(task.trial_id, outcome="fail_fast")
            return LiveDockerTestReport(
                launched=launched,
                image=task.runtime.image,
                mode=launch.mode,
                command=launch.command,
                returncode=launch.returncode,
                payload_config_path=task.payload_config_path,
                gpu_visible=launch.gpu_visible,
                gpu_probe_output=launch.probe_output,
                error=(
                    None
                    if launched
                    else launch.probe_output or f"Container exited with return code {launch.returncode}"
                ),
            )
        except Exception as exc:
            return LiveDockerTestReport(
                launched=False,
                image=task.runtime.image,
                mode="error",
                command=self.docker_runner.compose_command(task),
                payload_config_path=task.payload_config_path,
                gpu_visible=False,
                error=str(exc),
            )

    def status(self) -> Dict[str, Any]:
        self._sync_memory()
        payload = self.store.status_payload()
        telemetry = self.hardware.query_telemetry()
        anchor_manifest = self.store.load_dataset_manifest("anchor")
        research_manifest = self.store.load_dataset_manifest("research")
        payload["gpu"] = {
            "temperature_c": telemetry.temperature_c,
            "memory_temperature_c": telemetry.memory_temperature_c,
            "power_w": telemetry.power_w,
            "power_limit_w": telemetry.power_limit_w,
            "target_temperature_c": telemetry.target_temperature_c,
            "visible_devices": self.hardware.list_gpus(),
        }
        payload["datasets"] = {
            "anchor": anchor_manifest.model_dump(mode="json") if anchor_manifest else None,
            "research": research_manifest.model_dump(mode="json") if research_manifest else None,
        }
        payload["memory"] = self.vault.stats() if self.vault is not None else {"error": self.memory_error}
        payload["regime"] = self.check_regime()
        return payload

    def check_regime(self) -> Dict[str, Any]:
        latest = self._latest_metric()
        if latest is None:
            self.seed_mock_metrics()
            latest = self._latest_metric()
        if latest is None:
            return {"status": "unavailable"}

        warning = None
        regime = "searching"
        if (
            latest.dimensionality_ratio > 0.0
            and latest.training_loss is not None
            and latest.training_loss <= self.governor.thresholds.max_quenching_loss
            and latest.dimensionality_ratio < self.governor.thresholds.min_dimensionality_ratio
        ):
            regime = "thermodynamic_quenching"
            warning = "Degeneracy Warning: loss is holding while D_PR has collapsed."
        elif latest.equilibrium_gate:
            regime = "equilibrium"
        elif latest.equilibrium_fraction >= 0.5:
            regime = "stabilizing"

        return {
            "trial_id": latest.trial_id,
            "step": latest.step,
            "regime": regime,
            "effective_dimensionality": latest.effective_dimensionality,
            "dimensionality_ratio": latest.dimensionality_ratio,
            "equilibrium_fraction": latest.equilibrium_fraction,
            "equilibrium_gate": latest.equilibrium_gate,
            "training_loss": latest.training_loss,
            "warning": warning,
        }

    def chat(self, prompt: str) -> LabChatResponse:
        self._ensure_recent_metrics()
        self._ensure_data_prepared()
        self._sync_memory()
        prompt_lower = prompt.lower()
        if any(term in prompt_lower for term in ("current ai", "ai problems", "research landscape", "frontier")):
            existing_research = list(self.store.iter_research_documents())
            if not existing_research:
                try:
                    self.ingest_research(topic=prompt, max_results=6)
                except Exception:
                    pass
                self._sync_memory()
        memory_hits = self.vault.search(prompt, n_results=3) if self.vault is not None else []
        response = self.hierarchy.director_chat(
            self.store,
            prompt=prompt,
            memory_hits=memory_hits,
        )
        self.store.append_audit_event(
            "director",
            "chat",
            {
                "prompt": prompt,
                "response": response.model_dump(mode="json"),
            },
        )
        return response

    def pivot_force(self, force: bool = False) -> Dict[str, Any]:
        recovery = self.store.load_recovery()
        next_streak = max(recovery.consecutive_fail_fast, 3 if force else recovery.consecutive_fail_fast)
        updated = recovery.model_copy(update={"status": "pivoted", "consecutive_fail_fast": next_streak})
        self.store.save_recovery(updated)
        if updated.trial_id:
            self.store.update_knowledge_entry(updated.trial_id, outcome="pivoted", ended_at=_utc_now())
        self.store.append_audit_event("cli", "pivot_force", {"force": force})
        return updated.model_dump(mode="json")

    def explain_last_fail(self) -> FailureAutopsy:
        recovery = self.store.load_recovery()
        if recovery.status != "fail_fast" or recovery.last_fail_reason is None:
            return FailureAutopsy(status="no_failure")
        return FailureAutopsy(
            trial_id=recovery.trial_id,
            status="fail_fast",
            reason_of_death=recovery.last_fail_reason,
            strategy_family=recovery.last_strategy_family,
            metrics=recovery.last_fail_metrics,
            consecutive_fail_fast=recovery.consecutive_fail_fast,
        )

    def panic(self) -> Dict[str, Any]:
        commands = self.docker_runner.panic_kill(dry_run=True)
        self.store.append_audit_event("cli", "panic", {"commands": commands})
        return {"commands": commands}

    def shutdown(self) -> None:
        if self.memory_indexer is not None:
            self.memory_indexer.stop()
        if self.vault is not None:
            self.vault.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
