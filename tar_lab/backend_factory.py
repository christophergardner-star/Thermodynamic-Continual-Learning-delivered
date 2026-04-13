from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from tar_lab.schemas import (
    BackendProvenance,
    ExperimentArtifactLineage,
    ExperimentBackendRuntimeRecord,
    ExperimentBackendSpec,
    ExperimentLaunchPlan,
    ExperimentResumeInfo,
    RuntimeSpec,
)
from tar_lab.state import TARStateStore


def json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


class ExperimentBackendRegistry:
    def __init__(self, workspace: str = "."):
        self.store = TARStateStore(workspace)
        self.workspace = self.store.workspace
        self.backend_dir = self.store.state_dir / "experiment_backends"
        self.backend_dir.mkdir(parents=True, exist_ok=True)
        self._aliases = {
            "asc_text_payload": "asc_text",
        }
        self._backends = {
            "asc_text": ExperimentBackendSpec(
                backend_id="asc_text",
                summary="Manifest-backed ASC text payload used by TAR for governed research runs.",
                domain="language_modeling",
                entrypoint="tar_lab.train_template",
                status="executable",
                research_grade_capable=True,
                expected_data_type="token_sequence",
                requires_tokenizer=True,
                supports_resume=True,
                supports_distributed=False,
                requires_gpu=False,
                required_deps=["torch", "transformers", "pydantic"],
                required_metrics=["training_loss", "effective_dimensionality", "entropy_sigma", "drift_rho", "calibration_ece"],
                required_artifacts=["payload_summary.json", "thermo_metrics.jsonl", "checkpoint.pt"],
                valid_input_contract=[
                    "Requires TAR-prepared anchor and research manifests.",
                    "Requires complete dataset and tokenizer provenance for research intent.",
                ],
                notes=[
                    "Consumes TAR-prepared anchor and research manifests directly.",
                    "Runs valid ASC min-max optimization with thermodynamic telemetry and checkpoints.",
                ],
            ),
            "toy_anchor": ExperimentBackendSpec(
                backend_id="toy_anchor",
                summary="Thermodynamic toy payload used only for control-path validation.",
                domain="thermodynamic_control",
                entrypoint="tar_lab.train_template",
                status="executable",
                control_only=True,
                expected_data_type="toy_vector_batch",
                requires_tokenizer=False,
                supports_resume=True,
                supports_distributed=False,
                requires_gpu=False,
                required_deps=["torch", "pydantic"],
                required_metrics=["training_loss", "effective_dimensionality", "entropy_sigma", "drift_rho"],
                required_artifacts=["payload_summary.json", "thermo_metrics.jsonl"],
                valid_input_contract=["Only valid for control or plumbing runs."],
                notes=[
                    "Useful for governance and telemetry smoke tests.",
                    "Not a publication-grade training target.",
                ],
            ),
            "asc_full": ExperimentBackendSpec(
                backend_id="asc_full",
                summary="Full ASC min-max trainer for GPT-2 family backbones.",
                domain="language_modeling",
                entrypoint="asc_train_full.py",
                status="executable",
                research_grade_capable=True,
                expected_data_type="token_sequence",
                requires_tokenizer=True,
                supports_resume=True,
                supports_distributed=True,
                requires_gpu=True,
                required_deps=["torch", "transformers"],
                required_metrics=["task_loss", "consistency_loss"],
                required_artifacts=["training_log.json", "resume_state.pt", "model_checkpoint"],
                valid_input_contract=["Use a real text corpus and a real tokenizer for research runs."],
                notes=[
                    "Use this instead of asc_train.py/asc_train_cpu.py for valid ASC research runs.",
                    "Designed for real text-model training, not just TAR payload validation.",
                    "Checkpoint-aware relaunch is supported through resume_state.pt and deterministic per-epoch shuffling.",
                ],
            ),
            "coding_asc": ExperimentBackendSpec(
                backend_id="coding_asc",
                summary="Coding-oriented ASC fine-tune surface; still experimental at research scale.",
                domain="coding_models",
                entrypoint="coding_asc_finetune.py",
                status="scaffold",
                research_grade_capable=False,
                expected_data_type="code_sequence",
                requires_tokenizer=True,
                supports_resume=False,
                supports_distributed=False,
                requires_gpu=True,
                required_deps=["torch", "transformers"],
                required_metrics=["task_loss", "consistency_loss"],
                required_artifacts=["training_log.json", "model_checkpoint"],
                valid_input_contract=["Do not launch as a research-grade TAR run until masking/device placement work is complete."],
                notes=[
                    "Current implementation is lightweight and best suited to pod-backed experiments.",
                    "Large-model production training still needs further systems work.",
                ],
            ),
            "asc_cv": ExperimentBackendSpec(
                backend_id="asc_cv",
                summary="Computer-vision ASC backend with executable minimal image-classification loop and TCL governor hooks.",
                domain="computer_vision",
                entrypoint="tar_lab.multimodal_payloads",
                status="executable",
                research_grade_capable=True,
                expected_data_type="image_classification",
                requires_tokenizer=False,
                supports_resume=True,
                supports_distributed=False,
                requires_gpu=False,
                required_deps=["torch", "numpy"],
                required_metrics=["accuracy", "effective_dimensionality", "entropy_sigma", "drift_rho", "calibration_ece"],
                required_artifacts=["execution_summary.json", "thermo_metrics.jsonl"],
                valid_input_contract=[
                    "Requires labelled image-like tensors or a known vision benchmark source.",
                    "Must emit provenance for dataset, purity, and backend readiness.",
                ],
                notes=[
                    "Runs a real minimal supervised vision loop under the same D_PR, sigma, and rho governance contract.",
                ],
            ),
            "asc_rl": ExperimentBackendSpec(
                backend_id="asc_rl",
                summary="Reinforcement-learning ASC backend with executable policy-gradient loop and TCL governor telemetry.",
                domain="reinforcement_learning",
                entrypoint="tar_lab.multimodal_payloads",
                status="executable",
                research_grade_capable=True,
                expected_data_type="trajectory_buffer",
                requires_tokenizer=False,
                supports_resume=True,
                supports_distributed=False,
                requires_gpu=False,
                required_deps=["torch", "numpy"],
                required_metrics=["episodic_return", "effective_dimensionality", "entropy_sigma", "drift_rho"],
                required_artifacts=["execution_summary.json", "thermo_metrics.jsonl"],
                valid_input_contract=[
                    "Requires an explicit environment provenance block or a known RL environment contract.",
                    "Must report governor-compatible telemetry for policy updates.",
                ],
                notes=[
                    "Runs a real on-policy reinforcement-learning loop instead of a registration-only stub.",
                ],
            ),
            "asc_qml": ExperimentBackendSpec(
                backend_id="asc_qml",
                summary="Quantum-ML ASC backend with executable variational-circuit training loop under TCL governance.",
                domain="quantum_ml",
                entrypoint="tar_lab.multimodal_payloads",
                status="executable",
                research_grade_capable=True,
                expected_data_type="variational_circuit_dataset",
                requires_tokenizer=False,
                supports_resume=True,
                supports_distributed=False,
                requires_gpu=False,
                required_deps=["numpy", "torch"],
                required_metrics=["accuracy", "effective_dimensionality", "entropy_sigma", "drift_rho"],
                required_artifacts=["execution_summary.json", "thermo_metrics.jsonl"],
                valid_input_contract=[
                    "Requires circuit-ready numeric features and labels, or a declared local simulator path.",
                    "Must surface whether execution used PennyLane or the internal variational simulator.",
                ],
                notes=[
                    "Runs a real minimal variational-circuit optimization loop with thermodynamic observability.",
                ],
            ),
        }

    def list_backends(self) -> List[ExperimentBackendSpec]:
        return list(self._backends.values())

    def canonical_backend_id(self, backend_id: str) -> str:
        return self._aliases.get(backend_id, backend_id)

    def get(self, backend_id: str) -> ExperimentBackendSpec:
        backend_id = self.canonical_backend_id(backend_id)
        if backend_id not in self._backends:
            raise KeyError(f"Unknown experiment backend: {backend_id}")
        return self._backends[backend_id]

    def as_provenance(self, backend_id: str) -> BackendProvenance:
        spec = self.get(backend_id)
        return BackendProvenance.model_validate(spec.model_dump(mode="json"))

    def _state_path(self, trial_name: str, backend_id: str) -> Path:
        return self.store.experiment_backend_state_path(trial_name, backend_id)

    @staticmethod
    def _container_state_path(trial_name: str, backend_id: str) -> str:
        return f"/workspace/tar_state/experiment_backends/{trial_name}__{backend_id}.json"

    @staticmethod
    def _container_output_dir(trial_name: str, backend_id: str) -> str:
        return f"/workspace/tar_runs/{trial_name}/{backend_id}"

    def _build_artifact_lineage(
        self,
        spec: ExperimentBackendSpec,
        output_dir: Path,
        payload: Dict[str, Any],
    ) -> ExperimentArtifactLineage:
        if spec.backend_id == "asc_full":
            size = str(payload.get("size", "124M"))
            run_dir = output_dir / f"ASC-{size}"
            return ExperimentArtifactLineage(
                training_log_path=str(run_dir / "training_log.json"),
                latest_checkpoint_path=str(run_dir / "resume_state.pt"),
                final_checkpoint_path=str(run_dir / "final"),
            )
        if spec.backend_id == "asc_text":
            return ExperimentArtifactLineage(
                summary_path=str(output_dir / "payload_summary.json"),
                latest_checkpoint_path=str(output_dir / "asc_checkpoint.pt"),
            )
        if spec.backend_id in {"asc_cv", "asc_rl", "asc_qml"}:
            return ExperimentArtifactLineage(
                summary_path=str(output_dir / "execution_summary.json"),
                latest_checkpoint_path=str(output_dir / f"{spec.backend_id}_checkpoint.pt"),
            )
        return ExperimentArtifactLineage(
            training_log_path=str(output_dir / "training_log.json"),
            final_checkpoint_path=str(output_dir / "final"),
        )

    def _resolve_resume(
        self,
        spec: ExperimentBackendSpec,
        output_dir: Path,
        payload: Dict[str, Any],
        artifact_lineage: ExperimentArtifactLineage,
    ) -> ExperimentResumeInfo:
        requested_path = payload.get("resume_from_checkpoint")
        if requested_path is not None:
            requested = Path(str(requested_path))
            return ExperimentResumeInfo(
                supported=spec.supports_resume,
                requested=True,
                mode="checkpoint_resume",
                resume_from_checkpoint=str(requested),
                latest_checkpoint_path=str(requested),
                checkpoint_exists=requested.exists(),
                reason="explicit_resume_requested",
            )
        latest_path = artifact_lineage.latest_checkpoint_path
        if spec.supports_resume and latest_path:
            latest = Path(latest_path)
            if latest.exists():
                return ExperimentResumeInfo(
                    supported=True,
                    requested=True,
                    mode="checkpoint_resume",
                    resume_from_checkpoint=str(latest),
                    latest_checkpoint_path=str(latest),
                    checkpoint_exists=True,
                    reason="existing_checkpoint_detected",
                )
        return ExperimentResumeInfo(
            supported=spec.supports_resume,
            requested=False,
            mode="fresh_start",
            latest_checkpoint_path=latest_path,
            checkpoint_exists=bool(latest_path and Path(latest_path).exists()),
            reason="fresh_start",
        )

    def build_plan(
        self,
        backend_id: str,
        *,
        trial_name: str,
        config: Dict[str, Any] | None = None,
    ) -> ExperimentLaunchPlan:
        spec = self.get(backend_id)
        payload = dict(config or {})
        payload.setdefault("run_intent", "control")
        payload.setdefault("backend_provenance", self.as_provenance(spec.backend_id).model_dump(mode="json"))

        output_dir = self.workspace / "tar_runs" / trial_name / spec.backend_id
        output_dir.mkdir(parents=True, exist_ok=True)
        runtime = RuntimeSpec()
        artifact_lineage = self._build_artifact_lineage(spec, output_dir, payload)
        resume = self._resolve_resume(spec, output_dir, payload, artifact_lineage)
        backend_state_path = self._state_path(trial_name, spec.backend_id)
        container_output_dir = self._container_output_dir(trial_name, spec.backend_id)
        container_state_path = self._container_state_path(trial_name, spec.backend_id)

        if spec.backend_id == "asc_text":
            command = [
                "python",
                "-m",
                "tar_lab.train_template",
                "--config",
                f"/workspace/tar_runs/{trial_name}/config.json",
            ]
        elif spec.backend_id == "toy_anchor":
            command = [
                "python",
                "-m",
                "tar_lab.train_template",
                "--config",
                f"/workspace/tar_runs/{trial_name}/config.json",
                "--dry_run",
            ]
        elif spec.backend_id == "asc_full":
            command = [
                "python",
                "asc_train_full.py",
                "--size",
                str(payload.get("size", "124M")),
                "--dataset",
                str(payload.get("dataset", "wikitext-2-raw-v1")),
                "--epochs",
                str(payload.get("epochs", 1)),
                "--batch_size",
                str(payload.get("batch_size", 2)),
                "--max_steps",
                str(payload.get("max_steps", 100)),
                "--out_dir",
                container_output_dir,
                "--seed",
                str(payload.get("seed", 42)),
                "--backend_state_path",
                container_state_path,
            ]
            if resume.requested and resume.resume_from_checkpoint:
                command.extend(
                    [
                        "--resume_from_checkpoint",
                        f"{container_output_dir}/ASC-{str(payload.get('size', '124M'))}/resume_state.pt",
                    ]
                )
        elif spec.backend_id == "coding_asc":
            command = [
                "python",
                "coding_asc_finetune.py",
                "--model",
                str(payload.get("model", "tiny")),
                "--dataset",
                str(payload.get("dataset", "synthetic")),
                "--batch_size",
                str(payload.get("batch_size", 2)),
                "--max_steps",
                str(payload.get("max_steps", 10)),
            ]
        else:
            command = [
                "python",
                "-m",
                "tar_lab.multimodal_payloads",
                "--backend",
                spec.backend_id,
                "--trial-name",
                trial_name,
                "--config-json",
                f"/workspace/tar_runs/{trial_name}/{spec.backend_id}/backend_config.json",
            ]

        plan = ExperimentLaunchPlan(
            backend=spec,
            trial_name=trial_name,
            command=command,
            runtime=runtime,
            output_dir=str(output_dir),
            manifest_path=str(output_dir / "experiment_backend.json"),
            backend_state_path=str(backend_state_path),
            resume=resume,
            artifact_lineage=artifact_lineage,
            config=payload,
        )
        Path(plan.manifest_path).write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        self.store.save_experiment_backend_runtime(
            ExperimentBackendRuntimeRecord(
                trial_name=trial_name,
                backend_id=spec.backend_id,
                status="planned",
                output_dir=str(output_dir),
                manifest_path=plan.manifest_path,
                backend_state_path=str(backend_state_path),
                supports_resume=spec.supports_resume,
                resume=resume,
                artifact_lineage=artifact_lineage,
            )
        )
        if spec.backend_id in {"asc_cv", "asc_rl", "asc_qml"}:
            (output_dir / "backend_config.json").write_text(
                json_dumps(
                    {
                        "backend_id": spec.backend_id,
                        "trial_name": trial_name,
                        "run_intent": payload.get("run_intent", "control"),
                        "backend_provenance": payload.get("backend_provenance"),
                        "data_provenance": payload.get("data_provenance"),
                        "tokenizer_provenance": payload.get("tokenizer_provenance", {}),
                        "provenance_complete": payload.get("provenance_complete", False),
                        "research_grade": payload.get("research_grade", False),
                        "config": payload,
                    }
                ),
                encoding="utf-8",
            )
        return plan

    def status(self) -> dict[str, Any]:
        manifests = sorted((self.workspace / "tar_runs").glob("**/experiment_backend.json"))
        return {
            "available": [item.model_dump(mode="json") for item in self.list_backends()],
            "planned_runs": len(manifests),
            "runtime_records": [item.model_dump(mode="json") for item in self.store.list_experiment_backend_runtimes()],
        }
