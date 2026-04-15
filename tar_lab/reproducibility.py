from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from shutil import which
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, List, Optional

from tar_lab.errors import ReproducibilityLockError
from tar_lab.schemas import (
    BuildAttestation,
    DependencyPackageRecord,
    DependencyLockManifest,
    EnvironmentFingerprint,
    ImageManifest,
    PayloadEnvironmentReport,
    RunManifest,
    SandboxPolicy,
    ScienceEnvironmentBundle,
)
from tar_lab.state import TARStateStore


class PayloadEnvironmentBuilder:
    def __init__(self, workspace: str = "."):
        self.store = TARStateStore(workspace)
        self.root = self.store.state_dir / "payload_env"
        self.root.mkdir(parents=True, exist_ok=True)

    def prepare(self, image_tag: str = "tar-payload:locked") -> PayloadEnvironmentReport:
        requirements_path = self.root / "requirements-payload.txt"
        dockerfile_path = self.root / "Dockerfile"
        manifest_path = self.root / "payload_environment.json"
        existing_report = self.load()
        base_image = "pytorch/pytorch:latest"
        package_records = self._reuse_existing_payload_package_records(
            existing_report,
            expected_specs=self._payload_package_specs(),
            base_image=base_image,
        ) or self._resolve_package_records(
            self._payload_package_specs(),
            base_image=base_image,
        )
        packages = self._resolved_specs(package_records)
        unresolved_packages = self._unresolved_specs(package_records)
        lock_incomplete_reason = self._lock_incomplete_reason(package_records)

        report = PayloadEnvironmentReport(
            image_tag=image_tag,
            dockerfile_path=str(dockerfile_path),
            requirements_path=str(requirements_path),
            manifest_path=str(manifest_path),
            packages=packages,
            package_records=package_records,
            unresolved_packages=unresolved_packages,
            lock_incomplete_reason=lock_incomplete_reason,
            reproducibility_complete=lock_incomplete_reason is None,
        )

        if lock_incomplete_reason is None:
            requirements_path.write_text("\n".join(packages) + "\n", encoding="utf-8")
            dockerfile_path.write_text(self._render_dockerfile(requirements_path.name), encoding="utf-8")

            lock_manifest = self._dependency_lock_manifest(requirements_path, package_records)
            fingerprint = self._environment_fingerprint(
                dockerfile_path=dockerfile_path,
                requirements_path=requirements_path,
            )
            image_manifest = self._image_manifest(
                image_tag=image_tag,
                base_image=base_image,
                dockerfile_path=dockerfile_path,
                context_path=self.root,
                dependency_lock=lock_manifest,
                fingerprint=fingerprint,
            )
            self.store.write_image_manifest(image_manifest)
            run_manifest = self.create_run_manifest(
                kind="payload",
                image_manifest=image_manifest,
                sandbox_policy=self.default_sandbox_policy(artifact_dir="/workspace/tar_runs"),
                command=["python", "-m", "tar_lab.train_template"],
                config_path=str(self.root / "requirements-payload.txt"),
                trial_id="payload-environment",
            )
            report = report.model_copy(
                update={
                    "build_command": image_manifest.build_command,
                    "image_manifest": image_manifest,
                    "run_manifest": run_manifest,
                    "reproducibility_complete": True,
                }
            )
            report = self._preserve_build_state_if_unchanged(existing_report, report)
        manifest_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return report

    def load(self) -> Optional[PayloadEnvironmentReport]:
        manifest_path = self.root / "payload_environment.json"
        if not manifest_path.exists():
            return None
        return PayloadEnvironmentReport.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    def lock_science_bundle(self, bundle: ScienceEnvironmentBundle, *, base_image: Optional[str] = None) -> ScienceEnvironmentBundle:
        requirements_path = Path(bundle.requirements_path)
        dockerfile_path = Path(bundle.dockerfile_path)
        package_specs = [
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        package_records = self._resolve_package_records(
            package_specs,
            base_image=base_image or self._infer_base_image(dockerfile_path),
        )
        packages = self._resolved_specs(package_records)
        unresolved_packages = self._unresolved_specs(package_records)
        lock_incomplete_reason = self._lock_incomplete_reason(package_records)
        artifact_dir = self._workspace_relative_container_path(Path(bundle.execution_report_path).parent)
        sandbox_policy = self.default_sandbox_policy(
            artifact_dir=artifact_dir,
            writable_mounts=[artifact_dir] if artifact_dir is not None else None,
            read_only_mounts=["/workspace"],
        )

        if lock_incomplete_reason is not None:
            return bundle.model_copy(
                update={
                    "image_manifest_path": None,
                    "run_manifest_path": None,
                    "image_manifest": None,
                    "run_manifest": None,
                    "reproducibility_complete": False,
                    "locked_packages": packages,
                    "unresolved_packages": unresolved_packages,
                    "lock_incomplete_reason": lock_incomplete_reason,
                    "sandbox_policy": sandbox_policy,
                }
            )

        requirements_path.write_text("\n".join(packages) + "\n", encoding="utf-8")
        lock_manifest = self._dependency_lock_manifest(requirements_path, package_records)
        fingerprint = self._environment_fingerprint(
            dockerfile_path=dockerfile_path,
            requirements_path=requirements_path,
        )
        image_manifest = self._image_manifest(
            image_tag=bundle.docker_image_tag,
            base_image=base_image or self._infer_base_image(dockerfile_path),
            dockerfile_path=dockerfile_path,
            context_path=Path(bundle.build_context_path),
            dependency_lock=lock_manifest,
            fingerprint=fingerprint,
        )
        image_manifest_path = self.store.write_image_manifest(image_manifest)
        run_manifest = self.create_run_manifest(
            kind="science_bundle",
            image_manifest=image_manifest,
            sandbox_policy=sandbox_policy,
            command=bundle.run_command,
            config_path=bundle.study_plan_path,
            problem_id=bundle.problem_id,
        )
        return bundle.model_copy(
            update={
                "image_manifest_path": str(image_manifest_path),
                "run_manifest_path": str(self.store.manifests_dir / f"{run_manifest.manifest_id}.json"),
                "image_manifest": image_manifest,
                "run_manifest": run_manifest,
                "run_command": self._locked_science_run_command(
                    bundle.run_command,
                    image_manifest_path=image_manifest_path,
                    run_manifest=run_manifest,
                    workspace=self.store.workspace,
                ),
                "reproducibility_complete": True,
                "locked_packages": packages,
                "unresolved_packages": [],
                "lock_incomplete_reason": None,
                "sandbox_policy": sandbox_policy,
            }
        )

    def create_run_manifest(
        self,
        *,
        kind: str,
        image_manifest: ImageManifest,
        sandbox_policy: SandboxPolicy,
        command: List[str],
        config_path: Optional[str] = None,
        trial_id: Optional[str] = None,
        problem_id: Optional[str] = None,
    ) -> RunManifest:
        if not image_manifest.locked or not image_manifest.dependency_lock.fully_pinned:
            raise ReproducibilityLockError(
                image_manifest.dependency_lock.lock_incomplete_reason
                or "Run manifest creation requires a fully pinned image manifest."
            )
        payload = {
            "kind": kind,
            "command": command,
            "config_path": config_path,
            "trial_id": trial_id,
            "problem_id": problem_id,
            "image_hash": image_manifest.hash_sha256,
            "sandbox": sandbox_policy.model_dump(mode="json"),
        }
        manifest_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        manifest = RunManifest(
            manifest_id=f"run-{manifest_hash[:16]}",
            kind=kind,  # type: ignore[arg-type]
            trial_id=trial_id,
            problem_id=problem_id,
            command=command,
            config_path=config_path,
            image_manifest=image_manifest,
            sandbox_policy=sandbox_policy,
            hash_sha256=manifest_hash,
            reproducibility_complete=image_manifest.locked,
            unresolved_packages=list(image_manifest.dependency_lock.unresolved_packages),
            lock_incomplete_reason=image_manifest.dependency_lock.lock_incomplete_reason,
        )
        self.store.write_run_manifest(manifest)
        return manifest

    def attach_payload_build_attestation(
        self,
        report: PayloadEnvironmentReport,
        *,
        build_result: Any,
    ) -> PayloadEnvironmentReport:
        if report.image_manifest is None:
            return report
        attestation = self._build_attestation(
            scope_kind="payload_environment",
            image_manifest=report.image_manifest,
            run_manifest=report.run_manifest,
            build_result=build_result,
        )
        path = self.store.save_build_attestation(attestation)
        updated = report.model_copy(
            update={
                "build_status": attestation.build_status,
                "build_command": attestation.build_command,
                "build_attestation_path": str(path),
                "build_attestation": attestation,
            }
        )
        Path(updated.manifest_path).write_text(updated.model_dump_json(indent=2), encoding="utf-8")
        return updated

    def attach_science_build_attestation(
        self,
        bundle: ScienceEnvironmentBundle,
        *,
        build_result: Any,
    ) -> ScienceEnvironmentBundle:
        if bundle.image_manifest is None:
            return bundle
        attestation = self._build_attestation(
            scope_kind="science_bundle",
            image_manifest=bundle.image_manifest,
            run_manifest=bundle.run_manifest,
            build_result=build_result,
            trial_id=bundle.problem_id,
            problem_id=bundle.problem_id,
        )
        path = self.store.save_build_attestation(attestation)
        return bundle.model_copy(
            update={
                "build_status": attestation.build_status,
                "build_command": attestation.build_command,
                "build_attestation_path": str(path),
                "build_attestation": attestation,
            }
        )

    @staticmethod
    def default_sandbox_policy(
        *,
        artifact_dir: Optional[str] = None,
        writable_mounts: Optional[List[str]] = None,
        read_only_mounts: Optional[List[str]] = None,
        profile: str = "production",
    ) -> SandboxPolicy:
        writable_mounts = list(writable_mounts or ["/workspace/tar_runs", "/workspace/logs", "/workspace/anchors"])
        read_only_mounts = list(read_only_mounts or ["/workspace", "/data"])
        return SandboxPolicy(
            mode="docker_only",
            profile=profile,  # type: ignore[arg-type]
            network_policy="off",
            allowed_mounts=sorted(dict.fromkeys([*read_only_mounts, *writable_mounts])),
            read_only_mounts=read_only_mounts,
            writable_mounts=writable_mounts,
            cpu_limit=1,
            memory_limit_gb=1,
            timeout_s=30,
            artifact_dir=artifact_dir,
            workspace_root="/workspace",
        )

    @staticmethod
    def _render_dockerfile(requirements_name: str) -> str:
        return "\n".join(
            [
                "FROM pytorch/pytorch:latest",
                f"COPY {requirements_name} /tmp/{requirements_name}",
                f"RUN python -m pip install --no-cache-dir -r /tmp/{requirements_name}",
                "RUN python -m pip uninstall -y torchvision torchaudio || true",
                "ENV TRANSFORMERS_NO_TORCHVISION=1",
                "WORKDIR /workspace",
            ]
        ) + "\n"

    @staticmethod
    def _payload_package_specs() -> List[str]:
        return [
            "datasets",
            "numpy",
            "peft",
            "pydantic",
            "sentence-transformers",
            "torch",
            "transformers",
        ]

    @staticmethod
    def _normalize_package_name(name: str) -> str:
        return name.strip().lower().replace("_", "-")

    @classmethod
    def _extract_package_name(cls, spec: str) -> Optional[str]:
        cleaned = spec.split(";", 1)[0].strip()
        if not cleaned or any(token in cleaned for token in ("@", "[")):
            return None
        for operator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if operator in cleaned:
                cleaned = cleaned.split(operator, 1)[0].strip()
                break
        return cls._normalize_package_name(cleaned) if cleaned else None

    @staticmethod
    def _exact_version(spec: str) -> Optional[str]:
        cleaned = spec.split(";", 1)[0].strip()
        if "==" not in cleaned or any(token in cleaned for token in ("@", "[", ">=", "<=", "~=", "!=", ">", "<")):
            return None
        _, version = cleaned.split("==", 1)
        version = version.strip()
        return version or None

    @classmethod
    def _resolve_package_record_from_host(
        cls,
        spec: str,
        *,
        required: bool = True,
    ) -> DependencyPackageRecord:
        normalized_name = cls._extract_package_name(spec)
        if not normalized_name:
            return DependencyPackageRecord(
                requested_spec=spec,
                normalized_name=cls._normalize_package_name(spec),
                required=required,
                resolution_status="missing_version" if required else "optional_missing",
            )

        exact_version = cls._exact_version(spec)
        if exact_version is not None:
            return DependencyPackageRecord(
                requested_spec=spec,
                normalized_name=normalized_name,
                resolved_spec=f"{normalized_name}=={exact_version}",
                version=exact_version,
                required=required,
                resolution_status="pinned",
            )

        try:
            version = metadata.version(normalized_name)
        except metadata.PackageNotFoundError:
            return DependencyPackageRecord(
                requested_spec=spec,
                normalized_name=normalized_name,
                required=required,
                resolution_status="missing_version" if required else "optional_missing",
            )

        return DependencyPackageRecord(
            requested_spec=spec,
            normalized_name=normalized_name,
            resolved_spec=f"{normalized_name}=={version}",
            version=version,
            required=required,
            resolution_status="pinned",
        )

    def _resolve_package_records(
        self,
        specs: Iterable[str],
        *,
        required: bool = True,
        base_image: Optional[str] = None,
    ) -> List[DependencyPackageRecord]:
        cleaned_specs = [
            raw_spec.strip()
            for raw_spec in specs
            if raw_spec.strip() and not raw_spec.strip().startswith("#")
        ]
        mode = self._target_image_locking_mode()
        if base_image and mode != "host":
            try:
                return self._resolve_package_records_in_image(
                    cleaned_specs,
                    base_image=base_image,
                    required=required,
                )
            except RuntimeError:
                if mode == "target":
                    return self._unresolved_target_records(cleaned_specs, required=required)
            except subprocess.SubprocessError:
                return self._unresolved_target_records(cleaned_specs, required=required)

        records_by_name: dict[str, DependencyPackageRecord] = {}
        for spec in cleaned_specs:
            record = self._resolve_package_record_from_host(spec, required=required)
            key = record.normalized_name or self._normalize_package_name(record.requested_spec)
            existing = records_by_name.get(key)
            if existing is None:
                records_by_name[key] = record
                continue
            if existing.resolved_spec == record.resolved_spec and existing.resolution_status == record.resolution_status:
                continue
            records_by_name[key] = DependencyPackageRecord(
                requested_spec=f"{existing.requested_spec} | {record.requested_spec}",
                normalized_name=key,
                required=existing.required or record.required,
                resolution_status="missing_version" if (existing.required or record.required) else "optional_missing",
            )
        return sorted(records_by_name.values(), key=lambda item: item.normalized_name)

    def _resolve_package_records_in_image(
        self,
        specs: List[str],
        *,
        base_image: str,
        required: bool,
    ) -> List[DependencyPackageRecord]:
        if not specs:
            return []
        resolved_versions = self._resolve_versions_in_image(specs, base_image=base_image)
        records: List[DependencyPackageRecord] = []
        for spec in specs:
            normalized_name = self._extract_package_name(spec)
            if not normalized_name:
                records.append(
                    DependencyPackageRecord(
                        requested_spec=spec,
                        normalized_name=self._normalize_package_name(spec),
                        required=required,
                        resolution_status="missing_version" if required else "optional_missing",
                    )
                )
                continue
            exact_version = self._exact_version(spec)
            version = exact_version or resolved_versions.get(normalized_name)
            if version is None:
                records.append(
                    DependencyPackageRecord(
                        requested_spec=spec,
                        normalized_name=normalized_name,
                        required=required,
                        resolution_status="missing_version" if required else "optional_missing",
                    )
                )
                continue
            records.append(
                DependencyPackageRecord(
                    requested_spec=spec,
                    normalized_name=normalized_name,
                    resolved_spec=f"{normalized_name}=={version}",
                    version=version,
                    required=required,
                    resolution_status="pinned",
                )
            )
        if any(record.required and record.resolution_status != "pinned" for record in records):
            return self._unresolved_target_records(specs, required=required)
        return sorted(records, key=lambda item: item.normalized_name)

    def _resolve_versions_in_image(
        self,
        specs: List[str],
        *,
        base_image: str,
    ) -> dict[str, str]:
        docker_bin = self._docker_bin()
        with tempfile.TemporaryDirectory(prefix="tar-resolve-") as tmp:
            report_host_path = Path(tmp) / "pip-report.json"
            command = [
                docker_bin,
                "run",
                "--rm",
                "-v",
                f"{report_host_path.parent}:/tar-resolve",
                base_image,
                "python",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--ignore-installed",
                "--dry-run",
                "--report",
                "/tar-resolve/pip-report.json",
                *specs,
            ]
            try:
                proc = subprocess.run(command, capture_output=True, text=True, check=False)
            except FileNotFoundError as exc:
                raise RuntimeError("docker runtime unavailable") from exc
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip().lower()
                if any(
                    marker in stderr
                    for marker in (
                        "docker daemon",
                        "error during connect",
                        "is the docker daemon running",
                        "cannot connect",
                        "file not found",
                    )
                ):
                    raise RuntimeError("docker runtime unavailable")
                raise subprocess.SubprocessError(proc.stderr or proc.stdout or "target image resolution failed")
            if not report_host_path.exists():
                raise subprocess.SubprocessError("target image resolution did not emit a pip report")
            payload = json.loads(report_host_path.read_text(encoding="utf-8"))
        installs = payload.get("install") or []
        resolved: dict[str, str] = {}
        for item in installs:
            metadata_block = item.get("metadata") or {}
            name = metadata_block.get("name")
            version = metadata_block.get("version")
            if not name or not version:
                continue
            resolved[self._normalize_package_name(name)] = str(version)
        return resolved

    def _unresolved_target_records(
        self,
        specs: List[str],
        *,
        required: bool,
    ) -> List[DependencyPackageRecord]:
        records: List[DependencyPackageRecord] = []
        for spec in specs:
            normalized_name = self._extract_package_name(spec)
            exact_version = self._exact_version(spec)
            if normalized_name and exact_version is not None:
                records.append(
                    DependencyPackageRecord(
                        requested_spec=spec,
                        normalized_name=normalized_name,
                        resolved_spec=f"{normalized_name}=={exact_version}",
                        version=exact_version,
                        required=required,
                        resolution_status="pinned",
                    )
                )
                continue
            records.append(
                DependencyPackageRecord(
                    requested_spec=spec,
                    normalized_name=normalized_name or self._normalize_package_name(spec),
                    required=required,
                    resolution_status="missing_version" if required else "optional_missing",
                )
            )
        return sorted(records, key=lambda item: item.normalized_name)

    @staticmethod
    def _target_image_locking_mode() -> str:
        mode = os.environ.get("TAR_TARGET_IMAGE_LOCKING", "auto").strip().lower()
        if mode not in {"auto", "host", "target"}:
            return "auto"
        return mode

    def _preserve_build_state_if_unchanged(
        self,
        existing_report: Optional[PayloadEnvironmentReport],
        report: PayloadEnvironmentReport,
    ) -> PayloadEnvironmentReport:
        if existing_report is None:
            return report
        if existing_report.image_manifest is None or report.image_manifest is None:
            return report
        if existing_report.run_manifest is None or report.run_manifest is None:
            return report
        if existing_report.image_tag != report.image_tag:
            return report
        if existing_report.image_manifest.base_image != report.image_manifest.base_image:
            return report
        existing_lock = existing_report.image_manifest.dependency_lock
        current_lock = report.image_manifest.dependency_lock
        if existing_lock.hash_sha256 != current_lock.hash_sha256:
            return report
        existing_fp = existing_report.image_manifest.environment_fingerprint
        current_fp = report.image_manifest.environment_fingerprint
        if existing_fp.dockerfile_hash_sha256 != current_fp.dockerfile_hash_sha256:
            return report
        if existing_fp.requirements_hash_sha256 != current_fp.requirements_hash_sha256:
            return report
        if existing_report.run_manifest.kind != report.run_manifest.kind:
            return report
        if existing_report.run_manifest.command != report.run_manifest.command:
            return report
        if existing_report.run_manifest.config_path != report.run_manifest.config_path:
            return report
        return report.model_copy(
            update={
                "build_command": existing_report.build_command,
                "build_status": existing_report.build_status,
                "build_attestation_path": existing_report.build_attestation_path,
                "build_attestation": existing_report.build_attestation,
            }
        )

    def _reuse_existing_payload_package_records(
        self,
        existing_report: Optional[PayloadEnvironmentReport],
        *,
        expected_specs: List[str],
        base_image: str,
    ) -> Optional[List[DependencyPackageRecord]]:
        if existing_report is None:
            return None
        if not existing_report.reproducibility_complete:
            return None
        if existing_report.image_manifest is None:
            return None
        if existing_report.image_manifest.base_image != base_image:
            return None
        records = existing_report.package_records or existing_report.image_manifest.dependency_lock.package_records
        if not records or any(record.resolution_status != "pinned" for record in records):
            return None
        expected_names = sorted(
            self._extract_package_name(spec) or self._normalize_package_name(spec)
            for spec in expected_specs
        )
        existing_names = sorted(
            self._extract_package_name(record.requested_spec) or self._normalize_package_name(record.requested_spec)
            for record in records
        )
        if expected_names != existing_names:
            return None
        return [record.model_copy(deep=True) for record in records]

    @staticmethod
    def _resolved_specs(records: Iterable[DependencyPackageRecord]) -> List[str]:
        return sorted(
            {record.resolved_spec for record in records if record.resolved_spec},
            key=lambda item: item.lower(),
        )

    @staticmethod
    def _unresolved_specs(records: Iterable[DependencyPackageRecord]) -> List[str]:
        unresolved = []
        for record in records:
            if record.resolution_status == "pinned":
                continue
            unresolved.append(record.requested_spec)
        return sorted(dict.fromkeys(unresolved), key=str.lower)

    @staticmethod
    def _lock_incomplete_reason(records: Iterable[DependencyPackageRecord]) -> Optional[str]:
        missing = sorted(
            {
                record.requested_spec
                for record in records
                if record.required and record.resolution_status != "pinned"
            },
            key=str.lower,
        )
        if not missing:
            return None
        return "Missing pinned versions for required packages: " + ", ".join(missing) + "."

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _dependency_lock_manifest(
        self,
        requirements_path: Path,
        package_records: Iterable[DependencyPackageRecord],
    ) -> DependencyLockManifest:
        content = requirements_path.read_text(encoding="utf-8")
        hash_sha = self._hash_text(content)
        records = list(package_records)
        return DependencyLockManifest(
            lock_id=f"lock-{hash_sha[:16]}",
            requirements_path=str(requirements_path),
            packages=self._resolved_specs(records),
            package_records=records,
            unresolved_packages=self._unresolved_specs(records),
            fully_pinned=self._lock_incomplete_reason(records) is None,
            lock_incomplete_reason=self._lock_incomplete_reason(records),
            hash_sha256=hash_sha,
        )

    def _environment_fingerprint(self, *, dockerfile_path: Path, requirements_path: Path) -> EnvironmentFingerprint:
        source_hash = self._workspace_source_hash()
        dockerfile_hash = self._hash_text(dockerfile_path.read_text(encoding="utf-8"))
        requirements_hash = self._hash_text(requirements_path.read_text(encoding="utf-8"))
        combined = hashlib.sha256(
            json.dumps(
                {
                    "source": source_hash,
                    "dockerfile": dockerfile_hash,
                    "requirements": requirements_hash,
                    "python": platform.python_version(),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return EnvironmentFingerprint(
            fingerprint_id=f"env-{combined[:16]}",
            workspace_root=str(self.store.workspace),
            source_hash_sha256=source_hash,
            dockerfile_hash_sha256=dockerfile_hash,
            requirements_hash_sha256=requirements_hash,
            python_version=platform.python_version(),
        )

    def _image_manifest(
        self,
        *,
        image_tag: str,
        base_image: str,
        dockerfile_path: Path,
        context_path: Path,
        dependency_lock: DependencyLockManifest,
        fingerprint: EnvironmentFingerprint,
    ) -> ImageManifest:
        build_command = [
            self._docker_bin(),
            "build",
            "-t",
            image_tag,
            "-f",
            str(dockerfile_path),
            str(context_path),
        ]
        hash_sha = hashlib.sha256(
            json.dumps(
                {
                    "image_tag": image_tag,
                    "base_image": base_image,
                    "build_command": build_command,
                    "dependency_lock": dependency_lock.hash_sha256,
                    "fingerprint": fingerprint.fingerprint_id,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return ImageManifest(
            image_tag=image_tag,
            base_image=base_image,
            dockerfile_path=str(dockerfile_path),
            build_context_path=str(context_path),
            build_command=build_command,
            dependency_lock=dependency_lock,
            environment_fingerprint=fingerprint,
            hash_sha256=hash_sha,
            locked=dependency_lock.fully_pinned,
        )

    def _workspace_source_hash(self) -> str:
        excluded_parts = {
            "__pycache__",
            ".git",
            ".venv",
            ".pytest_cache",
            "tar_state",
            "tar_runs",
            "logs",
            "asc_model_fixed",
            "coding_ai_out",
            "dataset_artifacts",
            "eval_artifacts",
            "training_artifacts",
        }
        tracked = []
        for path in sorted(self.store.workspace.rglob("*")):
            if not path.is_file():
                continue
            if any(part in excluded_parts for part in path.parts):
                continue
            if path.suffix.lower() not in {".py", ".json", ".md", ".txt", ".yaml", ".yml"}:
                continue
            tracked.append(
                {
                    "path": str(path.relative_to(self.store.workspace)).replace("\\", "/"),
                    "sha": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        return hashlib.sha256(json.dumps(tracked, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _infer_base_image(dockerfile_path: Path) -> str:
        for line in dockerfile_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("FROM "):
                return stripped.split(maxsplit=1)[1]
        return "pytorch/pytorch:latest"

    @staticmethod
    def _docker_bin() -> str:
        for env_name in ("TAR_DOCKER_BIN", "DOCKER_BIN"):
            value = os.environ.get(env_name)
            if value:
                return value
        return which("docker") or "docker"

    @staticmethod
    def _locked_science_run_command(
        command: List[str],
        *,
        image_manifest_path: Path,
        run_manifest: RunManifest,
        workspace: Path,
    ) -> List[str]:
        locked = list(command)
        if len(locked) >= 3 and locked[0] == "docker" and locked[1] == "run":
            image_manifest_container_path = "/workspace/" + image_manifest_path.relative_to(workspace).as_posix()
            run_manifest_path = workspace / "tar_state" / "manifests" / f"{run_manifest.manifest_id}.json"
            run_manifest_container_path = "/workspace/" + run_manifest_path.relative_to(workspace).as_posix()
            insertion = [
                "--network",
                "none",
                "-e",
                f"TAR_IMAGE_MANIFEST={image_manifest_container_path}",
                "-e",
                f"TAR_RUN_MANIFEST={run_manifest_container_path}",
            ]
            locked = [locked[0], locked[1], *insertion, *locked[2:]]
        return locked

    def _workspace_relative_container_path(self, path: Path) -> Optional[str]:
        try:
            rel = path.resolve().relative_to(self.store.workspace.resolve())
        except ValueError:
            return None
        return f"/workspace/{rel.as_posix()}"

    def _build_attestation(
        self,
        *,
        scope_kind: str,
        image_manifest: ImageManifest,
        run_manifest: Optional[RunManifest],
        build_result: Any,
        trial_id: Optional[str] = None,
        problem_id: Optional[str] = None,
    ) -> BuildAttestation:
        build_status = "dry_run" if getattr(build_result, "mode", None) == "dry_run" else (
            "built" if getattr(build_result, "returncode", None) == 0 else "failed"
        )
        payload = {
            "scope_kind": scope_kind,
            "image_tag": image_manifest.image_tag,
            "build_command": list(getattr(build_result, "command", []) or []),
            "builder_backend": getattr(build_result, "mode", "subprocess"),
            "build_status": build_status,
            "returncode": getattr(build_result, "returncode", None),
            "image_manifest_hash": image_manifest.hash_sha256,
            "dependency_lock_hash": image_manifest.dependency_lock.hash_sha256,
            "environment_fingerprint_id": image_manifest.environment_fingerprint.fingerprint_id,
            "run_manifest_hash": run_manifest.hash_sha256 if run_manifest is not None else None,
            "image_digest": getattr(build_result, "image_digest", None),
            "image_id": getattr(build_result, "image_id", None),
            "digest_source": getattr(build_result, "digest_source", "unavailable"),
            "trial_id": trial_id,
            "problem_id": problem_id,
        }
        attestation_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return BuildAttestation(
            attestation_id=f"build-{attestation_hash[:16]}",
            scope_kind=scope_kind,  # type: ignore[arg-type]
            image_tag=image_manifest.image_tag,
            build_command=list(getattr(build_result, "command", []) or []),
            builder_backend=getattr(build_result, "mode", "subprocess"),  # type: ignore[arg-type]
            build_status=build_status,  # type: ignore[arg-type]
            returncode=getattr(build_result, "returncode", None),
            image_manifest_hash=image_manifest.hash_sha256,
            dependency_lock_hash=image_manifest.dependency_lock.hash_sha256,
            environment_fingerprint_id=image_manifest.environment_fingerprint.fingerprint_id,
            run_manifest_hash=run_manifest.hash_sha256 if run_manifest is not None else None,
            image_digest=getattr(build_result, "image_digest", None),
            image_id=getattr(build_result, "image_id", None),
            digest_source=getattr(build_result, "digest_source", "unavailable"),  # type: ignore[arg-type]
            trial_id=trial_id,
            problem_id=problem_id,
        )
