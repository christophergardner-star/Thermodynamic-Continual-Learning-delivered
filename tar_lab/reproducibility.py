from __future__ import annotations

import hashlib
import json
import os
import platform
from shutil import which
from importlib import metadata
from pathlib import Path
from typing import Iterable, List, Optional

from tar_lab.schemas import (
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
        packages = self._locked_packages()
        requirements_path.write_text("\n".join(packages) + "\n", encoding="utf-8")
        dockerfile_path.write_text(self._render_dockerfile(requirements_path.name), encoding="utf-8")

        lock_manifest = self._dependency_lock_manifest(requirements_path, packages)
        fingerprint = self._environment_fingerprint(
            dockerfile_path=dockerfile_path,
            requirements_path=requirements_path,
        )
        image_manifest = self._image_manifest(
            image_tag=image_tag,
            base_image="pytorch/pytorch:latest",
            dockerfile_path=dockerfile_path,
            context_path=self.root,
            dependency_lock=lock_manifest,
            fingerprint=fingerprint,
        )
        image_manifest_path = self.store.write_image_manifest(image_manifest)
        run_manifest = self.create_run_manifest(
            kind="payload",
            image_manifest=image_manifest,
            sandbox_policy=self.default_sandbox_policy(artifact_dir="/workspace/tar_runs"),
            command=["python", "-m", "tar_lab.train_template"],
            config_path=str(self.root / "requirements-payload.txt"),
            trial_id="payload-environment",
        )
        report = PayloadEnvironmentReport(
            image_tag=image_tag,
            dockerfile_path=str(dockerfile_path),
            requirements_path=str(requirements_path),
            manifest_path=str(manifest_path),
            build_command=image_manifest.build_command,
            packages=packages,
            image_manifest=image_manifest,
            run_manifest=run_manifest,
            reproducibility_complete=True,
        )
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
        packages = [line.strip() for line in requirements_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        lock_manifest = self._dependency_lock_manifest(requirements_path, packages)
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
            sandbox_policy=self.default_sandbox_policy(artifact_dir=str(Path(bundle.execution_report_path).parent)),
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
                "sandbox_policy": self.default_sandbox_policy(artifact_dir=str(Path(bundle.execution_report_path).parent)),
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
        )
        self.store.write_run_manifest(manifest)
        return manifest

    @staticmethod
    def default_sandbox_policy(*, artifact_dir: Optional[str] = None) -> SandboxPolicy:
        return SandboxPolicy(
            mode="docker_only",
            network_policy="off",
            cpu_limit=1,
            memory_limit_gb=1,
            timeout_s=30,
            artifact_dir=artifact_dir,
        )

    @staticmethod
    def _render_dockerfile(requirements_name: str) -> str:
        return "\n".join(
            [
                "FROM pytorch/pytorch:latest",
                f"COPY {requirements_name} /tmp/{requirements_name}",
                f"RUN python -m pip install --no-cache-dir -r /tmp/{requirements_name}",
                "WORKDIR /workspace",
            ]
        ) + "\n"

    @staticmethod
    def _locked_packages() -> List[str]:
        packages = []
        for name in ("pydantic", "torch", "numpy", "transformers", "peft", "datasets", "sentence-transformers"):
            try:
                version = metadata.version(name)
                packages.append(f"{name}=={version}")
            except metadata.PackageNotFoundError:
                packages.append(name)
        return packages

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _dependency_lock_manifest(self, requirements_path: Path, packages: Iterable[str]) -> DependencyLockManifest:
        content = requirements_path.read_text(encoding="utf-8")
        hash_sha = self._hash_text(content)
        return DependencyLockManifest(
            lock_id=f"lock-{hash_sha[:16]}",
            requirements_path=str(requirements_path),
            packages=list(packages),
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
        )

    def _workspace_source_hash(self) -> str:
        tracked = []
        for path in sorted(self.store.workspace.rglob("*")):
            if not path.is_file():
                continue
            if any(part in {"__pycache__", ".git", "tar_state", "logs", "asc_model_fixed", "coding_ai_out"} for part in path.parts):
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
