import tempfile
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

from tar_lab import reproducibility as reproducibility_module
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.docker_runner import BuildResult
from tar_lab.reproducibility import PayloadEnvironmentBuilder
from tar_lab.schemas import BuildAttestation, DependencyPackageRecord, ScienceEnvironmentBundle, TrainingPayloadConfig
from tar_lab.state import TARStateStore


def _copy_science_profiles(tmp: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "science_profiles"
    target_dir = Path(tmp) / "science_profiles"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.json"):
        (target_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def test_payload_environment_builder_writes_locked_manifests():
    with tempfile.TemporaryDirectory() as tmp:
        builder = PayloadEnvironmentBuilder(tmp)
        report = builder.prepare()
        assert report.reproducibility_complete
        assert all("==" in package for package in report.packages)
        assert not report.unresolved_packages
        assert report.image_manifest is not None
        assert report.run_manifest is not None
        assert report.image_manifest.dependency_lock.fully_pinned is True
        assert Path(report.manifest_path).exists()
        manifest_path = Path(tmp) / "tar_state" / "manifests" / f"{report.run_manifest.manifest_id}.json"
        assert manifest_path.exists()
        dockerfile = Path(report.dockerfile_path).read_text(encoding="utf-8")
        assert "pip uninstall -y torchvision torchaudio" in dockerfile
        assert "TRANSFORMERS_NO_TORCHVISION=1" in dockerfile


def test_payload_environment_builder_manifest_hash_is_stable():
    with tempfile.TemporaryDirectory() as tmp:
        builder = PayloadEnvironmentBuilder(tmp)
        first = builder.prepare()
        second = builder.prepare()

        assert first.reproducibility_complete
        assert second.reproducibility_complete
        assert first.packages == second.packages
        assert first.image_manifest is not None and second.image_manifest is not None
        assert first.run_manifest is not None and second.run_manifest is not None
        assert first.image_manifest.hash_sha256 == second.image_manifest.hash_sha256
        assert first.run_manifest.hash_sha256 == second.run_manifest.hash_sha256
        assert first.image_manifest.dependency_lock.hash_sha256 == second.image_manifest.dependency_lock.hash_sha256


def test_payload_environment_builder_prefers_target_image_resolution(monkeypatch):
    def fake_target_resolve(specs, *, base_image, required):
        assert base_image == "pytorch/pytorch:latest"
        assert required is True
        return [
            DependencyPackageRecord(
                requested_spec=spec,
                normalized_name=PayloadEnvironmentBuilder._extract_package_name(spec) or spec,
                resolved_spec=f"{PayloadEnvironmentBuilder._extract_package_name(spec)}==1.2.3",
                version="1.2.3",
                required=True,
                resolution_status="pinned",
            )
            for spec in specs
        ]

    def fake_host_version(name: str) -> str:
        return "9.9.9"

    monkeypatch.setenv("TAR_TARGET_IMAGE_LOCKING", "auto")
    monkeypatch.setattr(reproducibility_module.metadata, "version", fake_host_version)

    with tempfile.TemporaryDirectory() as tmp:
        builder = PayloadEnvironmentBuilder(tmp)
        monkeypatch.setattr(builder, "_resolve_package_records_in_image", fake_target_resolve)
        report = builder.prepare()

        assert report.reproducibility_complete is True
        assert all(package.endswith("==1.2.3") for package in report.packages)


def test_payload_environment_builder_reuses_existing_locked_payload_packages(monkeypatch):
    versions = {
        "datasets": "4.8.4",
        "numpy": "2.2.6",
        "peft": "0.18.1",
        "pydantic": "2.13.0",
        "sentence-transformers": "5.4.0",
        "torch": "2.11.0",
        "transformers": "5.5.4",
    }

    def fake_target_resolve(specs, *, base_image, required):
        return [
            DependencyPackageRecord(
                requested_spec=spec,
                normalized_name=PayloadEnvironmentBuilder._extract_package_name(spec) or spec,
                resolved_spec=f"{PayloadEnvironmentBuilder._extract_package_name(spec)}=={versions[PayloadEnvironmentBuilder._extract_package_name(spec)]}",
                version=versions[PayloadEnvironmentBuilder._extract_package_name(spec)],
                required=required,
                resolution_status="pinned",
            )
            for spec in specs
        ]

    monkeypatch.setenv("TAR_TARGET_IMAGE_LOCKING", "auto")

    with tempfile.TemporaryDirectory() as tmp:
        builder = PayloadEnvironmentBuilder(tmp)
        monkeypatch.setattr(builder, "_resolve_package_records_in_image", fake_target_resolve)
        first = builder.prepare()

        monkeypatch.setattr(
            builder,
            "_resolve_package_records_in_image",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("existing lock should be reused")),
        )
        second = builder.prepare()

        assert first.packages == second.packages
        assert second.reproducibility_complete is True


def test_payload_environment_builder_fails_closed_when_package_version_missing(monkeypatch):
    original_version = reproducibility_module.metadata.version

    def fake_version(name: str) -> str:
        if name == "peft":
            raise metadata.PackageNotFoundError
        return original_version(name)

    monkeypatch.setattr(reproducibility_module.metadata, "version", fake_version)

    with tempfile.TemporaryDirectory() as tmp:
        builder = PayloadEnvironmentBuilder(tmp)
        report = builder.prepare()

        assert report.reproducibility_complete is False
        assert report.image_manifest is None
        assert report.run_manifest is None
        assert report.unresolved_packages == ["peft"]
        assert report.lock_incomplete_reason is not None
        assert "peft" in report.lock_incomplete_reason
        assert all("==" in package for package in report.packages)
        assert Path(report.manifest_path).exists()


def test_payload_environment_builder_preserves_build_attestation_when_lock_is_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        builder = PayloadEnvironmentBuilder(tmp)
        first = builder.prepare()
        first = builder.attach_payload_build_attestation(
            first,
            build_result=BuildResult(
                mode="subprocess",
                command=["docker", "build", "-t", first.image_tag],
                image_tag=first.image_tag,
                returncode=0,
                stdout="built",
                stderr="",
                image_digest=f"{first.image_tag}@sha256:abc123",
                image_id="sha256:image123",
                digest_source="docker_inspect",
            ),
        )

        second = builder.prepare()

        assert second.build_status == "built"
        assert second.build_attestation is not None
        assert second.build_attestation_path == first.build_attestation_path
        assert second.build_attestation.attestation_id == first.build_attestation.attestation_id


def test_payload_environment_builder_preserves_build_attestation_when_tar_runs_change():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        builder = PayloadEnvironmentBuilder(tmp)
        first = builder.prepare()
        first = builder.attach_payload_build_attestation(
            first,
            build_result=BuildResult(
                mode="subprocess",
                command=["docker", "build", "-t", first.image_tag],
                image_tag=first.image_tag,
                returncode=0,
                stdout="built",
                stderr="",
                image_digest=f"{first.image_tag}@sha256:def456",
                image_id="sha256:image456",
                digest_source="docker_inspect",
            ),
        )

        trial_dir = root / "tar_runs" / "trial-1"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "config.json").write_text("{\"generated\": true}\n", encoding="utf-8")

        second = builder.prepare()

        assert second.build_status == "built"
        assert second.build_attestation is not None
        assert second.build_attestation_path == first.build_attestation_path
        assert second.build_attestation.attestation_id == first.build_attestation.attestation_id


def test_payload_environment_builder_preserves_build_attestation_when_source_only_changes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "notes.md").write_text("before\n", encoding="utf-8")
        builder = PayloadEnvironmentBuilder(tmp)
        first = builder.prepare()
        first = builder.attach_payload_build_attestation(
            first,
            build_result=BuildResult(
                mode="subprocess",
                command=["docker", "build", "-t", first.image_tag],
                image_tag=first.image_tag,
                returncode=0,
                stdout="built",
                stderr="",
                image_digest=f"{first.image_tag}@sha256:ghi789",
                image_id="sha256:image789",
                digest_source="docker_inspect",
            ),
        )

        (root / "notes.md").write_text("after\n", encoding="utf-8")
        second = builder.prepare()

        assert second.build_status == "built"
        assert second.build_attestation is not None
        assert second.build_attestation_path == first.build_attestation_path
        assert second.build_attestation.attestation_id == first.build_attestation.attestation_id


def test_workspace_source_hash_ignores_generated_artifact_trees():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "app.py").write_text("print('ok')\n", encoding="utf-8")
        (root / ".venv").mkdir(parents=True, exist_ok=True)
        (root / ".venv" / "noise.py").write_text("print('noise')\n", encoding="utf-8")
        (root / "training_artifacts").mkdir(parents=True, exist_ok=True)
        (root / "training_artifacts" / "metadata.json").write_text("{\"ignored\": true}\n", encoding="utf-8")
        (root / "tar_runs").mkdir(parents=True, exist_ok=True)
        (root / "tar_runs" / "config.json").write_text("{\"generated\": true}\n", encoding="utf-8")

        builder = PayloadEnvironmentBuilder(tmp)
        first = builder._workspace_source_hash()

        (root / ".venv" / "noise.py").write_text("print('different')\n", encoding="utf-8")
        (root / "training_artifacts" / "metadata.json").write_text("{\"ignored\": false}\n", encoding="utf-8")
        (root / "tar_runs" / "config.json").write_text("{\"generated\": false}\n", encoding="utf-8")
        second = builder._workspace_source_hash()

        assert first == second


def test_science_bundle_lock_fails_closed_when_requirement_cannot_be_pinned(monkeypatch):
    original_version = reproducibility_module.metadata.version

    def fake_version(name: str) -> str:
        if name == "missing-package":
            raise metadata.PackageNotFoundError
        return original_version(name)

    monkeypatch.setattr(reproducibility_module.metadata, "version", fake_version)

    with tempfile.TemporaryDirectory() as tmp:
        bundle = ScienceEnvironmentBundle(
            problem_id="problem-1",
            problem="Test problem",
            profile_id="generic_ml",
            domain="generic_ml",
            profile_hash="hash",
            docker_image_tag="tar-science:test",
            build_context_path=tmp,
            dockerfile_path=str(Path(tmp) / "Dockerfile"),
            requirements_path=str(Path(tmp) / "requirements.txt"),
            study_plan_path=str(Path(tmp) / "study_plan.json"),
            execution_report_path=str(Path(tmp) / "execution_report.json"),
        )
        Path(bundle.dockerfile_path).write_text("FROM python:3.11\n", encoding="utf-8")
        Path(bundle.requirements_path).write_text("numpy\nmissing-package\n", encoding="utf-8")

        locked = PayloadEnvironmentBuilder(tmp).lock_science_bundle(bundle)

        assert locked.reproducibility_complete is False
        assert locked.image_manifest is None
        assert locked.run_manifest is None
        assert locked.unresolved_packages == ["missing-package"]
        assert locked.lock_incomplete_reason is not None
        assert "missing-package" in locked.lock_incomplete_reason


def test_orchestrator_status_and_manifest_surface_incomplete_lock(monkeypatch):
    original_version = reproducibility_module.metadata.version

    def fake_version(name: str) -> str:
        if name == "peft":
            raise metadata.PackageNotFoundError
        return original_version(name)

    monkeypatch.setattr(reproducibility_module.metadata, "version", fake_version)

    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            report = orchestrator.prepare_payload_environment()
            status = orchestrator.status()
            manifest = orchestrator.show_manifest()

            assert report.reproducibility_complete is False
            assert status["reproducibility_complete"] is False
            assert status["unresolved_dependency_count"] == 1
            assert status["unresolved_dependencies"] == ["peft"]
            assert "peft" in (status["lock_incomplete_reason"] or "")
            assert manifest["manifest_found"] is False
            assert manifest["unresolved_packages"] == ["peft"]
        finally:
            orchestrator.shutdown()


def test_science_bundle_locking_injects_manifest_paths_and_network_policy():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            study = orchestrator.study_problem(
                "Investigate calibration in deep learning",
                build_env=False,
                max_results=0,
                benchmark_tier="validation",
            )
            bundle = study.environment
            assert bundle.reproducibility_complete
            assert bundle.image_manifest is not None
            assert bundle.run_manifest is not None
            assert bundle.image_manifest_path is not None
            assert bundle.run_manifest_path is not None
            assert "--network" in bundle.run_command
            assert "none" in bundle.run_command
            assert any(item.startswith("TAR_IMAGE_MANIFEST=/workspace/") for item in bundle.run_command)
            assert any(item.startswith("TAR_RUN_MANIFEST=/workspace/") for item in bundle.run_command)
        finally:
            orchestrator.shutdown()


def test_rebuild_locked_image_persists_build_attestation_and_status():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            def fake_build(report, dry_run=False):
                return BuildResult(
                    mode="subprocess",
                    command=["docker", "build", "-t", report.image_tag],
                    image_tag=report.image_tag,
                    returncode=0,
                    stdout="built",
                    stderr="",
                    image_digest=f"{report.image_tag}@sha256:abc123",
                    image_id="sha256:image123",
                    digest_source="docker_inspect",
                )

            orchestrator.docker_runner.build_payload_environment = fake_build  # type: ignore[assignment]
            report = orchestrator.rebuild_locked_image()
            status = orchestrator.status()
            runtime = orchestrator.runtime_status()

            assert report.build_status == "built"
            assert report.build_attestation is not None
            assert report.build_attestation.image_digest == f"{report.image_tag}@sha256:abc123"
            assert report.build_attestation_path is not None
            assert Path(report.build_attestation_path).exists()
            assert status["build_attestation_id"] == report.build_attestation.attestation_id
            assert status["image_digest"] == report.build_attestation.image_digest
            assert runtime["payload_build_status"] == "built"
            assert runtime["build_attestation_id"] == report.build_attestation.attestation_id
        finally:
            orchestrator.shutdown()


def test_prepare_science_environment_build_attaches_build_attestation():
    with tempfile.TemporaryDirectory() as tmp:
        _copy_science_profiles(tmp)
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            def fake_build(bundle, dry_run=False):
                return BuildResult(
                    mode="subprocess",
                    command=["docker", "build", "-t", bundle.docker_image_tag],
                    image_tag=bundle.docker_image_tag,
                    returncode=0,
                    stdout="built",
                    stderr="",
                    image_digest=f"{bundle.docker_image_tag}@sha256:def456",
                    image_id="sha256:science456",
                    digest_source="docker_inspect",
                )

            orchestrator.docker_runner.build_science_environment = fake_build  # type: ignore[assignment]
            bundle = orchestrator.prepare_science_environment(
                "Investigate calibration in deep learning",
                build=True,
                benchmark_tier="validation",
            )

            assert bundle.build_status == "built"
            assert bundle.build_attestation is not None
            assert bundle.build_attestation.scope_kind == "science_bundle"
            assert bundle.build_attestation.image_digest == f"{bundle.docker_image_tag}@sha256:def456"
            assert bundle.build_attestation_path is not None
            assert Path(bundle.build_attestation_path).exists()
        finally:
            orchestrator.shutdown()


def test_training_payload_split_validator_handles_missing_info_data():
    assert TrainingPayloadConfig.validate_split_sum(0.15, SimpleNamespace(data=None)) == 0.15


def test_latest_build_attestation_prefers_newest_built_at():
    with tempfile.TemporaryDirectory() as tmp:
        store = TARStateStore(tmp)
        older = BuildAttestation(
            attestation_id="build-zolder",
            scope_kind="payload_environment",
            image_tag="tar-payload:locked",
            build_command=["docker", "build"],
            builder_backend="subprocess",
            build_status="built",
            image_manifest_hash="hash-older",
            dependency_lock_hash="lock",
            environment_fingerprint_id="env-older",
            built_at="2026-04-13T10:00:00+00:00",
        )
        newer = BuildAttestation(
            attestation_id="build-anewer",
            scope_kind="payload_environment",
            image_tag="tar-payload:locked",
            build_command=["docker", "build"],
            builder_backend="subprocess",
            build_status="built",
            image_manifest_hash="hash-newer",
            dependency_lock_hash="lock",
            environment_fingerprint_id="env-newer",
            built_at="2026-04-15T10:00:00+00:00",
        )

        store.save_build_attestation(older)
        store.save_build_attestation(newer)

        latest = store.latest_build_attestation(scope_kind="payload_environment")

        assert latest is not None
        assert latest.attestation_id == "build-anewer"
