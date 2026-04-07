import tempfile
from importlib import metadata
from pathlib import Path

from tar_lab import reproducibility as reproducibility_module
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.reproducibility import PayloadEnvironmentBuilder
from tar_lab.schemas import ScienceEnvironmentBundle


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
