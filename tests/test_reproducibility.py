import tempfile
from pathlib import Path

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.reproducibility import PayloadEnvironmentBuilder


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
        assert report.image_manifest is not None
        assert report.run_manifest is not None
        assert Path(report.manifest_path).exists()
        manifest_path = Path(tmp) / "tar_state" / "manifests" / f"{report.run_manifest.manifest_id}.json"
        assert manifest_path.exists()


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
