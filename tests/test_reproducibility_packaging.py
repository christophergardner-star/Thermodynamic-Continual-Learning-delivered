from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import EnvironmentManifest, ReproducibilityPackage, SealedDatasetManifest


def test_environment_manifest_schema_valid():
    manifest = EnvironmentManifest(
        manifest_id="env-1",
        captured_at="2026-04-18T10:00:00",
        python_version="3.11.0",
        torch_version="2.0.0",
        platform="win32",
        package_hashes={"torch": "2.0.0"},
        dataset_checksums={},
        repo_commit="abc123",
        repo_dirty=False,
    )
    assert isinstance(manifest.repo_dirty, bool)
    with pytest.raises(ValidationError):
        EnvironmentManifest(
            manifest_id="env-1",
            captured_at="2026-04-18T10:00:00",
            python_version="3.11.0",
            torch_version="2.0.0",
            platform="win32",
            package_hashes={"torch": "2.0.0"},
            dataset_checksums={},
            repo_commit="abc123",
            repo_dirty=False,
            unknown="x",
        )


def test_sealed_dataset_manifest_schema_valid():
    manifest = SealedDatasetManifest(
        dataset_id="ds-1",
        name="split_cifar10",
        archive_sha256="",
        split_config={"n_tasks": 5},
        sealed_at="2026-04-18T10:00:00",
    )
    assert manifest.n_train_total == 0
    with pytest.raises(ValidationError):
        SealedDatasetManifest(
            dataset_id="ds-1",
            name="split_cifar10",
            archive_sha256="",
            split_config={"n_tasks": 5},
            sealed_at="2026-04-18T10:00:00",
            unknown="x",
        )


def test_reproducibility_package_schema_valid():
    package = ReproducibilityPackage(
        package_id="pkg-1",
        project_id="proj-1",
        created_at="2026-04-18T10:00:00",
        environment_manifest_id="env-1",
        dataset_manifest_id="ds-1",
        comparison_result_id="cmp-1",
        rerun_script_path="tar_state/reproducibility/packages/pkg-1/rerun.py",
        reviewer_summary_path="tar_state/reproducibility/packages/pkg-1/reviewer_summary.md",
        artifact_paths=[],
        package_sha256="deadbeef",
    )
    assert package.positioning_report_id == ""
    with pytest.raises(ValidationError):
        ReproducibilityPackage(
            package_id="pkg-1",
            project_id="proj-1",
            created_at="2026-04-18T10:00:00",
            environment_manifest_id="env-1",
            dataset_manifest_id="ds-1",
            comparison_result_id="cmp-1",
            rerun_script_path="tar_state/reproducibility/packages/pkg-1/rerun.py",
            reviewer_summary_path="tar_state/reproducibility/packages/pkg-1/reviewer_summary.md",
            artifact_paths=[],
            package_sha256="deadbeef",
            unknown="x",
        )


def test_capture_environment_manifest(tmp_path):
    orchestrator = TAROrchestrator(str(tmp_path))
    try:
        manifest = orchestrator.capture_environment_manifest()
        assert isinstance(manifest.python_version, str) and manifest.python_version
        assert isinstance(manifest.torch_version, str) and manifest.torch_version
        assert isinstance(manifest.manifest_id, str) and manifest.manifest_id
        manifest_path = tmp_path / "tar_state" / "reproducibility" / f"env_{manifest.manifest_id}.json"
        assert manifest_path.exists()
    finally:
        orchestrator.shutdown()


def test_create_reproducibility_package(tmp_path):
    orchestrator = TAROrchestrator(str(tmp_path))
    try:
        comparisons_dir = tmp_path / "tar_state" / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        result_id = "result-123"
        (comparisons_dir / f"{result_id}.json").write_text(
            json.dumps(
                {
                    "result_id": result_id,
                    "honest_assessment": "TCL does not significantly differ from baselines.",
                    "pairwise_pvalues": {"tcl_vs_ewc": 0.2},
                    "pairwise_effect_sizes": {"tcl_vs_ewc": -0.1},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        package = orchestrator.create_reproducibility_package("proj_test", result_id)
        package_dir = tmp_path / "tar_state" / "reproducibility" / "packages" / package.package_id
        assert package.package_id
        assert package.package_sha256
        assert (package_dir / "rerun.py").exists()
        assert (package_dir / "reviewer_summary.md").exists()
        assert (package_dir / "task_order_manifest.json").exists()
        assert (package_dir / "seed_manifest.json").exists()
    finally:
        orchestrator.shutdown()


def test_rerun_script_contains_project_id(tmp_path):
    orchestrator = TAROrchestrator(str(tmp_path))
    try:
        comparisons_dir = tmp_path / "tar_state" / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        result_id = "result-123"
        (comparisons_dir / f"{result_id}.json").write_text(
            json.dumps({"result_id": result_id, "honest_assessment": "Assessment"}, indent=2),
            encoding="utf-8",
        )
        package = orchestrator.create_reproducibility_package("proj_test", result_id)
        rerun_path = tmp_path / "tar_state" / "reproducibility" / "packages" / package.package_id / "rerun.py"
        assert "proj_test" in rerun_path.read_text(encoding="utf-8")
    finally:
        orchestrator.shutdown()
