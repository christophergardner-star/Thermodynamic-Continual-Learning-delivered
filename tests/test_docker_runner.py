import tempfile
from importlib import metadata
from types import SimpleNamespace

import pytest

from tar_lab import reproducibility as reproducibility_module
from tar_lab.docker_runner import DockerRunner
from tar_lab.reproducibility import PayloadEnvironmentBuilder
from tar_lab.schemas import RuntimeSpec


def test_build_payload_environment_uses_locked_manifest_command():
    with tempfile.TemporaryDirectory() as tmp:
        builder = PayloadEnvironmentBuilder(tmp)
        report = builder.prepare()
        runner = DockerRunner(docker_bin="docker-custom")
        build = runner.build_payload_environment(report, dry_run=True)
        assert build.command[0] == "docker-custom"
        assert build.command[1:3] == ["build", "-t"]


def test_build_payload_environment_refuses_incomplete_lock(monkeypatch):
    original_version = reproducibility_module.metadata.version

    def fake_version(name: str) -> str:
        if name == "peft":
            raise metadata.PackageNotFoundError
        return original_version(name)

    monkeypatch.setattr(reproducibility_module.metadata, "version", fake_version)

    with tempfile.TemporaryDirectory() as tmp:
        builder = PayloadEnvironmentBuilder(tmp)
        report = builder.prepare()
        runner = DockerRunner(docker_bin="docker-custom")
        build = runner.build_payload_environment(report, dry_run=True)

        assert report.reproducibility_complete is False
        assert build.returncode == 1
        assert build.stderr is not None
        assert "peft" in build.stderr


def test_build_payload_environment_records_image_metadata(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        builder = PayloadEnvironmentBuilder(tmp)
        report = builder.prepare()
        runner = DockerRunner(docker_bin="docker-custom")

        monkeypatch.setattr(
            "tar_lab.docker_runner.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="built", stderr=""),
        )
        monkeypatch.setattr(
            runner,
            "_inspect_image_metadata",
            lambda image_tag: (f"{image_tag}@sha256:abc123", "sha256:image123", "docker_inspect"),
        )

        build = runner.build_payload_environment(report, dry_run=False)

        assert build.returncode == 0
        assert build.image_digest == f"{report.image_tag}@sha256:abc123"
        assert build.image_id == "sha256:image123"
        assert build.digest_source == "docker_inspect"


def test_compose_runtime_command_refuses_workspace_write_in_production():
    runner = DockerRunner(docker_bin="docker")
    runtime = RuntimeSpec(
        image="tar-payload:locked",
        image_locked=True,
        image_manifest_path="/workspace/tar_state/manifests/image.json",
        run_manifest_path="/workspace/tar_state/manifests/run.json",
        volumes={"C:/host/workspace": "/workspace"},
    )
    with pytest.raises(RuntimeError, match="cannot mount /workspace read-write"):
        runner.compose_runtime_command(
            runtime=runtime,
            container_command=["python", "-m", "tar_lab.train_template", "--dry_run"],
            container_name="tar-test",
        )


def test_compose_runtime_command_uses_explicit_artifact_mounts():
    runner = DockerRunner(docker_bin="docker")
    runtime = RuntimeSpec(
        image="tar-payload:locked",
        image_locked=True,
        image_manifest_path="/workspace/tar_state/manifests/image.json",
        run_manifest_path="/workspace/tar_state/manifests/run.json",
        volumes={
            "C:/host/logs": "/workspace/logs",
            "C:/host/anchors": "/workspace/anchors",
            "C:/host/tar_runs": "/workspace/tar_runs",
        },
        read_only_volumes={
            "C:/host/workspace": "/workspace",
            "C:/host/data": "/data",
        },
    )
    command = runner.compose_runtime_command(
        runtime=runtime,
        container_command=["python", "-m", "tar_lab.train_template", "--dry_run"],
        container_name="tar-test",
    )
    assert "C:/host/workspace:/workspace:ro" in command
    assert "C:/host/data:/data:ro" in command
    assert "C:/host/logs:/workspace/logs" in command
    assert "C:/host/tar_runs:/workspace/tar_runs" in command


def test_compose_runtime_command_refuses_non_artifact_write_mount_in_production():
    runner = DockerRunner(docker_bin="docker")
    runtime = RuntimeSpec(
        image="tar-payload:locked",
        image_locked=True,
        image_manifest_path="/workspace/tar_state/manifests/image.json",
        run_manifest_path="/workspace/tar_state/manifests/run.json",
        volumes={"C:/host/cache": "/workspace/cache"},
        read_only_volumes={
            "C:/host/workspace": "/workspace",
            "C:/host/data": "/data",
        },
    )
    with pytest.raises(RuntimeError, match="explicit artifact paths"):
        runner.compose_runtime_command(
            runtime=runtime,
            container_command=["python", "-m", "tar_lab.train_template", "--dry_run"],
            container_name="tar-test",
        )
