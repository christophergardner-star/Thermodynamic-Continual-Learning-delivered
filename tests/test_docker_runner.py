import tempfile
from importlib import metadata
from types import SimpleNamespace

import pytest

from tar_lab import reproducibility as reproducibility_module
from tar_lab.docker_runner import DockerRunner
from tar_lab.reproducibility import PayloadEnvironmentBuilder
from tar_lab.schemas import (
    DependencyLockManifest,
    EnvironmentFingerprint,
    ImageManifest,
    RunManifest,
    RuntimeSpec,
    SandboxPolicy,
    ScienceEnvironmentBundle,
)


def _fake_image_manifest() -> ImageManifest:
    return ImageManifest(
        image_tag="tar-payload:locked",
        base_image="pytorch/pytorch:latest",
        dockerfile_path="/workspace/Dockerfile",
        build_context_path="/workspace",
        build_command=["docker", "build", "-t", "tar-payload:locked", "-f", "/workspace/Dockerfile", "/workspace"],
        dependency_lock=DependencyLockManifest(
            lock_id="lock-1234",
            requirements_path="/workspace/requirements.txt",
            packages=["torch==2.0.0"],
            fully_pinned=True,
            hash_sha256="abc123",
        ),
        environment_fingerprint=EnvironmentFingerprint(
            fingerprint_id="env-1234",
            workspace_root="/workspace",
            source_hash_sha256="source123",
            dockerfile_hash_sha256="docker123",
            requirements_hash_sha256="req123",
            python_version="3.11.0",
        ),
        hash_sha256="imagehash123",
        locked=True,
    )


def _fake_run_manifest(policy: SandboxPolicy) -> RunManifest:
    image_manifest = _fake_image_manifest()
    return RunManifest(
        manifest_id="run-1234",
        kind="science_bundle",
        command=["docker", "run", "--rm", "tar-payload:locked"],
        image_manifest=image_manifest,
        sandbox_policy=policy,
        hash_sha256="runhash123",
        reproducibility_complete=True,
    )


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


def test_pull_image_short_circuits_when_image_exists_locally(monkeypatch):
    runner = DockerRunner(docker_bin="docker-custom")
    monkeypatch.setattr(runner, "_image_exists_locally", lambda image: True)
    monkeypatch.setattr(
        "tar_lab.docker_runner.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pull should not be attempted")),
    )

    assert runner.pull_image("tar-payload:locked") == 0


def test_build_payload_environment_reuses_existing_local_image(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        builder = PayloadEnvironmentBuilder(tmp)
        report = builder.prepare().model_copy(update={"build_status": "built"})
        runner = DockerRunner(docker_bin="docker-custom")
        monkeypatch.setattr(runner, "_image_exists_locally", lambda image: True)
        monkeypatch.setattr(
            runner,
            "_inspect_image_metadata",
            lambda image_tag: (f"{image_tag}@sha256:cached", "sha256:cached", "docker_inspect"),
        )
        monkeypatch.setattr(
            "tar_lab.docker_runner.subprocess.run",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("docker build should not run")),
        )

        build = runner.build_payload_environment(report, dry_run=False)

        assert build.returncode == 0
        assert build.stdout == "reused_local_image"
        assert build.image_digest == f"{report.image_tag}@sha256:cached"


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
    assert "--read-only" in command
    assert "--tmpfs" in command
    assert "/tmp" in command
    assert "--cap-drop" in command
    assert "ALL" in command
    assert "no-new-privileges" in command
    assert any(str(item).startswith("seccomp=") for item in command)
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


def test_run_science_environment_dry_run_applies_sandbox_policy():
    runner = DockerRunner(docker_bin="docker")
    policy = SandboxPolicy(
        profile="production",
        network_policy="off",
        allowed_mounts=["/workspace", "/workspace/tar_runs"],
        read_only_mounts=["/workspace"],
        writable_mounts=["/workspace/tar_runs"],
        seccomp_profile_path="/workspace/seccomp.json",
        capability_drop=["ALL"],
        workspace_read_only=True,
        artifact_dir="/workspace/tar_runs",
        workspace_root="/workspace",
    )
    bundle = ScienceEnvironmentBundle(
        problem_id="problem-1",
        problem="test problem",
        profile_id="generic_ml",
        domain="generic_ml",
        profile_hash="profilehash123",
        docker_image_tag="tar-payload:locked",
        build_context_path="/workspace",
        dockerfile_path="/workspace/Dockerfile",
        requirements_path="/workspace/requirements.txt",
        study_plan_path="/workspace/tar_runs/study_plan.json",
        execution_report_path="/workspace/tar_runs/report.json",
        run_command=[
            "docker",
            "run",
            "--rm",
            "-v",
            "/host/workspace:/workspace",
            "-v",
            "/host/tar_runs:/workspace/tar_runs",
            "-w",
            "/workspace",
            "tar-payload:locked",
            "python",
            "-m",
            "tar_lab.problem_runner",
        ],
        image_manifest=_fake_image_manifest(),
        run_manifest=_fake_run_manifest(policy),
        reproducibility_complete=True,
        sandbox_policy=policy,
    )

    result = runner.run_science_environment(bundle, dry_run=True)

    assert "--read-only" in result.command
    assert "--tmpfs" in result.command
    assert "--cap-drop" in result.command
    assert "ALL" in result.command
    assert "no-new-privileges" in result.command
    assert "seccomp=/workspace/seccomp.json" in result.command
    assert "/host/workspace:/workspace:ro" in result.command
    assert any(item.startswith("workspace_mount=read-only:/workspace") for item in result.sandbox_audit_log)
    assert any(item == "security_opt=no-new-privileges" for item in result.sandbox_audit_log)
