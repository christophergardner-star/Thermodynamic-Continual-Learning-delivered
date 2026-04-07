import tempfile

from tar_lab.docker_runner import DockerRunner
from tar_lab.reproducibility import PayloadEnvironmentBuilder


def test_build_payload_environment_uses_locked_manifest_command():
    with tempfile.TemporaryDirectory() as tmp:
        builder = PayloadEnvironmentBuilder(tmp)
        report = builder.prepare()
        runner = DockerRunner(docker_bin="docker-custom")
        build = runner.build_payload_environment(report, dry_run=True)
        assert build.command[0] == "docker-custom"
        assert build.command[1:3] == ["build", "-t"]
