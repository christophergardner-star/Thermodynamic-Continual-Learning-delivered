import tempfile

from tar_lab.safe_exec import SandboxedPythonExecutor


def test_sandbox_executor_refuses_profile_required_without_profile():
    with tempfile.TemporaryDirectory() as tmp:
        executor = SandboxedPythonExecutor(workspace=tmp)
        report = executor.execute("print('hello')", network_policy="profile_required")
        assert not report.ok
        assert report.error == "Sandbox execution requires an explicit network profile."
        assert report.sandbox_policy.profile == "production"
        assert report.sandbox_policy.network_policy == "profile_required"
        assert report.sandbox_policy.workspace_root == "/sandbox"


def test_sandbox_executor_reports_explicit_writable_mount_scope():
    with tempfile.TemporaryDirectory() as tmp:
        executor = SandboxedPythonExecutor(workspace=tmp, docker_bin="definitely-not-docker")
        report = executor.execute("print('hello')")
        assert not report.ok
        assert report.sandbox_policy.profile == "production"
        assert report.sandbox_policy.workspace_root == "/sandbox"
        assert report.sandbox_policy.writable_mounts == ["/sandbox"]
        assert report.sandbox_policy.allowed_mounts == ["/sandbox"]
        assert "--read-only" in report.command
