import tempfile

from tar_lab.safe_exec import SandboxedPythonExecutor


def test_sandbox_executor_refuses_profile_required_without_profile():
    with tempfile.TemporaryDirectory() as tmp:
        executor = SandboxedPythonExecutor(workspace=tmp)
        report = executor.execute("print('hello')", network_policy="profile_required")
        assert not report.ok
        assert report.error == "Sandbox execution requires an explicit network profile."
        assert report.sandbox_policy.network_policy == "profile_required"
