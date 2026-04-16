import sys
import tempfile

import pytest

from tar_lab.errors import ExecutionPolicyViolation
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.safe_exec import SandboxedPythonExecutor
from tar_lab.schemas import TARExecutionPolicy
from tar_lab.state import TARStateStore


def test_sandbox_executor_refuses_profile_required_without_profile():
    with tempfile.TemporaryDirectory() as tmp:
        executor = SandboxedPythonExecutor(workspace=tmp)
        report = executor.execute("print('hello')", network_policy="profile_required")
        assert not report.ok
        assert report.error == "Sandbox execution requires an explicit network profile."
        assert report.sandbox_policy.profile == "production"
        assert report.sandbox_policy.network_policy == "profile_required"
        assert report.sandbox_policy.workspace_root == "/workspace"
        assert report.sandbox_policy.workspace_read_only is True
        assert report.sandbox_policy.capability_drop == ["ALL"]
        assert report.sandbox_policy.seccomp_profile_path is not None
        assert any(item.startswith("seccomp=") for item in report.sandbox_audit_log)


def test_sandbox_executor_reports_explicit_writable_mount_scope():
    with tempfile.TemporaryDirectory() as tmp:
        executor = SandboxedPythonExecutor(workspace=tmp, docker_bin="definitely-not-docker")
        report = executor.execute("print('hello')")
        assert not report.ok
        assert report.sandbox_policy.profile == "production"
        assert report.sandbox_policy.workspace_root == "/workspace"
        assert report.sandbox_policy.read_only_mounts == ["/workspace"]
        assert report.sandbox_policy.writable_mounts == ["/sandbox"]
        assert report.sandbox_policy.allowed_mounts == ["/sandbox", "/workspace"]
        assert "--read-only" in report.command
        assert "--cap-drop" in report.command
        assert "ALL" in report.command
        assert "no-new-privileges" in report.command
        assert any(str(item).startswith("seccomp=") for item in report.command)
        assert any(item.endswith(":/workspace:ro") for item in report.command)
        assert any("artifact_mount=read-write:/sandbox" == item for item in report.sandbox_audit_log)
        assert any("security_opt=no-new-privileges" == item for item in report.sandbox_audit_log)


def test_orchestrator_blocks_unsandboxed_generated_subprocess():
    with tempfile.TemporaryDirectory() as tmp:
        TARStateStore(tmp).save_execution_policy(TARExecutionPolicy())
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            with pytest.raises(ExecutionPolicyViolation, match="generated code"):
                orchestrator._run_subprocess(
                    [sys.executable, "-c", "print('unsafe')"],
                    source_path="tests.generated_code",
                    execution_kind="generated_code",
                    capture_output=True,
                    text=True,
                    check=False,
                )
        finally:
            orchestrator.shutdown()


def test_orchestrator_allows_documented_trusted_internal_subprocess():
    with tempfile.TemporaryDirectory() as tmp:
        TARStateStore(tmp).save_execution_policy(TARExecutionPolicy())
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            proc = orchestrator._run_subprocess(
                [sys.executable, "-c", "print('ok')"],
                source_path="tar_lab.problem_runner",
                execution_kind="trusted_internal",
                deliberate_exception_reason="trusted internal TAR study runner",
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 0
            assert (proc.stdout or "").strip() == "ok"
        finally:
            orchestrator.shutdown()


def test_runtime_status_surfaces_loaded_execution_policy():
    with tempfile.TemporaryDirectory() as tmp:
        store = TARStateStore(tmp)
        store.save_execution_policy(
            TARExecutionPolicy(allowed_unsandboxed_paths=["tar_lab.problem_runner"])
        )
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            runtime_status = orchestrator.runtime_status()
            assert runtime_status["execution_policy"]["policy_version"] == "ws35.v1"
            assert runtime_status["execution_policy"]["allowed_unsandboxed_paths"] == [
                "tar_lab.problem_runner"
            ]
        finally:
            orchestrator.shutdown()
