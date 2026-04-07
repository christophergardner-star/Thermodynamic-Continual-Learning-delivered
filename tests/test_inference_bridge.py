import tempfile
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

from tar_lab.inference_bridge import InferenceBridge
from tar_lab.schemas import EndpointHealth, EndpointRecord


def test_inference_bridge_registers_and_assigns_role():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"
        model_dir.mkdir()
        bridge = InferenceBridge(tmp)
        record = bridge.register_checkpoint(
            name="asc-local",
            model_path=str(model_dir),
            backend="transformers",
            role="assistant",
        )
        plan = bridge.build_endpoint("asc-local", role="director", port=8101)
        assert plan.endpoint_name == "director-asc-local"
        assert Path(plan.command[1]).exists()
        assignment = bridge.assign_role(role="director", checkpoint_name="asc-local")
        assert assignment.role == "director"
        assert assignment.checkpoint_name == record.name


def test_inference_bridge_start_and_health_refresh(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "mock-model"
        model_dir.mkdir()
        (model_dir / "mock_endpoint.json").write_text("{}", encoding="utf-8")
        bridge = InferenceBridge(tmp)
        bridge.register_checkpoint(name="asc-local", model_path=str(model_dir), backend="transformers", role="assistant")

        monkeypatch.setattr(
            "tar_lab.inference_bridge.subprocess.Popen",
            lambda *args, **kwargs: SimpleNamespace(pid=4321, poll=lambda: None, terminate=lambda: None, wait=lambda timeout=5: 0),
        )
        monkeypatch.setattr(
            bridge,
            "endpoint_health",
            lambda endpoint_name: EndpointHealth(endpoint_name=endpoint_name, status="healthy", ok=True, backend="mock", model_id="asc-local"),
        )
        monkeypatch.setattr(bridge, "_pid_alive", lambda pid: True)
        record = bridge.start_endpoint("asc-local", role="assistant")
        assert record.process_pid == 4321
        assert record.stdout_log_path is not None and Path(record.stdout_log_path).exists()
        assert record.stderr_log_path is not None and Path(record.stderr_log_path).exists()
        assert record.manifest_path is not None and Path(record.manifest_path).exists()
        stored = bridge.get_endpoint(record.endpoint_name)
        assert stored.status == "running"
        assert stored.health is not None and stored.health.ok
        assert stored.last_health_at is not None


def test_inference_bridge_start_failure_persists_logs_and_error(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "mock-model"
        model_dir.mkdir()
        (model_dir / "mock_endpoint.json").write_text("{}", encoding="utf-8")
        bridge = InferenceBridge(tmp)
        bridge.register_checkpoint(
            name="asc-local",
            model_path=str(model_dir),
            backend="transformers",
            role="assistant",
            metadata={"trust_remote_code": True},
        )

        def fail_popen(*args, **kwargs):
            raise OSError("launch exploded")

        monkeypatch.setattr("tar_lab.inference_bridge.subprocess.Popen", fail_popen)
        record = bridge.start_endpoint("asc-local", role="assistant")
        assert record.status == "failed"
        assert record.trust_remote_code is True
        assert record.last_error is not None and "launch exploded" in record.last_error
        assert record.stderr_log_path is not None
        stderr_path = Path(record.stderr_log_path)
        assert stderr_path.exists()
        assert "launch exploded" in stderr_path.read_text(encoding="utf-8")


def test_inference_bridge_wait_for_health_timeout_records_explicit_failure(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "mock-model"
        model_dir.mkdir()
        (model_dir / "mock_endpoint.json").write_text("{}", encoding="utf-8")
        bridge = InferenceBridge(tmp)
        bridge.register_checkpoint(name="asc-local", model_path=str(model_dir), backend="transformers", role="assistant")

        process = SimpleNamespace(pid=4321, poll=lambda: None, terminate=lambda: None, wait=lambda timeout=5: 0)
        monkeypatch.setattr("tar_lab.inference_bridge.subprocess.Popen", lambda *args, **kwargs: process)

        clock = {"now": 1000.0}
        monkeypatch.setattr("tar_lab.inference_bridge.time.time", lambda: clock["now"])
        monkeypatch.setattr(
            "tar_lab.inference_bridge.time.sleep",
            lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
        )
        monkeypatch.setattr(
            bridge,
            "endpoint_health",
            lambda endpoint_name: EndpointHealth(
                endpoint_name=endpoint_name,
                status="unhealthy",
                ok=False,
                detail="health probe still failing",
                backend="mock",
                model_id="asc-local",
            ),
        )
        monkeypatch.setattr(bridge, "_pid_alive", lambda pid: True)

        record = bridge.start_endpoint(
            "asc-local",
            role="assistant",
            wait_for_health=True,
            startup_timeout_s=0.5,
        )
        assert record.status == "failed"
        assert record.process_pid is None
        assert record.last_error is not None
        assert "did not become healthy within 0.5s" in record.last_error
        assert "health probe still failing" in record.last_error
        assert record.stderr_log_path is not None and Path(record.stderr_log_path).exists()


def test_endpoint_health_failure_uses_stderr_tail_context(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"
        model_dir.mkdir()
        bridge = InferenceBridge(tmp)
        bridge.register_checkpoint(name="asc-local", model_path=str(model_dir), backend="transformers", role="assistant")
        plan = bridge.build_endpoint("asc-local", role="assistant", port=8105)

        stdout_path = bridge.store.endpoint_stdout_log_path(plan.endpoint_name)
        stderr_path = bridge.store.endpoint_stderr_log_path(plan.endpoint_name)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("RuntimeError: missing weights\ncheckpoint corrupted\n", encoding="utf-8")
        bridge.store.upsert_endpoint(
            EndpointRecord(
                endpoint_name=plan.endpoint_name,
                checkpoint_name="asc-local",
                role="assistant",
                host="127.0.0.1",
                port=8105,
                backend="transformers",
                base_url="http://127.0.0.1:8105/v1",
                command=plan.command,
                env=plan.env,
                status="running",
                process_pid=9999,
                started_at="2026-04-07T18:00:00Z",
                stdout_log_path=str(stdout_path),
                stderr_log_path=str(stderr_path),
                trust_remote_code=False,
            )
        )

        def fail_open(*args, **kwargs):
            raise URLError("connection refused")

        monkeypatch.setattr("tar_lab.inference_bridge.urlopen", fail_open)
        monkeypatch.setattr(bridge, "_pid_alive", lambda pid: False)

        health = bridge.endpoint_health(plan.endpoint_name)
        assert health.status == "failed"
        assert health.detail is not None
        assert "stderr tail:" in health.detail
        assert "missing weights" in health.detail


def test_inference_bridge_uses_repo_serve_local_path_from_non_repo_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "custom-workspace"
        workspace.mkdir()
        model_dir = workspace / "model"
        model_dir.mkdir()
        bridge = InferenceBridge(str(workspace))
        bridge.register_checkpoint(name="asc-local", model_path=str(model_dir), backend="transformers", role="assistant")

        plan = bridge.build_endpoint("asc-local", role="assistant", port=8120)

        assert Path(plan.command[1]) == Path(__file__).resolve().parents[1] / "serve_local.py"
