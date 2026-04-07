import tempfile
from pathlib import Path
from types import SimpleNamespace

from tar_lab.inference_bridge import InferenceBridge
from tar_lab.schemas import EndpointHealth


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
        stored = bridge.get_endpoint(record.endpoint_name)
        assert stored.status == "running"
        assert stored.health is not None and stored.health.ok
