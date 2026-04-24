from __future__ import annotations

from fastapi.testclient import TestClient

import tar_api
from tar_lab.schemas import ControlResponse


def test_root_exposes_routes_without_auth():
    app = tar_api.create_app(control_host="127.0.0.1", control_port=8765, api_key=None)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "tar-http-api"
    assert payload["auth_required"] is False
    assert "/comparison/{project_id}" in payload["routes"]


def test_runtime_route_proxies_control_command(monkeypatch):
    seen: dict[str, object] = {}

    def fake_send_command(command, payload, host, port):
        seen.update(
            {
                "command": command,
                "payload": payload,
                "host": host,
                "port": port,
            }
        )
        return ControlResponse(
            ok=True,
            command=command,
            payload={"heartbeat": {"status": "ok"}},
        )

    monkeypatch.setattr(tar_api, "send_command", fake_send_command)
    app = tar_api.create_app(control_host="127.0.0.1", control_port=9001, api_key=None)
    client = TestClient(app)

    response = client.get("/runtime")

    assert response.status_code == 200
    assert response.json()["heartbeat"]["status"] == "ok"
    assert seen == {
        "command": "runtime_status",
        "payload": {},
        "host": "127.0.0.1",
        "port": 9001,
    }


def test_comparison_route_passes_project_id(monkeypatch):
    seen: dict[str, object] = {}

    def fake_send_command(command, payload, host, port):
        seen.update({"command": command, "payload": payload})
        return ControlResponse(
            ok=True,
            command=command,
            payload={"project_id": payload["project_id"], "honest_assessment": "locked"},
        )

    monkeypatch.setattr(tar_api, "send_command", fake_send_command)
    app = tar_api.create_app(api_key=None)
    client = TestClient(app)

    response = client.get("/comparison/proj-123")

    assert response.status_code == 200
    assert response.json()["project_id"] == "proj-123"
    assert seen == {
        "command": "get_comparison_result",
        "payload": {"project_id": "proj-123"},
    }


def test_http_error_when_control_command_fails(monkeypatch):
    def fake_send_command(command, payload, host, port):
        return ControlResponse(ok=False, command=command, error="control exploded")

    monkeypatch.setattr(tar_api, "send_command", fake_send_command)
    app = tar_api.create_app(api_key=None)
    client = TestClient(app)

    response = client.get("/projects")

    assert response.status_code == 502
    assert response.json()["detail"] == "control exploded"


def test_comparison_route_returns_404_for_missing_result(monkeypatch):
    def fake_send_command(command, payload, host, port):
        return ControlResponse(ok=True, command=command, payload={})

    monkeypatch.setattr(tar_api, "send_command", fake_send_command)
    app = tar_api.create_app(api_key=None)
    client = TestClient(app)

    response = client.get("/comparison/proj-missing")

    assert response.status_code == 404
    assert "proj-missing" in response.json()["detail"]


def test_api_key_is_enforced_when_configured(monkeypatch):
    called = {"count": 0}

    def fake_send_command(command, payload, host, port):
        called["count"] += 1
        return ControlResponse(ok=True, command=command, payload={"ok": True})

    monkeypatch.setattr(tar_api, "send_command", fake_send_command)
    app = tar_api.create_app(api_key="secret")
    client = TestClient(app)

    unauthorized = client.get("/status")
    authorized = client.get("/status", headers={"x-api-key": "secret"})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert called["count"] == 1
