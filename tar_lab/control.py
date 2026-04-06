from __future__ import annotations

import socket
import socketserver
from typing import Any, Dict, Optional

from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import ControlRequest, ControlResponse


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def handle_request(orchestrator: TAROrchestrator, request: ControlRequest) -> ControlResponse:
    try:
        if request.command == "status":
            payload = orchestrator.status()
        elif request.command == "dry_run":
            payload = orchestrator.run_dry_run(
                force_fail_fast=bool(request.payload.get("force_fail_fast", False))
            ).model_dump(mode="json")
        elif request.command == "pivot":
            payload = orchestrator.pivot_force(force=bool(request.payload.get("force", False)))
        elif request.command == "explain_last_fail":
            payload = orchestrator.explain_last_fail().model_dump(mode="json")
        elif request.command == "panic":
            payload = orchestrator.panic()
        elif request.command == "live_docker_test":
            payload = orchestrator.live_docker_test().model_dump(mode="json")
        elif request.command == "chat":
            payload = orchestrator.chat(str(request.payload.get("prompt", ""))).model_dump(mode="json")
        elif request.command == "check_regime":
            payload = orchestrator.check_regime()
        elif request.command == "ingest_research":
            payload = orchestrator.ingest_research(
                topic=str(request.payload.get("topic", "frontier ai")),
                max_results=int(request.payload.get("max_results", 6)),
            ).model_dump(mode="json")
        elif request.command == "verify_last_trial":
            payload = orchestrator.verify_last_trial(
                trial_id=request.payload.get("trial_id"),
            ).model_dump(mode="json")
        elif request.command == "breakthrough_report":
            payload = orchestrator.breakthrough_report(
                trial_id=request.payload.get("trial_id"),
            ).model_dump(mode="json")
        else:  # pragma: no cover
            raise ValueError(f"Unsupported command: {request.command}")
        return ControlResponse(ok=True, command=request.command, payload=payload)
    except Exception as exc:  # pragma: no cover - defensive path
        return ControlResponse(ok=False, command=request.command, error=str(exc))


class _ControlHandler(socketserver.StreamRequestHandler):
    orchestrator: TAROrchestrator

    def handle(self) -> None:
        raw = self.rfile.readline().decode("utf-8").strip()
        request = ControlRequest.model_validate_json(raw)
        response = handle_request(self.orchestrator, request)
        self.wfile.write((response.model_dump_json() + "\n").encode("utf-8"))


class TARControlServer:
    def __init__(self, orchestrator: TAROrchestrator, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.orchestrator = orchestrator
        handler = type("BoundControlHandler", (_ControlHandler,), {"orchestrator": orchestrator})
        self.server = socketserver.ThreadingTCPServer((host, port), handler)

    def serve_forever(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def send_command(
    command: str,
    payload: Optional[Dict[str, Any]] = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> ControlResponse:
    request = ControlRequest(command=command, payload=payload or {})
    with socket.create_connection((host, port), timeout=5.0) as sock:
        sock.sendall((request.model_dump_json() + "\n").encode("utf-8"))
        data = sock.recv(65536).decode("utf-8").strip()
    return ControlResponse.model_validate_json(data)
