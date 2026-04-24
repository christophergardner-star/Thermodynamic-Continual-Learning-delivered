from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status

from tar_lab.control import DEFAULT_HOST, DEFAULT_PORT, send_command


def _auth_dependency(expected_api_key: Optional[str]):
    def _authorize(x_api_key: Optional[str] = Header(default=None)) -> None:
        if expected_api_key is None:
            return
        if x_api_key != expected_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key.",
            )

    return _authorize


def create_app(
    *,
    control_host: Optional[str] = None,
    control_port: Optional[int] = None,
    api_key: Optional[str] = None,
) -> FastAPI:
    resolved_control_host = control_host or os.getenv("TAR_CONTROL_HOST", DEFAULT_HOST)
    resolved_control_port = int(control_port or os.getenv("TAR_CONTROL_PORT", DEFAULT_PORT))
    resolved_api_key = api_key if api_key is not None else os.getenv("TAR_API_KEY")
    authorize = _auth_dependency(resolved_api_key)
    auth = [Depends(authorize)]

    app = FastAPI(
        title="TAR HTTP API",
        version="0.1.0",
        description="Thin HTTP adapter over TAR's existing control server.",
    )

    def call(command: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        response = send_command(
            command,
            payload or {},
            host=resolved_control_host,
            port=resolved_control_port,
        )
        if not response.ok:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=response.error or f"TAR command failed: {command}",
            )
        return response.payload

    @app.get("/", dependencies=auth)
    def root() -> dict[str, Any]:
        return {
            "service": "tar-http-api",
            "control_host": resolved_control_host,
            "control_port": resolved_control_port,
            "auth_required": resolved_api_key is not None,
            "routes": [
                "/health",
                "/status",
                "/runtime",
                "/projects",
                "/queue-health",
                "/frontier/status",
                "/positioning/reports",
                "/comparison/{project_id}",
                "/publication-handoff/{project_id}",
            ],
        }

    @app.get("/health", dependencies=auth)
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "status": call("status"),
        }

    @app.get("/status", dependencies=auth)
    def get_status() -> dict[str, Any]:
        return call("status")

    @app.get("/runtime", dependencies=auth)
    def runtime() -> dict[str, Any]:
        return call("runtime_status")

    @app.get("/projects", dependencies=auth)
    def projects() -> dict[str, Any]:
        return call("list_projects")

    @app.get("/queue-health", dependencies=auth)
    def queue_health() -> dict[str, Any]:
        return call("queue_health")

    @app.get("/frontier/status", dependencies=auth)
    def frontier_status() -> dict[str, Any]:
        return call("frontier_status")

    @app.get("/positioning/reports", dependencies=auth)
    def positioning_reports() -> dict[str, Any]:
        return call("get_positioning_reports")

    @app.get("/comparison/{project_id}", dependencies=auth)
    def comparison_result(project_id: str) -> dict[str, Any]:
        result = call("get_comparison_result", {"project_id": project_id})
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No comparison result found for project_id={project_id}.",
            )
        return result

    @app.get("/publication-handoff/{project_id}", dependencies=auth)
    def publication_handoff(project_id: str) -> dict[str, Any]:
        return call("publication_handoff", {"project_id": project_id})

    return app


app = create_app()
