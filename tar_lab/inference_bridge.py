from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import urlopen

from tar_lab.schemas import (
    CheckpointRecord,
    CheckpointRegistryState,
    ClaimAcceptancePolicy,
    EndpointHealth,
    EndpointRecord,
    InferenceEndpointPlan,
    RoleAssignment,
)
from tar_lab.state import TARStateStore


VALID_BACKENDS = {"transformers", "vllm"}
VALID_ROLES = {"assistant", "director", "strategist", "scout"}


class InferenceBridge:
    def __init__(self, workspace: str = "."):
        self.store = TARStateStore(workspace)
        self.registry_path = self.store.state_dir / "checkpoint_registry.json"
        self._processes: dict[str, subprocess.Popen[Any]] = {}

    def load_registry(self) -> CheckpointRegistryState:
        if not self.registry_path.exists():
            return CheckpointRegistryState()
        return CheckpointRegistryState.model_validate_json(self.registry_path.read_text(encoding="utf-8"))

    def save_registry(self, registry: CheckpointRegistryState) -> None:
        self.registry_path.write_text(registry.model_dump_json(indent=2), encoding="utf-8")

    def register_checkpoint(
        self,
        *,
        name: str,
        model_path: str,
        backend: str = "transformers",
        role: str = "assistant",
        metadata: Optional[dict[str, Any]] = None,
    ) -> CheckpointRecord:
        backend_name = str(backend).strip().lower()
        role_name = str(role).strip().lower()
        if backend_name not in VALID_BACKENDS:
            raise ValueError(f"Unsupported inference backend: {backend}")
        if role_name not in VALID_ROLES:
            raise ValueError(f"Unsupported inference role: {role}")
        record = CheckpointRecord(
            name=name,
            model_path=str(Path(model_path).expanduser().resolve()),
            backend=backend_name,  # type: ignore[arg-type]
            role=role_name,  # type: ignore[arg-type]
            metadata=metadata or {},
        )
        registry = self.load_registry()
        entries = [item for item in registry.entries if item.name != name]
        entries.append(record)
        self.save_registry(CheckpointRegistryState(entries=entries))
        return record

    def list_checkpoints(self) -> list[CheckpointRecord]:
        return self.load_registry().entries

    def get(self, name: str) -> CheckpointRecord:
        for item in self.load_registry().entries:
            if item.name == name:
                return item
        raise KeyError(f"Unknown checkpoint: {name}")

    def build_endpoint(
        self,
        name: str,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        role: Optional[str] = None,
    ) -> InferenceEndpointPlan:
        checkpoint = self.get(name)
        resolved_role = str(role or checkpoint.role).strip().lower()
        if resolved_role not in VALID_ROLES:
            raise ValueError(f"Unsupported endpoint role: {resolved_role}")
        endpoint_name = f"{resolved_role}-{name}"
        env = {
            "TAR_ENDPOINT_NAME": endpoint_name,
            "TAR_ENDPOINT_ROLE": resolved_role,
            "TAR_ENDPOINT_MODEL": checkpoint.name,
        }
        command = [
            sys.executable,
            str(self.store.workspace / "serve_local.py"),
            "--backend",
            checkpoint.backend,
            "--model",
            checkpoint.model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--role",
            resolved_role,
            "--served-model-name",
            checkpoint.name,
        ]
        return InferenceEndpointPlan(
            checkpoint=checkpoint,
            host=host,
            port=port,
            base_url=f"http://{host}:{port}/v1",
            command=command,
            env=env,
            endpoint_name=endpoint_name,
            role=resolved_role,  # type: ignore[arg-type]
        )

    def list_endpoints(self) -> list[EndpointRecord]:
        return self.store.list_endpoints()

    def get_endpoint(self, endpoint_name: str) -> EndpointRecord:
        record = self.store.get_endpoint(endpoint_name)
        if record is None:
            raise KeyError(f"Unknown endpoint: {endpoint_name}")
        return record

    def start_endpoint(
        self,
        name: str,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        role: Optional[str] = None,
        wait_for_health: bool = False,
        startup_timeout_s: float = 5.0,
    ) -> EndpointRecord:
        plan = self.build_endpoint(name, host=host, port=port, role=role)
        existing = self.store.get_endpoint(plan.endpoint_name or "")
        if existing is not None and existing.status == "running":
            return self.refresh_endpoint_health(existing.endpoint_name)
        env = dict(os.environ)
        env.update(plan.env)
        process = subprocess.Popen(
            plan.command,
            cwd=self.store.workspace,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._processes[plan.endpoint_name or name] = process
        record = EndpointRecord(
            endpoint_name=plan.endpoint_name or name,
            checkpoint_name=plan.checkpoint.name,
            role=plan.role or plan.checkpoint.role,
            host=plan.host,
            port=plan.port,
            backend=plan.checkpoint.backend,
            base_url=plan.base_url,
            command=plan.command,
            env=plan.env,
            status="starting",
            process_pid=process.pid,
            started_at=self._now(),
        )
        self.store.upsert_endpoint(record)
        if wait_for_health:
            deadline = time.time() + max(0.5, startup_timeout_s)
            while time.time() < deadline:
                refreshed = self.refresh_endpoint_health(record.endpoint_name)
                if refreshed.health is not None and refreshed.health.ok:
                    return refreshed
                time.sleep(0.25)
        return self.refresh_endpoint_health(record.endpoint_name)

    def stop_endpoint(self, endpoint_name: str) -> EndpointRecord:
        record = self.get_endpoint(endpoint_name)
        pid = record.process_pid
        if pid:
            proc = self._processes.get(endpoint_name)
            try:
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)
                else:
                    os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        updated = record.model_copy(
            update={
                "status": "stopped",
                "process_pid": None,
                "stopped_at": self._now(),
                "health": EndpointHealth(
                    endpoint_name=record.endpoint_name,
                    status="stopped",
                    ok=False,
                    detail="endpoint stopped",
                    backend=record.backend,
                ),
            }
        )
        self._processes.pop(endpoint_name, None)
        self.store.upsert_endpoint(updated)
        return updated

    def restart_endpoint(
        self,
        endpoint_name: str,
        *,
        wait_for_health: bool = False,
        startup_timeout_s: float = 5.0,
    ) -> EndpointRecord:
        record = self.stop_endpoint(endpoint_name)
        return self.start_endpoint(
            record.checkpoint_name,
            host=record.host,
            port=record.port,
            role=record.role,
            wait_for_health=wait_for_health,
            startup_timeout_s=startup_timeout_s,
        )

    def refresh_endpoint_health(self, endpoint_name: str) -> EndpointRecord:
        record = self.get_endpoint(endpoint_name)
        health = self.endpoint_health(endpoint_name)
        status = "running" if health.ok else ("failed" if record.status == "running" else record.status)
        if record.process_pid and not self._pid_alive(record.process_pid):
            status = "failed" if health.status != "stopped" else "stopped"
        updated = record.model_copy(update={"health": health, "status": status, "last_error": None if health.ok else health.detail})
        self.store.upsert_endpoint(updated)
        return updated

    def endpoint_health(self, endpoint_name: str) -> EndpointHealth:
        record = self.get_endpoint(endpoint_name)
        start = time.perf_counter()
        try:
            with urlopen(f"{record.base_url.removesuffix('/v1')}/health", timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
                latency_ms = (time.perf_counter() - start) * 1000.0
                return EndpointHealth(
                    endpoint_name=endpoint_name,
                    status="healthy",
                    ok=True,
                    http_status=response.status,
                    latency_ms=round(latency_ms, 3),
                    detail=payload.get("detail"),
                    model_id=payload.get("model"),
                    backend=payload.get("backend"),
                )
        except URLError as exc:
            return EndpointHealth(
                endpoint_name=endpoint_name,
                status="unhealthy" if record.status != "stopped" else "stopped",
                ok=False,
                detail=str(exc.reason),
                backend=record.backend,
            )
        except Exception as exc:
            return EndpointHealth(
                endpoint_name=endpoint_name,
                status="failed",
                ok=False,
                detail=str(exc),
                backend=record.backend,
            )

    def assign_role(
        self,
        *,
        role: str,
        checkpoint_name: str,
        endpoint_name: Optional[str] = None,
    ) -> RoleAssignment:
        role_name = str(role).strip().lower()
        if role_name not in VALID_ROLES:
            raise ValueError(f"Unsupported role assignment: {role}")
        checkpoint = self.get(checkpoint_name)
        if endpoint_name is not None:
            endpoint = self.get_endpoint(endpoint_name)
            if endpoint.checkpoint_name != checkpoint.name:
                raise ValueError("Endpoint checkpoint does not match the requested checkpoint assignment.")
        assignment = RoleAssignment(
            role=role_name,  # type: ignore[arg-type]
            checkpoint_name=checkpoint.name,
            endpoint_name=endpoint_name,
        )
        self.store.upsert_role_assignment(assignment)
        return assignment

    def list_role_assignments(self) -> list[RoleAssignment]:
        return self.store.list_role_assignments()

    @staticmethod
    def default_claim_policy() -> ClaimAcceptancePolicy:
        return ClaimAcceptancePolicy()

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            if sys.platform == "win32":
                os.kill(pid, 0)
            else:
                os.kill(pid, 0)
            return True
        except OSError:
            return False
