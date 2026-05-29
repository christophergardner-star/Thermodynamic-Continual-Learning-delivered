"""
TAR Experiment Scheduler
=========================
Hardware-aware scheduler that decides which experiments can run in parallel
given current GPU/CPU/RAM availability, and generates human-language rationale
for every decision.

Called by ExperimentOrchestrator.run_parallel() and by the dashboard
to show what TAR is holding and why.

Output: {workspace}/tar_state/scheduler_state.json
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent
from tar_lab.human_review import approved_experiment_ids
from tar_storage import ensure_workspace_layout, resolve_workspace

# ── VRAM estimates per dataset (conservative, includes overhead) ──────────────
_VRAM_BUDGET: dict[str, float] = {
    "split_cifar10":      2.5,   # ResNet-18 + 5-task CIFAR-10 on 4 GB-class GPUs
    "split_cifar100":     3.2,   # calibrated from the empirical local Phase 16 partial run
    "split_tinyimagenet": 3.3,   # conservative 4 GB-class target for serialized scale-up runs
    "permuted_mnist":     1.5,   # MLP on 28x28 grayscale — minimal VRAM
}

# GPU headroom to reserve (never fill VRAM completely — leave room for temp spikes)
_MIN_VRAM_HEADROOM_GB = 0.5
_MAX_VRAM_HEADROOM_GB = 2.0


def _vram_headroom_gb(total_vram_gb: float) -> float:
    if total_vram_gb <= 0:
        return _MAX_VRAM_HEADROOM_GB
    return min(_MAX_VRAM_HEADROOM_GB, max(_MIN_VRAM_HEADROOM_GB, total_vram_gb * 0.15))


def _spec_vram_budget(spec: Any) -> float:
    # Dataset-specific calibrated estimates take precedence — they are empirically
    # validated and more accurate than the generic hardware_budget default (3.5 GB).
    dataset = getattr(spec, "dataset", "")
    if dataset in _VRAM_BUDGET:
        return _VRAM_BUDGET[dataset]
    # Fall back to explicit hardware_budget if set to something non-default
    budget = getattr(spec, "hardware_budget", {}) or {}
    if isinstance(budget, dict) and budget.get("vram_gb"):
        try:
            return float(budget["vram_gb"])
        except Exception:
            pass
    return 4.0

# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class HardwareSnapshot:
    gpu_available:     bool  = False
    gpu_name:          str   = ""
    gpu_util_pct:      int   = 0
    vram_used_gb:      float = 0.0
    vram_total_gb:     float = 0.0
    vram_free_gb:      float = 0.0
    gpu_temp_c:        int   = 0
    cpu_util_pct:      float = 0.0
    cpu_cores:         int   = 0
    ram_used_gb:       float = 0.0
    ram_total_gb:      float = 0.0
    ram_available_gb:  float = 0.0
    timestamp:         str   = ""


@dataclass
class HoldReason:
    experiment_id:   str
    experiment_name: str
    reason:          str    # human text


@dataclass
class SchedulerDecision:
    timestamp:            str
    can_start:            list[str]          # experiment IDs to start now
    hold_reasons:         list[HoldReason]   # why others are held
    running_ids:          list[str]          # already running
    rationale:            str               # full human-language paragraph
    hardware_used:        dict              # {"vram_gb": X, "cpu_pct": Y}
    hardware_available:   dict              # {"vram_gb": X, "cpu_pct": Y}


# ── scheduler ────────────────────────────────────────────────────────────────
class TARScheduler:
    """
    Decides which pending experiments to start given hardware state.
    Rules:
      - At most 1 GPU experiment running at once (prevents VRAM exhaustion)
      - CPU-only tasks (method=sgd_baseline with tiny config) can run in parallel
      - Reserve _VRAM_HEADROOM_GB on GPU
      - Never start if GPU temp > 85°C
      - Never start if RAM available < 4 GB
    """

    def __init__(self, workspace: Path):
        self.workspace   = workspace
        self._state_path = workspace / "tar_state" / "scheduler_state.json"

    def _director_frontier_ranks(self) -> dict[str, int]:
        path = self.workspace / "tar_state" / "research_director_state.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        directives = data.get("frontier_directives", []) if isinstance(data, dict) else []
        ranks: dict[str, int] = {}
        for idx, rec in enumerate(directives, start=1):
            pid = str(rec.get("problem_id", "") or "")
            if not pid:
                continue
            try:
                ranks[pid] = int(rec.get("scheduler_rank", idx))
            except Exception:
                ranks[pid] = idx
        return ranks

    def _director_experiment_ranks(self) -> dict[str, int]:
        path = self.workspace / "tar_state" / "research_director_state.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        directives = data.get("experiment_directives", []) if isinstance(data, dict) else []
        ranks: dict[str, int] = {}
        for idx, rec in enumerate(directives, start=1):
            exp_id = str(rec.get("experiment_id", "") or "")
            if not exp_id:
                continue
            try:
                ranks[exp_id] = int(rec.get("scheduler_rank", idx))
            except Exception:
                ranks[exp_id] = idx
        return ranks

    @staticmethod
    def _priority_key(spec: Any, experiment_ranks: dict[str, int], frontier_ranks: dict[str, int]) -> tuple[int, int, int, int, str]:
        stage = str(getattr(spec, "stage", "") or "")
        exp_id = str(getattr(spec, "id", "") or "")
        frontier = str(getattr(spec, "frontier_problem_id", "") or "")
        experiment_rank = experiment_ranks.get(exp_id, 999)
        stalled_scaleup = int(not (stage == "stalled" and frontier == "fp-scale-up"))
        frontier_rank = frontier_ranks.get(frontier, 999)
        priority = int(getattr(spec, "priority", 50) or 50)
        submitted_at = str(getattr(spec, "submitted_at", "") or "")
        return (experiment_rank, stalled_scaleup, frontier_rank, priority, submitted_at)

    def read_hardware(self) -> HardwareSnapshot:
        hw_path = self.workspace / "tar_state" / "hardware_state.json"
        if not hw_path.exists():
            return HardwareSnapshot()
        try:
            data = json.loads(hw_path.read_text(encoding="utf-8"))
            gpu  = data.get("gpu", {})
            cpu  = data.get("cpu", {})
            ram  = data.get("ram", {})
            return HardwareSnapshot(
                gpu_available    = gpu.get("available", False),
                gpu_name         = gpu.get("name", ""),
                gpu_util_pct     = gpu.get("utilization_pct", 0),
                vram_used_gb     = gpu.get("vram_used_gb", 0.0),
                vram_total_gb    = gpu.get("vram_total_gb", 0.0),
                vram_free_gb     = gpu.get("vram_free_gb", 0.0),
                gpu_temp_c       = gpu.get("temperature_c", 0),
                cpu_util_pct     = cpu.get("utilization_pct", 0.0),
                cpu_cores        = cpu.get("core_count", 0),
                ram_used_gb      = ram.get("used_gb", 0.0),
                ram_total_gb     = ram.get("total_gb", 0.0),
                ram_available_gb = ram.get("available_gb", 0.0),
                timestamp        = data.get("timestamp", ""),
            )
        except Exception:
            return HardwareSnapshot()

    def _queue_status(self) -> dict[str, dict]:
        queue_path = self.workspace / "tar_state" / "experiment_queue.json"
        if not queue_path.exists():
            return {}
        try:
            data = json.loads(queue_path.read_text(encoding="utf-8"))
            experiments = data.get("experiments", []) if isinstance(data, dict) else []
            return {exp.get("id", ""): exp for exp in experiments if exp.get("id")}
        except Exception:
            return {}

    def _archive_status(self) -> dict[str, dict]:
        archive_path = self.workspace / "tar_state" / "experiment_archive.json"
        if not archive_path.exists():
            return {}
        try:
            data = json.loads(archive_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        experiments = data.get("experiments", []) if isinstance(data, dict) else []
        return {exp.get("id", ""): exp for exp in experiments if isinstance(exp, dict) and exp.get("id")}

    @staticmethod
    def _record_is_complete(record: dict[str, Any]) -> bool:
        if not isinstance(record, dict):
            return False
        status = str(record.get("status", "") or "").lower()
        stage = str(record.get("stage", "") or "").lower()
        if status == "complete" or stage == "complete":
            return True
        result_path = str(record.get("result_path", "") or "")
        if not result_path:
            return False
        path = Path(result_path)
        if not path.exists():
            return False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        verdict = str(raw.get("verdict", "") or "").upper()
        status_hint = str(raw.get("status", "") or "").upper()
        return bool(raw) and verdict != "ERROR" and status_hint != "ERROR"

    def _dependency_complete(self, experiment_id: str, known_status: dict[str, dict], archived_status: dict[str, dict]) -> bool:
        if not experiment_id:
            return False
        live = known_status.get(experiment_id, {})
        if self._record_is_complete(live):
            return True
        archived = archived_status.get(experiment_id, {})
        if self._record_is_complete(archived):
            return True
        direct_path = self.workspace / "tar_state" / "experiments" / experiment_id / "result.json"
        if not direct_path.exists():
            return False
        try:
            raw = json.loads(direct_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        verdict = str(raw.get("verdict", "") or "").upper()
        status_hint = str(raw.get("status", "") or "").upper()
        return bool(raw) and verdict != "ERROR" and status_hint != "ERROR"

    def get_runnable(
        self,
        pending_specs: list[Any],
        snapshot: HardwareSnapshot | None = None,
        running_specs: list[Any] | None = None,
    ) -> list[Any]:
        decision = self.decide(
            pending_specs=pending_specs,
            running_specs=running_specs or [],
            hw=snapshot,
        )
        runnable_ids = set(decision.can_start)
        return [spec for spec in pending_specs if spec.id in runnable_ids]

    def decide(
        self,
        pending_specs: list[Any],      # list of ExperimentSpec (duck-typed to avoid circular import)
        running_specs: list[Any],
        hw: HardwareSnapshot | None = None,
    ) -> SchedulerDecision:
        if hw is None:
            hw = self.read_hardware()

        now = datetime.now(timezone.utc).isoformat()
        can_start:    list[str]       = []
        hold_reasons: list[HoldReason] = []

        # Count currently running GPU experiments
        gpu_running = sum(
            1 for s in running_specs
            if s.dataset != "cpu_only"
        )
        vram_committed = sum(
            _spec_vram_budget(s)
            for s in running_specs
            if s.dataset != "cpu_only"
        )
        # Default to 0.0 (hold all GPU) when VRAM detection fails — safe fallback.
        # 24.0 GB was the former default, which incorrectly granted unlimited budget.
        _vram_detected = hw.vram_total_gb if hw.vram_total_gb and hw.vram_total_gb > 0 else 0.0
        reserve_headroom = _vram_headroom_gb(_vram_detected)
        vram_headroom = _vram_detected - reserve_headroom

        # Thermal gate
        gpu_too_hot = hw.gpu_temp_c > 85
        ram_critical = hw.ram_available_gb > 0 and hw.ram_available_gb < 4.0

        running_ids = [s.id for s in running_specs]
        known_status = self._queue_status()
        archived_status = self._archive_status()
        experiment_ranks = self._director_experiment_ranks()
        frontier_ranks = self._director_frontier_ranks()
        approved_ids = approved_experiment_ids(self.workspace)

        for spec in sorted(pending_specs, key=lambda rec: self._priority_key(rec, experiment_ranks, frontier_ranks)):
            exp_vram = _spec_vram_budget(spec)
            is_cpu_only = spec.dataset == "cpu_only"
            if str(getattr(spec, "id", "") or "") not in approved_ids:
                hold_reasons.append(HoldReason(
                    experiment_id=spec.id,
                    experiment_name=spec.name,
                    reason="Awaiting explicit human approval in the dashboard and a committed execution manifest.",
                ))
                continue
            unmet = []
            for dep_id in getattr(spec, "depends_on", []) or []:
                if not self._dependency_complete(str(dep_id or ""), known_status, archived_status):
                    unmet.append(dep_id)
            if unmet:
                hold_reasons.append(HoldReason(
                    experiment_id=spec.id,
                    experiment_name=spec.name,
                    reason=f"Waiting on dependencies to finish: {', '.join(unmet)}.",
                ))
                continue

            # Thermal gate
            if gpu_too_hot and not is_cpu_only:
                hold_reasons.append(HoldReason(
                    experiment_id   = spec.id,
                    experiment_name = spec.name,
                    reason = f"GPU at {hw.gpu_temp_c}°C — holding until temp drops below 85°C."
                ))
                continue

            # RAM gate
            if ram_critical:
                hold_reasons.append(HoldReason(
                    experiment_id   = spec.id,
                    experiment_name = spec.name,
                    reason = f"Only {hw.ram_available_gb:.1f} GB RAM available — need ≥4 GB."
                ))
                continue

            if is_cpu_only:
                can_start.append(spec.id)
                continue

            # GPU experiment: check slot and VRAM
            if gpu_running >= 1:
                running_name = running_specs[0].name if running_specs else "current"
                est_h = getattr(running_specs[0], "estimated_runtime_h", 0) if running_specs else 0
                hold_reasons.append(HoldReason(
                    experiment_id   = spec.id,
                    experiment_name = spec.name,
                    reason = (
                        f"GPU slot occupied by '{running_name}' "
                        f"(est. {est_h:.1f}h remaining). "
                        f"Needs {exp_vram:.1f} GB VRAM — queued."
                    )
                ))
                continue

            if vram_committed + exp_vram > vram_headroom:
                hold_reasons.append(HoldReason(
                    experiment_id   = spec.id,
                    experiment_name = spec.name,
                    reason = (
                        f"Needs {exp_vram:.1f} GB VRAM but only "
                        f"{vram_headroom - vram_committed:.1f} GB free "
                        f"(keeping {reserve_headroom:.1f} GB headroom)."
                    )
                ))
                continue

            # Clear to run
            can_start.append(spec.id)
            gpu_running       += 1
            vram_committed    += exp_vram

        # Build rationale paragraph
        rationale = self._rationale(hw, running_specs, can_start, hold_reasons, vram_committed, vram_headroom)

        used_vram = vram_committed + hw.vram_used_gb
        decision = SchedulerDecision(
            timestamp          = now,
            can_start          = can_start,
            hold_reasons       = hold_reasons,
            running_ids        = running_ids,
            rationale          = rationale,
            hardware_used      = {
                "vram_gb":  round(used_vram, 1),
                "cpu_pct":  hw.cpu_util_pct,
                "ram_gb":   hw.ram_used_gb,
            },
            hardware_available = {
                "vram_gb":  round(hw.vram_total_gb, 1),
                "cpu_pct":  100.0,
                "ram_gb":   hw.ram_total_gb,
            },
        )
        self._save(decision)
        return decision

    def _rationale(
        self,
        hw:           HardwareSnapshot,
        running:      list[Any],
        can_start:    list[str],
        holds:        list[HoldReason],
        used_vram:    float,
        total_vram:   float,
    ) -> str:
        parts: list[str] = []

        # Hardware summary
        if hw.gpu_available:
            parts.append(
                f"GPU: {hw.gpu_name} at {hw.gpu_util_pct}% utilisation, "
                f"{hw.vram_used_gb:.1f}/{hw.vram_total_gb:.1f} GB VRAM used, "
                f"{hw.gpu_temp_c}°C."
            )
        else:
            parts.append("No GPU detected — all experiments will run on CPU.")

        # Running
        if running:
            r_names = ", ".join(f"'{s.name}'" for s in running)
            parts.append(f"Currently running: {r_names}.")

        # Starting
        if can_start:
            parts.append(f"Cleared to start: {len(can_start)} experiment(s) ({', '.join(can_start)}).")
        else:
            parts.append("No new experiments starting this cycle.")

        # Holds
        if holds:
            hold_summary = "; ".join(f"'{h.experiment_name}': {h.reason}" for h in holds[:3])
            if len(holds) > 3:
                hold_summary += f" (+ {len(holds)-3} more held)"
            parts.append(f"Holding: {hold_summary}")

        # Thermal
        if hw.gpu_temp_c > 75:
            parts.append(f"Note: GPU temp {hw.gpu_temp_c}°C — monitoring.")

        template = " ".join(parts)
        try:
            from tar_lab.llm_bridge import narrate_scheduler_decision
            narrated = narrate_scheduler_decision(
                self.workspace,
                gpu_name=hw.gpu_name,
                gpu_util_pct=hw.gpu_util_pct,
                vram_used_gb=used_vram,
                vram_total_gb=hw.vram_total_gb,
                gpu_temp_c=hw.gpu_temp_c,
                running_names=[s.name for s in running],
                can_start=can_start,
                hold_reasons=[{"experiment_name": h.experiment_name, "reason": h.reason} for h in holds],
                template_fallback=template,
            )
            if narrated:
                return narrated
        except Exception:
            pass
        return template

    def _save(self, decision: SchedulerDecision) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        can_start = list(decision.can_start)
        hold_reasons = [asdict(h) for h in decision.hold_reasons]
        data = {
            "timestamp":            decision.timestamp,
            "saved_at":             decision.timestamp,
            "state_version":        2,
            "rationale":            decision.rationale,
            "rationale_human_text": decision.rationale,
            "can_start":            can_start,
            "experiments_to_start": can_start,
            "running_ids":          decision.running_ids,
            "running_count":        len(decision.running_ids),
            "can_start_count":      len(can_start),
            "next_experiment_id":   can_start[0] if can_start else "",
            "hardware_used":        decision.hardware_used,
            "hardware_available":   decision.hardware_available,
            "hold_reasons":         hold_reasons,
            "hold_count":           len(hold_reasons),
        }
        try:
            self._state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def read_state(self) -> dict:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ws  = ensure_workspace_layout(resolve_workspace(_REPO), repo_root=_REPO)
    sch = TARScheduler(ws)
    hw  = sch.read_hardware()
    print(f"\nHardware: GPU={hw.gpu_name}  VRAM={hw.vram_used_gb:.1f}/{hw.vram_total_gb:.1f}GB  "
          f"Temp={hw.gpu_temp_c}C  CPU={hw.cpu_util_pct:.0f}%  RAM={hw.ram_used_gb:.1f}/{hw.ram_total_gb:.1f}GB")
    state = sch.read_state()
    if state:
        print(f"\nLast decision ({state.get('timestamp','')[:16]}):")
        print(f"  {state.get('rationale','(none)')}")
