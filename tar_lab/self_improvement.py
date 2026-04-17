from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tar_lab.schemas import (
    CuratedDeltaRecord,
    FrozenAnchorPackManifest,
    SelfImprovementCycleRecord,
    SelfImprovementPolicy,
    TrainingSignalRecord,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class SelfImprovementEngine:
    """
    Curates TAR's own signal and enforces the local self-improvement safety gates.
    """

    SIGNALS_PATH = "tar_state/self_improvement/signals"
    DELTAS_PATH = "tar_state/self_improvement/deltas"
    CYCLES_PATH = "tar_state/self_improvement/cycles"
    ANCHOR_PATH = "tar_state/self_improvement/anchor_manifest.json"

    def __init__(
        self,
        workspace_root: str,
        policy: Optional[SelfImprovementPolicy] = None,
    ) -> None:
        self._workspace = Path(workspace_root).resolve()
        self._policy = policy or SelfImprovementPolicy()
        self._signals_dir = self._workspace / self.SIGNALS_PATH
        self._deltas_dir = self._workspace / self.DELTAS_PATH
        self._cycles_dir = self._workspace / self.CYCLES_PATH
        self._anchor_path = self._workspace / self.ANCHOR_PATH
        for directory in (self._signals_dir, self._deltas_dir, self._cycles_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def initialize_anchor_pack(
        self,
        pack_path: str,
        run_manifest_path: str,
        baseline_mean_score: float,
        baseline_overclaim_rate: float,
    ) -> FrozenAnchorPackManifest:
        if self._anchor_path.exists():
            raise RuntimeError(
                "Anchor pack already initialized. "
                "Delete tar_state/self_improvement/anchor_manifest.json manually if you need to reseal."
            )
        run_manifest_hash = self._sha256_file(run_manifest_path)
        item_ids = self._extract_item_ids(pack_path)
        manifest = FrozenAnchorPackManifest(
            manifest_id=f"anchor-{uuid.uuid4().hex[:8]}",
            pack_path=pack_path,
            run_manifest_hash_sha256=run_manifest_hash,
            item_ids=item_ids,
            item_count=len(item_ids),
            baseline_mean_score=baseline_mean_score,
            baseline_overclaim_rate=baseline_overclaim_rate,
        )
        self._anchor_path.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return manifest

    def load_anchor_manifest(self) -> FrozenAnchorPackManifest:
        if not self._anchor_path.exists():
            raise RuntimeError("Anchor pack not initialized. Run initialize_anchor_pack first.")
        return FrozenAnchorPackManifest.model_validate_json(
            self._anchor_path.read_text(encoding="utf-8")
        )

    def verify_anchor_integrity(self) -> bool:
        manifest = self.load_anchor_manifest()
        run_manifest_path = self._resolve_path(manifest.pack_path) / "run_manifest.json"
        current_hash = self._sha256_file(str(run_manifest_path))
        return current_hash == manifest.run_manifest_hash_sha256

    def curate_signal(self, signal: TrainingSignalRecord) -> bool:
        if signal.overclaim_present:
            return False
        anchor = self.load_anchor_manifest()
        if signal.source_id in anchor.item_ids:
            signal = signal.model_copy(update={"anchor_pack_overlap": True})
            return False
        path = self._signals_dir / f"{signal.signal_id}.json"
        path.write_text(signal.model_dump_json(indent=2), encoding="utf-8")
        return True

    def list_signals(self) -> list[TrainingSignalRecord]:
        signals: list[TrainingSignalRecord] = []
        for path in self._signals_dir.glob("*.json"):
            signals.append(
                TrainingSignalRecord.model_validate_json(path.read_text(encoding="utf-8"))
            )
        return sorted(signals, key=lambda item: (item.created_at, item.signal_id))

    def assemble_delta(self, cycle_id: str) -> CuratedDeltaRecord:
        signals = self.list_signals()
        anchor = self.load_anchor_manifest()
        anchor_overlaps_excluded = sum(
            1
            for signal in signals
            if signal.anchor_pack_overlap or signal.source_id in anchor.item_ids
        )
        overclaim_excluded = sum(1 for signal in signals if signal.overclaim_present)
        clean = [
            signal
            for signal in signals
            if not signal.overclaim_present
            and not signal.anchor_pack_overlap
            and signal.source_id not in anchor.item_ids
        ]
        kind_distribution: dict[str, int] = {}
        for signal in clean:
            kind_distribution[signal.kind] = kind_distribution.get(signal.kind, 0) + 1
        diversity_score = min(1.0, len(kind_distribution) / 5.0)
        delta = CuratedDeltaRecord(
            delta_id=f"delta-{uuid.uuid4().hex[:8]}",
            cycle_id=cycle_id,
            signal_ids=[signal.signal_id for signal in clean],
            signal_count=len(clean),
            anchor_overlaps_excluded=anchor_overlaps_excluded,
            overclaim_excluded=overclaim_excluded,
            diversity_score=diversity_score,
            kind_distribution=kind_distribution,
            ready=(
                len(clean) >= self._policy.min_delta_signals
                and diversity_score >= self._policy.min_diversity_score
            ),
        )
        path = self._deltas_dir / f"{delta.delta_id}.json"
        path.write_text(delta.model_dump_json(indent=2), encoding="utf-8")
        return delta

    def evaluate_gate(
        self,
        probe_mean_score: float,
        probe_overclaim_rate: float,
        anchor_hash_verified: bool,
    ) -> tuple[bool, str]:
        if not anchor_hash_verified:
            return False, "anchor_integrity_failed"
        assert probe_overclaim_rate == 0.0, (
            f"overclaim_rate={probe_overclaim_rate} violated hard zero invariant"
        )
        if probe_mean_score < self._policy.min_mean_score_floor:
            return False, (
                f"mean_score={probe_mean_score} below floor "
                f"{self._policy.min_mean_score_floor}"
            )
        return True, "passed"

    def start_cycle(self) -> SelfImprovementCycleRecord:
        existing = self._load_latest_cycle()
        if existing and existing.status not in {
            "completed",
            "idle",
            "paused_consecutive_failures",
            "paused_cycle_limit",
            "gate_failed",
        }:
            raise RuntimeError(
                f"Cannot start new cycle: current cycle {existing.cycle_id} is in status {existing.status}"
            )
        if existing and existing.human_resume_required:
            raise RuntimeError(
                "Human review required before starting a new cycle. "
                "Call resume_self_improvement() after manual review."
            )
        cycle_number = (existing.cycle_number + 1) if existing else 1
        total_cycles_completed = existing.total_cycles_completed if existing else 0
        consecutive_failures = existing.consecutive_gate_failures if existing else 0
        if existing and existing.total_cycles_completed >= self._policy.max_auto_cycles:
            cycle = SelfImprovementCycleRecord(
                cycle_id=f"cycle-{uuid.uuid4().hex[:8]}",
                cycle_number=cycle_number,
                status="paused_cycle_limit",
                total_cycles_completed=existing.total_cycles_completed,
                consecutive_gate_failures=consecutive_failures,
                human_resume_required=True,
                paused_reason=f"max_auto_cycles={self._policy.max_auto_cycles} reached",
            )
            self.save_cycle(cycle)
            return cycle
        cycle = SelfImprovementCycleRecord(
            cycle_id=f"cycle-{uuid.uuid4().hex[:8]}",
            cycle_number=cycle_number,
            status="curating",
            total_cycles_completed=total_cycles_completed,
            consecutive_gate_failures=consecutive_failures,
        )
        self.save_cycle(cycle)
        return cycle

    def record_gate_failure(
        self,
        cycle: SelfImprovementCycleRecord,
        reason: str,
    ) -> SelfImprovementCycleRecord:
        cycle = cycle.model_copy(
            update={
                "consecutive_gate_failures": cycle.consecutive_gate_failures + 1,
                "updated_at": utc_now_iso(),
            }
        )
        if cycle.consecutive_gate_failures >= self._policy.max_consecutive_gate_failures:
            cycle = cycle.model_copy(
                update={
                    "status": "paused_consecutive_failures",
                    "human_resume_required": True,
                    "paused_reason": (
                        f"{cycle.consecutive_gate_failures} consecutive gate failures: "
                        f"last reason={reason}"
                    ),
                }
            )
        else:
            cycle = cycle.model_copy(update={"status": "gate_failed"})
        self.save_cycle(cycle)
        return cycle

    def resume_self_improvement(self, cycle_id: str) -> SelfImprovementCycleRecord:
        cycle = self.load_cycle(cycle_id)
        if cycle is None:
            raise ValueError(f"Cycle not found: {cycle_id}")
        resumed = cycle.model_copy(
            update={
                "human_resume_required": False,
                "consecutive_gate_failures": 0,
                "status": "curating",
                "updated_at": utc_now_iso(),
            }
        )
        self.save_cycle(resumed)
        return resumed

    def current_status(self) -> SelfImprovementCycleRecord:
        cycle = self._load_latest_cycle()
        if cycle is not None:
            return cycle
        return SelfImprovementCycleRecord(
            cycle_id="cycle-idle",
            status="idle",
            cycle_number=1,
            total_cycles_completed=0,
        )

    def load_cycle(self, cycle_id: str) -> Optional[SelfImprovementCycleRecord]:
        path = self._cycles_dir / f"{cycle_id}.json"
        if not path.exists():
            return None
        return SelfImprovementCycleRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def save_cycle(self, cycle: SelfImprovementCycleRecord) -> None:
        path = self._cycles_dir / f"{cycle.cycle_id}.json"
        path.write_text(cycle.model_dump_json(indent=2), encoding="utf-8")

    def _sha256_file(self, path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _extract_item_ids(self, pack_path: str) -> list[str]:
        item_ids: set[str] = set()
        for filename in ("eval_items.jsonl", "predictions.jsonl", "eval_core.jsonl", "results.jsonl"):
            path = self._resolve_path(pack_path) / filename
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                item_id = payload.get("item_id")
                if item_id:
                    item_ids.add(str(item_id))
        return sorted(item_ids)

    def _load_latest_cycle(self) -> Optional[SelfImprovementCycleRecord]:
        cycles: list[SelfImprovementCycleRecord] = []
        for path in self._cycles_dir.glob("*.json"):
            cycles.append(
                SelfImprovementCycleRecord.model_validate_json(path.read_text(encoding="utf-8"))
            )
        if not cycles:
            return None
        return max(cycles, key=lambda item: (item.cycle_number, item.started_at, item.cycle_id))

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self._workspace / candidate
