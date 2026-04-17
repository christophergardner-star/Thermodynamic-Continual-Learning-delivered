from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

from tar_lab.schemas import (
    FrontierModelConfig,
    LocalLLMConfig,
    ModelRoutingRecord,
    RoutingSummary,
)


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ModelRouter:
    """
    Cost-aware tier router for efficient/frontier model selection.
    """

    ROUTING_LOG_PATH = "tar_state/routing"

    def __init__(self, workspace_root: str, config: FrontierModelConfig) -> None:
        self._workspace = Path(workspace_root).resolve()
        self._config = config
        self._log_dir = self._workspace / self.ROUTING_LOG_PATH
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._session_cost_usd: float = 0.0

    def select_config(self, decision_type: str) -> LocalLLMConfig:
        if self._config.routing_policy == "always_frontier":
            selected_tier = "frontier"
        elif self._config.routing_policy == "always_efficient":
            selected_tier = "efficient"
        else:
            selected_tier = (
                "frontier"
                if decision_type in self._config.frontier_decisions
                else "efficient"
            )
        if (
            selected_tier == "frontier"
            and self._config.max_frontier_budget_usd > 0.0
            and self._current_total_cost() >= self._config.max_frontier_budget_usd
        ):
            selected_tier = "efficient"
        return (
            self._config.frontier_role_config
            if selected_tier == "frontier"
            else self._config.efficient_role_config
        )

    def log_call(
        self,
        decision_type: str,
        tier_selected: str,
        model_id: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> ModelRoutingRecord:
        config = (
            self._config.frontier_role_config
            if tier_selected == "frontier"
            else self._config.efficient_role_config
        )
        cost = (
            tokens_in * config.cost_per_token_input
            + tokens_out * config.cost_per_token_output
        )
        self._session_cost_usd = self._current_total_cost() + cost
        budget_remaining: Optional[float] = None
        if self._config.max_frontier_budget_usd > 0.0:
            budget_remaining = max(
                0.0,
                self._config.max_frontier_budget_usd - self._session_cost_usd,
            )
        record = ModelRoutingRecord(
            record_id=f"route-{uuid.uuid4().hex[:8]}",
            decision_type=decision_type,
            tier_selected=tier_selected,  # type: ignore[arg-type]
            model_id=model_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            budget_remaining_usd=budget_remaining,
        )
        path = self._log_dir / f"{record.record_id}.json"
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return record

    def get_summary(self) -> RoutingSummary:
        records = self._load_all_records()
        total_cost = sum(record.cost_usd for record in records)
        budget_remaining: Optional[float] = None
        budget_exhausted = False
        if self._config.max_frontier_budget_usd > 0.0:
            budget_remaining = max(
                0.0,
                self._config.max_frontier_budget_usd - total_cost,
            )
            budget_exhausted = total_cost >= self._config.max_frontier_budget_usd
        return RoutingSummary(
            frontier_calls=sum(1 for record in records if record.tier_selected == "frontier"),
            efficient_calls=sum(1 for record in records if record.tier_selected == "efficient"),
            total_cost_usd=total_cost,
            budget_remaining_usd=budget_remaining,
            budget_exhausted=budget_exhausted,
        )

    def load_log(self) -> List[ModelRoutingRecord]:
        return self._load_all_records()

    def _load_all_records(self) -> List[ModelRoutingRecord]:
        records: List[ModelRoutingRecord] = []
        for path in sorted(self._log_dir.glob("route-*.json")):
            records.append(
                ModelRoutingRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            )
        return records

    def _current_total_cost(self) -> float:
        return sum(record.cost_usd for record in self._load_all_records())
