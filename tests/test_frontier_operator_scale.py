import tempfile
from pathlib import Path

import pytest

from tar_lab.model_router import ModelRouter
from tar_lab.schemas import FrontierModelConfig, LocalLLMConfig, ModelRoutingRecord


def _frontier_config(*, max_budget: float = 0.0) -> FrontierModelConfig:
    return FrontierModelConfig(
        frontier_role_config=LocalLLMConfig(
            base_url="http://frontier.local/v1",
            api_key="local",
            model="frontier-model",
            model_tier="frontier",
            cost_per_token_input=0.01,
            cost_per_token_output=0.02,
        ),
        efficient_role_config=LocalLLMConfig(
            base_url="http://efficient.local/v1",
            api_key="local",
            model="efficient-model",
            model_tier="efficient",
            cost_per_token_input=0.001,
            cost_per_token_output=0.002,
        ),
        routing_policy="stakes_aware",
        max_frontier_budget_usd=max_budget,
    )


def test_efficient_tier_selected_for_unknown_decision():
    with tempfile.TemporaryDirectory() as tmp:
        router = ModelRouter(tmp, _frontier_config())
        selected = router.select_config("scout_reasoning")
        assert selected.model == "efficient-model"
        assert selected.model_tier == "efficient"


def test_frontier_tier_selected_for_director_propose():
    with tempfile.TemporaryDirectory() as tmp:
        router = ModelRouter(tmp, _frontier_config())
        selected = router.select_config("director_propose")
        assert selected.model == "frontier-model"
        assert selected.model_tier == "frontier"


def test_routing_record_persisted_after_log_call():
    with tempfile.TemporaryDirectory() as tmp:
        router = ModelRouter(tmp, _frontier_config())
        record = router.log_call("director_propose", "frontier", "model-x", 100, 50)
        routing_dir = Path(tmp) / "tar_state" / "routing"
        files = list(routing_dir.glob("route-*.json"))
        assert len(files) == 1
        parsed = ModelRoutingRecord.model_validate_json(files[0].read_text(encoding="utf-8"))
        assert parsed.record_id == record.record_id


def test_cost_accumulates_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        router = ModelRouter(tmp, _frontier_config())
        record = router.log_call("director_propose", "frontier", "model-x", 10, 5)
        assert record.cost_usd == pytest.approx(0.20)


def test_budget_cap_falls_back_to_efficient():
    with tempfile.TemporaryDirectory() as tmp:
        router = ModelRouter(tmp, _frontier_config(max_budget=0.10))
        router.log_call("director_propose", "frontier", "model-x", 10, 5)
        selected = router.select_config("director_propose")
        assert selected.model == "efficient-model"
        assert selected.model_tier == "efficient"
