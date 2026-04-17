from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from tar_lab.schemas import (
    DirectorPolicy,
    GenerativeDirectorProposal,
    ProposedExperimentFamily,
)

if TYPE_CHECKING:
    from tar_lab.hierarchy import LocalOpenAIRole


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


PROPOSAL_PROMPT_TEMPLATE = """You are the Director of a thermodynamic continual-learning research system.

The current experiment is stuck. The three standard families
(elastic_anchor, ou_drift_jitter, layer_freeze) have all been tried and
are not resolving the current failure pattern.

Objective: {objective_slug}
Failure context: {trigger_reason}
Recent failure streak: {failure_streak}

Propose ONE new experiment family. Respond in this exact JSON format:
{{
  "name": "<short_slug_no_spaces>",
  "description": "<one sentence describing the approach>",
  "config_delta": {{"key": "value"}},
  "rationale": "<why this would address the failure pattern>"
}}

Be concrete. Do not propose a variant of the three existing families.
"""


class GenerativeDirector:
    """
    Wraps RuleDirector and adds operator-backed family proposal when the
    rule-based approach is exhausted or confidence is low.
    """

    PROPOSAL_TRIGGER_STREAK: int = 5

    def __init__(
        self,
        workspace_root: str,
        operator_role: Optional["LocalOpenAIRole"] = None,
    ) -> None:
        self._workspace = workspace_root
        self._operator = operator_role

    def should_propose(self, policy: DirectorPolicy) -> bool:
        return policy.pivot_required and policy.failure_streak >= self.PROPOSAL_TRIGGER_STREAK

    def propose_family(
        self,
        policy: DirectorPolicy,
        trigger_reason: str,
    ) -> GenerativeDirectorProposal:
        proposal_id = f"gdp-{uuid.uuid4().hex[:8]}"
        operator_available = False
        operator_prompt_used = None
        family: ProposedExperimentFamily

        if self._operator is not None:
            prompt = PROPOSAL_PROMPT_TEMPLATE.format(
                objective_slug=policy.objective_slug,
                trigger_reason=trigger_reason,
                failure_streak=policy.failure_streak,
            )
            try:
                response = self._call_operator(prompt)
                parsed = self._parse_operator_response(response)
                family = ProposedExperimentFamily(
                    family_id=f"fam-{uuid.uuid4().hex[:8]}",
                    name=str(parsed.get("name", "operator_proposed")),
                    description=str(parsed.get("description", "")),
                    config_delta=parsed.get("config_delta", {}) if isinstance(parsed.get("config_delta", {}), dict) else {},
                    rationale=str(parsed.get("rationale", "")),
                    proposed_by="operator",
                )
                operator_available = True
                operator_prompt_used = prompt
            except Exception as exc:
                family = self._rule_heuristic_proposal(policy, trigger_reason)
                family = family.model_copy(
                    update={
                        "feasibility_note": f"operator_unavailable: {exc}",
                        "updated_at": utc_now_iso(),
                    }
                )
        else:
            family = self._rule_heuristic_proposal(policy, trigger_reason)

        return GenerativeDirectorProposal(
            proposal_id=proposal_id,
            objective_slug=policy.objective_slug,
            trigger_reason=trigger_reason,
            proposed_family=family,
            operator_available=operator_available,
            operator_prompt_used=operator_prompt_used,
        )

    def _rule_heuristic_proposal(
        self,
        policy: DirectorPolicy,
        trigger_reason: str,
    ) -> ProposedExperimentFamily:
        return ProposedExperimentFamily(
            family_id=f"fam-{uuid.uuid4().hex[:8]}",
            name="elastic_anchor_conservative",
            description=(
                "Tightened elastic anchor with reduced drift budget "
                "and lower fim_lambda for high-streak failure patterns."
            ),
            config_delta={
                "drift_budget_multiplier": 0.5,
                "fim_lambda_multiplier": 0.7,
            },
            rationale=(
                f"Rule heuristic: high failure streak ({policy.failure_streak}) "
                f"on {policy.objective_slug}. Reduce exploration aggressiveness. "
                f"Trigger: {trigger_reason}"
            ),
            proposed_by="rule_heuristic",
        )

    def _parse_operator_response(self, response: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError("no JSON object in operator response")
        parsed = json.loads(match.group())
        if not isinstance(parsed, dict):
            raise ValueError("operator response was not a JSON object")
        return parsed

    def _call_operator(self, prompt: str) -> str:
        if self._operator is None:
            raise RuntimeError("operator role unavailable")
        client = self._operator._client()
        return self._operator._chat(
            client,
            [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
