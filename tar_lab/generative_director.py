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


# Step 1 — Diagnose the root cause before proposing anything.
# This prevents unconditional novelty-seeking when the real problem is
# a wrong hyperparameter or an architecture that is too small.
_DIAGNOSIS_PROMPT = """\
You are diagnosing why a continual learning experiment has failed.
Given the failure pattern below, classify the most likely root cause:

(a) hyperparameter mis-specification — the algorithm family is correct but λ, lr, or momentum are wrong
(b) architecture limitation — the backbone is too small or the wrong inductive bias for this task
(c) dataset characteristic — the data distribution violates the method's assumptions
(d) algorithmic limitation — no tuning of this family can address this failure mode

Objective: {objective_slug}
Failure context: {trigger_reason}
Recent failure streak: {failure_streak}
Latest governor metrics: energy={energy:.4f}, σ={sigma:.4f}, ρ={rho:.4f}

Respond with ONLY this JSON object (no other text):
{{"root_cause": "<a|b|c|d>", "reasoning": "<one sentence>"}}"""

# Step 2 — Only reached when diagnosis is (c) or (d).
# For (a)/(b) a tuning action is returned directly without this call.
_PROPOSAL_PROMPT = """\
Diagnosed root cause: {root_cause} — {reasoning}

The three standard families (elastic_anchor, ou_drift_jitter, layer_freeze) have
been tried and cannot resolve this failure pattern.  Propose ONE new experiment
family that directly addresses the diagnosed root cause.

Objective: {objective_slug}
Failure context: {trigger_reason}

Respond with ONLY this JSON object (no other text):
{{
  "name": "<short_slug_no_spaces>",
  "description": "<one sentence describing the approach>",
  "config_delta": {{"key": "value"}},
  "rationale": "<why this addresses the diagnosed root cause>"
}}"""


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
            # ── Step 1: Diagnose the root cause ──────────────────────────────
            latest = policy.data_anchor[-1]
            diagnosis_prompt = _DIAGNOSIS_PROMPT.format(
                objective_slug=policy.objective_slug,
                trigger_reason=trigger_reason,
                failure_streak=policy.failure_streak,
                energy=latest.energy_e,
                sigma=latest.entropy_sigma,
                rho=latest.drift_rho,
            )
            try:
                diag_response = self._call_operator(diagnosis_prompt)
                diag_parsed = self._parse_operator_response(diag_response)
                root_cause = str(diag_parsed.get("root_cause", "d")).strip().lower()
                reasoning = str(diag_parsed.get("reasoning", ""))
            except Exception as exc:
                # Diagnosis failed — fall through to heuristic
                root_cause = "d"
                reasoning = f"diagnosis_unavailable: {exc}"

            # ── Step 2: Act on the diagnosis ──────────────────────────────────
            if root_cause in ("a", "b"):
                # Hyperparameter or architecture issue — no new family needed.
                # Return a tuning recommendation instead.
                tune_target = "architecture_backbone" if root_cause == "b" else "lambda_lr_momentum"
                family = ProposedExperimentFamily(
                    family_id=f"fam-{uuid.uuid4().hex[:8]}",
                    name="tune_hyperparameters",
                    description=(
                        f"Tuning recommendation (root cause {root_cause}): "
                        f"adjust {tune_target} rather than proposing a new algorithm family."
                    ),
                    config_delta={
                        "action": "tune_hyperparameters",
                        "root_cause": root_cause,
                        "tune_target": tune_target,
                    },
                    rationale=reasoning,
                    proposed_by="operator",
                    feasibility_note=f"diagnosis={root_cause}: {reasoning}",
                )
                operator_available = True
                operator_prompt_used = diagnosis_prompt
            else:
                # Root cause is dataset or algorithmic — propose a new family.
                proposal_prompt = _PROPOSAL_PROMPT.format(
                    root_cause=root_cause,
                    reasoning=reasoning,
                    objective_slug=policy.objective_slug,
                    trigger_reason=trigger_reason,
                )
                try:
                    response = self._call_operator(proposal_prompt)
                    parsed = self._parse_operator_response(response)
                    family = ProposedExperimentFamily(
                        family_id=f"fam-{uuid.uuid4().hex[:8]}",
                        name=str(parsed.get("name", "operator_proposed")),
                        description=str(parsed.get("description", "")),
                        config_delta=parsed.get("config_delta", {}) if isinstance(parsed.get("config_delta", {}), dict) else {},
                        rationale=str(parsed.get("rationale", "")),
                        proposed_by="operator",
                        feasibility_note=f"diagnosis={root_cause}: {reasoning}",
                    )
                    operator_available = True
                    operator_prompt_used = f"{diagnosis_prompt}\n\n---\n\n{proposal_prompt}"
                except Exception as exc:
                    family = self._rule_heuristic_proposal(policy, trigger_reason)
                    family = family.model_copy(
                        update={
                            "feasibility_note": f"operator_unavailable: {exc} (diagnosis={root_cause})",
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
