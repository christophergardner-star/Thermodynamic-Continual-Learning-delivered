from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Protocol

from tar_lab.eval_schemas import EvalAggregate, EvalFileFingerprint, EvalItem, EvalItemResult
from tar_lab.eval_scorers import (
    FAMILY_RUBRICS,
    aggregate_results,
    describe_rubrics,
    render_prediction_from_summary,
    score_prediction,
    scoring_target_for_family,
    suite_for_family,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_nonempty_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _fingerprint(path: Path, *, root: Path) -> EvalFileFingerprint:
    records = _count_nonempty_lines(path) if path.suffix == ".jsonl" else None
    return EvalFileFingerprint(
        path=_relative_or_name(path, root),
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        records=records,
    )


def _suite_names_for_family(task_family: str) -> list[str]:
    return sorted({"core", suite_for_family(task_family)})


def _evaluation_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    trimmed = list(messages)
    while trimmed and str(trimmed[-1].get("role", "")).strip() == "assistant":
        trimmed.pop()
    if not trimmed:
        raise ValueError("Evaluation items must retain at least one non-assistant prompt message.")
    return trimmed


def _lookup_key(value: Any, needle: str) -> Any:
    if isinstance(value, dict):
        if needle in value:
            return value[needle]
        for inner in value.values():
            found = _lookup_key(inner, needle)
            if found is not None:
                return found
    elif isinstance(value, list):
        for inner in value:
            found = _lookup_key(inner, needle)
            if found is not None:
                return found
    return None


def _heuristic_summary(item: EvalItem) -> dict[str, Any]:
    summary = {key: None for key in item.scoring_target}
    context = item.input_context
    for key in list(summary):
        found = _lookup_key(context, key)
        if found is not None:
            summary[key] = found

    if item.task_family == "benchmark_honesty":
        requested_tier = _lookup_key(context, "requested_benchmark_tier")
        canonical_comparable = _lookup_key(context, "canonical_comparable")
        truthful_statuses = _lookup_key(context, "benchmark_truth_statuses")
        if truthful_statuses is not None:
            summary["truthful_statuses"] = truthful_statuses
        if canonical_comparable is not None:
            summary["canonical_comparable"] = canonical_comparable
        if summary.get("benchmark_alignment") is None:
            if canonical_comparable is False and requested_tier == "canonical":
                summary["benchmark_alignment"] = "refused"
            elif canonical_comparable is True and requested_tier == "canonical":
                summary["benchmark_alignment"] = "canonical_ready"
    elif item.task_family == "reproducibility_refusal":
        complete = _lookup_key(context, "reproducibility_complete")
        unresolved = _lookup_key(context, "unresolved_packages") or []
        if complete is not None:
            summary["should_refuse_promotion"] = (not bool(complete)) or bool(unresolved)
        if summary.get("operator_language") is None and summary.get("should_refuse_promotion") is True:
            summary["operator_language"] = "manifest_lock_incomplete_refuse_or_downgrade"
    elif item.task_family == "sandbox_policy_reasoning":
        scope = _lookup_key(context, "write_scope")
        if summary.get("artifact_only_write_scope") is None and isinstance(scope, str):
            summary["artifact_only_write_scope"] = scope == "artifacts_only"
    elif item.task_family == "falsification_planning":
        trigger_reason = _lookup_key(context, "trigger_reason")
        if summary.get("test_kinds") in (None, []):
            summary["test_kinds"] = (
                ["contradiction_resolution"]
                if trigger_reason and "contradiction" in str(trigger_reason)
                else ["mechanism_ablation"]
            )
        if summary.get("test_count") is None:
            summary["test_count"] = len(summary.get("test_kinds", []))
        if summary.get("coverage_overall_sufficient") is None:
            summary["coverage_overall_sufficient"] = False
        if summary.get("overall_sufficient") is None:
            summary["overall_sufficient"] = False
    elif item.task_family == "verification_judgement":
        replication = _lookup_key(context, "replication_status")
        if isinstance(replication, dict):
            summary["replication_num_runs"] = replication.get("num_runs", summary.get("replication_num_runs"))
            summary["replication_stable"] = replication.get("stable", summary.get("replication_stable"))
    elif item.task_family == "portfolio_staleness_recovery":
        if summary.get("resume_candidate") is None:
            summary["resume_candidate"] = bool(_lookup_key(context, "resume_candidate"))
        if summary.get("closure_candidate") is None:
            summary["closure_candidate"] = bool(_lookup_key(context, "closure_candidate"))
    elif item.task_family == "tcl_failure_mode_classification":
        metrics = _lookup_key(context, "last_metrics") or {}
        reasons = set(_lookup_key(context, "governor_reasons") or [])
        drift_rho = float(metrics.get("drift_rho") or 0.0)
        dimensionality_ratio = float(metrics.get("dimensionality_ratio") or 0.0)
        training_loss = metrics.get("training_loss")
        calibration = _lookup_key(context, "calibration") or {}
        calibration_ece = calibration.get("ece")
        if dimensionality_ratio > 0.0 and training_loss is not None and float(training_loss) <= 0.80 and dimensionality_ratio < 0.55:
            summary.update(
                {
                    "failure_mode": "dimensionality_collapse",
                    "severity": "high",
                    "primary_signal": "dimensionality_ratio",
                    "claim_promotion_safe": False,
                }
            )
        elif drift_rho >= 0.12 or "weight_drift_limit" in reasons:
            summary.update(
                {
                    "failure_mode": "drift_instability",
                    "severity": "high",
                    "primary_signal": "drift_rho",
                    "claim_promotion_safe": False,
                }
            )
        elif calibration_ece is not None and float(calibration_ece) >= 0.16:
            summary.update(
                {
                    "failure_mode": "calibration_degradation",
                    "severity": "medium",
                    "primary_signal": "calibration_ece",
                    "claim_promotion_safe": False,
                }
            )
        else:
            summary.update(
                {
                    "failure_mode": "stable_control",
                    "severity": "low",
                    "primary_signal": "equilibrium_gate",
                    "claim_promotion_safe": bool(metrics.get("equilibrium_gate")),
                }
            )
    elif item.task_family == "tcl_anchor_policy_judgement":
        regime = str(_lookup_key(context, "regime") or "searching")
        reasons = set(_lookup_key(context, "governor_reasons") or [])
        anchor_dim = _lookup_key(context, "anchor_effective_dimensionality")
        last_metrics = _lookup_key(context, "last_metrics") or {}
        current_dim = last_metrics.get("effective_dimensionality")
        if not anchor_dim:
            summary.update(
                {
                    "anchor_policy": "anchor_absent_review",
                    "anchor_reuse_recommended": False,
                    "rationale_signals": ["anchor_missing"],
                }
            )
        elif regime == "warming_up":
            summary.update(
                {
                    "anchor_policy": "suppress_anchor_changes_during_warmup",
                    "anchor_reuse_recommended": False,
                    "rationale_signals": ["warmup_incomplete"],
                }
            )
        elif regime in {"equilibrium", "stabilizing"} and current_dim is not None and float(anchor_dim) >= float(current_dim):
            summary.update(
                {
                    "anchor_policy": "reuse_last_stable_anchor",
                    "anchor_reuse_recommended": True,
                    "rationale_signals": ["stable_regime", "anchor_capacity_available"],
                }
            )
        elif "weight_drift_limit" in reasons:
            summary.update(
                {
                    "anchor_policy": "reset_anchor_state",
                    "anchor_reuse_recommended": False,
                    "rationale_signals": ["quenching_or_drift_limit"],
                }
            )
        else:
            summary.update(
                {
                    "anchor_policy": "hold_anchor_constant",
                    "anchor_reuse_recommended": False,
                    "rationale_signals": ["monitor_anchor_pressure"],
                }
            )
    elif item.task_family == "tcl_intervention_selection":
        failure_mode = str(_lookup_key(context, "failure_mode") or "stable_control")
        has_fallback_data = bool(_lookup_key(context, "has_fallback_data"))
        regime = str(_lookup_key(context, "regime") or "searching")
        if failure_mode == "dimensionality_collapse":
            summary.update(
                {
                    "recommended_tcl_action": "reduce_anchor_pressure_and_run_short_recovery_probe",
                    "intervention_reason": "quenching_signature",
                    "claim_promotion_safe": False,
                }
            )
        elif failure_mode == "drift_instability":
            summary.update(
                {
                    "recommended_tcl_action": "tighten_governor_and_debug_drift_limit",
                    "intervention_reason": "drift_instability",
                    "claim_promotion_safe": False,
                }
            )
        elif failure_mode == "calibration_degradation":
            summary.update(
                {
                    "recommended_tcl_action": "run_calibration_repair_before_promotion",
                    "intervention_reason": "calibration_gap",
                    "claim_promotion_safe": False,
                }
            )
        elif regime == "equilibrium" and not has_fallback_data:
            summary.update(
                {
                    "recommended_tcl_action": "replicate_before_promotion",
                    "intervention_reason": "equilibrium_without_fallback",
                    "claim_promotion_safe": True,
                }
            )
        else:
            summary.update(
                {
                    "recommended_tcl_action": "continue_monitoring",
                    "intervention_reason": "no_high_risk_signal",
                    "claim_promotion_safe": False,
                }
            )
    elif item.task_family == "tcl_trace_anomaly_diagnosis":
        first_metric = _lookup_key(context, "first_metric") or {}
        last_metric = _lookup_key(context, "last_metric") or {}
        flags: list[str] = []
        first_drift = float(first_metric.get("drift_rho") or 0.0)
        last_drift = float(last_metric.get("drift_rho") or 0.0)
        first_ratio = float(first_metric.get("dimensionality_ratio") or 0.0)
        last_ratio = float(last_metric.get("dimensionality_ratio") or 0.0)
        positive_ratios = [value for value in (first_ratio, last_ratio) if value > 0.0]
        first_eq = float(first_metric.get("equilibrium_fraction") or 0.0)
        last_eq = float(last_metric.get("equilibrium_fraction") or 0.0)
        first_loss = float(first_metric.get("training_loss") or 0.0)
        last_loss = float(last_metric.get("training_loss") or 0.0)
        if max(first_drift, last_drift) >= 0.12:
            flags.append("drift_spike")
        if positive_ratios and min(positive_ratios) < 0.55 and max(first_loss, last_loss) <= 0.85:
            flags.append("quenching_signature")
        if first_eq >= 0.50 and last_eq <= first_eq - 0.25:
            flags.append("equilibrium_backslide")
        if first_loss > 0.0 and last_loss > first_loss * 1.15:
            flags.append("loss_instability")
        dominant = next((name for name in ("quenching_signature", "drift_spike", "equilibrium_backslide", "loss_instability") if name in flags), "none")
        summary.update(
            {
                "anomaly_present": bool(flags),
                "dominant_anomaly": dominant,
                "anomaly_flags": flags,
            }
        )
    elif item.task_family == "tcl_regime_transition_forecast":
        current_regime = str(_lookup_key(context, "current_regime") or "searching")
        anomaly = _lookup_key(context, "anomaly") or {}
        dominant_anomaly = str((anomaly or {}).get("dominant_anomaly") or "none")
        if dominant_anomaly in {"quenching_signature", "drift_spike"}:
            summary.update(
                {
                    "predicted_next_regime": "thermodynamic_quenching",
                    "intervention_urgency": "high",
                    "confidence_band": "high",
                }
            )
        elif current_regime == "warming_up":
            summary.update(
                {
                    "predicted_next_regime": "stabilizing",
                    "intervention_urgency": "medium",
                    "confidence_band": "medium",
                }
            )
        elif current_regime in {"searching", "stabilizing"}:
            summary.update(
                {
                    "predicted_next_regime": "equilibrium",
                    "intervention_urgency": "medium",
                    "confidence_band": "medium",
                }
            )
        else:
            summary.update(
                {
                    "predicted_next_regime": current_regime,
                    "intervention_urgency": "low",
                    "confidence_band": "medium",
                }
            )
    elif item.task_family == "tcl_recovery_confidence_estimation":
        status = str(_lookup_key(context, "status") or "unknown")
        consecutive_fail_fast = int(_lookup_key(context, "consecutive_fail_fast") or 0)
        stable = bool(_lookup_key(context, "last_known_stable_hyperparameters"))
        achieved_dim = float(_lookup_key(context, "max_effective_dimensionality_achieved") or 0.0)
        if status == "completed" and stable:
            summary.update(
                {
                    "recovery_outlook": "strong",
                    "resume_confidence_band": "high",
                    "requires_human_review": False,
                }
            )
        elif status == "fail_fast" and consecutive_fail_fast >= 3:
            summary.update(
                {
                    "recovery_outlook": "poor",
                    "resume_confidence_band": "low",
                    "requires_human_review": True,
                }
            )
        elif status == "pivoted":
            summary.update(
                {
                    "recovery_outlook": "guarded",
                    "resume_confidence_band": "medium",
                    "requires_human_review": True,
                }
            )
        elif stable and achieved_dim >= 7.0:
            summary.update(
                {
                    "recovery_outlook": "recoverable",
                    "resume_confidence_band": "medium",
                    "requires_human_review": False,
                }
            )
    elif item.task_family == "tcl_run_triage":
        status = str(_lookup_key(context, "status") or "unknown")
        consecutive_fail_fast = int(_lookup_key(context, "consecutive_fail_fast") or 0)
        recovery_confidence = _lookup_key(context, "recovery_confidence") or {}
        if status == "completed" and recovery_confidence.get("resume_confidence_band") == "high":
            summary.update(
                {
                    "operator_decision": "resume_controlled",
                    "urgency": "medium",
                    "human_review_required": False,
                }
            )
        elif status == "fail_fast" and consecutive_fail_fast >= 3:
            summary.update(
                {
                    "operator_decision": "pivot_or_terminate",
                    "urgency": "high",
                    "human_review_required": True,
                }
            )
        elif status == "fail_fast":
            summary.update(
                {
                    "operator_decision": "debug_before_resume",
                    "urgency": "high",
                    "human_review_required": True,
                }
            )
        elif status == "pivoted":
            summary.update(
                {
                    "operator_decision": "prepare_alternative_strategy",
                    "urgency": "medium",
                    "human_review_required": True,
                }
            )
    return summary


class Predictor(Protocol):
    def metadata(self) -> dict[str, Any]:
        ...

    def predict(self, item: EvalItem) -> str:
        ...


class GoldPredictor:
    def metadata(self) -> dict[str, Any]:
        return {"predictor_type": "gold"}

    def predict(self, item: EvalItem) -> str:
        return json.dumps(item.gold_target, sort_keys=True)


class HeuristicPredictor:
    def metadata(self) -> dict[str, Any]:
        return {"predictor_type": "heuristic"}

    def predict(self, item: EvalItem) -> str:
        summary = _heuristic_summary(item)
        return json.dumps(
            render_prediction_from_summary(item.task_family, summary),
            sort_keys=True,
        )


class StaticPredictor:
    def __init__(self, responses: dict[str, str] | Callable[[EvalItem], str]):
        self._responses = responses

    def metadata(self) -> dict[str, Any]:
        return {"predictor_type": "static"}

    def predict(self, item: EvalItem) -> str:
        if callable(self._responses):
            return self._responses(item)
        return self._responses[item.item_id]


class HFCausalLMPredictor:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        adapter_path: str | None,
        local_files_only: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        use_4bit: bool,
        bf16: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self._tokenizer.chat_template is None:
            raise ValueError("WS24 evaluation requires a chat-template tokenizer.")

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": False,
            "local_files_only": local_files_only,
        }
        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16 if bf16 else torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        if not use_4bit:
            model = model.to(self._device)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self._model = model.eval()
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._do_sample = temperature > 0.0
        self._device = next(model.parameters()).device
        self._model_name_or_path = model_name_or_path
        self._adapter_path = adapter_path

    def metadata(self) -> dict[str, Any]:
        return {
            "predictor_type": "hf_causal_lm",
            "model_name_or_path": self._model_name_or_path,
            "adapter_path": self._adapter_path,
            "max_new_tokens": self._max_new_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }

    def predict(self, item: EvalItem) -> str:
        with self._torch.no_grad():
            prompt = self._tokenizer.apply_chat_template(
                item.messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self._device)
            attention_mask = self._torch.ones_like(prompt, device=self._device)
            generated = self._model.generate(
                input_ids=prompt,
                attention_mask=attention_mask,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._do_sample,
                temperature=self._temperature if self._do_sample else None,
                top_p=self._top_p if self._do_sample else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            new_tokens = generated[0][prompt.shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class ASCCausalLMPredictor:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        tokenizer_name_or_path: str | None,
        local_files_only: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        bf16: bool,
    ) -> None:
        import torch
        from asc_model import ASCForCausalLM
        from transformers import AutoTokenizer

        model_path = Path(model_name_or_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"ASC checkpoint directory does not exist: {model_name_or_path}")
        asc_config_path = model_path / "asc_config.json"
        if not asc_config_path.exists():
            raise FileNotFoundError(
                f"ASC checkpoint is missing asc_config.json: {model_name_or_path}"
            )

        asc_config = _load_json(asc_config_path)
        tokenizer_source = tokenizer_name_or_path or self._infer_tokenizer_source(
            model_path=model_path,
            asc_config=asc_config,
        )

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self._tokenizer.chat_template is None:
            raise ValueError("WS27 ASC evaluation requires a chat-template tokenizer.")

        model = ASCForCausalLM.load(str(model_path))
        if torch.cuda.is_available():
            model = model.to(self._device)
        self._model = model.eval()
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._do_sample = temperature > 0.0
        self._device = next(model.parameters()).device
        self._model_name_or_path = str(model_path)
        self._tokenizer_name_or_path = str(tokenizer_source)
        self._asc_base_model_name = asc_config.get("base_model_name")
        self._bf16 = bf16

    @staticmethod
    def _infer_tokenizer_source(*, model_path: Path, asc_config: dict[str, Any]) -> str:
        tokenizer_markers = (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
        if any((model_path / name).exists() for name in tokenizer_markers):
            return str(model_path)
        base_model_name = asc_config.get("base_model_name")
        if isinstance(base_model_name, str) and base_model_name.strip():
            return base_model_name
        return str(model_path)

    def metadata(self) -> dict[str, Any]:
        return {
            "predictor_type": "asc_causal_lm",
            "model_name_or_path": self._model_name_or_path,
            "tokenizer_name_or_path": self._tokenizer_name_or_path,
            "asc_base_model_name": self._asc_base_model_name,
            "max_new_tokens": self._max_new_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "bf16": self._bf16,
        }

    def predict(self, item: EvalItem) -> str:
        with self._torch.no_grad():
            prompt = self._tokenizer.apply_chat_template(
                item.messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self._device)
            attention_mask = self._torch.ones_like(prompt, device=self._device)
            generated = self._model.generate(
                input_ids=prompt,
                attention_mask=attention_mask,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._do_sample,
                temperature=self._temperature if self._do_sample else None,
                top_p=self._top_p if self._do_sample else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            new_tokens = generated[0][prompt.shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_eval_pack(
    *,
    dataset_dir: Path,
    eval_pack_dir: Path,
    eval_version: str,
    test_split_file: str = "tar_master_dataset_test.jsonl",
    family_quotas: dict[str, int] | None = None,
    max_examples_per_family: int | None = None,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    eval_pack_dir = eval_pack_dir.resolve()
    eval_pack_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest = _load_json(dataset_dir / "manifest.json")
    test_path = dataset_dir / test_split_file
    test_examples = _load_jsonl(test_path)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in sorted(test_examples, key=lambda row: row["example_id"]):
        grouped[example["task_family"]].append(example)

    selected: list[EvalItem] = []
    for family, examples in sorted(grouped.items()):
        quota = family_quotas.get(family) if family_quotas else None
        if quota is None:
            quota = max_examples_per_family if max_examples_per_family is not None else len(examples)
        for example in examples[:quota]:
            selected.append(
                EvalItem(
                    item_id=f"{eval_version}:{example['example_id']}",
                    example_id=example["example_id"],
                    task_family=family,
                    task_name=example["task_name"],
                    suite_names=_suite_names_for_family(family),
                    lineage_key=str(example.get("lineage_key") or example["example_id"]),
                    source_kind=example["source_kind"],
                    source_id=str(example["provenance"]["source_id"]),
                    messages=_evaluation_messages(list(example["messages"])),
                    input_context=dict(example["input_context"]),
                    gold_target=dict(example["target"]),
                    scoring_target=scoring_target_for_family(family, dict(example["target"])),
                    provenance={
                        **dict(example["provenance"]),
                        "source_dataset_version": dataset_manifest.get("dataset_version"),
                        "source_dataset_dir": str(dataset_dir),
                    },
                    tags=list(example.get("tags") or []),
                )
            )

    selected = sorted(selected, key=lambda item: item.item_id)
    core_path = eval_pack_dir / "eval_core.jsonl"
    suite_paths = {
        "resume": eval_pack_dir / "eval_resume.jsonl",
        "honesty": eval_pack_dir / "eval_honesty.jsonl",
        "falsification": eval_pack_dir / "eval_falsification.jsonl",
        "portfolio": eval_pack_dir / "eval_portfolio.jsonl",
        "tcl": eval_pack_dir / "eval_tcl.jsonl",
    }
    rubric_path = eval_pack_dir / "scoring_rubrics.json"
    manifest_path = eval_pack_dir / "eval_manifest.json"

    _write_jsonl(core_path, [item.to_dict() for item in selected])
    for suite_name, suite_path in suite_paths.items():
        suite_items = [item.to_dict() for item in selected if suite_name in item.suite_names]
        _write_jsonl(suite_path, suite_items)
    _write_json(rubric_path, describe_rubrics())

    task_families = Counter(item.task_family for item in selected)
    suite_counts = Counter(
        suite_name
        for item in selected
        for suite_name in item.suite_names
        if suite_name != "core"
    )
    lineages = {item.lineage_key for item in selected}
    manifest = {
        "eval_version": eval_version,
        "source_dataset": {
            "dataset_version": dataset_manifest.get("dataset_version"),
            "dataset_dir": str(dataset_dir),
            "records": dataset_manifest.get("records"),
            "test_records": dataset_manifest.get("splits", {}).get("test"),
            "manifest_sha256": _sha256_file(dataset_dir / "manifest.json"),
            "test_sha256": _sha256_file(test_path),
        },
        "selection": {
            "family_quotas": dict(sorted((family_quotas or {}).items())),
            "max_examples_per_family": max_examples_per_family,
        },
        "items": len(selected),
        "task_families": dict(sorted(task_families.items())),
        "suites": dict(sorted(suite_counts.items())),
        "lineage_count": len(lineages),
        "files": {
            "core": _fingerprint(core_path, root=eval_pack_dir).to_dict(),
            "resume": _fingerprint(suite_paths["resume"], root=eval_pack_dir).to_dict(),
            "honesty": _fingerprint(suite_paths["honesty"], root=eval_pack_dir).to_dict(),
            "falsification": _fingerprint(suite_paths["falsification"], root=eval_pack_dir).to_dict(),
            "portfolio": _fingerprint(suite_paths["portfolio"], root=eval_pack_dir).to_dict(),
            "tcl": _fingerprint(suite_paths["tcl"], root=eval_pack_dir).to_dict(),
            "rubrics": _fingerprint(rubric_path, root=eval_pack_dir).to_dict(),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def load_eval_items(eval_pack_dir: Path) -> list[EvalItem]:
    eval_pack_dir = eval_pack_dir.resolve()
    items = _load_jsonl(eval_pack_dir / "eval_core.jsonl")
    return [EvalItem(**item) for item in items]


def aggregate_by_family(results: list[EvalItemResult]) -> dict[str, EvalAggregate]:
    grouped: dict[str, list[EvalItemResult]] = defaultdict(list)
    for result in results:
        grouped[result.task_family].append(result)
    return {family: aggregate_results(rows) for family, rows in sorted(grouped.items())}


def aggregate_by_suite(results: list[EvalItemResult]) -> dict[str, EvalAggregate]:
    grouped: dict[str, list[EvalItemResult]] = defaultdict(list)
    for result in results:
        for suite_name in result.suite_names:
            if suite_name == "core":
                continue
            grouped[suite_name].append(result)
    return {suite: aggregate_results(rows) for suite, rows in sorted(grouped.items())}


def evaluate_eval_pack(
    *,
    eval_pack_dir: Path,
    output_dir: Path,
    predictor: Predictor,
    max_items: int | None = None,
) -> dict[str, Any]:
    eval_pack_dir = eval_pack_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(eval_pack_dir / "eval_manifest.json")
    items = load_eval_items(eval_pack_dir)
    if max_items is not None:
        items = items[:max_items]

    results: list[EvalItemResult] = []
    for item in items:
        prediction_text = predictor.predict(item)
        results.append(
            score_prediction(
                item_id=item.item_id,
                example_id=item.example_id,
                task_family=item.task_family,
                suite_names=item.suite_names,
                gold_target=item.gold_target,
                prediction_text=prediction_text,
            )
        )

    overall = aggregate_results(results)
    family_breakdown = aggregate_by_family(results)
    suite_breakdown = aggregate_by_suite(results)

    results_path = output_dir / "results.jsonl"
    family_path = output_dir / "family_breakdown.json"
    suite_path = output_dir / "suite_breakdown.json"
    summary_path = output_dir / "results.json"
    errors_path = output_dir / "errors.jsonl"
    run_manifest_path = output_dir / "run_manifest.json"

    _write_jsonl(results_path, [result.to_dict() for result in results])
    _write_json(
        family_path,
        {family: aggregate.to_dict() for family, aggregate in family_breakdown.items()},
    )
    _write_json(
        suite_path,
        {suite: aggregate.to_dict() for suite, aggregate in suite_breakdown.items()},
    )
    _write_json(
        summary_path,
        {
            "eval_version": manifest.get("eval_version"),
            "items_evaluated": len(results),
            "overall": overall.to_dict(),
            "family_breakdown": {
                family: aggregate.to_dict() for family, aggregate in family_breakdown.items()
            },
            "suite_breakdown": {
                suite: aggregate.to_dict() for suite, aggregate in suite_breakdown.items()
            },
        },
    )
    _write_jsonl(
        errors_path,
        [
            result.to_dict()
            for result in results
            if result.error_bucket != "none"
        ],
    )

    run_manifest = {
        "timestamp": int(time.time()),
        "eval_pack": {
            "eval_version": manifest.get("eval_version"),
            "manifest_sha256": _sha256_file(eval_pack_dir / "eval_manifest.json"),
            "items": manifest.get("items"),
        },
        "predictor": predictor.metadata(),
        "items_evaluated": len(results),
        "overall": overall.to_dict(),
        "files": {
            "results": _fingerprint(results_path, root=output_dir).to_dict(),
            "family_breakdown": _fingerprint(family_path, root=output_dir).to_dict(),
            "suite_breakdown": _fingerprint(suite_path, root=output_dir).to_dict(),
            "summary": _fingerprint(summary_path, root=output_dir).to_dict(),
            "errors": _fingerprint(errors_path, root=output_dir).to_dict(),
        },
    }
    _write_json(run_manifest_path, run_manifest)
    return {
        "overall": overall.to_dict(),
        "family_breakdown": {family: agg.to_dict() for family, agg in family_breakdown.items()},
        "suite_breakdown": {suite: agg.to_dict() for suite, agg in suite_breakdown.items()},
        "run_manifest": run_manifest,
    }
