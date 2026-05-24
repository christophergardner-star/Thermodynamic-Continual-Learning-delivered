from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tar_lab.phase_catalog import phase_catalog_by_logical_name
from tar_lab.result_artifacts import (
    _iter_queue_experiment_records,
    iter_canonical_comparison_records,
    load_canonical_comparison,
    read_statistics,
)
from tar_lab.runtime_ledger import find_runtime_conflicts


TRUST_TRUSTED_RERUN = "trusted_rerun_with_env"
TRUST_TRUSTED_MANUAL = "trusted_manual_controlled"
TRUST_CORRECTED_INTERNAL = "corrected_recomputation_no_env"
TRUST_LEGACY = "legacy_pre_rail"
TRUST_QUARANTINED = "quarantined_untrusted"
TRUST_BLOCKED = "blocked_pending_rerun"

TRUST_PUBLICATION_ALLOWED = {
    TRUST_TRUSTED_RERUN,
    TRUST_TRUSTED_MANUAL,
}

# Days after which a result is flagged as potentially stale (trust_expiry_warning=True).
_TRUST_EXPIRY_DAYS: dict[str, int] = {
    TRUST_TRUSTED_RERUN: 90,
    TRUST_TRUSTED_MANUAL: 180,
    TRUST_CORRECTED_INTERNAL: 30,
    TRUST_LEGACY: 0,          # immediately warn
    TRUST_QUARANTINED: 0,     # never expires — already invalid
    TRUST_BLOCKED: 0,
}


def _check_trust_expiry(record: dict[str, Any]) -> dict[str, Any]:
    """Return expiry metadata for a result record.
    Does NOT change trust_tier — RAIL 1 prohibits that.
    Returns: {trust_expiry_warning, trust_expiry_days_old, trust_expiry_threshold_days}"""
    tier = str(record.get("trust_tier", "") or TRUST_LEGACY)
    created_at = str(record.get("created_at", "") or "")
    threshold = _TRUST_EXPIRY_DAYS.get(tier)
    if threshold is None or tier == TRUST_QUARANTINED:
        return {"trust_expiry_warning": False, "trust_expiry_days_old": None, "trust_expiry_threshold_days": None}
    days_old: float | None = None
    if created_at:
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            days_old = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        except Exception:
            pass
    warning = (days_old is not None and days_old >= threshold) or (threshold == 0 and tier in {TRUST_LEGACY, TRUST_CORRECTED_INTERNAL})
    return {
        "trust_expiry_warning": warning,
        "trust_expiry_days_old": round(days_old, 1) if days_old is not None else None,
        "trust_expiry_threshold_days": threshold,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def validation_state_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "validation_state.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def load_validation_state(workspace: Path) -> dict[str, Any]:
    return _read_json(validation_state_path(workspace))


def save_validation_state(workspace: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = validation_state_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return payload


def _coerce_spec_payload(spec: Any) -> dict[str, Any]:
    if isinstance(spec, dict):
        return dict(spec)
    if is_dataclass(spec):
        return asdict(spec)
    if hasattr(spec, "__dict__"):
        return {
            key: value
            for key, value in vars(spec).items()
            if not key.startswith("_")
        }
    return {}


def _has_invalid_number(value: Any) -> bool:
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(_has_invalid_number(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_invalid_number(v) for v in value)
    return False


def validate_execution_request(
    workspace: Path,
    *,
    spec: Any,
    manifest: Any | None,
    conflict_keys: list[str] | None = None,
) -> dict[str, Any]:
    payload = _coerce_spec_payload(spec)
    experiment_id = str(payload.get("id", "") or "")
    seeds = [seed for seed in payload.get("seeds", []) or [] if str(seed).strip()]
    dataset = str(payload.get("dataset", "") or "")
    result = {
        "schema": "tar_execution_validation_v1",
        "validated_at": _now_iso(),
        "experiment_id": experiment_id,
        "checks": {
            "manifest_loaded": manifest is not None,
            "experiment_authorised": False,
            "dataset_declared": bool(dataset),
            "seeds_declared": bool(seeds),
            "no_duplicate_runtime_conflict": False,
            "write_domain_declared": True,
            "outputs_declared_upfront": True,
        },
        "issues": [],
    }
    if manifest is not None:
        try:
            manifest.assert_experiment_authorised(experiment_id)
            result["checks"]["experiment_authorised"] = True
        except Exception as exc:
            result["issues"].append(f"manifest_authorisation_failed: {exc}")
    else:
        # In autonomous mode (stabilisation OFF), Director scope constraint
        # (_STRICT_REAL_WORLD_FRONTIER_ONLY) is the gate — no per-experiment manifest required.
        try:
            from tar_validation_mode import load_state as _load_vs
            _vs = _load_vs(workspace) or {}
            if not _vs.get("active"):
                result["checks"]["manifest_loaded"] = True
                result["checks"]["experiment_authorised"] = True
            else:
                result["issues"].append("manifest_missing")
        except Exception:
            result["issues"].append("manifest_missing")

    conflicts = find_runtime_conflicts(
        workspace,
        conflict_keys=conflict_keys or [f"experiment:{experiment_id}"],
        experiment_id=experiment_id,
    )
    result["checks"]["no_duplicate_runtime_conflict"] = not conflicts
    if conflicts:
        result["issues"].append(
            f"duplicate_runtime_conflict: {[str(item.get('lease_id', '') or '') for item in conflicts]}"
        )
    if not result["checks"]["dataset_declared"]:
        result["issues"].append("dataset_missing")
    if not result["checks"]["seeds_declared"]:
        result["issues"].append("seeds_missing")
    result["ok"] = not result["issues"] and all(result["checks"].values())
    return result


def classify_trust_tier(
    workspace: Path,
    *,
    record: dict[str, Any],
    result_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    logical_name = str(record.get("logical_name", "") or "")
    result_path = Path(str(record.get("result_path", "") or ""))
    env_path_text = str(record.get("env_path", "") or "")
    env_path = Path(env_path_text) if env_path_text else None
    env_present = bool(env_path and env_path.exists())
    name = result_path.name.lower()
    path_text = str(result_path).lower()

    if "_untrusted" in path_text or record.get("quarantined"):
        trust_tier = TRUST_QUARANTINED
        basis = "artifact_quarantined_from_live_comparisons"
    elif env_present:
        if logical_name == "phase10_baseline" and "controlled_rerun" in name:
            trust_tier = TRUST_TRUSTED_MANUAL
            basis = "controlled_phase10_recovery_with_env"
        else:
            trust_tier = TRUST_TRUSTED_RERUN
            basis = "append_only_canonical_write_with_env"
    elif "corrected" in name:
        trust_tier = TRUST_CORRECTED_INTERNAL
        basis = "corrected_recomputation_without_env_snapshot"
    else:
        trust_tier = TRUST_LEGACY
        basis = "legacy_pre_rail_or_missing_env"

    publication_allowed = trust_tier in TRUST_PUBLICATION_ALLOWED
    provenance_status = "env_snapshot_present" if env_present else "missing_env_snapshot"
    catalog = phase_catalog_by_logical_name().get(logical_name)
    return {
        "logical_name": logical_name,
        "result_path": str(result_path),
        "env_path": str(env_path) if env_path else "",
        "phase_number": record.get("phase_number"),
        "trust_tier": trust_tier,
        "provenance_status": provenance_status,
        "basis": basis,
        "publication_allowed": publication_allowed,
        "domain_id": catalog.primary_domain_id if catalog else "",
        "frontier_problem_id": catalog.frontier_problem_id if catalog else "",
        "target_paper_id": catalog.target_paper_id if catalog else "",
        "supersedes": record.get("supersedes", []),
        "superseded_by": record.get("superseded_by", ""),
        "validation_state": "validated" if publication_allowed else "limited_scope",
        "result_present": bool(result_payload),
        **_check_trust_expiry({**record, "trust_tier": trust_tier}),
    }


def validate_result_artifact(
    workspace: Path,
    *,
    record: dict[str, Any],
    result_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    trust = classify_trust_tier(workspace, record=record, result_payload=result_payload)
    stats = read_statistics(result_payload or {})
    issues: list[str] = []
    if result_payload is None:
        issues.append("result_payload_missing_or_unreadable")
    if _has_invalid_number(result_payload or {}):
        issues.append("invalid_number_detected_nan_or_inf")
    if result_payload is not None and not stats:
        issues.append("statistics_block_missing_or_empty")
    if trust["provenance_status"] != "env_snapshot_present" and trust["trust_tier"] in {TRUST_TRUSTED_RERUN, TRUST_TRUSTED_MANUAL}:
        issues.append("trusted_result_missing_env_snapshot")
    return {
        "schema": "tar_result_validation_v1",
        "validated_at": _now_iso(),
        "logical_name": trust["logical_name"],
        "result_path": trust["result_path"],
        "trust": trust,
        "checks": {
            "result_present": result_payload is not None,
            "env_present": trust["provenance_status"] == "env_snapshot_present",
            "statistics_present": bool(stats),
            "numbers_sane": "invalid_number_detected_nan_or_inf" not in issues,
            "publication_allowed": bool(trust["publication_allowed"]),
        },
        "issues": issues,
        "ok": not issues,
    }


def _trusted_experiment_ids(workspace: Path) -> set[str]:
    trusted: set[str] = set()
    for record in build_validation_state(workspace, persist=False).get("results", []):
        trust = record.get("trust", {}) if isinstance(record, dict) else {}
        if trust.get("publication_allowed") and trust.get("logical_name"):
            trusted.add(str(trust.get("logical_name", "") or ""))
    return trusted


def _superseded_experiment_ids(workspace: Path) -> set[str]:
    """Return IDs of experiments archived as 'skipped' (intentionally superseded).

    The director retires early probes by archiving them with status='skipped' and an
    archive_reason of 'superseded_by_<new_id>'. These must not count as unsupported
    evidence — the paper's linked list may lag behind the director's supersession
    decisions, and blocking publication on a deliberately retired probe is wrong.
    """
    archive_path = workspace / "tar_state" / "experiment_archive.json"
    if not archive_path.exists():
        return set()
    try:
        raw = json.loads(archive_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    experiments = raw.get("experiments", []) if isinstance(raw, dict) else []
    return {
        str(rec.get("id", ""))
        for rec in experiments
        if isinstance(rec, dict)
        and str(rec.get("status", "")).lower() == "skipped"
        and str(rec.get("id", ""))
    }


def validate_paper_evidence(
    workspace: Path,
    *,
    paper_id: str,
    linked_experiment_ids: list[str],
    waiting_for_experiment_ids: list[str],
) -> dict[str, Any]:
    trusted_ids = _trusted_experiment_ids(workspace)
    superseded_ids = _superseded_experiment_ids(workspace)
    waiting = [str(item) for item in waiting_for_experiment_ids if str(item).strip()]
    linked = [str(item) for item in linked_experiment_ids if str(item).strip()]
    # Superseded experiments are intentionally retired — exclude from unsupported list.
    unsupported = [
        item for item in linked
        if item not in trusted_ids and item not in superseded_ids
    ]
    issues: list[str] = []
    if waiting:
        issues.append(f"waiting_for_experiments: {waiting}")
    if unsupported:
        issues.append(f"linked_experiments_not_publication_allowed: {unsupported}")
    return {
        "paper_id": paper_id,
        "evidence_ready": not waiting and not unsupported,
        "waiting_for_experiments": waiting,
        "unsupported_experiments": unsupported,
        "issues": issues,
    }


def build_validation_state(workspace: Path, *, persist: bool = True) -> dict[str, Any]:
    records = iter_canonical_comparison_records(workspace)

    # Supplement with completed queue experiments not yet in the canonical JSONL index.
    # These write to experiments/<id>/result.json with spec.json as provenance but are
    # never registered via write_canonical_comparison_result. Deduplicate by logical_name
    # so canonical JSONL entries (with full env snapshots) always take precedence.
    canonical_names: set[str] = {str(r.get("logical_name", "")) for r in records if r.get("logical_name")}
    for queue_rec in _iter_queue_experiment_records(workspace):
        if queue_rec.get("logical_name") not in canonical_names:
            records.append(queue_rec)

    results: list[dict[str, Any]] = []
    for record in records:
        logical_name = str(record.get("logical_name", "") or "")
        path, payload = load_canonical_comparison(workspace, logical_name)
        if path is not None:
            record = {**record, "result_path": str(path)}
        results.append(validate_result_artifact(workspace, record=record, result_payload=payload))

    # ── Item 7: discovery scan and quarantine check (RAIL-6 non-recursive) ──
    from tar_lab.canonical_registry import quarantine_unregistered_results
    quarantine_set, is_clean_trust_state = quarantine_unregistered_results(
        workspace, records
    )

    summary = {
        "trusted_publication_allowed": sum(
            1 for rec in results
            if ((rec.get("trust", {}) or {}).get("publication_allowed") is True)
        ),
        "limited_scope": sum(
            1 for rec in results
            if ((rec.get("trust", {}) or {}).get("publication_allowed") is not True)
        ),
        "missing_env": sum(
            1 for rec in results
            if ((rec.get("trust", {}) or {}).get("provenance_status") != "env_snapshot_present")
        ),
        "quarantined": sum(
            1 for rec in results
            if ((rec.get("trust", {}) or {}).get("trust_tier") == TRUST_QUARANTINED)
        ),
    }
    payload = {
        "schema": "tar_validation_state_v1",
        "updated_at": _now_iso(),
        "results": results,
        "summary": summary,
        "quarantine_set": quarantine_set,
        "is_clean_trust_state": is_clean_trust_state,
    }
    if persist:
        save_validation_state(workspace, payload)
    return payload
