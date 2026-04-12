from __future__ import annotations

import argparse
import json
from pathlib import Path


SCENARIOS = (
    {
        "name": "equilibrium_lock",
        "governor_action": "continue",
        "governor_reasons": ["equilibrium_gate"],
        "strategy_family": "elastic_anchor",
        "has_fallback": False,
        "recovery_status": "completed",
        "last_fail_reason": None,
        "consecutive_fail_fast": 0,
        "anchor_path": "anchors/equilibrium_anchor.safetensors",
        "trace": [
            {"step": 1, "effective_dimensionality": 4.4, "dimensionality_ratio": 0.88, "equilibrium_fraction": 0.58, "equilibrium_gate": False, "drift_rho": 0.05, "training_loss": 0.46},
            {"step": 2, "effective_dimensionality": 4.6, "dimensionality_ratio": 0.82, "equilibrium_fraction": 0.71, "equilibrium_gate": True, "drift_rho": 0.04, "training_loss": 0.41},
            {"step": 3, "effective_dimensionality": 4.8, "dimensionality_ratio": 0.78, "equilibrium_fraction": 0.82, "equilibrium_gate": True, "drift_rho": 0.03, "training_loss": 0.39},
        ],
    },
    {
        "name": "stabilizing_climb",
        "governor_action": "continue",
        "governor_reasons": ["equilibrium_fraction_rising"],
        "strategy_family": "elastic_anchor",
        "has_fallback": False,
        "recovery_status": "completed",
        "last_fail_reason": None,
        "consecutive_fail_fast": 0,
        "anchor_path": "anchors/stabilizing_anchor.safetensors",
        "trace": [
            {"step": 1, "effective_dimensionality": 4.2, "dimensionality_ratio": 0.79, "equilibrium_fraction": 0.24, "equilibrium_gate": False, "drift_rho": 0.07, "training_loss": 0.52},
            {"step": 2, "effective_dimensionality": 4.4, "dimensionality_ratio": 0.73, "equilibrium_fraction": 0.41, "equilibrium_gate": False, "drift_rho": 0.06, "training_loss": 0.48},
            {"step": 3, "effective_dimensionality": 4.6, "dimensionality_ratio": 0.69, "equilibrium_fraction": 0.56, "equilibrium_gate": False, "drift_rho": 0.05, "training_loss": 0.44},
        ],
    },
    {
        "name": "quenching_collapse",
        "governor_action": "terminate",
        "governor_reasons": ["weight_drift_limit", "dimensionality_ratio_breach"],
        "strategy_family": "elastic_anchor",
        "has_fallback": True,
        "recovery_status": "fail_fast",
        "last_fail_reason": "weight_drift_limit",
        "consecutive_fail_fast": 3,
        "anchor_path": "anchors/quenching_anchor.safetensors",
        "trace": [
            {"step": 1, "effective_dimensionality": 4.9, "dimensionality_ratio": 0.74, "equilibrium_fraction": 0.47, "equilibrium_gate": False, "drift_rho": 0.09, "training_loss": 0.66},
            {"step": 2, "effective_dimensionality": 3.4, "dimensionality_ratio": 0.53, "equilibrium_fraction": 0.31, "equilibrium_gate": False, "drift_rho": 0.13, "training_loss": 0.64},
            {"step": 3, "effective_dimensionality": 2.7, "dimensionality_ratio": 0.46, "equilibrium_fraction": 0.18, "equilibrium_gate": False, "drift_rho": 0.16, "training_loss": 0.61},
        ],
    },
    {
        "name": "drift_limit_trip",
        "governor_action": "terminate",
        "governor_reasons": ["weight_drift_limit"],
        "strategy_family": "elastic_anchor",
        "has_fallback": True,
        "recovery_status": "fail_fast",
        "last_fail_reason": "weight_drift_limit",
        "consecutive_fail_fast": 2,
        "anchor_path": "anchors/drift_anchor.safetensors",
        "trace": [
            {"step": 1, "effective_dimensionality": 4.5, "dimensionality_ratio": 0.77, "equilibrium_fraction": 0.39, "equilibrium_gate": False, "drift_rho": 0.08, "training_loss": 0.55},
            {"step": 2, "effective_dimensionality": 4.4, "dimensionality_ratio": 0.75, "equilibrium_fraction": 0.36, "equilibrium_gate": False, "drift_rho": 0.12, "training_loss": 0.57},
            {"step": 3, "effective_dimensionality": 4.2, "dimensionality_ratio": 0.73, "equilibrium_fraction": 0.29, "equilibrium_gate": False, "drift_rho": 0.15, "training_loss": 0.6},
        ],
    },
    {
        "name": "calibration_decay",
        "governor_action": "continue",
        "governor_reasons": ["calibration_review_required"],
        "strategy_family": "elastic_anchor",
        "has_fallback": False,
        "recovery_status": "pivoted",
        "last_fail_reason": "calibration_ece_high",
        "consecutive_fail_fast": 1,
        "anchor_path": "anchors/calibration_anchor.safetensors",
        "trace": [
            {"step": 1, "effective_dimensionality": 4.7, "dimensionality_ratio": 0.81, "equilibrium_fraction": 0.63, "equilibrium_gate": True, "drift_rho": 0.05, "training_loss": 0.44},
            {"step": 2, "effective_dimensionality": 4.8, "dimensionality_ratio": 0.79, "equilibrium_fraction": 0.66, "equilibrium_gate": True, "drift_rho": 0.05, "training_loss": 0.43},
            {"step": 3, "effective_dimensionality": 4.8, "dimensionality_ratio": 0.78, "equilibrium_fraction": 0.64, "equilibrium_gate": True, "drift_rho": 0.06, "training_loss": 0.43},
        ],
    },
    {
        "name": "warmup_stall",
        "governor_action": "continue",
        "governor_reasons": ["statistical_warmup_incomplete"],
        "strategy_family": "elastic_anchor",
        "has_fallback": True,
        "recovery_status": "fail_fast",
        "last_fail_reason": "statistical_warmup_incomplete",
        "consecutive_fail_fast": 1,
        "anchor_path": "",
        "trace": [
            {"step": 1, "effective_dimensionality": 3.1, "dimensionality_ratio": 0.66, "equilibrium_fraction": 0.04, "equilibrium_gate": False, "drift_rho": 0.07, "training_loss": 0.69},
            {"step": 2, "effective_dimensionality": 3.2, "dimensionality_ratio": 0.64, "equilibrium_fraction": 0.07, "equilibrium_gate": False, "drift_rho": 0.08, "training_loss": 0.68},
            {"step": 3, "effective_dimensionality": 3.3, "dimensionality_ratio": 0.63, "equilibrium_fraction": 0.09, "equilibrium_gate": False, "drift_rho": 0.08, "training_loss": 0.67},
        ],
    },
    {
        "name": "equilibrium_backslide",
        "governor_action": "continue",
        "governor_reasons": ["equilibrium_backslide_detected"],
        "strategy_family": "elastic_anchor",
        "has_fallback": True,
        "recovery_status": "pivoted",
        "last_fail_reason": "equilibrium_backslide",
        "consecutive_fail_fast": 2,
        "anchor_path": "anchors/backslide_anchor.safetensors",
        "trace": [
            {"step": 1, "effective_dimensionality": 4.8, "dimensionality_ratio": 0.79, "equilibrium_fraction": 0.69, "equilibrium_gate": True, "drift_rho": 0.04, "training_loss": 0.42},
            {"step": 2, "effective_dimensionality": 4.4, "dimensionality_ratio": 0.73, "equilibrium_fraction": 0.51, "equilibrium_gate": False, "drift_rho": 0.09, "training_loss": 0.48},
            {"step": 3, "effective_dimensionality": 4.0, "dimensionality_ratio": 0.67, "equilibrium_fraction": 0.32, "equilibrium_gate": False, "drift_rho": 0.11, "training_loss": 0.54},
        ],
    },
    {
        "name": "anchor_reuse_resume",
        "governor_action": "continue",
        "governor_reasons": ["stable_anchor_reuse"],
        "strategy_family": "elastic_anchor",
        "has_fallback": False,
        "recovery_status": "completed",
        "last_fail_reason": None,
        "consecutive_fail_fast": 0,
        "anchor_path": "anchors/reuse_anchor.safetensors",
        "trace": [
            {"step": 1, "effective_dimensionality": 4.1, "dimensionality_ratio": 0.72, "equilibrium_fraction": 0.43, "equilibrium_gate": False, "drift_rho": 0.06, "training_loss": 0.51},
            {"step": 2, "effective_dimensionality": 4.4, "dimensionality_ratio": 0.70, "equilibrium_fraction": 0.58, "equilibrium_gate": False, "drift_rho": 0.05, "training_loss": 0.46},
            {"step": 3, "effective_dimensionality": 4.7, "dimensionality_ratio": 0.68, "equilibrium_fraction": 0.72, "equilibrium_gate": True, "drift_rho": 0.04, "training_loss": 0.42},
        ],
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic WS26 TCL-focused state.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trials-per-scenario", type=int, default=12)
    parser.add_argument("--prefix", default="ws26")
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _scenario_metric(base_metric: dict[str, float | bool], replica: int, *, trial_id: str) -> dict:
    metric = dict(base_metric)
    metric["trial_id"] = trial_id
    metric["entropy_sigma"] = round(0.0015 + (replica % 4) * 0.0004, 4)
    metric["grad_norm"] = round(0.42 + (replica % 5) * 0.06, 4)
    metric["gpu_temperature_c"] = 31.0 + (replica % 6)
    metric["gpu_power_w"] = 34.0 + (replica % 7) * 1.8
    metric["statistically_ready"] = metric.get("equilibrium_fraction", 0.0) >= 0.1 or metric.get("step", 0) >= 2
    return metric


def build_tcl_workspace(output_root: Path, *, trials_per_scenario: int, prefix: str) -> dict:
    workspace = output_root.resolve()
    state_dir = workspace / "tar_state"
    tar_runs_root = workspace / "tar_runs"
    for path in (state_dir, tar_runs_root):
        path.mkdir(parents=True, exist_ok=True)

    recovery_history: list[dict] = []
    trial_count = 0
    for scenario in SCENARIOS:
        for replica in range(trials_per_scenario):
            trial_count += 1
            trial_id = f"trial-{prefix}-{scenario['name']}-{replica:03d}"
            trial_dir = tar_runs_root / trial_id / "verification" / "control"
            trial_dir.mkdir(parents=True, exist_ok=True)

            metrics = [
                _scenario_metric(base_metric, replica, trial_id=trial_id)
                for base_metric in scenario["trace"]
            ]
            last_metric = metrics[-1]
            calibration_ece = round(0.07 + (replica % 4) * 0.03 + (0.06 if scenario["name"] == "calibration_decay" else 0.0), 4)
            calibration = {
                "ece": calibration_ece,
                "accuracy": round(0.41 + (replica % 5) * 0.05, 4),
                "mean_confidence": round(0.54 + (replica % 4) * 0.04, 4),
            }

            config = {
                "backend_id": "asc_text",
                "strategy_family": scenario["strategy_family"],
                "governor_thresholds": {
                    "max_quenching_loss": 0.8,
                    "min_dimensionality_ratio": 0.55,
                },
                "data_provenance": {
                    "research_grade": False,
                    "has_fallback": scenario["has_fallback"],
                },
                "backend_provenance": {
                    "required_metrics": [
                        "training_loss",
                        "effective_dimensionality",
                        "entropy_sigma",
                        "drift_rho",
                    ]
                },
            }
            summary = {
                "trial_id": trial_id,
                "strategy_family": scenario["strategy_family"],
                "governor_action": scenario["governor_action"],
                "governor_reasons": list(scenario["governor_reasons"]),
                "anchor_effective_dimensionality": round((last_metric["effective_dimensionality"] + 0.6), 4),
                "last_metrics": last_metric,
                "calibration": calibration,
            }
            _write_json(trial_dir / "config.json", config)
            _write_json(trial_dir / "payload_summary.json", summary)
            _write_jsonl(trial_dir / "thermo_metrics.jsonl", metrics)

            recovery_history.append(
                {
                    "trial_id": trial_id,
                    "status": scenario["recovery_status"],
                    "last_known_stable_hyperparameters": {"alpha": 0.04, "eta": 0.01} if scenario["recovery_status"] != "fail_fast" or replica % 2 == 0 else {},
                    "last_fail_reason": scenario["last_fail_reason"],
                    "last_fail_metrics": None if scenario["last_fail_reason"] is None else {"drift_rho": last_metric["drift_rho"], "dimensionality_ratio": last_metric["dimensionality_ratio"]},
                    "consecutive_fail_fast": scenario["consecutive_fail_fast"],
                    "last_strategy_family": scenario["strategy_family"],
                    "last_anchor_path": scenario["anchor_path"] or None,
                    "max_effective_dimensionality_achieved": round(max(metric["effective_dimensionality"] for metric in metrics), 4),
                }
            )

    _write_json(
        state_dir / "recovery.json",
        {
            "trial_id": f"trial-{prefix}-quenching-canonical",
            "status": "fail_fast",
            "last_known_stable_hyperparameters": {"alpha": 0.04, "eta": 0.01},
            "last_fail_reason": "weight_drift_limit",
            "last_fail_metrics": {"drift_rho": 0.16, "dimensionality_ratio": 0.46},
            "consecutive_fail_fast": 3,
            "last_strategy_family": "elastic_anchor",
            "last_anchor_path": "anchors/quenching_anchor.safetensors",
            "max_effective_dimensionality_achieved": 4.9,
        },
    )
    _write_jsonl(state_dir / "recovery_history.jsonl", recovery_history)
    _write_json(
        workspace / "ws26_tcl_campaign_manifest.json",
        {
            "prefix": prefix,
            "scenarios": [scenario["name"] for scenario in SCENARIOS],
            "trials_per_scenario": trials_per_scenario,
            "trial_count": trial_count,
            "recovery_records": len(recovery_history),
        },
    )
    return {
        "workspace": str(workspace),
        "tar_state": str(state_dir),
        "trial_count": trial_count,
        "recovery_records": len(recovery_history),
    }


def main() -> int:
    args = parse_args()
    summary = build_tcl_workspace(
        Path(args.output_root),
        trials_per_scenario=args.trials_per_scenario,
        prefix=args.prefix,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
