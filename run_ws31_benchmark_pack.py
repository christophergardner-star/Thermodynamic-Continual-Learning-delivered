from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tar_lab.science_exec import execute_study_payload
from tar_lab.science_profiles import ScienceProfileRegistry


REPO_ROOT = Path(__file__).resolve().parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_commit(repo_root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _find_template(profile: Any, template_id: str) -> Any:
    for template in profile.experiment_templates:
        if template.template_id == template_id:
            return template
    raise ValueError(f"template_id={template_id!r} is not defined for profile={profile.profile_id!r}")


def _suite_slug(index: int, suite: dict[str, Any]) -> str:
    return f"{index:02d}_{suite['profile_id']}__{suite['benchmark_id']}"


def _build_payload(
    *,
    pack_id: str,
    suite: dict[str, Any],
    suite_index: int,
    registry: ScienceProfileRegistry,
) -> tuple[dict[str, Any], Any, Any, Any]:
    profile = registry.get(str(suite["profile_id"]))
    template = _find_template(profile, str(suite["template_id"]))
    spec = registry.resolve_benchmark_suite(
        profile,
        str(suite["benchmark_family"]),
        tier=str(suite["tier"]),
    )
    availability = registry.benchmark_availability(spec)

    if spec.benchmark_id != suite["benchmark_id"]:
        raise ValueError(
            f"Frozen suite benchmark mismatch: expected {suite['benchmark_id']!r} "
            f"but registry resolved {spec.benchmark_id!r}"
        )
    if bool(suite.get("proxy_allowed", True)):
        raise ValueError(f"Frozen suite {spec.benchmark_id!r} is proxy-allowed; WS31 pack requires real execution.")
    if suite["tier"] == "canonical" and not availability.canonical_ready:
        raise ValueError(
            f"Frozen canonical suite {spec.benchmark_id!r} is not canonical-ready at availability time."
        )

    payload = {
        "problem_id": f"{pack_id}-{suite_index:02d}",
        "problem": f"Execute frozen WS31 benchmark pack suite {spec.benchmark_id}.",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "benchmark_tier": suite["tier"],
        "requested_benchmark": spec.benchmark_id,
        "canonical_only": bool(suite.get("canonical_comparable", False)),
        "no_proxy_benchmarks": True,
        "environment": {"validation_imports": list(profile.validation_imports)},
        "benchmark_availability": [availability.model_dump(mode="json")],
        "experiments": [
            {
                "template_id": template.template_id,
                "name": template.name,
                "benchmark": template.benchmark,
                "benchmark_tier": suite["tier"],
                "benchmark_spec": spec.model_dump(mode="json"),
                "benchmark_availability": availability.model_dump(mode="json"),
                "metrics": list(template.metrics),
                "parameter_grid": dict(template.parameter_grid),
                "success_criteria": list(template.success_criteria),
            }
        ],
    }
    return payload, profile, template, spec


def _suite_manifest(
    *,
    suite: dict[str, Any],
    payload_path: Path,
    report_path: Path,
    elapsed_s: float,
    report: Any,
) -> dict[str, Any]:
    statistical_summary = (
        report.benchmark_statistical_summary.model_dump(mode="json")
        if report.benchmark_statistical_summary is not None
        else None
    )
    experiment_summaries = []
    for experiment in report.experiments:
        experiment_summaries.append(
            {
                "template_id": experiment.template_id,
                "status": experiment.status,
                "execution_mode": experiment.execution_mode,
                "primary_metric": (
                    experiment.statistical_summary.primary_metric
                    if experiment.statistical_summary is not None
                    else None
                ),
                "statistical_summary": (
                    experiment.statistical_summary.model_dump(mode="json")
                    if experiment.statistical_summary is not None
                    else None
                ),
            }
        )

    return {
        "profile_id": suite["profile_id"],
        "benchmark_id": suite["benchmark_id"],
        "tier": suite["tier"],
        "template_id": suite["template_id"],
        "primary_metric": suite["primary_metric"],
        "required_seed_runs": suite["required_seed_runs"],
        "payload_path": str(payload_path),
        "payload_sha256": _sha256(payload_path),
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "elapsed_s": round(elapsed_s, 3),
        "status": report.status,
        "benchmark_alignment": report.benchmark_alignment,
        "proxy_benchmarks_used": report.proxy_benchmarks_used,
        "canonical_comparable": report.canonical_comparable,
        "benchmark_statistical_summary": statistical_summary,
        "experiment_summaries": experiment_summaries,
    }


def run_pack(pack_config_path: Path, output_dir: Path) -> dict[str, Any]:
    pack = _load_json(pack_config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = ScienceProfileRegistry(str(REPO_ROOT))
    suite_manifests: list[dict[str, Any]] = []
    runtime_assessment = pack.get("local_runtime_assessment") or pack.get("eligibility_assessment") or {}

    pack_started_at = _utc_now()
    total_started = time.perf_counter()
    for index, suite in enumerate(pack["suites"], start=1):
        suite_dir = output_dir / _suite_slug(index, suite)
        suite_dir.mkdir(parents=True, exist_ok=True)
        payload_path = suite_dir / "payload.json"
        report_path = suite_dir / "report.json"
        benchmark_manifest_path = suite_dir / "benchmark_manifest.json"

        payload, _profile, _template, _spec = _build_payload(
            pack_id=str(pack["pack_id"]),
            suite=suite,
            suite_index=index,
            registry=registry,
        )
        _write_json(payload_path, payload)
        started = time.perf_counter()
        report = execute_study_payload(payload, report_path)
        elapsed_s = time.perf_counter() - started
        manifest = _suite_manifest(
            suite=suite,
            payload_path=payload_path,
            report_path=report_path,
            elapsed_s=elapsed_s,
            report=report,
        )
        _write_json(benchmark_manifest_path, manifest)
        suite_manifests.append(
            {
                **manifest,
                "benchmark_manifest_path": str(benchmark_manifest_path),
            }
        )

    total_elapsed_s = time.perf_counter() - total_started
    ready_count = sum(
        1
        for item in suite_manifests
        if item["benchmark_statistical_summary"] is not None
        and item["benchmark_statistical_summary"]["statistically_ready"]
    )
    aggregate = {
        "suite_count": len(suite_manifests),
        "completed_suite_count": sum(1 for item in suite_manifests if item["status"] == "completed"),
        "statistically_ready_suite_count": ready_count,
        "all_statistically_ready": ready_count == len(suite_manifests),
        "all_non_proxy": all(not item["proxy_benchmarks_used"] for item in suite_manifests),
        "all_aligned": all(item["benchmark_alignment"] == "aligned" for item in suite_manifests),
        "total_elapsed_s": round(total_elapsed_s, 3),
    }
    pod_justified_now = bool(runtime_assessment.get("pod_justified_now"))
    if not pod_justified_now and aggregate["all_statistically_ready"]:
        pod_reason = "Frozen pack is statistically ready and still fits comfortably on local hardware under the WS31 scale criteria."
    elif runtime_assessment.get("reason"):
        pod_reason = str(runtime_assessment.get("reason"))
    elif not pod_justified_now:
        pod_reason = "Frozen pack still fits comfortably on local hardware under the WS31 scale criteria."
    else:
        pod_reason = "Frozen pack exceeded the local trigger and requires pod-scale execution."
    pod_decision = {
        "pod_justified_after_local_run": pod_justified_now,
        "reason": pod_reason,
    }
    manifest = {
        "pack_id": pack["pack_id"],
        "source_pack_config_path": str(pack_config_path),
        "source_pack_config_sha256": _sha256(pack_config_path),
        "executed_at": pack_started_at,
        "repo_commit": _current_commit(REPO_ROOT),
        "execution_mode": "local_only",
        "suite_manifests": suite_manifests,
        "aggregate": aggregate,
        "pod_decision": pod_decision,
    }
    _write_json(output_dir / "pack_manifest.json", manifest)
    _write_json(
        output_dir / "pack_summary.json",
        {
            "pack_id": manifest["pack_id"],
            "repo_commit": manifest["repo_commit"],
            "aggregate": aggregate,
            "pod_decision": pod_decision,
        },
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen WS31 benchmark pack locally.")
    parser.add_argument(
        "--pack-config",
        default=str(REPO_ROOT / "configs" / "ws31_benchmark_pack_v1.json"),
        help="Path to the frozen benchmark pack config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "eval_artifacts" / "ws31_validation_pack_v1"),
        help="Directory where payloads, reports, and manifests will be written.",
    )
    args = parser.parse_args()

    manifest = run_pack(Path(args.pack_config), Path(args.output_dir))
    print(json.dumps(manifest["aggregate"], indent=2))


if __name__ == "__main__":
    main()
