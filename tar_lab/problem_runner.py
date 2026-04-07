from __future__ import annotations

import argparse
import json
from pathlib import Path

from tar_lab.science_exec import execute_study_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAR science-environment problem runner")
    parser.add_argument("--study-plan", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--benchmark-tier", choices=["smoke", "validation", "canonical"], default=None)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--no-proxy-benchmarks", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    study_plan_path = Path(args.study_plan)
    payload = json.loads(study_plan_path.read_text(encoding="utf-8"))
    if args.benchmark_tier:
        payload["benchmark_tier"] = args.benchmark_tier
    if args.benchmark:
        payload["requested_benchmark"] = args.benchmark
    if args.canonical_only:
        payload["canonical_only"] = True
    if args.no_proxy_benchmarks:
        payload["no_proxy_benchmarks"] = True
    output_path = Path(args.output) if args.output else study_plan_path.with_name("environment_probe.json")
    report = execute_study_payload(payload, output_path)
    print(report.model_dump_json(indent=2))
    return 0 if report.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
