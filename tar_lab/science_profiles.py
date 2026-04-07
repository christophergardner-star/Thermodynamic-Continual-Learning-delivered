from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from tar_lab.schemas import (
    BenchmarkAvailability,
    BenchmarkSpec,
    BenchmarkTier,
    HypothesisRecord,
    ProblemExperimentPlan,
    ProblemResolutionReport,
    ProblemStudyReport,
    ScienceEnvironmentBundle,
    ScienceExperimentTemplate,
    ScienceProfile,
)


def _slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug or "problem")[:max_len].rstrip("-")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class ScienceProfileRegistry:
    def __init__(self, workspace: str = "."):
        self.workspace = Path(workspace).resolve()
        self.profile_dir = self.workspace / "science_profiles"
        if not self.profile_dir.exists():
            self.profile_dir = Path(__file__).resolve().parent.parent / "science_profiles"
        self.profiles = self._load_profiles()
        self._by_id = {profile.profile_id: profile for profile in self.profiles}
        if "generic_ml" not in self._by_id:
            raise RuntimeError("science_profiles/generic_ml.json is required")

    def _load_profiles(self) -> List[ScienceProfile]:
        if not self.profile_dir.exists():
            raise RuntimeError(f"science profile directory not found: {self.profile_dir}")
        profiles: List[ScienceProfile] = []
        for path in sorted(self.profile_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            profiles.append(ScienceProfile.model_validate(payload))
        if not profiles:
            raise RuntimeError("No science profiles were found")
        return profiles

    def get(self, profile_id: str) -> ScienceProfile:
        return self._by_id[profile_id]

    def list_benchmarks(
        self,
        *,
        profile_id: Optional[str] = None,
        tier: Optional[BenchmarkTier] = None,
    ) -> List[BenchmarkSpec]:
        profiles = [self.get(profile_id)] if profile_id else self.profiles
        suites: list[BenchmarkSpec] = []
        for profile in profiles:
            for suite in profile.benchmark_suites:
                if tier is None or suite.tier == tier:
                    suites.append(suite)
        return suites

    def benchmark_profile_counts(self) -> dict[str, int]:
        return {profile.profile_id: len(profile.benchmark_suites) for profile in self.profiles}

    def resolve_benchmark_suite(
        self,
        profile: ScienceProfile,
        benchmark_family: str,
        *,
        tier: BenchmarkTier = "validation",
        requested_benchmark: Optional[str] = None,
    ) -> BenchmarkSpec:
        requested = (requested_benchmark or "").strip().lower()
        if requested:
            for suite in profile.benchmark_suites:
                if suite.benchmark_id.lower() == requested and suite.family == benchmark_family:
                    return suite
        family_matches = [suite for suite in profile.benchmark_suites if suite.family == benchmark_family]
        if not family_matches:
            raise RuntimeError(f"No benchmark suite is registered for family '{benchmark_family}' in profile '{profile.profile_id}'")
        exact = [suite for suite in family_matches if suite.tier == tier]
        if exact:
            return exact[0]
        fallback_order: dict[BenchmarkTier, list[BenchmarkTier]] = {
            "canonical": ["canonical", "validation", "smoke"],
            "validation": ["validation", "smoke"],
            "smoke": ["smoke"],
        }
        for candidate_tier in fallback_order[tier]:
            for suite in family_matches:
                if suite.tier == candidate_tier:
                    return suite
        return family_matches[0]

    def benchmark_availability(self, suite: BenchmarkSpec) -> BenchmarkAvailability:
        imports_ready, missing_imports = _validate_modules(suite.required_imports)
        allow_download = os.environ.get("TAR_ALLOW_DATA_DOWNLOAD", "").strip() == "1"
        dataset_ready = False
        reason: Optional[str] = None
        dataset_target = suite.dataset_or_env.lower()
        if suite.tier == "smoke":
            dataset_ready = True
        elif dataset_target.startswith("local:") or dataset_target.startswith("builtin:"):
            dataset_ready = True
        elif suite.requires_download:
            dataset_ready = allow_download
            if not dataset_ready:
                reason = "dataset download not enabled"
        else:
            dataset_ready = True
        canonical_ready = imports_ready and dataset_ready and suite.canonical_comparable
        if reason is None and not imports_ready:
            reason = "missing imports"
        if reason is None and not dataset_ready:
            reason = "dataset or environment unavailable"
        if reason is None and suite.tier == "canonical" and not canonical_ready:
            reason = "canonical benchmark not ready"
        return BenchmarkAvailability(
            benchmark_id=suite.benchmark_id,
            tier=suite.tier,
            imports_ready=imports_ready,
            dataset_ready=dataset_ready,
            canonical_ready=canonical_ready,
            reason=reason,
            missing_imports=missing_imports,
        )

    def resolve_problem(
        self,
        problem: str,
        *,
        benchmark_tier: BenchmarkTier = "validation",
        requested_benchmark: Optional[str] = None,
    ) -> ProblemResolutionReport:
        prompt = problem.lower()
        best_profile = self.get("generic_ml")
        best_score = -1.0
        best_matches: List[str] = []

        for profile in self.profiles:
            score = 0.0
            matched: List[str] = []
            for keyword in profile.keywords:
                token = keyword.lower()
                if " " in token:
                    if token in prompt:
                        score += 2.5
                        matched.append(keyword)
                elif token and token in prompt:
                    score += 1.0
                    matched.append(keyword)

            if profile.domain == "quantum_ml":
                if any(
                    phrase in prompt
                    for phrase in (
                        "quantum",
                        "qml",
                        "barren plateau",
                        "barren landscapes",
                        "ansatz",
                        "variational circuit",
                        "pennylane",
                        "qiskit",
                    )
                ):
                    score += 5.0
            elif profile.domain == "reinforcement_learning" and any(
                phrase in prompt for phrase in ("policy gradient", "mdp", "ppo", "rl", "actor critic", "return")
            ):
                score += 3.0
            elif profile.domain == "computer_vision" and any(
                phrase in prompt for phrase in ("image", "vision", "segmentation", "detection", "cifar", "imagenet")
            ):
                score += 3.0
            elif profile.domain == "natural_language_processing" and any(
                phrase in prompt for phrase in ("language model", "token", "prompt", "nlp", "translation", "rag")
            ):
                score += 3.0

            if score > best_score:
                best_profile = profile
                best_score = score
                best_matches = matched

        confidence = min(0.99, max(0.15, 0.18 + 0.11 * len(best_matches) + 0.04 * best_score))
        return ProblemResolutionReport(
            problem=problem,
            profile_id=best_profile.profile_id,
            domain=best_profile.domain,
            confidence=confidence,
            matched_keywords=best_matches[:12],
            summary=best_profile.summary,
            benchmark_targets=best_profile.benchmark_targets,
            benchmark_catalog=best_profile.benchmark_suites,
            benchmark_tier=benchmark_tier,
            requested_benchmark=requested_benchmark,
            metric_hooks=best_profile.metric_hooks,
            validation_imports=best_profile.validation_imports,
        )


class ProblemResearchEngine:
    def __init__(self, workspace: str = ".", registry: Optional[ScienceProfileRegistry] = None):
        self.workspace = Path(workspace).resolve()
        self.registry = registry or ScienceProfileRegistry(workspace)
        self.env_root = self.workspace / "tar_state" / "science_envs"
        self.env_root.mkdir(parents=True, exist_ok=True)

    def prepare_environment(
        self,
        problem: str,
        resolution: Optional[ProblemResolutionReport] = None,
        *,
        benchmark_tier: BenchmarkTier = "validation",
        requested_benchmark: Optional[str] = None,
        canonical_only: bool = False,
        no_proxy_benchmarks: bool = False,
    ) -> ScienceEnvironmentBundle:
        resolution = resolution or self.registry.resolve_problem(
            problem,
            benchmark_tier=benchmark_tier,
            requested_benchmark=requested_benchmark,
        )
        profile = self.registry.get(resolution.profile_id)
        problem_id = f"{_slugify(problem)}-{_utc_stamp()}"
        bundle_dir = self.env_root / resolution.domain / problem_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

        profile_hash = hashlib.sha256(
            json.dumps(profile.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]

        requirements_path = bundle_dir / "requirements-profile.txt"
        dockerfile_path = bundle_dir / "Dockerfile"
        study_plan_path = bundle_dir / "study_plan.json"
        execution_report_path = bundle_dir / "execution_report.json"
        profile_path = bundle_dir / "profile.json"

        requirements = "\n".join(pkg.requirement_line() for pkg in profile.pip_packages) + "\n"
        requirements_path.write_text(requirements, encoding="utf-8")
        profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        placeholder_study = {
            "problem_id": problem_id,
            "problem": problem,
            "profile_id": profile.profile_id,
            "domain": profile.domain,
            "benchmark_tier": benchmark_tier,
            "requested_benchmark": requested_benchmark,
            "canonical_only": canonical_only,
            "no_proxy_benchmarks": no_proxy_benchmarks,
            "status": "environment_prepared",
            "experiments": [],
        }
        study_plan_path.write_text(json.dumps(placeholder_study, indent=2), encoding="utf-8")

        dockerfile = self._render_dockerfile(profile)
        dockerfile_path.write_text(dockerfile, encoding="utf-8")

        image_tag = f"tar-science-{resolution.domain}:{profile_hash}"
        build_command = [
            "docker",
            "build",
            "-t",
            image_tag,
            "-f",
            str(dockerfile_path),
            str(bundle_dir),
        ]
        run_command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.workspace}:/workspace",
            "-w",
            "/workspace",
            image_tag,
            "python",
            "-m",
            "tar_lab.problem_runner",
            "--study-plan",
            f"/workspace/{study_plan_path.relative_to(self.workspace).as_posix()}",
            "--output",
            f"/workspace/{execution_report_path.relative_to(self.workspace).as_posix()}",
        ]

        return ScienceEnvironmentBundle(
            problem_id=problem_id,
            problem=problem,
            profile_id=profile.profile_id,
            domain=profile.domain,
            profile_hash=profile_hash,
            docker_image_tag=image_tag,
            build_context_path=str(bundle_dir),
            dockerfile_path=str(dockerfile_path),
            requirements_path=str(requirements_path),
            study_plan_path=str(study_plan_path),
            execution_report_path=str(execution_report_path),
            build_command=build_command,
            run_command=run_command,
            pip_packages=[pkg.requirement_line() for pkg in profile.pip_packages],
            apt_packages=profile.apt_packages,
            validation_imports=profile.validation_imports,
            benchmark_tier=benchmark_tier,
            requested_benchmark=requested_benchmark,
            canonical_only=canonical_only,
            no_proxy_benchmarks=no_proxy_benchmarks,
        )

    def build_study_report(
        self,
        problem: str,
        resolution: ProblemResolutionReport,
        environment: ScienceEnvironmentBundle,
        research_ids: Iterable[str],
        memory_ids: Iterable[str],
        *,
        benchmark_tier: BenchmarkTier = "validation",
        requested_benchmark: Optional[str] = None,
        canonical_only: bool = False,
        no_proxy_benchmarks: bool = False,
    ) -> ProblemStudyReport:
        profile = self.registry.get(resolution.profile_id)
        hypotheses = [
            HypothesisRecord(
                hypothesis_id=f"{environment.problem_id}-hypothesis-{index + 1}",
                problem=problem,
                hypothesis=template.hypothesis,
                rationale=(
                    f"Generated from the '{template.name}' experiment template in profile "
                    f"'{profile.profile_id}'."
                ),
                confidence=max(0.2, min(0.9, resolution.confidence)),
                evidence_bundle_id=f"{environment.problem_id}-template-evidence",
                proposed_benchmark_ids=[],
                unresolved_assumptions=[
                    "This hypothesis is template-derived and should be refined with retrieved evidence."
                ],
            )
            for index, template in enumerate(profile.experiment_templates[:3])
        ]
        experiments: list[ProblemExperimentPlan] = []
        availability: list[BenchmarkAvailability] = []
        for template in profile.experiment_templates:
            suite = self.registry.resolve_benchmark_suite(
                profile,
                template.benchmark_family or template.benchmark,
                tier=benchmark_tier,
                requested_benchmark=requested_benchmark,
            )
            suite_availability = self.registry.benchmark_availability(suite)
            availability.append(suite_availability)
            experiments.append(
                ProblemExperimentPlan(
                    template_id=template.template_id,
                    name=template.name,
                    hypothesis=template.hypothesis,
                    benchmark=template.benchmark,
                    benchmark_tier=benchmark_tier,
                    benchmark_spec=suite,
                    benchmark_availability=suite_availability,
                    canonical_comparable=bool(suite.canonical_comparable and benchmark_tier == "canonical"),
                    metrics=template.metrics,
                    parameter_grid=template.parameter_grid,
                    success_criteria=template.success_criteria,
                )
            )
        availability = [item.benchmark_availability for item in experiments if item.benchmark_availability is not None]
        benchmark_ids = [item.benchmark_spec.benchmark_id for item in experiments if item.benchmark_spec is not None]
        benchmark_names = [item.benchmark_spec.name for item in experiments if item.benchmark_spec is not None]
        actual_benchmark_tiers = [item.benchmark_spec.tier for item in experiments if item.benchmark_spec is not None]
        for hypothesis in hypotheses:
            hypothesis.proposed_benchmark_ids = benchmark_ids[:3]
        payload = {
            "problem_id": environment.problem_id,
            "problem": problem,
            "profile_id": profile.profile_id,
            "domain": profile.domain,
            "resolution_confidence": resolution.confidence,
            "hypotheses": [item.model_dump(mode="json") for item in hypotheses],
            "benchmark_targets": profile.benchmark_targets,
            "benchmark_tier": benchmark_tier,
            "requested_benchmark": requested_benchmark,
            "canonical_only": canonical_only,
            "no_proxy_benchmarks": no_proxy_benchmarks,
            "canonical_comparable": bool(
                benchmark_tier == "canonical" and experiments and all(item.canonical_comparable for item in experiments)
            ),
            "benchmark_ids": benchmark_ids,
            "benchmark_names": benchmark_names,
            "actual_benchmark_tiers": actual_benchmark_tiers,
            "benchmark_availability": [item.model_dump(mode="json") for item in availability],
            "metric_hooks": profile.metric_hooks,
            "cited_research_ids": list(research_ids),
            "retrieved_memory_ids": list(memory_ids),
            "experiments": [item.model_dump(mode="json") for item in experiments],
            "environment": environment.model_dump(mode="json"),
            "next_action": self._next_action(profile, benchmark_tier=benchmark_tier),
            "status": "environment_ready" if environment.build_status in {"built", "dry_run"} else "planned",
        }
        Path(environment.study_plan_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ProblemStudyReport.model_validate(payload)

    @staticmethod
    def _next_action(profile: ScienceProfile, *, benchmark_tier: BenchmarkTier = "validation") -> str:
        top_template = profile.experiment_templates[0] if profile.experiment_templates else None
        if top_template is None:
            return "Build the environment, ingest papers, and draft the first named benchmark."
        return (
            f"Build the {profile.profile_id} environment, run the "
            f"'{top_template.name}' template, and verify the primary metrics "
            f"{', '.join(top_template.metrics[:3])} at benchmark_tier={benchmark_tier}."
        )

    @staticmethod
    def _render_dockerfile(profile: ScienceProfile) -> str:
        apt_line = ""
        if profile.apt_packages:
            apt_line = (
                "RUN apt-get update && apt-get install -y --no-install-recommends "
                + " ".join(profile.apt_packages)
                + " && rm -rf /var/lib/apt/lists/*\n"
            )
        return (
            f"FROM {profile.base_image}\n"
            "ENV PYTHONDONTWRITEBYTECODE=1\n"
            "ENV PYTHONUNBUFFERED=1\n"
            f"ENV TAR_SCIENCE_DOMAIN={profile.domain}\n"
            f"ENV TAR_PROFILE_ID={profile.profile_id}\n"
            + apt_line
            + "COPY requirements-profile.txt /tmp/requirements-profile.txt\n"
            + "RUN python -m pip install --no-cache-dir -r /tmp/requirements-profile.txt\n"
            + "WORKDIR /workspace\n"
        )


def _validate_modules(modules: List[str]) -> tuple[bool, List[str]]:
    missing: list[str] = []
    for module_name in modules:
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)
    return len(missing) == 0, missing
