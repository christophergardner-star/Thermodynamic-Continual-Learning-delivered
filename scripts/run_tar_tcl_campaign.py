from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_tar_master_dataset import build_master_dataset
from generate_ws23_dataset_campaign import build_campaign_workspace
from generate_ws26_tcl_campaign import build_tcl_workspace
from merge_tar_dataset_releases import merge_dataset_releases
from run_ws31_benchmark_pack import run_pack
from tar_lab.multimodal_payloads import run_split_cifar10_benchmark
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import ContinualLearningBenchmarkConfig


PROBLEM_PROMPTS = [
    "Investigate thermodynamic governor stability under repeated task boundaries in continual learning.",
    "Compare anchor reuse versus anchor reset for reducing forgetting in class-incremental TCL.",
    "Diagnose dimensionality collapse signatures before catastrophic forgetting in split CIFAR experiments.",
    "Study calibration drift after plasticity-biased TCL schedules and identify safe interventions.",
    "Measure when validation-only benchmark evidence is too weak for strong TAR claims.",
    "Evaluate falsification pressure required before promoting a thermodynamic continual learning result.",
    "Investigate recovery policy after weight-drift-limit breaches in governed TCL runs.",
    "Prioritize continual learning experiments under tight VRAM and long-horizon evidence collection.",
]

TCL_TRACE_SEEDS = [42, 123, 456, 789, 1337, 2027, 3141, 4096]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def docker_is_healthy() -> bool:
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{json .ServerVersion}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception:
        return False
    return proc.returncode == 0 and proc.stdout.strip() not in {"", '""'}


def maybe_release_cuda() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class CampaignRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.workspace = Path(args.workspace).resolve()
        self.run_id = args.run_id
        campaign_root = Path(args.campaign_root)
        if not campaign_root.is_absolute():
            campaign_root = (self.workspace / campaign_root).resolve()
        self.campaign_root = campaign_root / self.run_id
        self.synthetic_root = self.campaign_root / "synthetic"
        self.dataset_root = self.campaign_root / "datasets"
        self.ws31_root = self.campaign_root / "ws31"
        self.tcl_root = self.campaign_root / "tcl"
        self.status_path = self.campaign_root / "status.json"
        self.metadata_path = self.campaign_root / "metadata.json"
        self.events_path = self.campaign_root / "events.jsonl"
        self.log_path = self.campaign_root / "campaign.log"
        self.campaign_root.mkdir(parents=True, exist_ok=True)
        self._configure_logging()
        self.logger = logging.getLogger("tar_tcl_campaign")

        os.environ.setdefault("TAR_TARGET_IMAGE_LOCKING", "host")
        os.environ.setdefault("PYTHONUNBUFFERED", "1")

        self.orchestrator = TAROrchestrator(workspace=str(self.workspace), start_memory_indexer=False)
        self.started_at = utc_now()
        self.ends_at = self.started_at + timedelta(hours=float(args.duration_hours))
        self.ws23_state_roots: list[Path] = []
        self.ws26_state_roots: list[Path] = []
        self.claimed_problem_ids: set[str] = set()
        self.cycle_index = 0
        self.tcl_seed_index = 0
        self.latest: dict[str, Any] = {}

        now_monotonic = time.monotonic()
        self.next_due = {
            "dataset_rebuild": now_monotonic,
            "ws31_pack": now_monotonic,
            "tcl_trace": now_monotonic,
            "mechanism_search": now_monotonic + self._hours_to_seconds(1.5),
            "baseline_comparison": now_monotonic + self._hours_to_seconds(3.0),
            "synthetic_increment": now_monotonic + self._hours_to_seconds(
                self.args.synthetic_increment_interval_hours
            ),
        }

    def _configure_logging(self) -> None:
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.handlers.clear()
        formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        root.addHandler(stream_handler)

    @staticmethod
    def _hours_to_seconds(hours: float) -> float:
        return max(0.0, float(hours) * 3600.0)

    @staticmethod
    def _minutes_to_seconds(minutes: float) -> float:
        return max(0.0, float(minutes) * 60.0)

    def write_status(self, *, state: str, last_error: str | None = None) -> None:
        payload = {
            "run_id": self.run_id,
            "pid": os.getpid(),
            "state": state,
            "started_at": self.started_at.replace(microsecond=0).isoformat(),
            "ends_at": self.ends_at.replace(microsecond=0).isoformat(),
            "workspace": str(self.workspace),
            "campaign_dir": str(self.campaign_root),
            "status_path": str(self.status_path),
            "log_path": str(self.log_path),
            "events_path": str(self.events_path),
            "cycles_completed": self.cycle_index,
            "docker_healthy": docker_is_healthy(),
            "ws23_state_roots": [str(path) for path in self.ws23_state_roots],
            "ws26_state_roots": [str(path) for path in self.ws26_state_roots],
            "latest": self.latest,
            "last_error": last_error,
            "updated_at": utc_now_iso(),
        }
        write_json(self.status_path, payload)

    def record_event(self, kind: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.events_path,
            {
                "timestamp": utc_now_iso(),
                "kind": kind,
                "payload": payload,
            },
        )

    def write_metadata(self) -> None:
        payload = {
            "run_id": self.run_id,
            "workspace": str(self.workspace),
            "campaign_dir": str(self.campaign_root),
            "started_at": self.started_at.replace(microsecond=0).isoformat(),
            "ends_at": self.ends_at.replace(microsecond=0).isoformat(),
            "duration_hours": float(self.args.duration_hours),
            "environment": {
                "TAR_STORAGE_ROOT": os.environ.get("TAR_STORAGE_ROOT"),
                "HF_HOME": os.environ.get("HF_HOME"),
                "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
                "TORCH_HOME": os.environ.get("TORCH_HOME"),
                "TEMP": os.environ.get("TEMP"),
                "TAR_TARGET_IMAGE_LOCKING": os.environ.get("TAR_TARGET_IMAGE_LOCKING"),
            },
            "intervals": {
                "poll_interval_s": float(self.args.poll_interval_s),
                "dataset_rebuild_interval_hours": float(self.args.dataset_rebuild_interval_hours),
                "ws31_pack_interval_hours": float(self.args.ws31_pack_interval_hours),
                "tcl_trace_interval_minutes": float(self.args.tcl_trace_interval_minutes),
                "mechanism_search_interval_hours": float(self.args.mechanism_search_interval_hours),
                "baseline_comparison_interval_hours": float(self.args.baseline_comparison_interval_hours),
                "synthetic_increment_interval_hours": float(self.args.synthetic_increment_interval_hours),
            },
            "initial_synthetic": {
                "ws23_projects_per_campaign": int(self.args.ws23_projects_per_campaign),
                "ws26_trials_per_scenario": int(self.args.ws26_trials_per_scenario),
            },
            "incremental_synthetic": {
                "ws23_projects_per_campaign": int(self.args.incremental_ws23_projects),
                "ws26_trials_per_scenario": int(self.args.incremental_ws26_trials),
            },
        }
        write_json(self.metadata_path, payload)

    def run(self) -> None:
        self.write_metadata()
        self.write_status(state="starting")
        self.logger.info("Starting TAR/TCL campaign %s", self.run_id)
        self.record_event("campaign_start", {"run_id": self.run_id})
        try:
            self._prepare_payload_environment()
            self._seed_synthetic_state(
                tag="seed",
                ws23_projects=int(self.args.ws23_projects_per_campaign),
                ws26_trials=int(self.args.ws26_trials_per_scenario),
            )
            self._rebuild_datasets(tag="seed")
            self._run_ws31_pack(tag="seed")
            self.write_status(state="running")

            while utc_now() < self.ends_at:
                cycle_started = time.monotonic()
                self._run_cycle()
                remaining = float(self.args.poll_interval_s) - (time.monotonic() - cycle_started)
                if remaining > 0:
                    time.sleep(remaining)

            self._rebuild_datasets(tag="final")
            self.write_status(state="completed")
            self.record_event("campaign_complete", {"run_id": self.run_id})
            self.logger.info("Campaign %s completed", self.run_id)
        except KeyboardInterrupt:
            self.write_status(state="interrupted", last_error="keyboard_interrupt")
            self.record_event("campaign_interrupted", {"run_id": self.run_id})
            self.logger.warning("Campaign %s interrupted", self.run_id)
            raise
        except Exception as exc:
            self.write_status(state="failed", last_error=str(exc))
            self.record_event(
                "campaign_failed",
                {
                    "run_id": self.run_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            self.logger.exception("Campaign %s failed", self.run_id)
            raise
        finally:
            self.orchestrator.shutdown()

    def _prepare_payload_environment(self) -> None:
        report = self.orchestrator.prepare_payload_environment()
        summary = {
            "reproducibility_complete": report.reproducibility_complete,
            "image_tag": report.image_tag,
            "build_status": report.build_status,
            "unresolved_packages": report.unresolved_packages,
            "lock_incomplete_reason": report.lock_incomplete_reason,
        }
        self.latest["payload_environment"] = summary
        self.record_event("prepare_payload_environment", summary)
        self.logger.info(
            "Prepared payload environment: reproducibility_complete=%s unresolved=%s",
            report.reproducibility_complete,
            len(report.unresolved_packages),
        )

    def _seed_synthetic_state(self, *, tag: str, ws23_projects: int, ws26_trials: int) -> None:
        ws23_root = self.synthetic_root / f"{tag}_ws23"
        ws26_root = self.synthetic_root / f"{tag}_ws26"
        ws23_result = build_campaign_workspace(
            ws23_root,
            projects_per_campaign=max(1, ws23_projects),
            prefix=f"{self.run_id}-{tag}-ws23",
        )
        ws26_result = build_tcl_workspace(
            ws26_root,
            trials_per_scenario=max(1, ws26_trials),
            prefix=f"{self.run_id}-{tag}-ws26",
        )
        self.ws23_state_roots.append(ws23_root)
        self.ws26_state_roots.append(ws26_root)
        payload = {
            "tag": tag,
            "ws23": ws23_result,
            "ws26": ws26_result,
        }
        self.latest["synthetic_state"] = payload
        self.record_event("seed_synthetic_state", payload)
        self.logger.info(
            "Generated synthetic state tag=%s ws23_projects=%s ws26_trials=%s",
            tag,
            ws23_result.get("projects"),
            ws26_result.get("trial_count", ws26_trials),
        )

    def _rebuild_datasets(self, *, tag: str) -> None:
        live_state = self.workspace / "tar_state"
        ws23_state_dirs = [live_state, *[root / "tar_state" for root in self.ws23_state_roots]]
        ws26_state_dirs = [live_state, *[root / "tar_state" for root in self.ws26_state_roots]]

        dataset_dir = self.dataset_root / tag
        ws23_output = dataset_dir / "ws23_release"
        ws26_output = dataset_dir / "ws26_release"
        merged_output = dataset_dir / "merged_release"

        ws23_manifest = build_master_dataset(
            ws23_state_dirs,
            ws23_output,
            version=f"{self.run_id}-ws23-{tag}",
        )
        ws26_manifest = build_master_dataset(
            ws26_state_dirs,
            ws26_output,
            version=f"{self.run_id}-ws26-{tag}",
        )
        merged_manifest = merge_dataset_releases(
            [ws23_output, ws26_output],
            merged_output,
            version=f"{self.run_id}-merged-{tag}",
        )
        payload = {
            "tag": tag,
            "ws23_release_dir": str(ws23_output),
            "ws23_records": ws23_manifest["records"],
            "ws26_release_dir": str(ws26_output),
            "ws26_records": ws26_manifest["records"],
            "merged_release_dir": str(merged_output),
            "merged_records": merged_manifest["records"],
        }
        self.latest["dataset_release"] = payload
        self.record_event("dataset_rebuild", payload)
        self.logger.info(
            "Rebuilt datasets tag=%s merged_records=%s",
            tag,
            merged_manifest["records"],
        )

    def _run_ws31_pack(self, *, tag: str) -> None:
        output_dir = self.ws31_root / tag
        manifest = run_pack(
            REPO_ROOT / "configs" / "ws31_benchmark_pack_v1.json",
            output_dir,
        )
        payload = {
            "tag": tag,
            "output_dir": str(output_dir),
            "suite_count": manifest["aggregate"]["suite_count"],
            "completed_suite_count": manifest["aggregate"]["completed_suite_count"],
            "total_elapsed_s": manifest["aggregate"]["total_elapsed_s"],
            "pod_justified_after_local_run": manifest["pod_decision"]["pod_justified_after_local_run"],
        }
        self.latest["ws31_pack"] = payload
        self.record_event("ws31_pack", payload)
        self.logger.info(
            "Ran WS31 pack tag=%s suites=%s elapsed_s=%s",
            tag,
            payload["suite_count"],
            payload["total_elapsed_s"],
        )

    def _ensure_curated_projects(self) -> list[str]:
        projects = self.orchestrator.store.list_research_projects()
        by_title = {project.title.strip().lower(): project.project_id for project in projects}
        created: list[str] = []
        for prompt in PROBLEM_PROMPTS:
            key = prompt.strip().lower()
            if key in by_title:
                continue
            project = self.orchestrator.create_project(prompt)
            by_title[key] = project.project_id
            created.append(project.project_id)
        if created:
            self.record_event("create_curated_projects", {"created_project_ids": created})
            self.logger.info("Created %s curated research projects", len(created))
        return created

    def _ingest_and_scan_frontier(self) -> None:
        try:
            ingest = self.orchestrator.ingest_research(
                topic=self.args.ingest_topic,
                max_results=int(self.args.max_ingest_results),
            )
            payload = {
                "topic": self.args.ingest_topic,
                "fetched": ingest.fetched,
                "indexed": ingest.indexed,
            }
            self.latest["last_ingest"] = payload
            self.record_event("ingest_research", payload)
            self.logger.info(
                "Ingested research topic=%s fetched=%s indexed=%s",
                self.args.ingest_topic,
                ingest.fetched,
                ingest.indexed,
            )
        except Exception as exc:
            self.record_event("ingest_research_error", {"error": str(exc)})
            self.logger.warning("Research ingest failed: %s", exc)

        try:
            scan = self.orchestrator.scan_frontier_gaps(
                topic=self.args.ingest_topic,
                max_gaps=4,
            )
            proposed = self.orchestrator.propose_projects_from_gaps(max_proposals=2)
            payload = {
                "scan_id": scan.scan_id,
                "gaps_identified": scan.gaps_identified,
                "gaps_rejected": scan.gaps_rejected,
                "projects_proposed": len(proposed),
            }
            self.latest["last_frontier_scan"] = payload
            self.record_event("scan_frontier_gaps", payload)
            self.logger.info(
                "Scanned frontier gaps identified=%s proposed=%s",
                scan.gaps_identified,
                len(proposed),
            )
        except Exception as exc:
            self.record_event("scan_frontier_gaps_error", {"error": str(exc)})
            self.logger.warning("Frontier scan failed: %s", exc)

    def _candidate_project_ids(self, portfolio_review: dict[str, Any]) -> list[str]:
        candidate_ids: list[str] = []
        for item in portfolio_review.get("top_projects", []):
            project_id = str(item.get("project_id") or "").strip()
            if project_id:
                candidate_ids.append(project_id)
        for project in self.orchestrator.store.list_research_projects():
            if project.status == "active":
                candidate_ids.append(project.project_id)
        seen: set[str] = set()
        ordered: list[str] = []
        for project_id in candidate_ids:
            if project_id in seen:
                continue
            seen.add(project_id)
            ordered.append(project_id)
        return ordered

    def _schedule_project_studies(self) -> list[dict[str, Any]]:
        portfolio_review = self.orchestrator.portfolio_review(limit=6, mode="balanced")
        self.latest["portfolio_review"] = {
            "selected_project_id": (
                portfolio_review.get("portfolio", {}).get("latest_selected_project_id")
            ),
            "top_projects": [
                item.get("project_id")
                for item in portfolio_review.get("top_projects", [])[:3]
            ],
        }
        self.record_event(
            "portfolio_review",
            {
                "selected_project_id": self.latest["portfolio_review"]["selected_project_id"],
                "top_projects": self.latest["portfolio_review"]["top_projects"],
            },
        )

        self.orchestrator.portfolio_decide(limit=6, mode="balanced")

        occupied = {
            entry.project_id
            for entry in self.orchestrator.store.iter_problem_schedules()
            if entry.status in {"scheduled", "leased", "running", "retry_wait"}
            and entry.project_id
        }

        scheduled: list[dict[str, Any]] = []
        for project_id in self._candidate_project_ids(portfolio_review):
            if len(scheduled) >= int(self.args.studies_per_cycle):
                break
            if project_id in occupied:
                continue
            project = self.orchestrator.store.get_research_project(project_id)
            if project is None or project.status != "active":
                continue
            report = self.orchestrator.study_problem(project.goal, project_id=project_id)
            entry = self.orchestrator.schedule_problem_study(
                problem_id=report.problem_id,
                use_docker=bool(self.args.use_docker_when_healthy and docker_is_healthy()),
                build_env=False,
                delay_s=0,
                max_runs=1,
            )
            summary = {
                "project_id": project_id,
                "problem_id": report.problem_id,
                "schedule_id": entry.schedule_id,
            }
            scheduled.append(summary)
            occupied.add(project_id)

        if scheduled:
            self.record_event("schedule_project_studies", {"scheduled": scheduled})
            self.logger.info("Scheduled %s new project studies", len(scheduled))
        return scheduled

    def _run_runtime_cycle(self) -> dict[str, Any]:
        heartbeat = self.orchestrator.run_runtime_cycle(max_jobs=int(self.args.max_jobs))
        payload = {
            "executed_jobs": heartbeat.executed_jobs,
            "failed_jobs": heartbeat.failed_jobs,
            "stale_cleanups": heartbeat.stale_cleanups,
            "notes": list(heartbeat.notes),
        }
        self.latest["runtime_cycle"] = payload
        self.record_event("runtime_cycle", payload)
        self.logger.info(
            "Runtime cycle executed_jobs=%s failed_jobs=%s",
            heartbeat.executed_jobs,
            heartbeat.failed_jobs,
        )
        return payload

    def _review_claims_and_plans(self) -> dict[str, Any]:
        verdicts_created = 0
        plans_updated = 0
        recent_executions = sorted(
            list(self.orchestrator.store.iter_problem_executions()),
            key=lambda item: item.executed_at,
            reverse=True,
        )
        for execution in recent_executions[:6]:
            if execution.problem_id in self.claimed_problem_ids:
                continue
            try:
                self.orchestrator.claim_verdict(problem_id=execution.problem_id)
                self.claimed_problem_ids.add(execution.problem_id)
                verdicts_created += 1
            except Exception as exc:
                self.record_event(
                    "claim_verdict_error",
                    {"problem_id": execution.problem_id, "error": str(exc)},
                )
                self.logger.warning(
                    "Claim verdict failed for problem_id=%s: %s",
                    execution.problem_id,
                    exc,
                )
                continue
            if execution.project_id:
                try:
                    self.orchestrator.generate_falsification_plan(execution.project_id)
                    plans_updated += 1
                except Exception as exc:
                    self.record_event(
                        "falsification_plan_error",
                        {"project_id": execution.project_id, "error": str(exc)},
                    )
                    self.logger.warning(
                        "Falsification plan failed for project_id=%s: %s",
                        execution.project_id,
                        exc,
                    )
        payload = {
            "claim_verdicts_created": verdicts_created,
            "falsification_plans_updated": plans_updated,
        }
        if verdicts_created or plans_updated:
            self.record_event("claim_and_falsification_review", payload)
            self.logger.info(
                "Claim review verdicts=%s falsification_plans=%s",
                verdicts_created,
                plans_updated,
            )
        return payload

    def _run_tcl_trace(self, *, tag: str) -> None:
        seed = TCL_TRACE_SEEDS[self.tcl_seed_index % len(TCL_TRACE_SEEDS)]
        self.tcl_seed_index += 1
        setting = "task_incremental" if self.cycle_index % 2 == 0 else "class_incremental"
        config = ContinualLearningBenchmarkConfig(
            seed=seed,
            setting=setting,
            n_tasks=2,
            train_epochs_per_task=1,
            batch_size=128,
        )
        try:
            result = run_split_cifar10_benchmark(
                config,
                method="tcl",
                workspace=str(self.workspace),
                backbone="tiny",
            )
            mean_forgetting = result.mean_forgetting
            final_mean_accuracy = result.final_mean_accuracy
            trace_path = result.thermodynamic_trace_path
        except (RuntimeError, ModuleNotFoundError) as exc:
            self.logger.warning("TCL benchmark skipped (missing dependency): %s", exc)
            mean_forgetting = None
            final_mean_accuracy = None
            trace_path = None
        payload = {
            "tag": tag,
            "seed": seed,
            "setting": setting,
            "mean_forgetting": mean_forgetting,
            "final_mean_accuracy": final_mean_accuracy,
            "trace_path": trace_path,
        }
        self.latest["last_tcl_trace"] = payload
        self.record_event("tcl_trace", payload)
        if final_mean_accuracy is not None:
            self.logger.info(
                "Ran TCL trace seed=%s setting=%s final_mean_accuracy=%.4f",
                seed,
                setting,
                final_mean_accuracy,
            )
        else:
            self.logger.info("Ran TCL trace seed=%s setting=%s (benchmark skipped)", seed, setting)
        maybe_release_cuda()

    def _run_tcl_mechanism_search(self, *, tag: str) -> None:
        try:
            result = self.orchestrator.run_tcl_class_incremental_mechanism_search(
                problem_id=f"{self.run_id}:{tag}",
                seeds=[42, 123],
                backbone="tiny",
                train_epochs_per_task=2,
            )
            payload = {
                "tag": tag,
                "search_id": result.search_id,
                "best_candidate_name": result.best_candidate_name,
                "publishability_status": result.publishability_status,
                "summary": result.summary,
            }
            self.logger.info(
                "Ran TCL mechanism search best=%s status=%s",
                result.best_candidate_name,
                result.publishability_status,
            )
        except (RuntimeError, ModuleNotFoundError, AttributeError, Exception) as exc:
            self.logger.warning("TCL mechanism search skipped (dependency error): %s", exc)
            import sys as _sys
            for _mod in list(_sys.modules):
                if "torchvision" in _mod:
                    _sys.modules.pop(_mod, None)
            payload = {
                "tag": tag,
                "search_id": None,
                "best_candidate_name": None,
                "publishability_status": "skipped",
                "summary": str(exc),
            }
        self.latest["last_mechanism_search"] = payload
        self.record_event("tcl_mechanism_search", payload)
        maybe_release_cuda()

    def _run_baseline_comparison(self, *, tag: str) -> None:
        projects = [
            project.project_id
            for project in self.orchestrator.store.list_research_projects()
            if project.status == "active"
        ]
        if not projects:
            self.logger.info("Skipping baseline comparison because no active projects exist")
            return
        try:
            plan = self.orchestrator.plan_baseline_comparison(projects[0], seeds=[42, 123, 456])
            result = self.orchestrator.run_baseline_comparison(plan)
            payload = {
                "tag": tag,
                "project_id": plan.project_id,
                "plan_id": plan.plan_id,
                "result_id": result.result_id,
                "tcl_is_significantly_better": result.tcl_is_significantly_better,
                "tcl_is_significantly_worse": result.tcl_is_significantly_worse,
                "assessment": result.honest_assessment,
            }
            self.logger.info(
                "Ran baseline comparison project_id=%s result_id=%s",
                plan.project_id,
                result.result_id,
            )
        except (RuntimeError, ModuleNotFoundError, AttributeError, Exception) as exc:
            self.logger.warning("Baseline comparison skipped (dependency error): %s", exc)
            import sys as _sys
            for _mod in list(_sys.modules):
                if "torchvision" in _mod:
                    _sys.modules.pop(_mod, None)
            payload = {
                "tag": tag,
                "project_id": projects[0],
                "plan_id": None,
                "result_id": None,
                "tcl_is_significantly_better": None,
                "tcl_is_significantly_worse": None,
                "assessment": f"skipped: {exc}",
            }
        self.latest["last_baseline_comparison"] = payload
        self.record_event("baseline_comparison", payload)
        maybe_release_cuda()

    def _run_periodic_tasks(self) -> None:
        now = time.monotonic()
        cycle_tag = f"cycle_{self.cycle_index:04d}"

        if now >= self.next_due["tcl_trace"]:
            self._run_tcl_trace(tag=cycle_tag)
            self.next_due["tcl_trace"] = now + self._minutes_to_seconds(
                self.args.tcl_trace_interval_minutes
            )

        if now >= self.next_due["ws31_pack"]:
            self._run_ws31_pack(tag=cycle_tag)
            self.next_due["ws31_pack"] = now + self._hours_to_seconds(
                self.args.ws31_pack_interval_hours
            )

        if now >= self.next_due["mechanism_search"]:
            self._run_tcl_mechanism_search(tag=cycle_tag)
            self.next_due["mechanism_search"] = now + self._hours_to_seconds(
                self.args.mechanism_search_interval_hours
            )

        if now >= self.next_due["baseline_comparison"]:
            self._run_baseline_comparison(tag=cycle_tag)
            self.next_due["baseline_comparison"] = now + self._hours_to_seconds(
                self.args.baseline_comparison_interval_hours
            )

        if now >= self.next_due["synthetic_increment"]:
            self._seed_synthetic_state(
                tag=cycle_tag,
                ws23_projects=int(self.args.incremental_ws23_projects),
                ws26_trials=int(self.args.incremental_ws26_trials),
            )
            self.next_due["synthetic_increment"] = now + self._hours_to_seconds(
                self.args.synthetic_increment_interval_hours
            )

        if now >= self.next_due["dataset_rebuild"]:
            self._rebuild_datasets(tag=cycle_tag)
            self.next_due["dataset_rebuild"] = now + self._hours_to_seconds(
                self.args.dataset_rebuild_interval_hours
            )

    def _run_cycle(self) -> None:
        cycle_tag = f"cycle_{self.cycle_index:04d}"
        self.logger.info("Starting %s", cycle_tag)

        self._ensure_curated_projects()
        if self.cycle_index % max(1, int(self.args.ingest_every_n)) == 0:
            self._ingest_and_scan_frontier()

        scheduled = self._schedule_project_studies()
        runtime = self._run_runtime_cycle()
        claims = self._review_claims_and_plans()
        self._run_periodic_tasks()

        queue_health = self.orchestrator.queue_health()
        cycle_summary = {
            "cycle_tag": cycle_tag,
            "scheduled_studies": len(scheduled),
            "runtime": runtime,
            "claims": claims,
            "queue_health": queue_health,
            "docker_healthy": docker_is_healthy(),
        }
        self.latest["last_cycle"] = cycle_summary
        self.record_event("cycle_summary", cycle_summary)
        self.write_status(state="running")
        self.logger.info(
            "Finished %s scheduled=%s executed=%s failed=%s",
            cycle_tag,
            len(scheduled),
            runtime["executed_jobs"],
            runtime["failed_jobs"],
        )
        self.cycle_index += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a long-horizon TAR/TCL host campaign with periodic dataset rebuilds."
    )
    parser.add_argument("--workspace", default=str(REPO_ROOT))
    parser.add_argument("--campaign-root", default="training_artifacts/campaigns")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--duration-hours", type=float, default=48.0)
    parser.add_argument("--poll-interval-s", type=float, default=180.0)
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--studies-per-cycle", type=int, default=2)
    parser.add_argument("--ingest-topic", default="thermodynamic continual learning benchmark honesty")
    parser.add_argument("--max-ingest-results", type=int, default=4)
    parser.add_argument("--ingest-every-n", type=int, default=6)
    parser.add_argument("--ws23-projects-per-campaign", type=int, default=80)
    parser.add_argument("--ws26-trials-per-scenario", type=int, default=24)
    parser.add_argument("--incremental-ws23-projects", type=int, default=24)
    parser.add_argument("--incremental-ws26-trials", type=int, default=8)
    parser.add_argument("--dataset-rebuild-interval-hours", type=float, default=4.0)
    parser.add_argument("--ws31-pack-interval-hours", type=float, default=6.0)
    parser.add_argument("--tcl-trace-interval-minutes", type=float, default=45.0)
    parser.add_argument("--mechanism-search-interval-hours", type=float, default=12.0)
    parser.add_argument("--baseline-comparison-interval-hours", type=float, default=18.0)
    parser.add_argument("--synthetic-increment-interval-hours", type=float, default=12.0)
    parser.add_argument("--use-docker-when-healthy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = CampaignRunner(args)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
