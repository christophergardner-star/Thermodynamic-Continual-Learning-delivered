#!/usr/bin/env python3
"""
TAR Health Check — Phase 6.7
Run:  python tar_health_check.py
      python tar_health_check.py --json   (machine-readable output)
Output: tar_state/health_report.json + terminal summary.
Exit code: 0 if all pass/warn/skip, 1 if any fail.
"""
from __future__ import annotations
import argparse, ast, json, os, sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent
_WS   = _REPO  # workspace is repo root for this project


@dataclass
class CheckResult:
    name: str
    status: str  # pass | fail | warn | skip
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class HealthReport:
    checks: list[CheckResult] = field(default_factory=list)
    generated_at: str = ""
    passed: int = 0
    failed: int = 0
    warned: int = 0
    skipped: int = 0


class HealthChecker:
    def __init__(self, workspace: Path) -> None:
        self.ws = workspace

    # ── helpers ──────────────────────────────────────────────────────────────

    def _jload(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None

    # ── checks ───────────────────────────────────────────────────────────────

    def check_governance_files_present(self) -> CheckResult:
        name = "governance_files_present"
        required = [
            _REPO / "tar_lab" / "human_review.py",
            _REPO / "tar_scheduler.py",
            _REPO / "tar_research_director.py",
            _REPO / "tar_experiment_orchestrator.py",
            _REPO / "tar_lab" / "llm_bridge.py",
        ]
        missing = []
        syntax_errors = []
        for p in required:
            if not p.exists():
                missing.append(p.name)
            else:
                try:
                    ast.parse(p.read_text(encoding="utf-8", errors="replace"))
                except SyntaxError as exc:
                    syntax_errors.append(f"{p.name}: {exc}")
        if missing or syntax_errors:
            detail: dict = {}
            if missing:
                detail["missing"] = missing
            if syntax_errors:
                detail["syntax_errors"] = syntax_errors
            return CheckResult(name, "fail",
                               f"{len(missing)} missing, {len(syntax_errors)} syntax errors",
                               detail)
        return CheckResult(name, "pass", "All 5 governance files present and parse cleanly")

    def check_evidence_inventory_coherence(self) -> CheckResult:
        name = "evidence_inventory_coherence"
        inv_path = self.ws / "tar_state" / "honest_evidence_inventory.json"
        if not inv_path.exists():
            return CheckResult(name, "skip", "honest_evidence_inventory.json not found")
        data = self._jload(inv_path)
        if not isinstance(data, (dict, list)):
            return CheckResult(name, "fail", "Cannot parse honest_evidence_inventory.json")
        results = data.get("results", data) if isinstance(data, dict) else data
        if not isinstance(results, list):
            return CheckResult(name, "fail", "Results key is not a list")
        violations = []
        for r in results:
            if not isinstance(r, dict):
                continue
            eid = r.get("experiment_id", "(unknown)")
            if "experiment_id" not in r:
                violations.append(f"missing experiment_id: {r}")
            if "honest_verdict" not in r:
                violations.append(f"{eid}: missing honest_verdict")
            verdict = str(r.get("honest_verdict", "") or "").upper()
            if verdict == "PUBLICATION_ALLOWED":
                n = r.get("n_seeds", 0)
                if int(n or 0) < 5:
                    violations.append(f"{eid}: PUBLICATION_ALLOWED but n_seeds={n} (< 5)")
            if str(r.get("honest_verdict", "") or "").upper() == "EXPLORATION_GRADE":
                if r.get("paper_citation_allowed") is True:
                    violations.append(f"{eid}: EXPLORATION_GRADE but paper_citation_allowed=true")
        if violations:
            return CheckResult(name, "fail",
                               f"{len(violations)} coherence violation(s)",
                               {"violations": violations[:20]})
        return CheckResult(name, "pass",
                           f"Evidence inventory coherent ({len(results)} entries checked)")

    def check_canonical_result_paths(self) -> CheckResult:
        name = "canonical_result_paths"
        inv_path = self.ws / "tar_state" / "honest_evidence_inventory.json"
        if not inv_path.exists():
            return CheckResult(name, "skip", "honest_evidence_inventory.json not found")
        data = self._jload(inv_path)
        if not isinstance(data, (dict, list)):
            return CheckResult(name, "skip", "Cannot parse inventory")
        results = data.get("results", data) if isinstance(data, dict) else data
        if not isinstance(results, list):
            return CheckResult(name, "skip", "No results list")
        missing = []
        for r in results:
            if not isinstance(r, dict):
                continue
            rpath = r.get("result_path")
            if rpath and r.get("paper_citation_allowed") is True:
                p = Path(str(rpath))
                if not p.is_absolute():
                    p = self.ws / p
                if not p.exists():
                    missing.append(str(rpath))
        if missing:
            return CheckResult(name, "warn",
                               f"{len(missing)} citation-allowed result paths missing",
                               {"missing": missing[:20]})
        return CheckResult(name, "pass", "All citation-allowed result paths exist")

    def check_active_session(self) -> CheckResult:
        name = "active_session"
        sess_path = self.ws / "tar_state" / "active_session.json"
        if not sess_path.exists():
            return CheckResult(name, "warn",
                               "active_session.json not found — session may not have started")
        data = self._jload(sess_path)
        if not isinstance(data, dict):
            return CheckResult(name, "warn", "active_session.json is malformed")
        status = str(data.get("status", "") or "")
        if status == "DORMANT_NO_MANIFEST":
            try:
                import psutil
                python_procs = [p for p in psutil.process_iter(["name", "cmdline"])
                                if "python" in (p.info.get("name") or "").lower()]
                if python_procs:
                    return CheckResult(name, "warn",
                                       "active_session=DORMANT_NO_MANIFEST but Python processes running",
                                       {"python_process_count": len(python_procs)})
            except Exception:
                pass
            return CheckResult(name, "warn", "Session is DORMANT_NO_MANIFEST")
        return CheckResult(name, "pass", f"Session status: {status}")

    def check_gpu_temperature(self) -> CheckResult:
        name = "gpu_temperature"
        hw_path = self.ws / "tar_state" / "hardware_state.json"
        if not hw_path.exists():
            return CheckResult(name, "skip", "hardware_state.json not found")
        data = self._jload(hw_path)
        if not isinstance(data, dict):
            return CheckResult(name, "skip", "hardware_state.json malformed")
        try:
            age_s = datetime.now().timestamp() - hw_path.stat().st_mtime
        except Exception:
            age_s = 0.0
        if age_s > 120:
            return CheckResult(name, "warn",
                               f"hardware_state.json is {int(age_s)}s old (> 120s)",
                               {"age_s": age_s})
        gpu = data.get("gpu") or {}
        temp = gpu.get("temperature_c")
        if temp is None:
            return CheckResult(name, "skip", "No GPU temperature data")
        temp = float(temp)
        if temp > 85:
            return CheckResult(name, "fail", f"GPU temperature critical: {temp:.0f}°C (> 85°C)",
                               {"temperature_c": temp})
        if temp > 75:
            return CheckResult(name, "warn", f"GPU temperature high: {temp:.0f}°C (> 75°C)",
                               {"temperature_c": temp})
        return CheckResult(name, "pass", f"GPU temperature OK: {temp:.0f}°C")

    def check_api_key_configured(self) -> CheckResult:
        name = "api_key_configured"
        try:
            sys.path.insert(0, str(_REPO))
            from tar_lab.llm_bridge import _api_key  # type: ignore[attr-defined]
            key = _api_key()
        except ImportError:
            return CheckResult(name, "skip", "tar_lab.llm_bridge not importable")
        except Exception as exc:
            return CheckResult(name, "warn", f"Could not call _api_key(): {exc}")
        if not key:
            return CheckResult(name, "fail", "API key is None/empty")
        if not str(key).startswith("sk-ant-"):
            return CheckResult(name, "warn",
                               "API key present but does not start with 'sk-ant-'")
        return CheckResult(name, "pass", "API key configured and looks valid")

    def check_orphan_processes(self) -> CheckResult:
        name = "orphan_processes"
        queue_path = self.ws / "tar_state" / "experiment_queue.json"
        if not queue_path.exists():
            return CheckResult(name, "skip", "experiment_queue.json not found")
        data = self._jload(queue_path)
        if not isinstance(data, dict):
            return CheckResult(name, "skip", "experiment_queue.json malformed")
        experiments = data.get("experiments", [])
        if not isinstance(experiments, list):
            return CheckResult(name, "skip", "No experiments list")
        running = [e for e in experiments
                   if isinstance(e, dict)
                   and str(e.get("stage") or e.get("status") or "") == "running"]
        if not running:
            return CheckResult(name, "pass", "No running experiments in queue")
        try:
            import psutil
        except ImportError:
            return CheckResult(name, "skip", "psutil unavailable — cannot check PIDs")
        orphans = []
        for exp in running:
            pid = exp.get("pid")
            if not pid:
                continue
            try:
                if not psutil.pid_exists(int(pid)):
                    orphans.append({"id": exp.get("id"), "pid": pid})
            except Exception:
                pass
        if orphans:
            return CheckResult(name, "warn",
                               f"{len(orphans)} running experiment(s) have dead PIDs",
                               {"orphans": orphans})
        return CheckResult(name, "pass",
                           f"{len(running)} running experiment(s), all PIDs alive")

    def check_stale_leases(self) -> CheckResult:
        name = "stale_leases"
        tar_state = self.ws / "tar_state"
        lease_files = list(tar_state.glob("*.lease")) + list(tar_state.glob("runtime_lease*.json"))
        if not lease_files:
            return CheckResult(name, "skip", "No lease files found")
        stale = []
        now = datetime.now().timestamp()
        for lf in lease_files:
            try:
                age_h = (now - lf.stat().st_mtime) / 3600.0
                if age_h > 24:
                    stale.append({"file": lf.name, "age_h": round(age_h, 1)})
            except Exception:
                pass
        if stale:
            return CheckResult(name, "warn",
                               f"{len(stale)} lease file(s) older than 24h",
                               {"stale": stale})
        return CheckResult(name, "pass", f"{len(lease_files)} lease file(s), all fresh")

    def check_queue_schema(self) -> CheckResult:
        name = "queue_schema"
        queue_path = self.ws / "tar_state" / "experiment_queue.json"
        if not queue_path.exists():
            return CheckResult(name, "warn", "experiment_queue.json not found")
        data = self._jload(queue_path)
        if not isinstance(data, dict):
            return CheckResult(name, "fail", "experiment_queue.json is not a JSON object")
        if "experiments" not in data:
            return CheckResult(name, "fail", "experiment_queue.json missing 'experiments' key")
        experiments = data["experiments"]
        if not isinstance(experiments, list):
            return CheckResult(name, "fail", "'experiments' is not a list")
        bad = []
        for i, e in enumerate(experiments):
            if not isinstance(e, dict):
                bad.append(f"index {i}: not a dict")
                continue
            if "id" not in e:
                bad.append(f"index {i}: missing 'id'")
            if "stage" not in e and "status" not in e:
                bad.append(f"index {i}: missing both 'stage' and 'status'")
        if bad:
            return CheckResult(name, "fail",
                               f"{len(bad)} malformed experiment(s) in queue",
                               {"bad_entries": bad[:10]})
        return CheckResult(name, "pass",
                           f"Queue schema valid ({len(experiments)} experiments)")

    def check_director_proposals_parseable(self) -> CheckResult:
        name = "director_proposals_parseable"
        dp_path = self.ws / "tar_state" / "director_proposals.json"
        if not dp_path.exists():
            return CheckResult(name, "skip", "director_proposals.json not found")
        data = self._jload(dp_path)
        if not isinstance(data, list):
            return CheckResult(name, "fail", "director_proposals.json is not a JSON array")
        required_keys = {"experiment_id", "status", "proposed_at", "auto_approve_at"}
        bad = []
        for i, p in enumerate(data):
            if not isinstance(p, dict):
                bad.append(f"index {i}: not a dict")
                continue
            missing = required_keys - set(p.keys())
            if missing:
                bad.append(
                    f"index {i} ({p.get('experiment_id', '?')}): missing {sorted(missing)}"
                )
        if bad:
            return CheckResult(name, "warn",
                               f"{len(bad)} proposal(s) missing required keys",
                               {"issues": bad[:10]})
        return CheckResult(name, "pass",
                           f"director_proposals.json parseable ({len(data)} entries)")

    # ── runner ────────────────────────────────────────────────────────────────

    def run_all_checks(self) -> HealthReport:
        check_methods = [
            self.check_governance_files_present,
            self.check_evidence_inventory_coherence,
            self.check_canonical_result_paths,
            self.check_active_session,
            self.check_gpu_temperature,
            self.check_api_key_configured,
            self.check_orphan_processes,
            self.check_stale_leases,
            self.check_queue_schema,
            self.check_director_proposals_parseable,
        ]
        report = HealthReport()
        report.generated_at = datetime.now(timezone.utc).isoformat()
        for method in check_methods:
            try:
                result = method()
            except Exception as exc:
                result = CheckResult(
                    name=method.__name__.replace("check_", ""),
                    status="fail",
                    message=f"Check raised unexpected exception: {exc}",
                )
            report.checks.append(result)
            if result.status == "pass":
                report.passed += 1
            elif result.status == "fail":
                report.failed += 1
            elif result.status == "warn":
                report.warned += 1
            else:
                report.skipped += 1

        # Write report
        try:
            report_path = self.ws / "tar_state" / "health_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(asdict(report), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checker = HealthChecker(_WS)
    report = checker.run_all_checks()

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        # Print coloured terminal summary
        STATUS_ICON  = {"pass": "[OK]", "fail": "[FAIL]", "warn": "[WARN]", "skip": "[SKIP]"}
        STATUS_STYLE = {"pass": "\033[32m", "fail": "\033[31m", "warn": "\033[33m",
                        "skip": "\033[90m"}
        RESET = "\033[0m"
        SEP = "-" * 60
        print(f"\nTAR Health Check - {report.generated_at}")
        print(SEP)
        for c in report.checks:
            icon  = STATUS_ICON.get(c.status, "?")
            style = STATUS_STYLE.get(c.status, "")
            print(f"  {style}{icon:<6}{RESET} {c.name:<40} {c.message}")
        print(SEP)
        print(f"  {report.passed} passed | {report.warned} warnings | "
              f"{report.failed} failed | {report.skipped} skipped")
        print()

    sys.exit(0 if report.failed == 0 else 1)
