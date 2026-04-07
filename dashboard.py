from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from tar_lab.orchestrator import TAROrchestrator


st.set_page_config(page_title="TAR Sidecar", layout="wide")

workspace = st.sidebar.text_input("Workspace", ".")
orchestrator = TAROrchestrator(workspace=workspace)
status = orchestrator.status()
recovery = status["recovery"]
metrics = status["last_three_metrics"]
datasets = status.get("datasets", {})
memory = status.get("memory", {})
regime = status.get("regime", {})
runtime = status.get("runtime", {})
sandbox_policy = runtime.get("sandbox_policy", {})
latest_breakthrough = status.get("latest_breakthrough_report")
latest_claim_verdict = status.get("latest_claim_verdict")
latest_research_decision = status.get("latest_research_decision")
latest_problem_study = status.get("latest_problem_study")
latest_problem_execution = status.get("latest_problem_execution")
latest_problem_schedule = status.get("latest_problem_schedule")
endpoints = status.get("endpoints", [])
role_assignments = status.get("role_assignments", [])
healthy_endpoints = [item for item in endpoints if (item.get("health") or {}).get("ok")]
failed_endpoints = [item for item in endpoints if item.get("status") == "failed"]
trusted_endpoints = [item for item in endpoints if item.get("trust_remote_code")]

st.title("TAR Sidecar")

col1, col2, col3 = st.columns(3)
col1.metric("Trial ID", recovery.get("trial_id") or "none")
col2.metric("Status", recovery.get("status"))
col3.metric("Fail-Fast Streak", recovery.get("consecutive_fail_fast", 0))

reg_col1, reg_col2, reg_col3 = st.columns(3)
reg_col1.metric("D_PR", regime.get("effective_dimensionality", "n/a"))
reg_col2.metric("Eq Fraction", regime.get("equilibrium_fraction", "n/a"))
reg_col3.metric("Regime", regime.get("regime", "unknown"))

bench_col1, bench_col2, bench_col3 = st.columns(3)
bench_col1.metric("Benchmark", status.get("benchmark_name") or "n/a")
bench_col2.metric("Benchmark Tier", status.get("benchmark_tier", "n/a"))
bench_col3.metric("Benchmark Alignment", status.get("benchmark_alignment", "n/a"))

bench_meta_col1, bench_meta_col2 = st.columns(2)
bench_meta_col1.caption(f"Actual Benchmark Tiers: {', '.join(status.get('actual_benchmark_tiers', [])) or 'n/a'}")
bench_meta_col2.caption(f"Truth Statuses: {', '.join(status.get('benchmark_truth_statuses', [])) or 'n/a'}")

bench_meta_col3, bench_meta_col4 = st.columns(2)
bench_meta_col3.caption(f"Canonical Comparable: {status.get('canonical_comparable', False)}")
bench_meta_col4.caption(f"Benchmark Profiles: {len(status.get('frontier', {}).get('benchmark_profiles', {}))}")

intel_col1, intel_col2, intel_col3, intel_col4, intel_col5, intel_col6 = st.columns(6)
intel_col1.metric("Research Docs", status.get("research_documents", 0))
intel_col2.metric("Verifications", status.get("verification_reports", 0))
intel_col3.metric("Breakthroughs", status.get("breakthrough_reports", 0))
intel_col4.metric("Problem Studies", status.get("problem_studies", 0))
intel_col5.metric("Problem Runs", status.get("problem_executions", 0))
intel_col6.metric("Schedules", status.get("active_problem_schedules", 0))

runtime_col1, runtime_col2, runtime_col3, runtime_col4 = st.columns(4)
runtime_col1.metric("Reproducible", status.get("reproducibility_complete", False))
runtime_col2.metric("Active Leases", len(runtime.get("active_leases", [])))
runtime_col3.metric("Retry Waiting", len(runtime.get("retry_waiting", [])))
runtime_col4.metric("Alerts", len(runtime.get("alerts", [])))

runtime_meta_col1, runtime_meta_col2, runtime_meta_col3, runtime_meta_col4 = st.columns(4)
runtime_meta_col1.caption(f"Image Tag: {status.get('image_tag') or 'n/a'}")
runtime_meta_col2.caption(f"Manifest Hash: {status.get('manifest_hash') or 'n/a'}")
runtime_meta_col3.caption(f"Sandbox Mode: {status.get('safe_execution_mode') or 'n/a'}")
runtime_meta_col4.caption(f"Sandbox Profile: {sandbox_policy.get('profile') or status.get('sandbox_profile') or 'n/a'}")

agent_col1, agent_col2, agent_col3 = st.columns(3)
agent_col1.metric("Endpoints", len(endpoints))
agent_col2.metric("Role Assignments", len(role_assignments))
agent_col3.metric("Claim Verdicts", status.get("claim_verdicts", 0))

endpoint_col1, endpoint_col2, endpoint_col3 = st.columns(3)
endpoint_col1.metric("Healthy Endpoints", len(healthy_endpoints))
endpoint_col2.metric("Failed Endpoints", len(failed_endpoints))
endpoint_col3.metric("Remote-Code Trusted", len(trusted_endpoints))

st.subheader("Thermodynamic State")
if metrics:
    st.line_chart(
        {
            "E": [item["energy_e"] for item in metrics],
            "sigma": [item["entropy_sigma"] for item in metrics],
            "rho": [item["drift_rho"] for item in metrics],
            "D_PR": [item.get("effective_dimensionality", 0.0) for item in metrics],
            "Eq Fraction": [item.get("equilibrium_fraction", 0.0) for item in metrics],
            "Loss": [item.get("training_loss", 0.0) or 0.0 for item in metrics],
        }
    )
else:
    st.info("No metrics recorded yet.")

if regime.get("warning"):
    st.error(regime["warning"])

st.subheader("Recovery")
st.json(recovery)

st.subheader("Regime Check")
st.json(regime)

st.subheader("Runtime")
st.json(runtime)

st.subheader("Sandbox Policy")
st.json(sandbox_policy)

info_col1, info_col2 = st.columns(2)
with info_col1:
    st.subheader("Datasets")
    st.json(datasets)
with info_col2:
    st.subheader("Vector Memory")
    st.json(memory)

st.subheader("Latest Breakthrough")
if latest_breakthrough:
    st.json(latest_breakthrough)
else:
    st.info("No breakthrough report generated yet.")

st.subheader("Latest Claim Verdict")
if latest_claim_verdict:
    st.json(latest_claim_verdict)
else:
    st.info("No claim verdict generated yet.")

st.subheader("Latest Research Decision")
if latest_research_decision:
    st.json(latest_research_decision)
else:
    st.info("No research decision logged yet.")

st.subheader("Latest Problem Study")
if latest_problem_study:
    st.json(latest_problem_study)
else:
    st.info("No problem study generated yet.")

st.subheader("Latest Problem Execution")
if latest_problem_execution:
    st.json(latest_problem_execution)
else:
    st.info("No problem execution generated yet.")

st.subheader("Latest Problem Schedule")
if latest_problem_schedule:
    st.json(latest_problem_schedule)
else:
    st.info("No scheduled problem execution yet.")

st.subheader("Inference Endpoints")
if endpoints:
    st.json(endpoints)
else:
    st.info("No managed inference endpoints registered.")

st.subheader("Role Assignments")
if role_assignments:
    st.json(role_assignments)
else:
    st.info("No role assignments recorded.")

st.subheader("Knowledge Graph")
graph_path = Path(workspace) / "tar_state" / "knowledge_graph.json"
if graph_path.exists():
    st.json(json.loads(graph_path.read_text(encoding="utf-8")))
else:
    st.info("knowledge_graph.json not created yet.")

button_col1, button_col2, button_col3 = st.columns(3)
if button_col1.button("Dry Run"):
    report = orchestrator.run_dry_run()
    st.json(report.model_dump(mode="json"))

if button_col2.button("Force Pivot"):
    st.json(orchestrator.pivot_force(force=True))

if button_col3.button("Panic Button"):
    st.json(orchestrator.panic())

action_col1, action_col2, action_col3 = st.columns(3)
if action_col1.button("Ingest Research"):
    st.json(orchestrator.ingest_research(topic="frontier ai", max_results=6).model_dump(mode="json"))

if action_col2.button("Verify Trial"):
    st.json(orchestrator.verify_last_trial().model_dump(mode="json"))

if action_col3.button("Breakthrough Report"):
    st.json(orchestrator.breakthrough_report().model_dump(mode="json"))

st.subheader("Talk to Lab")
prompt = st.text_input("Director Query", "Analyze the current stability")
if st.button("Submit Query"):
    st.json(orchestrator.chat(prompt).model_dump(mode="json"))

st.subheader("Problem-Driven Research")
problem_prompt = st.text_input("Problem Prompt", "Investigate barren plateaus in quantum AI")
prob_col1, prob_col2, prob_col3 = st.columns(3)
if prob_col1.button("Resolve Domain"):
    st.json(orchestrator.resolve_problem(problem_prompt).model_dump(mode="json"))
if prob_col2.button("Prepare Science Env"):
    st.json(orchestrator.prepare_science_environment(problem_prompt).model_dump(mode="json"))
if prob_col3.button("Create Study Plan"):
    st.json(orchestrator.study_problem(problem_prompt).model_dump(mode="json"))

exec_col1, exec_col2 = st.columns(2)
if exec_col1.button("Run Study Local"):
    st.json(orchestrator.run_problem_study(use_docker=False).model_dump(mode="json"))
if exec_col2.button("Run Study In Docker"):
    st.json(orchestrator.run_problem_study(use_docker=True, build_env=True).model_dump(mode="json"))

sched_col1, sched_col2, sched_col3 = st.columns(3)
if sched_col1.button("Schedule Study"):
    st.json(orchestrator.schedule_problem_study(delay_s=0, max_runs=1).model_dump(mode="json"))
if sched_col2.button("Scheduler Status"):
    st.json(orchestrator.scheduler_status())
if sched_col3.button("Run Scheduler Once"):
    st.json(orchestrator.run_scheduler_once(max_jobs=1).model_dump(mode="json"))

orchestrator.shutdown()
