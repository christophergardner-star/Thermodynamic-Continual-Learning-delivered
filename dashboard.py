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
latest_breakthrough = status.get("latest_breakthrough_report")

st.title("TAR Sidecar")

col1, col2, col3 = st.columns(3)
col1.metric("Trial ID", recovery.get("trial_id") or "none")
col2.metric("Status", recovery.get("status"))
col3.metric("Fail-Fast Streak", recovery.get("consecutive_fail_fast", 0))

reg_col1, reg_col2, reg_col3 = st.columns(3)
reg_col1.metric("D_PR", regime.get("effective_dimensionality", "n/a"))
reg_col2.metric("Eq Fraction", regime.get("equilibrium_fraction", "n/a"))
reg_col3.metric("Regime", regime.get("regime", "unknown"))

intel_col1, intel_col2, intel_col3 = st.columns(3)
intel_col1.metric("Research Docs", status.get("research_documents", 0))
intel_col2.metric("Verifications", status.get("verification_reports", 0))
intel_col3.metric("Breakthroughs", status.get("breakthrough_reports", 0))

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

orchestrator.shutdown()
