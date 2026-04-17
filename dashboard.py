from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from tar_lab.orchestrator import TAROrchestrator


def _project_label(payload: dict[str, Any]) -> str:
    project = payload.get("project") or {}
    if not isinstance(project, dict):
        return "unknown project"
    title = str(project.get("title") or project.get("project_id") or "unknown")
    status = str(project.get("status") or "n/a")
    return f"{title} [{status}]"


def _frontier_gap_label(payload: dict[str, Any]) -> str:
    gap_id = str(payload.get("gap_id") or "unknown-gap")
    status = str(payload.get("status") or "n/a")
    description = str(payload.get("description") or gap_id)
    short = description if len(description) <= 72 else f"{description[:69]}..."
    return f"{short} [{status}]"


def build_dashboard_context(
    orchestrator: TAROrchestrator,
    *,
    include_blocked: bool = True,
    max_results: int = 8,
    selected_project_id: str = "",
    frontier_gap_status_filter: str = "",
    frontier_gap_min_confidence: float = 0.0,
    selected_frontier_gap_id: str = "",
) -> dict[str, Any]:
    status = orchestrator.status()
    projects_payload = orchestrator.list_projects()
    project_views = projects_payload.get("projects", [])
    project_ids = [
        item.get("project", {}).get("project_id")
        for item in project_views
        if item.get("project", {}).get("project_id")
    ]
    project_labels = {
        item.get("project", {}).get("project_id"): _project_label(item)
        for item in project_views
        if item.get("project", {}).get("project_id")
    }

    operator_view = orchestrator.operator_view(
        include_blocked=include_blocked,
        limit=max_results,
    )
    portfolio_status = orchestrator.portfolio_status(
        include_blocked=include_blocked,
        limit=max_results,
    )
    portfolio_review = orchestrator.portfolio_review(
        include_blocked=include_blocked,
        limit=max_results,
    )
    experiment_backend_runtime_status = orchestrator.experiment_backend_runtime_status(limit=max_results * 3)
    literature_status = orchestrator.literature_status()
    literature_artifacts = orchestrator.list_paper_artifacts(limit=max_results)
    literature_conflicts = orchestrator.literature_conflicts(limit=max_results)
    frontier_gap_status = orchestrator.frontier_gap_status(
        status=frontier_gap_status_filter or None,
        limit=max_results,
        min_confidence=frontier_gap_min_confidence,
    )
    frontier_gap_views = frontier_gap_status.get("gaps", [])
    frontier_gap_ids = [item.get("gap_id") for item in frontier_gap_views if item.get("gap_id")]
    frontier_gap_labels = {
        item.get("gap_id"): _frontier_gap_label(item)
        for item in frontier_gap_views
        if item.get("gap_id")
    }
    selected_frontier_gap = None
    if selected_frontier_gap_id:
        gap = orchestrator.store.get_frontier_gap(selected_frontier_gap_id)
        if gap is not None:
            selected_frontier_gap = gap.model_dump(mode="json")

    selected_project_status = None
    selected_resume_dashboard = None
    selected_evidence_map = None
    selected_claim_lineage = None
    selected_timeline = None
    selected_falsification_status = None
    selected_publication_handoff = None
    if selected_project_id:
        selected_project_status = orchestrator.project_status(selected_project_id)
        selected_resume_dashboard = orchestrator.resume_dashboard(selected_project_id)
        selected_evidence_map = orchestrator.project_evidence_map(selected_project_id)
        selected_claim_lineage = orchestrator.claim_lineage(selected_project_id)
        selected_timeline = orchestrator.project_timeline(selected_project_id, limit=max_results * 3)
        selected_falsification_status = orchestrator.falsification_status(selected_project_id)
        selected_publication_handoff = orchestrator.publication_handoff(selected_project_id)

    return {
        "workspace": orchestrator.workspace,
        "status": status,
        "projects_payload": projects_payload,
        "project_views": project_views,
        "project_ids": project_ids,
        "project_labels": project_labels,
        "selected_project_id": selected_project_id,
        "operator_view": operator_view,
        "portfolio_status": portfolio_status,
        "portfolio_review": portfolio_review,
        "experiment_backend_runtime_status": experiment_backend_runtime_status,
        "literature_status": literature_status,
        "literature_artifacts": literature_artifacts,
        "literature_conflicts": literature_conflicts,
        "frontier_gap_status": frontier_gap_status,
        "frontier_gap_views": frontier_gap_views,
        "frontier_gap_ids": frontier_gap_ids,
        "frontier_gap_labels": frontier_gap_labels,
        "selected_frontier_gap_id": selected_frontier_gap_id,
        "selected_frontier_gap": selected_frontier_gap,
        "selected_project_status": selected_project_status,
        "selected_resume_dashboard": selected_resume_dashboard,
        "selected_evidence_map": selected_evidence_map,
        "selected_claim_lineage": selected_claim_lineage,
        "selected_timeline": selected_timeline,
        "selected_falsification_status": selected_falsification_status,
        "selected_publication_handoff": selected_publication_handoff,
    }


def _show_dataframe(title: str, rows: list[dict[str, Any]], empty_message: str) -> None:
    st.subheader(title)
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info(empty_message)


def _show_json(title: str, payload: Any, empty_message: str) -> None:
    st.subheader(title)
    if payload:
        st.json(payload)
    else:
        st.info(empty_message)


def _render_overview_tab(context: dict[str, Any]) -> None:
    status = context["status"]
    recovery = status["recovery"]
    metrics = status["last_three_metrics"]
    regime = status.get("regime", {})
    benchmark_profiles = status.get("frontier", {}).get("benchmark_profiles", {})

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Trial ID", recovery.get("trial_id") or "none")
    top2.metric("Recovery Status", recovery.get("status"))
    top3.metric("Projects", status.get("research_projects", 0))
    top4.metric("Active Projects", status.get("active_research_projects", 0))

    reg1, reg2, reg3, reg4 = st.columns(4)
    reg1.metric("D_PR", regime.get("effective_dimensionality", "n/a"))
    reg2.metric("Eq Fraction", regime.get("equilibrium_fraction", "n/a"))
    reg3.metric("Regime", regime.get("regime", "unknown"))
    reg4.metric("Fail-Fast Streak", recovery.get("consecutive_fail_fast", 0))

    intel1, intel2, intel3, intel4, intel5, intel6 = st.columns(6)
    intel1.metric("Problem Studies", status.get("problem_studies", 0))
    intel2.metric("Problem Runs", status.get("problem_executions", 0))
    intel3.metric("Priority Snapshots", status.get("prioritization_snapshots", 0))
    intel4.metric("Falsification Plans", status.get("falsification_plans", 0))
    intel5.metric("Portfolio Decisions", status.get("portfolio_decisions", 0))
    intel6.metric("Evidence Debt", status.get("evidence_debt_records", 0))

    bench1, bench2, bench3, bench4 = st.columns(4)
    bench1.metric("Benchmark", status.get("benchmark_name") or "n/a")
    bench2.metric("Tier", status.get("benchmark_tier", "n/a"))
    bench3.metric("Alignment", status.get("benchmark_alignment", "n/a"))
    bench4.metric("Profiles", len(benchmark_profiles))

    if metrics:
        st.subheader("Thermodynamic State")
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

    left, right = st.columns(2)
    with left:
        _show_json("Recovery", recovery, "No recovery state recorded.")
    with right:
        _show_json("Regime Check", regime, "No regime state recorded.")


def _render_operator_tab(context: dict[str, Any]) -> None:
    operator_view = context["operator_view"]
    portfolio_status = context["portfolio_status"]
    portfolio_review = context["portfolio_review"]
    counts = operator_view.get("project_counts", {})
    health = operator_view.get("portfolio_health", {})
    retrieval = operator_view.get("retrieval_mode_breakdown", {})
    verdicts = operator_view.get("claim_verdict_lifecycle", {})
    frontier = operator_view.get("frontier_gap_counts", {})

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total", counts.get("total", 0))
    col2.metric("Active", counts.get("active", 0))
    col3.metric("Blocked", counts.get("blocked", 0))
    col4.metric("Stale", counts.get("stale", 0))
    col5.metric("Resume Candidates", health.get("resume_candidates", 0))
    col6.metric("Escalated Verdicts", verdicts.get("escalated", 0))

    ret1, ret2, ret3, ret4 = st.columns(4)
    ret1.metric("Semantic Studies", retrieval.get("semantic", 0))
    ret2.metric("Lexical Fallback", retrieval.get("lexical_fallback", 0))
    ret3.metric("Degraded Studies", operator_view.get("degraded_retrieval_studies", 0))
    ret4.metric("Aging Verdicts", verdicts.get("aging", 0))

    frontier1, frontier2, frontier3, frontier4 = st.columns(4)
    frontier1.metric("Frontier Identified", frontier.get("identified", 0))
    frontier2.metric("Frontier Proposed", frontier.get("proposed", 0))
    frontier3.metric("Frontier Promoted", frontier.get("promoted", 0))
    frontier4.metric("Frontier Scans", operator_view.get("frontier_gap_scan_count", 0))

    left, right = st.columns(2)
    with left:
        _show_dataframe(
            "Top Action Candidates",
            operator_view.get("top_candidates", []),
            "No prioritized action candidates available.",
        )
        _show_dataframe(
            "Active Projects",
            operator_view.get("active_projects", []),
            "No active projects.",
        )
    with right:
        _show_dataframe(
            "Top Portfolio Projects",
            portfolio_review.get("top_projects", []),
            "No portfolio project rankings available.",
        )
        _show_json(
            "Portfolio Status",
            portfolio_status,
            "No portfolio status recorded.",
        )

    stale_col, blocked_col = st.columns(2)
    with stale_col:
        _show_dataframe(
            "Stale Projects",
            operator_view.get("stale_projects", []),
            "No stale projects.",
        )
    with blocked_col:
        _show_dataframe(
            "Promotion Blocked",
            operator_view.get("promotion_blocked_projects", []),
            "No promotion-blocked projects.",
        )

    _show_json(
        "Resume Candidates",
        operator_view.get("resume_candidates", []),
        "No resume candidates.",
    )
    _show_dataframe(
        "Frontier Gaps",
        operator_view.get("frontier_gaps", []),
        "No frontier gaps recorded.",
    )


def _render_project_tab(context: dict[str, Any]) -> None:
    project_id = context["selected_project_id"]
    if not project_id:
        st.info("Select a project from the sidebar to inspect its WS21 surfaces.")
        return

    status_payload = context["selected_project_status"] or {}
    resume_dashboard = context["selected_resume_dashboard"] or {}
    evidence_map = context["selected_evidence_map"] or {}
    claim_lineage = context["selected_claim_lineage"] or {}
    timeline = context["selected_timeline"] or {}
    falsification_status = context["selected_falsification_status"] or {}

    project = status_payload.get("project", {})
    thread = status_payload.get("active_thread", {})
    question = status_payload.get("current_question", {})
    action = status_payload.get("next_action", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Project Status", project.get("status", "n/a"))
    col2.metric("Thread Status", thread.get("status", "n/a"))
    col3.metric("Confidence", thread.get("confidence_state", "n/a"))
    col4.metric("Resume State", resume_dashboard.get("resume_state", "n/a"))

    st.subheader("Current Thread")
    st.write(project.get("latest_decision_summary", "No latest decision summary recorded."))
    st.caption(f"Question: {question.get('question', 'n/a')}")
    st.caption(f"Next Action: {action.get('description', 'n/a')}")

    resume_col, evidence_col = st.columns(2)
    with resume_col:
        _show_json("Resume Dashboard", resume_dashboard, "No resume state recorded.")
    with evidence_col:
        _show_json("Evidence Map", evidence_map, "No evidence map recorded.")

    lineage_col, falsify_col = st.columns(2)
    with lineage_col:
        _show_json("Claim Lineage", claim_lineage, "No claim lineage recorded.")
    with falsify_col:
        _show_json("Falsification Status", falsification_status, "No falsification state recorded.")

    _show_dataframe(
        "Project Timeline",
        timeline.get("events", []),
        "No project timeline events recorded.",
    )


def _render_publication_tab(context: dict[str, Any]) -> None:
    project_id = context["selected_project_id"]
    if not project_id:
        st.info("Select a project from the sidebar to inspect its publication handoff package.")
        return

    publication = context["selected_publication_handoff"] or {}
    package = publication.get("package", publication)
    accepted = package.get("accepted_claims", [])
    provisional = package.get("provisional_claims", [])
    rejected = package.get("rejected_alternatives", [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Package Status", package.get("package_status", "n/a"))
    col2.metric("Accepted Claims", len(accepted))
    col3.metric("Provisional Claims", len(provisional))
    col4.metric("Rejected Alternatives", len(rejected))

    left, right = st.columns(2)
    with left:
        _show_json("Publication Package", package, "No publication handoff package recorded.")
        _show_dataframe(
            "Experiment Lineage",
            package.get("experiment_lineage", []),
            "No publication lineage events recorded.",
        )
    with right:
        _show_dataframe(
            "Benchmark Truth Attachments",
            package.get("benchmark_truth_attachments", []),
            "No benchmark truth attachments recorded.",
        )
        _show_json(
            "Writer Cautions",
            package.get("writer_cautions", []),
            "No writer cautions recorded.",
        )


def _render_infrastructure_tab(context: dict[str, Any]) -> None:
    status = context["status"]
    runtime = status.get("runtime", {})
    sandbox_policy = runtime.get("sandbox_policy", {})
    runtime_policy = runtime.get("runtime_policy", {})
    verdicts = runtime.get("claim_verdict_lifecycle", {})
    queue_health = runtime.get("queue_health", {})
    endpoints = status.get("endpoints", [])
    role_assignments = status.get("role_assignments", [])
    operator_serving = status.get("operator_serving", {})
    operator_state = operator_serving.get("state", {})
    backend_runtime = context.get("experiment_backend_runtime_status", {})
    backend_counts = backend_runtime.get("counts", {})
    backend_records = backend_runtime.get("records", [])
    build_attestation = runtime.get("build_attestation", {})

    runtime_col1, runtime_col2, runtime_col3, runtime_col4, runtime_col5, runtime_col6 = st.columns(6)
    runtime_col1.metric("Reproducible", status.get("reproducibility_complete", False))
    runtime_col2.metric("Active Leases", len(runtime.get("active_leases", [])))
    runtime_col3.metric("Retry Waiting", len(runtime.get("retry_waiting", [])))
    runtime_col4.metric("Alerts", len(runtime.get("alerts", [])))
    runtime_col5.metric("Backend Runs", backend_counts.get("total", 0))
    runtime_col6.metric("Escalated Verdicts", verdicts.get("escalated", 0))

    endpoint_col1, endpoint_col2, endpoint_col3, endpoint_col4, endpoint_col5, endpoint_col6 = st.columns(6)
    endpoint_col1.metric("Endpoints", len(endpoints))
    endpoint_col2.metric("Role Assignments", len(role_assignments))
    endpoint_col3.metric("Operator Mode", operator_state.get("mode", "n/a"))
    endpoint_col4.metric("Resumable Backends", backend_counts.get("resumable", 0))
    endpoint_col5.metric("Recoverable Crashes", queue_health.get("recoverable_crash", 0))
    endpoint_col6.metric("Verdict Aging Days", runtime_policy.get("verdict_aging_days", 0))

    queue_col1, queue_col2, queue_col3, queue_col4 = st.columns(4)
    queue_col1.metric("Queue Orphans", queue_health.get("orphan_count", 0))
    queue_col2.metric("Stale Leases", queue_health.get("stale_lease_count", 0))
    queue_col3.metric("Running Jobs", queue_health.get("running", 0))
    queue_col4.metric("Oldest Pending (min)", queue_health.get("oldest_pending_age_minutes", 0.0))

    infra_left, infra_right = st.columns(2)
    with infra_left:
        _show_json("Runtime", runtime, "No runtime state recorded.")
        _show_json("Sandbox Policy", sandbox_policy, "No sandbox policy recorded.")
    with infra_right:
        _show_json("Queue Health", queue_health, "No queue health recorded.")
        _show_json("Build Attestation", build_attestation, "No build attestation recorded.")
        _show_json("Operator Serving", operator_serving, "No operator serving state recorded.")
        _show_json("Inference Endpoints", endpoints, "No managed inference endpoints registered.")
        _show_json("Role Assignments", role_assignments, "No role assignments recorded.")

    _show_dataframe(
        "Experiment Backend Runs",
        backend_records,
        "No backend runtime records recorded.",
    )


def _render_literature_tab(context: dict[str, Any]) -> None:
    literature_status = context.get("literature_status", {})
    literature_artifacts = context.get("literature_artifacts", {})
    literature_conflicts = context.get("literature_conflicts", {})
    latest_manifest = literature_status.get("latest_manifest", {})
    capability = literature_status.get("capability_report", {})

    top1, top2, top3, top4, top5 = st.columns(5)
    top1.metric("Artifacts", literature_status.get("artifacts", 0))
    top2.metric("Conflicts", literature_status.get("conflicts", 0))
    top3.metric("Manifests", literature_status.get("manifests", 0))
    top4.metric("Tables", literature_status.get("tables", 0))
    top5.metric("Figures", literature_status.get("figures", 0))

    left, right = st.columns(2)
    with left:
        _show_json("Literature Status", literature_status, "No literature status recorded.")
        _show_json("Latest Manifest", latest_manifest, "No literature ingest manifest recorded.")
    with right:
        _show_json("Literature Capabilities", capability, "No literature capability report recorded.")
        _show_json("Recent Conflicts", literature_conflicts, "No literature conflicts recorded.")

    _show_dataframe(
        "Paper Artifacts",
        literature_artifacts.get("artifacts", []),
        "No literature artifacts recorded.",
    )


def _render_frontier_gap_tab(
    orchestrator: TAROrchestrator,
    context: dict[str, Any],
    *,
    frontier_topic: str,
    max_results: int,
    frontier_gap_min_confidence: float,
) -> None:
    payload = context.get("frontier_gap_status", {})
    counts = payload.get("counts", {})
    latest_scan = payload.get("latest_scan", {})
    gaps = payload.get("gaps", [])
    selected_gap = context.get("selected_frontier_gap") or {}

    notice = st.session_state.pop("frontier_gap_notice", None)
    if notice:
        st.success(str(notice))

    top1, top2, top3, top4, top5, top6 = st.columns(6)
    top1.metric("Total", counts.get("total", 0))
    top2.metric("Identified", counts.get("identified", 0))
    top3.metric("Proposed", counts.get("proposed", 0))
    top4.metric("Rejected", counts.get("rejected", 0))
    top5.metric("Promoted", counts.get("promoted", 0))
    top6.metric("Scans", payload.get("scan_count", 0))

    left, right = st.columns(2)
    with left:
        _show_json("Latest Frontier Scan", latest_scan, "No frontier gap scan recorded.")
    with right:
        _show_json("Frontier Gap Status", payload, "No frontier gap status recorded.")

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Scan Frontier Now", key="frontier_scan_now"):
            report = orchestrator.scan_frontier_gaps(topic=frontier_topic, max_gaps=max_results)
            st.session_state["frontier_gap_notice"] = (
                f"Frontier scan {report.scan_id} completed: identified={report.gaps_identified} rejected={report.gaps_rejected}."
            )
            st.rerun()
        if st.button("Propose From Frontier Gaps", key="frontier_propose_now"):
            created = orchestrator.propose_projects_from_gaps(
                max_proposals=max_results,
                confidence_threshold=frontier_gap_min_confidence,
            )
            st.session_state["frontier_gap_notice"] = (
                f"Created {len(created)} proposed project(s) from frontier gaps."
            )
            st.rerun()
    with action_col2:
        rejection_reason = st.text_input(
            "Reject Reason",
            value="operator review rejected this frontier gap",
            key="frontier_gap_rejection_reason",
        )
        if selected_gap and selected_gap.get("status") == "proposed":
            if st.button("Promote Selected Gap", key="frontier_promote_selected"):
                project = orchestrator.promote_gap_project(str(selected_gap.get("gap_id", "")))
                st.session_state["frontier_gap_notice"] = (
                    f"Promoted frontier gap {selected_gap.get('gap_id', 'n/a')} into active project {project.project_id}."
                )
                st.rerun()
        if selected_gap and selected_gap.get("status") in {"identified", "proposed"}:
            if st.button("Reject Selected Gap", key="frontier_reject_selected"):
                gap = orchestrator.reject_gap_project(str(selected_gap.get("gap_id", "")), rejection_reason)
                st.session_state["frontier_gap_notice"] = (
                    f"Rejected frontier gap {gap.gap_id}."
                )
                st.rerun()

    _show_json("Selected Frontier Gap", selected_gap, "No frontier gap selected.")
    _show_dataframe(
        "Frontier Gaps",
        gaps,
        "No frontier gaps recorded.",
    )


def _render_actions_tab(orchestrator: TAROrchestrator) -> None:
    button_col1, button_col2, button_col3 = st.columns(3)
    if button_col1.button("Dry Run"):
        st.json(orchestrator.run_dry_run().model_dump(mode="json"))
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


def _render_raw_tab(context: dict[str, Any]) -> None:
    status = context["status"]
    raw_col1, raw_col2 = st.columns(2)
    with raw_col1:
        _show_json("Status", status, "No status payload recorded.")
        _show_json("Operator View", context["operator_view"], "No operator view recorded.")
        _show_json("Portfolio Review", context["portfolio_review"], "No portfolio review recorded.")
    with raw_col2:
        _show_json(
            "Selected Project Status",
            context["selected_project_status"],
            "No project selected.",
        )
        _show_json(
            "Selected Resume Dashboard",
            context["selected_resume_dashboard"],
            "No project selected.",
        )
        _show_json(
            "Selected Evidence Map",
            context["selected_evidence_map"],
            "No project selected.",
        )
        _show_json(
            "Selected Publication Handoff",
            context["selected_publication_handoff"],
            "No project selected.",
        )

    graph_path = Path(context["workspace"]) / "tar_state" / "knowledge_graph.json"
    if graph_path.exists():
        _show_json(
            "Knowledge Graph",
            json.loads(graph_path.read_text(encoding="utf-8")),
            "knowledge_graph.json not created yet.",
        )
    else:
        st.info("knowledge_graph.json not created yet.")


def main() -> None:
    st.set_page_config(page_title="TAR Operator Surface", layout="wide")
    st.title("TAR Operator Surface")

    workspace = st.sidebar.text_input("Workspace", ".")
    max_results = st.sidebar.slider("View Limit", min_value=3, max_value=20, value=8)
    include_blocked = st.sidebar.checkbox("Include Blocked", value=True)
    frontier_topic = st.sidebar.text_input("Frontier Topic", "thermodynamic continual learning")
    frontier_gap_status_filter = st.sidebar.selectbox(
        "Gap Status Filter",
        options=["", "identified", "proposed", "rejected", "promoted"],
        format_func=lambda item: "All statuses" if not item else item,
    )
    frontier_gap_min_confidence = st.sidebar.slider(
        "Gap Min Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )

    orchestrator = TAROrchestrator(workspace=workspace)
    try:
        base_context = build_dashboard_context(
            orchestrator,
            include_blocked=include_blocked,
            max_results=max_results,
            selected_project_id="",
            frontier_gap_status_filter=frontier_gap_status_filter,
            frontier_gap_min_confidence=frontier_gap_min_confidence,
            selected_frontier_gap_id="",
        )
        selected_project_id = st.sidebar.selectbox(
            "Project",
            options=[""] + base_context["project_ids"],
            format_func=lambda item: "No project selected"
            if not item
            else base_context["project_labels"].get(item, item),
        )
        selected_frontier_gap_id = st.sidebar.selectbox(
            "Frontier Gap",
            options=[""] + base_context["frontier_gap_ids"],
            format_func=lambda item: "No frontier gap selected"
            if not item
            else base_context["frontier_gap_labels"].get(item, item),
        )
        context = (
            base_context
            if not selected_project_id and not selected_frontier_gap_id
            else build_dashboard_context(
                orchestrator,
                include_blocked=include_blocked,
                max_results=max_results,
                selected_project_id=selected_project_id,
                frontier_gap_status_filter=frontier_gap_status_filter,
                frontier_gap_min_confidence=frontier_gap_min_confidence,
                selected_frontier_gap_id=selected_frontier_gap_id,
            )
        )

        tabs = st.tabs(["Overview", "Operator", "Project", "Publication", "Infrastructure", "Literature", "Frontier Gaps", "Actions", "Raw"])
        with tabs[0]:
            _render_overview_tab(context)
        with tabs[1]:
            _render_operator_tab(context)
        with tabs[2]:
            _render_project_tab(context)
        with tabs[3]:
            _render_publication_tab(context)
        with tabs[4]:
            _render_infrastructure_tab(context)
        with tabs[5]:
            _render_literature_tab(context)
        with tabs[6]:
            _render_frontier_gap_tab(
                orchestrator,
                context,
                frontier_topic=frontier_topic,
                max_results=max_results,
                frontier_gap_min_confidence=frontier_gap_min_confidence,
            )
        with tabs[7]:
            _render_actions_tab(orchestrator)
        with tabs[8]:
            _render_raw_tab(context)
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
