"""
Shared Claude API bridge for TAR subsystems.

Six capabilities, all optional — each degrades to '' / no-op when the
Anthropic key is absent or the `anthropic` package is not installed.

Cache: tar_state/llm_cache/*.json  keyed by content hash.
TTLs: scheduler=10 min; synthesis/eval/verify=4 h; findings/failures=permanent.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

_SONNET = "claude-sonnet-4-6"
_HAIKU  = "claude-haiku-4-5-20251001"
_TTL_LONG_S  = 4 * 3600   # synthesis, eval, verify
_TTL_SCHED_S = 10 * 60    # scheduler rationale


# ── API key resolution ─────────────────────────────────────────────────────────

def _api_key() -> str:
    for name in ("ANTHROPIC_API_KEY", "TAR_LLM_API_KEY"):
        val = os.environ.get(name, "").strip()
        if val:
            return val
    if os.name == "nt":
        try:
            import winreg  # type: ignore
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                for name in ("ANTHROPIC_API_KEY", "TAR_LLM_API_KEY"):
                    try:
                        val, _ = winreg.QueryValueEx(key, name)
                        if val:
                            return str(val).strip()
                    except Exception:
                        pass
        except Exception:
            pass
    return ""


# ── Raw API call ───────────────────────────────────────────────────────────────

def call_claude(
    prompt: str,
    *,
    system: str = "",
    model: str = _SONNET,
    max_tokens: int = 1024,
    tag: str = "llm_bridge",
) -> str:
    """Call Claude; return '' on failure or missing key. Never raises."""
    try:
        import anthropic  # type: ignore
    except ImportError:
        return ""
    api_key = _api_key()
    if not api_key:
        return ""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return resp.content[0].text.strip()
    except Exception as exc:
        print(f"[{tag}] Claude call failed: {exc}", flush=True)
        return ""


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_dir(workspace: Path) -> Path:
    path = workspace / "tar_state" / "llm_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _content_hash(*parts: str) -> str:
    combined = "\n".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _cache_read(cache_dir: Path, cache_key: str, ttl_s: float = _TTL_LONG_S) -> str | None:
    path = cache_dir / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if ttl_s > 0:
            age = time.time() - float(data.get("written_at", 0))
            if age > ttl_s:
                return None
        return str(data.get("content", "") or "")
    except Exception:
        return None


def _cache_write(cache_dir: Path, cache_key: str, content: str) -> None:
    path = cache_dir / f"{cache_key}.json"
    try:
        path.write_text(
            json.dumps({"written_at": time.time(), "content": content}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


# ── 1. Findings memo ──────────────────────────────────────────────────────────

_FINDINGS_SYSTEM = """\
You are TAR-Analyst, the experiment interpretation component of the Thermodynamic Autonomous Researcher.
After an experiment completes, write a concise scientific findings memo.

Rules:
- State what the numbers actually showed. Never extrapolate beyond the data.
- Identify what the result means for the frontier problem being studied.
- Suggest one concrete next step that follows logically from this result.
- Write in neutral, third-person scientific register. No hype.
- 3-5 sentences maximum.
"""


def generate_findings_memo(
    workspace: Path,
    experiment_id: str,
    result_summary: dict[str, Any],
    *,
    frontier_title: str = "",
    dataset: str = "",
    method: str = "",
) -> str:
    """
    Generate a findings memo for a completed experiment.
    Written once to tar_state/llm_cache/findings_{experiment_id}.json; permanent.
    Returns the memo text (may be '' if Claude unavailable).
    """
    memo_path = _cache_dir(workspace) / f"findings_{experiment_id}.json"
    # Permanent: if we already wrote one for this run, reuse it.
    if memo_path.exists():
        try:
            existing = json.loads(memo_path.read_text(encoding="utf-8"))
            if existing.get("content"):
                return str(existing["content"])
        except Exception:
            pass

    stats = result_summary.get("statistics") or {}
    verdict        = str(result_summary.get("verdict", "") or "")
    mean_forgetting = result_summary.get("mean_forgetting") or stats.get("mean_forgetting")
    mean_delta      = result_summary.get("mean_delta") or stats.get("mean_delta")
    p_val           = result_summary.get("p_val") or stats.get("p_val")
    seeds_done      = (result_summary.get("progress") or {}).get("seeds_done", "?")
    seeds_total     = (result_summary.get("progress") or {}).get("seeds_total", "?")

    prompt = (
        f"Experiment completed: {experiment_id}\n"
        f"Frontier problem: {frontier_title or 'not specified'}\n"
        f"Dataset: {dataset or 'not specified'}, Method: {method or 'TCL/TAR'}\n"
        f"Seeds: {seeds_done}/{seeds_total}\n"
        f"Verdict: {verdict}\n"
        f"Mean forgetting: {mean_forgetting}\n"
        f"Mean delta vs baseline: {mean_delta}\n"
        f"p-value: {p_val}\n"
        f"Result summary (JSON excerpt): "
        f"{json.dumps(result_summary, default=str)[:1200]}\n\n"
        "Write a 3-5 sentence scientific findings memo: what the numbers showed, "
        "what it means for the frontier, and one concrete next step."
    )
    memo = call_claude(prompt, system=_FINDINGS_SYSTEM, model=_SONNET, max_tokens=400, tag="findings_memo")
    if memo:
        try:
            memo_path.write_text(
                json.dumps({
                    "written_at": time.time(),
                    "experiment_id": experiment_id,
                    "content": memo,
                }, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
    return memo


# ── 2. Failure diagnosis ───────────────────────────────────────────────────────

_FAILURE_SYSTEM = """\
You are TAR-Diagnostician. When a training experiment fails, diagnose the root cause
and suggest one specific recovery step.

Rules:
- Identify the most likely root cause from the error text.
- Distinguish: data issues, model/config issues, OOM/hardware, dependency, code bug.
- Suggest one specific remediation (config change, dataset fix, memory reduction, etc.).
- 2-4 sentences. No speculation beyond the evidence. Direct.
"""


def generate_failure_diagnosis(
    workspace: Path,
    experiment_id: str,
    error_text: str,
    *,
    dataset: str = "",
    method: str = "",
    log_tail: str = "",
) -> str:
    """
    Generate a root-cause diagnosis for a failed experiment.
    Written to tar_state/llm_cache/failure_{experiment_id}.json; permanent.
    """
    diag_path = _cache_dir(workspace) / f"failure_{experiment_id}.json"
    prompt = (
        f"Experiment FAILED: {experiment_id}\n"
        f"Dataset: {dataset or 'not specified'}, Method: {method or 'TCL/TAR'}\n"
        f"Error: {error_text[:800]}\n"
        f"Log tail: {log_tail[-600:] if log_tail else 'not available'}\n\n"
        "Write a 2-4 sentence root-cause hypothesis and one specific recovery suggestion."
    )
    diagnosis = call_claude(prompt, system=_FAILURE_SYSTEM, model=_SONNET, max_tokens=300, tag="failure_diagnosis")
    if diagnosis:
        try:
            diag_path.write_text(
                json.dumps({
                    "written_at": time.time(),
                    "experiment_id": experiment_id,
                    "error_text": error_text[:400],
                    "content": diagnosis,
                }, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
    return diagnosis


# ── 3. Frontier literature synthesis ──────────────────────────────────────────

_SYNTHESIS_SYSTEM = """\
You are TAR-Synthesizer. Given a frontier research problem and what the literature says about it,
write a brief synthesis paragraph for the research director.

Rules:
- Summarize what is already known from the literature about this frontier.
- Identify the most important gap that TAR's current evidence does NOT yet fill.
- End with one specific experiment suggestion that would fill that gap.
- 4-6 sentences. Factual, no hype. Do not invent citations or paper titles.
"""


def synthesize_frontier_literature(
    workspace: Path,
    frontier_id: str,
    frontier_title: str,
    *,
    evidence_notes: list[str] | None = None,
    domain_learned_summary: str = "",
    top_verified_titles: list[str] | None = None,
    truth_status: str = "weak",
) -> str:
    """
    Synthesize what literature says about a frontier and what gap TAR should fill next.
    Cached per (frontier_id, content_hash) with 4h TTL.
    """
    cache_dir = _cache_dir(workspace)
    content_key = _content_hash(
        frontier_id, truth_status,
        " ".join((evidence_notes or [])[:6]),
        (domain_learned_summary or "")[:200],
    )
    cached = _cache_read(cache_dir, f"synthesis_{frontier_id}_{content_key}")
    if cached is not None:
        return cached

    titles_text = "\n".join(f"- {t}" for t in (top_verified_titles or [])[:5]) or "- None indexed"
    notes_text  = "\n".join(f"- {n}" for n in (evidence_notes or [])[:6]) or "- None yet"
    prompt = (
        f'Frontier problem: "{frontier_title}" (ID: {frontier_id})\n'
        f"Current evidence status: {truth_status}\n"
        f"TAR's experimental evidence notes:\n{notes_text}\n\n"
        f"What the literature says about this domain:\n"
        f"{domain_learned_summary or 'No literature summary available.'}\n\n"
        f"Top verified paper titles in this area:\n{titles_text}\n\n"
        "Write a 4-6 sentence synthesis: what is already known, what gap TAR has not yet filled, "
        "and one specific experiment suggestion to fill it."
    )
    synthesis = call_claude(prompt, system=_SYNTHESIS_SYSTEM, model=_SONNET, max_tokens=500, tag="frontier_synthesis")
    if synthesis:
        _cache_write(cache_dir, f"synthesis_{frontier_id}_{content_key}", synthesis)
    return synthesis


# ── 4. Experiment proposal evaluation ─────────────────────────────────────────

_EVAL_SYSTEM = """\
You are TAR-Evaluator. Assess whether an experiment proposal would genuinely advance
the frontier or is redundant or mis-scoped.

Rules:
- Score the proposal on: (1) addresses the frontier gap, (2) not redundant, (3) feasible.
- Flag any mismatch between experiment design and the frontier problem.
- End with exactly one sentence: "Run now.", "Run after X.", or "Deprioritize because Y."
- 3-4 sentences total. Direct. No hedging.
"""


def evaluate_experiment_proposal(
    workspace: Path,
    experiment_id: str,
    *,
    frontier_title: str = "",
    experiment_goal: str = "",
    mechanism_focus: str = "",
    dataset: str = "",
    method: str = "",
    status: str = "proposed",
    evidence_notes: list[str] | None = None,
) -> str:
    """
    Evaluate whether a proposed/queued experiment advances its frontier.
    Skips running/complete experiments. Cached per (experiment_id, content_hash) with 4h TTL.
    """
    if status in {"running", "complete", "failed"}:
        return ""
    cache_dir = _cache_dir(workspace)
    content_key = _content_hash(experiment_id, frontier_title, experiment_goal, status)
    cached = _cache_read(cache_dir, f"eval_{experiment_id}_{content_key}")
    if cached is not None:
        return cached

    notes_text = "\n".join(f"- {n}" for n in (evidence_notes or [])[:5]) or "- None yet"
    prompt = (
        f"Proposed experiment: {experiment_id}\n"
        f"Frontier problem: {frontier_title or 'not specified'}\n"
        f"Experiment goal: {experiment_goal or 'not specified'}\n"
        f"Mechanism focus: {mechanism_focus or 'not specified'}\n"
        f"Dataset: {dataset or 'not specified'}, Method: {method or 'TCL/TAR'}\n"
        f"Current frontier evidence:\n{notes_text}\n\n"
        "In 3-4 sentences, assess: Does it address the frontier gap? Is it redundant? Is it feasible?\n"
        "End with exactly one sentence: 'Run now.', 'Run after X.', or 'Deprioritize because Y.'"
    )
    evaluation = call_claude(prompt, system=_EVAL_SYSTEM, model=_HAIKU, max_tokens=250, tag="exp_eval")
    if evaluation:
        _cache_write(cache_dir, f"eval_{experiment_id}_{content_key}", evaluation)
    return evaluation


# ── 5. Claim verification ──────────────────────────────────────────────────────

_VERIFY_SYSTEM = """\
You are TAR-ClaimChecker. Given a frontier claim and the available evidence,
assess whether the claim is Consistent, Contradicted, or Unsupported.

Rules:
- State a one-sentence verdict: Consistent / Contradicted / Unsupported.
- Explain the key reason in 1-2 sentences.
- Flag the strongest piece of counter-evidence if any exists.
- 3-5 sentences total. No hedging beyond what the evidence warrants.
"""


def verify_result_claim(
    workspace: Path,
    frontier_id: str,
    claim_text: str,
    *,
    evidence_notes: list[str] | None = None,
    truth_status: str = "weak",
    domain_learned_summary: str = "",
) -> str:
    """
    Verify whether a frontier claim is consistent with available evidence and literature.
    Cached per (frontier_id, claim_hash, truth_status) with 4h TTL.
    """
    cache_dir = _cache_dir(workspace)
    content_key = _content_hash(frontier_id, claim_text, truth_status)
    cached = _cache_read(cache_dir, f"verify_{frontier_id}_{content_key}")
    if cached is not None:
        return cached

    notes_text = "\n".join(f"- {n}" for n in (evidence_notes or [])[:6]) or "- None yet"
    prompt = (
        f'Frontier claim: "{claim_text}"\n'
        f"Current evidence status: {truth_status}\n"
        f"TAR's experimental evidence:\n{notes_text}\n\n"
        f"Literature context: {(domain_learned_summary or 'Not available')[:400]}\n\n"
        "In 3-5 sentences: Is this claim Consistent, Contradicted, or Unsupported?\n"
        "State the verdict, explain the key reason, flag the strongest counter-evidence if any."
    )
    verdict_text = call_claude(prompt, system=_VERIFY_SYSTEM, model=_HAIKU, max_tokens=200, tag="claim_verify")
    if verdict_text:
        _cache_write(cache_dir, f"verify_{frontier_id}_{content_key}", verdict_text)
    return verdict_text


# ── 6. Scheduler rationale ────────────────────────────────────────────────────

_SCHEDULER_SYSTEM = """\
You are TAR-Planner. Write one clear paragraph explaining TAR's current scheduling decision.

Rules:
- State what is running, what will start, and what is held and why.
- Mention hardware constraints (GPU%, VRAM, temp) if they are the limiting factor.
- Be specific about experiment names. No generic filler.
- 3-5 sentences. Direct, no hedging, no lists — one flowing paragraph.
"""


def narrate_scheduler_decision(
    workspace: Path,
    *,
    gpu_name: str = "",
    gpu_util_pct: int = 0,
    vram_used_gb: float = 0.0,
    vram_total_gb: float = 0.0,
    gpu_temp_c: int = 0,
    running_names: list[str] | None = None,
    can_start: list[str] | None = None,
    hold_reasons: list[dict[str, str]] | None = None,
    template_fallback: str = "",
) -> str:
    """
    Generate a Claude-powered scheduler rationale paragraph.
    Cached with a 10-minute TTL (scheduler cycles are frequent).
    Falls back to template_fallback if Claude is unavailable.
    """
    cache_dir = _cache_dir(workspace)
    content_key = _content_hash(
        str(gpu_util_pct), f"{vram_used_gb:.1f}",
        ",".join(sorted(running_names or [])),
        ",".join(sorted(can_start or [])),
        str(len(hold_reasons or [])),
    )
    cached = _cache_read(cache_dir, f"scheduler_{content_key}", ttl_s=_TTL_SCHED_S)
    if cached is not None:
        return cached

    hold_text = "; ".join(
        f"'{h.get('experiment_name', '?')}': {h.get('reason', '?')}"
        for h in (hold_reasons or [])[:4]
    )
    prompt = (
        f"Hardware: {gpu_name or 'GPU'} at {gpu_util_pct}% utilization, "
        f"{vram_used_gb:.1f}/{vram_total_gb:.1f} GB VRAM, {gpu_temp_c}°C.\n"
        f"Currently running: {', '.join(running_names or []) or 'nothing'}\n"
        f"Starting now: {', '.join(can_start or []) or 'nothing'}\n"
        f"Held (with reasons): {hold_text or 'none'}\n\n"
        "Write one 3-5 sentence paragraph explaining TAR's scheduling decision. "
        "Be specific about experiment names and hardware numbers."
    )
    narration = call_claude(
        prompt, system=_SCHEDULER_SYSTEM, model=_HAIKU, max_tokens=200, tag="scheduler_rationale"
    )
    if narration:
        _cache_write(cache_dir, f"scheduler_{content_key}", narration)
        return narration
    return template_fallback
