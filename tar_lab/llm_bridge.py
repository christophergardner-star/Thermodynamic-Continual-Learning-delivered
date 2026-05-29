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
import re
import threading
import time
from pathlib import Path
from typing import Any

_SONNET = "claude-sonnet-4-6"
_HAIKU  = "claude-haiku-4-5-20251001"
_TTL_LONG_S  = 4 * 3600   # synthesis, eval, verify
_TTL_SCHED_S = 10 * 60    # scheduler rationale

# Module-level 529 backoff — protected by _overload_lock for Flask thread safety
_overloaded_until: float = 0.0
_overload_lock = threading.Lock()


def _is_overloaded() -> bool:
    with _overload_lock:
        return time.time() < _overloaded_until


def _set_overloaded(backoff_s: float = 300.0) -> None:
    with _overload_lock:
        global _overloaded_until
        _overloaded_until = time.time() + backoff_s


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
    if _is_overloaded():
        return ""  # API overloaded backoff active — skip silently
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
        msg = str(exc)
        if "529" in msg or "overloaded" in msg.lower():
            _set_overloaded(300.0)  # 5-minute backoff on 529
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


def _cache_read_structured(
    cache_dir: Path, cache_key: str, ttl_s: float = _TTL_LONG_S
) -> tuple[str | None, dict | None]:
    """Return (content_str, structured_dict). Either may be None on miss or expiry."""
    path = cache_dir / f"{cache_key}.json"
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if ttl_s > 0:
            age = time.time() - float(data.get("written_at", 0))
            if age > ttl_s:
                return None, None
        content = str(data.get("content", "") or "")
        structured = data.get("structured") or None
        return content, structured
    except Exception:
        return None, None


def _cache_write(cache_dir: Path, cache_key: str, content: str) -> None:
    path = cache_dir / f"{cache_key}.json"
    try:
        path.write_text(
            json.dumps({"written_at": time.time(), "content": content}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _cache_write_structured(cache_dir: Path, cache_key: str, content: str, structured: dict) -> None:
    path = cache_dir / f"{cache_key}.json"
    try:
        path.write_text(
            json.dumps({"written_at": time.time(), "content": content, "structured": structured}, indent=2),
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
) -> dict:
    """
    Synthesize what literature says about a frontier and what gap TAR should fill next.
    Returns dict: {narrative, priority_delta, gap_signals, needs_more_evidence}.
    Cached per (frontier_id, content_hash) with 4h TTL.
    """
    cache_dir = _cache_dir(workspace)
    content_key = _content_hash(
        frontier_id, truth_status,
        " ".join((evidence_notes or [])[:6]),
        (domain_learned_summary or "")[:200],
    )
    cached_content, cached_structured = _cache_read_structured(cache_dir, f"synthesis_{frontier_id}_{content_key}")
    if cached_content is not None:
        return {"narrative": cached_content, **(cached_structured or {})}

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
        "and one specific experiment suggestion to fill it.\n\n"
        "After your synthesis paragraph, on a new line output a JSON block:\n"
        "```json\n"
        "{\"priority_delta\": <float -20 to +20>, \"gap_signals\": [<1-3 short gap strings>], "
        "\"needs_more_evidence\": <true/false>}\n"
        "```\n"
        "positive priority_delta = urgently needs more research; "
        "negative = well-covered, system should move to other problems."
    )
    raw = call_claude(prompt, system=_SYNTHESIS_SYSTEM, model=_SONNET, max_tokens=600, tag="frontier_synthesis")
    return _parse_synthesis_response(raw, cache_dir, f"synthesis_{frontier_id}_{content_key}")


def _parse_synthesis_response(raw: str, cache_dir: Path, cache_key: str) -> dict:
    fallback: dict = {"priority_delta": 0.0, "gap_signals": [], "needs_more_evidence": False}
    if not raw:
        return {"narrative": "", **fallback}

    narrative = raw
    structured = dict(fallback)
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            structured = {
                "priority_delta":      float(parsed.get("priority_delta", 0.0)),
                "gap_signals":         [str(s) for s in (parsed.get("gap_signals") or [])[:5]],
                "needs_more_evidence": bool(parsed.get("needs_more_evidence", False)),
            }
            narrative = raw[:json_match.start()].strip()
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    if narrative:
        _cache_write_structured(cache_dir, cache_key, narrative, structured)
    return {"narrative": narrative, **structured}


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
) -> dict:
    """
    Evaluate whether a proposed/queued experiment advances its frontier.
    Returns dict: {narrative, recommendation, priority_delta}.
    Skips running/complete experiments. Cached per (experiment_id, content_hash) with 4h TTL.
    """
    if status in {"running", "complete", "failed"}:
        return {"narrative": "", "recommendation": "neutral", "priority_delta": 0.0}
    cache_dir = _cache_dir(workspace)
    content_key = _content_hash(experiment_id, frontier_title, experiment_goal, status)
    cached_content, cached_structured = _cache_read_structured(cache_dir, f"eval_{experiment_id}_{content_key}")
    if cached_content is not None:
        return {"narrative": cached_content, **(cached_structured or {})}

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
    raw = call_claude(prompt, system=_EVAL_SYSTEM, model=_HAIKU, max_tokens=250, tag="exp_eval")
    return _parse_eval_response(raw, cache_dir, f"eval_{experiment_id}_{content_key}")


def _parse_eval_response(raw: str, cache_dir: Path, cache_key: str) -> dict:
    if not raw:
        return {"narrative": "", "recommendation": "neutral", "priority_delta": 0.0}
    last_sentence = raw.strip().rstrip(".").split(".")[-1].strip().lower()
    if "run now" in last_sentence:
        recommendation, delta = "run_now", 10.0
    elif "deprioritize" in last_sentence:
        recommendation, delta = "deprioritize", -15.0
    elif "run after" in last_sentence:
        recommendation, delta = "run_after", 0.0
    else:
        recommendation, delta = "neutral", 0.0
    structured = {"recommendation": recommendation, "priority_delta": delta}
    _cache_write_structured(cache_dir, cache_key, raw, structured)
    return {"narrative": raw, **structured}


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
) -> dict:
    """
    Verify whether a frontier claim is consistent with available evidence and literature.
    Returns dict: {narrative, confidence, needs_replication}.
    Cached per (frontier_id, claim_hash, truth_status) with 4h TTL.
    """
    cache_dir = _cache_dir(workspace)
    content_key = _content_hash(frontier_id, claim_text, truth_status)
    cached_content, cached_structured = _cache_read_structured(cache_dir, f"verify_{frontier_id}_{content_key}")
    if cached_content is not None:
        return {"narrative": cached_content, **(cached_structured or {})}

    notes_text = "\n".join(f"- {n}" for n in (evidence_notes or [])[:6]) or "- None yet"
    prompt = (
        f'Frontier claim: "{claim_text}"\n'
        f"Current evidence status: {truth_status}\n"
        f"TAR's experimental evidence:\n{notes_text}\n\n"
        f"Literature context: {(domain_learned_summary or 'Not available')[:400]}\n\n"
        "In 3-5 sentences: Is this claim Consistent, Contradicted, or Unsupported?\n"
        "State the verdict, explain the key reason, flag the strongest counter-evidence if any."
    )
    raw = call_claude(prompt, system=_VERIFY_SYSTEM, model=_HAIKU, max_tokens=200, tag="claim_verify")
    return _parse_verify_response(raw, cache_dir, f"verify_{frontier_id}_{content_key}")


def _parse_verify_response(raw: str, cache_dir: Path, cache_key: str) -> dict:
    if not raw:
        return {"narrative": "", "confidence": "low", "needs_replication": False}
    text_lower = raw.lower()
    if "contradicted" in text_lower:
        confidence, needs_replication = "low", True
    elif "unsupported" in text_lower:
        confidence, needs_replication = "medium", True
    else:
        confidence, needs_replication = "high", False
    structured = {"confidence": confidence, "needs_replication": needs_replication}
    _cache_write_structured(cache_dir, cache_key, raw, structured)
    return {"narrative": raw, **structured}


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
        prompt, system=_SCHEDULER_SYSTEM, model=_HAIKU, max_tokens=600, tag="scheduler_rationale"
    )
    if narration:
        _cache_write(cache_dir, f"scheduler_{content_key}", narration)
        return narration
    return template_fallback


_TTL_PROPOSAL_S = 2 * 3600


def propose_followup_experiments(
    workspace: Path,
    frontier_id: str,
    frontier_title: str,
    global_problem_statement: str,
    candidate_datasets: list[str],
    candidate_backbones: list[str],
    external_baselines: list[str],
    completed_summaries: list[str],
    exclude_ids: set[str],
    max_proposals: int = 2,
) -> list[dict]:
    """Ask Claude to propose follow-up experiments when the static catalog is exhausted.

    Returns [] if the API is unavailable or proposals don't validate.
    Cache TTL: 2 h, keyed on frontier + completed experiment set.
    """
    if not candidate_datasets or not candidate_backbones:
        return []

    cache_dir = _cache_dir(workspace)
    content_key = _content_hash(
        frontier_id,
        ",".join(sorted(completed_summaries)),
        ",".join(sorted(exclude_ids)),
    )
    cached = _cache_read(cache_dir, f"proposals_{content_key}", ttl_s=_TTL_PROPOSAL_S)
    if cached is not None:
        try:
            result = json.loads(cached)
            if isinstance(result, list):
                return result
        except Exception:
            pass

    summaries_str = "\n".join(f"- {s}" for s in completed_summaries) or "None yet"
    exclude_str = ", ".join(sorted(str(x) for x in exclude_ids)[:20]) or "none"

    prompt = f"""You are the TAR Research Director generating follow-up experiments for a machine learning research frontier.

Frontier: {frontier_title}
Problem: {global_problem_statement}

Completed experiments so far:
{summaries_str}

Available datasets: {', '.join(candidate_datasets)}
Available backbones: {', '.join(candidate_backbones)}
External baselines to compare against: {', '.join(external_baselines) or 'ewc, sgd_baseline'}

Already queued or done (do NOT propose these): {exclude_str}

RULES:
1. Only propose experiments using the listed available datasets and backbones
2. Always compare TCL against real external baselines — never test TCL alone
3. Each proposal must explore a meaningfully different axis than what is already done
4. TAR/TCL/ASC are unpublished internal candidate methods under evaluation — not assumed solutions
5. Prefer the cheapest dataset/backbone combination that still produces useful evidence

Return a JSON array of up to {max_proposals} experiments with these exact keys:
[
  {{
    "experiment_id": "unique-lowercase-slug-no-spaces",
    "title": "Short descriptive title",
    "dataset": "<one of the available datasets>",
    "backbone": "<one of the available backbones>",
    "estimated_runtime_h": 6.0,
    "config_overrides": {{}},
    "hypothesis": "One falsifiable sentence this experiment tests",
    "why": "Why this is the most valuable next step given results so far"
  }}
]

Return ONLY the JSON array, no other text."""

    system = (
        "You are the TAR Research Director. Propose follow-up ML experiments that are "
        "concrete, falsifiable, and grounded in completed results. "
        "Return only a valid JSON array."
    )
    raw = call_claude(prompt, system=system, model=_HAIKU, max_tokens=800, tag="followup_proposals")
    if not raw:
        return []

    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[^\n]*\n?", "", raw)
        raw = re.sub(r"```$", "", raw).strip()

    try:
        proposals = json.loads(raw)
        if not isinstance(proposals, list):
            return []
        valid: list[dict] = []
        for p in proposals[:max_proposals]:
            if not isinstance(p, dict):
                continue
            exp_id = str(p.get("experiment_id", "") or "").strip().lower().replace(" ", "-")
            if not exp_id or exp_id in exclude_ids:
                continue
            dataset = str(p.get("dataset", "") or "")
            backbone = str(p.get("backbone", "") or "")
            if dataset not in candidate_datasets or backbone not in candidate_backbones:
                continue
            valid.append({
                "experiment_id": exp_id,
                "title": str(p.get("title", exp_id) or exp_id)[:80],
                "dataset": dataset,
                "backbone": backbone,
                "estimated_runtime_h": float(p.get("estimated_runtime_h", 6.0) or 6.0),
                "config_overrides": dict(p.get("config_overrides") or {}),
                "hypothesis": str(p.get("hypothesis", "") or ""),
                "why": str(p.get("why", "") or ""),
            })
        if valid:
            _cache_write(cache_dir, f"proposals_{content_key}", json.dumps(valid))
        return valid
    except Exception:
        return []
