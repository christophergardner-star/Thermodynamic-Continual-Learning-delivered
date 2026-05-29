"""
tar_lab/canonical_registry.py — Canonical result registry for validation-suite artifacts.

Registers validation experiment results (tar_state/experiments/*/result.json) into
canonical_results_index.jsonl via a three-gate verification pipeline and a
lockfile-protected atomic write.

ACCEPTED RESIDUAL BEHAVIOURS
----------------------------
1. Without psutil, stale-lock reclaim never fires: a dead-process lock requires
   manual clearance (remove canonical_results_index.jsonl.lock). Accepted: loud
   and safe over silent and clever.

2. A concurrent registration may receive a spurious RegistryLockTimeout if it
   arrives during the holder's close→unlink release window. Intentional: the
   failure is loud, safe, and retryable; it is not fixed because doing so adds
   risk to the critical section for a benign outcome.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO

import numpy


REGISTRY_VERSION = "1"


# ── Exception classes ─────────────────────────────────────────────────────────

class ProvenanceSiblingInvalidError(Exception):
    """Gate 1 failure: result_env.json sibling is missing, unparseable, or
    does not contain the required provenance fields; also raised on path
    layout assumption violations in register_canonical_result."""


class ManifestIntegrityMismatchedError(Exception):
    """Gate 2 failure: manifest is dirty, untracked, or its computed hash
    does not match the frozen hash recorded in result_env.json."""


class DeterministicRecomputeFailure(Exception):
    """Gate 3 failure: standalone numpy recompute of aggregate statistics
    does not match the summary block in the result artifact to atol=1e-9."""


class RegistryLockTimeout(Exception):
    """Raised when lockfile acquisition fails due to a concurrent holder or
    a stale-lock race loss."""


# ── Supporting helpers ────────────────────────────────────────────────────────

def _compute_env_hash(env_path: Path) -> str:
    """SHA-256 of the raw bytes of result_env.json, returned as hex digest."""
    return hashlib.sha256(env_path.read_bytes()).hexdigest()


def _pid_alive(pid: int) -> bool:
    """Return True if the process with *pid* is still running.

    Uses psutil when available (Windows-safe). Falls back to True when psutil
    is absent — conservative: treat-as-alive prevents aggressive stale reclaim.
    See Accepted Residual Behaviour #1 in this module's docstring.
    """
    try:
        import psutil  # type: ignore[import]
        return bool(psutil.pid_exists(pid))
    except ImportError:
        return True


def _acquire_registry_lock(lock_path: Path) -> IO:
    """Acquire the registry lockfile via atomic exclusive creation.

    On success writes {pid, acquired_at} JSON to the returned open handle.
    The caller MUST release via the try/finally pattern from Section 3:

        lock_handle = _acquire_registry_lock(lock_path)
        try:
            ...critical section...
        finally:
            lock_handle.close()
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass

    Raises RegistryLockTimeout on all race-loser paths.
    IMPLEMENTATION NOTE: Never calls os.path.exists() before os.unlink() or
    before open('x') — the exception-trapping IS the synchronisation.
    """
    try:
        handle = open(lock_path, "x", encoding="utf-8")
        handle.write(json.dumps({"pid": os.getpid(), "acquired_at": time.time()}))
        handle.flush()
        return handle
    except FileExistsError:
        pass

    # Lock file exists — read mtime for staleness check.
    try:
        mtime = os.stat(lock_path).st_mtime
        age = time.time() - mtime
    except FileNotFoundError:
        # Lock vanished between our FileExistsError and the stat; retry once.
        try:
            handle = open(lock_path, "x", encoding="utf-8")
            handle.write(json.dumps({"pid": os.getpid(), "acquired_at": time.time()}))
            handle.flush()
            return handle
        except FileExistsError:
            raise RegistryLockTimeout(
                f"Registry lock held by another process: {lock_path}"
            )

    if age <= 60:
        raise RegistryLockTimeout(
            f"Registry lock is fresh (age={age:.1f}s <= 60s): {lock_path}"
        )

    # Read PID from lock content for liveness check.
    pid: int | None = None
    try:
        raw_content = json.loads(lock_path.read_text(encoding="utf-8"))
        raw_pid = raw_content.get("pid")
        if raw_pid is not None:
            pid = int(raw_pid)
    except Exception:
        pid = None

    if pid is None or _pid_alive(pid):
        raise RegistryLockTimeout(
            f"Registry lock held by live or unreadable PID {pid}: {lock_path}"
        )

    # Lock is stale — attempt reclaim.  No pre-existence check: exception IS the sync.
    try:
        os.unlink(lock_path)
    except FileNotFoundError:
        # Race loser #1: another process already cleared the lock.
        raise RegistryLockTimeout(
            f"Stale-lock reclaim race lost (FileNotFoundError): {lock_path}"
        )
    except PermissionError:
        # Race loser #2: another process holds an active open handle.
        raise RegistryLockTimeout(
            f"Stale-lock reclaim race lost (PermissionError): {lock_path}"
        )

    # Unlink succeeded — atomic re-acquire.  No pre-existence check.
    try:
        handle = open(lock_path, "x", encoding="utf-8")
        handle.write(json.dumps({"pid": os.getpid(), "acquired_at": time.time()}))
        handle.flush()
        return handle
    except FileExistsError:
        # Race loser #3: a third process slipped in after our unlink.
        raise RegistryLockTimeout(
            f"Stale-lock reclaim race lost (FileExistsError on re-acquire): {lock_path}"
        )


# ── Gate helpers (Section 2 — all execute entirely outside any lock) ──────────

def _check_gate_1_sibling(result_path: Path) -> dict[str, str]:
    """Gate 1 — Environment sibling coupling.

    Verifies result_env.json sibling exists, parses as JSON, and explicitly
    contains the required provenance fields: git.head, authorization.manifest_hash,
    and at least one of authorization.manifest_path / manifest_id.

    Returns extracted env_fields dict.  Raises ProvenanceSiblingInvalidError.
    """
    env_path = result_path.with_name("result_env.json")
    if not env_path.exists():
        raise ProvenanceSiblingInvalidError(
            f"result_env.json sibling not found: {env_path}"
        )

    try:
        raw = json.loads(env_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProvenanceSiblingInvalidError(
            f"result_env.json is not valid JSON ({env_path}): {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ProvenanceSiblingInvalidError(
            f"result_env.json root is not a JSON object: {env_path}"
        )

    git_block = raw.get("git") if isinstance(raw.get("git"), dict) else {}
    git_head = (git_block or {}).get("head")
    if not git_head:
        raise ProvenanceSiblingInvalidError(
            f"result_env.json missing git.head: {env_path}"
        )

    auth_block = raw.get("authorization") or {}
    if not isinstance(auth_block, dict):
        raise ProvenanceSiblingInvalidError(
            f"result_env.json authorization block is not a dict: {env_path}"
        )

    manifest_hash = str(auth_block.get("manifest_hash") or "")
    if not manifest_hash:
        raise ProvenanceSiblingInvalidError(
            f"result_env.json missing authorization.manifest_hash: {env_path}"
        )

    manifest_path_str = str(auth_block.get("manifest_path") or "")
    manifest_id = str(
        auth_block.get("manifest_id")
        or (raw.get("extra") or {}).get("manifest_id")
        or ""
    )
    if not manifest_path_str and not manifest_id:
        raise ProvenanceSiblingInvalidError(
            f"result_env.json missing both authorization.manifest_path and "
            f"manifest_id: {env_path}"
        )

    return {
        "git_head": str(git_head),
        "manifest_hash": manifest_hash,
        "manifest_path": manifest_path_str,
        "manifest_id": manifest_id,
    }


def _check_gate_2_manifest(env_fields: dict[str, str], repo_root: Path) -> None:
    """Gate 2 — Manifest hash verification and hierarchy.

    Precondition: manifest must be cleanly committed in git.  Two checks are
    required:
      1. git status --porcelain -- <path>  must return strictly empty stdout.
      2. git ls-files --error-unmatch -- <path>  must exit 0.

    Note on the ls-files check: git status --porcelain alone does not reliably
    return non-empty output for untracked files in all git configurations. The
    ls-files check honours the stated intent that untracked manifests must fail.
    This is a spec correction surfaced against existing codebase practice
    (see tar_lab/manifest.py _verify_git_committed).

    Resolves manifest path ONLY from frozen env_fields["manifest_path"].
    Never falls back to live workspace state or ambient configuration.

    Raises ManifestIntegrityMismatchedError on any failure.
    """
    manifest_path_str = env_fields.get("manifest_path", "")
    if not manifest_path_str:
        raise ManifestIntegrityMismatchedError(
            "env_fields contains no manifest_path; cannot verify manifest integrity."
        )

    manifest_path = Path(manifest_path_str)
    if not manifest_path.exists():
        raise ManifestIntegrityMismatchedError(
            f"Manifest file not found at frozen path: {manifest_path}"
        )

    # Precondition 1: git status must be strictly empty.
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--", str(manifest_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        raise ManifestIntegrityMismatchedError(
            f"Cannot run git status for manifest '{manifest_path.name}': {exc}"
        ) from exc

    status_output = (status_result.stdout or "").strip()
    if status_output:
        raise ManifestIntegrityMismatchedError(
            f"Manifest '{manifest_path.name}' is not cleanly committed "
            f"(git status: {repr(status_output)})."
        )

    # Precondition 2: manifest must be tracked by git.
    try:
        ls_result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(manifest_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        raise ManifestIntegrityMismatchedError(
            f"Cannot run git ls-files for manifest '{manifest_path.name}': {exc}"
        ) from exc

    if ls_result.returncode != 0:
        raise ManifestIntegrityMismatchedError(
            f"Manifest '{manifest_path.name}' is not tracked by git "
            f"(ls-files returncode={ls_result.returncode})."
        )

    # Authoritative invariant: fresh recomputed hash must match frozen hash.
    from tar_lab.manifest import compute_manifest_hash
    try:
        fresh_hash = compute_manifest_hash(manifest_path)
    except Exception as exc:
        raise ManifestIntegrityMismatchedError(
            f"Cannot compute manifest hash for '{manifest_path.name}': {exc}"
        ) from exc

    frozen_hash = env_fields.get("manifest_hash", "")
    if fresh_hash != frozen_hash:
        raise ManifestIntegrityMismatchedError(
            f"Manifest hash mismatch for '{manifest_path.name}': "
            f"frozen={frozen_hash[:16]}...  recomputed={fresh_hash[:16]}..."
        )


def _check_gate_3_recompute(result_path: Path) -> dict[str, float]:
    """Gate 3 — Deterministic seed-to-aggregate recompute.

    Independence invariant: uses only numpy (stdlib + numpy). Does NOT import
    or call any TAR aggregation function.

    Raises DeterministicRecomputeFailure if recomputed aggregates deviate from
    stored values by more than atol=1e-9 on any checked field.

    Returns dict of recomputed aggregate values.
    """
    try:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DeterministicRecomputeFailure(
            f"Cannot parse result artifact {result_path}: {exc}"
        ) from exc

    # Handle both wrap_verdict_separation schema (statistics sub-dict) and flat schema.
    if "statistics" in raw and isinstance(raw["statistics"], dict):
        stats_block: dict[str, Any] = raw["statistics"]
    else:
        stats_block = raw

    seed_results = (
        stats_block.get("seed_results")
        or raw.get("seed_results")
        or []
    )
    if not seed_results:
        raise DeterministicRecomputeFailure(
            f"No seed_results found in {result_path}."
        )

    forgetting_vals = [
        float(r["forgetting"])
        for r in seed_results
        if r.get("forgetting") is not None
    ]
    accuracy_vals = [
        float(r["accuracy"])
        for r in seed_results
        if r.get("accuracy") is not None
    ]

    if not forgetting_vals:
        raise DeterministicRecomputeFailure(
            f"No valid forgetting values in seed_results of {result_path}."
        )

    # Standalone numpy recompute — no TAR aggregation imports.
    f_arr = numpy.array(forgetting_vals, dtype=numpy.float64)
    recomputed_mean_f = float(numpy.mean(f_arr))
    recomputed_std_f = float(numpy.std(f_arr, ddof=1)) if len(f_arr) > 1 else 0.0

    if accuracy_vals:
        a_arr = numpy.array(accuracy_vals, dtype=numpy.float64)
        recomputed_mean_a = float(numpy.mean(a_arr))
        recomputed_std_a = float(numpy.std(a_arr, ddof=1)) if len(a_arr) > 1 else 0.0
    else:
        recomputed_mean_a = 0.0
        recomputed_std_a = 0.0

    recomputed = {
        "mean_forgetting": recomputed_mean_f,
        "std_forgetting": recomputed_std_f,
        "mean_accuracy": recomputed_mean_a,
        "std_accuracy": recomputed_std_a,
    }

    # Retrieve stored aggregates from stats_block or flat schema.
    def _get_stored(field: str) -> float | None:
        v = stats_block.get(field) if field in stats_block else raw.get(field)
        return None if v is None else float(v)

    atol = 1e-9
    for field_name, recomputed_val in recomputed.items():
        stored_val = _get_stored(field_name)
        if stored_val is None:
            if "accuracy" in field_name and not accuracy_vals:
                continue
            raise DeterministicRecomputeFailure(
                f"Stored field '{field_name}' is absent in {result_path}."
            )
        if abs(recomputed_val - stored_val) > atol:
            raise DeterministicRecomputeFailure(
                f"Aggregate mismatch on '{field_name}' in {result_path.name}: "
                f"recomputed={recomputed_val:.15g}  stored={stored_val:.15g}  "
                f"delta={abs(recomputed_val - stored_val):.3e}  atol={atol:.3e}"
            )

    return recomputed


# ── Main entry point ──────────────────────────────────────────────────────────

def register_canonical_result(validation_result_path: Path) -> dict[str, Any]:
    """Register a validation experiment result into canonical_results_index.jsonl.

    Executes the three-gate verification pipeline (entirely outside the lock),
    then acquires the lockfile mutex and writes a self-contained audit record.

    Path layout assumption: validation_result_path must be at
    workspace/tar_state/experiments/<run_id>/result.json.
    An explicit assertion validates this assumption before any gate runs; a
    layout mismatch raises ProvenanceSiblingInvalidError rather than silently
    resolving the wrong workspace (Binding 3 from authorised spec review).

    Returns the written audit record dict.
    """
    validation_result_path = validation_result_path.resolve()
    run_id = validation_result_path.parent.name
    workspace = validation_result_path.parents[3]

    # Self-validate the positional derivation — fail loud, not silent.
    expected = (
        workspace / "tar_state" / "experiments" / run_id / "result.json"
    ).resolve()
    if expected != validation_result_path:
        raise ProvenanceSiblingInvalidError(
            f"Path layout assumption violated: expected {expected}, "
            f"got {validation_result_path}. "
            f"register_canonical_result requires "
            f"workspace/tar_state/experiments/<run_id>/result.json."
        )

    comparisons_dir = workspace / "tar_state" / "comparisons"
    if not comparisons_dir.is_dir():
        raise ProvenanceSiblingInvalidError(
            f"Comparisons directory not found: {comparisons_dir}. "
            f"Workspace layout assumption violated."
        )

    repo_root = workspace
    env_path = validation_result_path.with_name("result_env.json")

    # ── Gates 1, 2, 3 — entirely outside the lock ─────────────────────────────
    env_fields = _check_gate_1_sibling(validation_result_path)
    _check_gate_2_manifest(env_fields, repo_root)
    recomputed_aggregates = _check_gate_3_recompute(validation_result_path)

    env_snapshot_hash = _compute_env_hash(env_path)
    index_path = comparisons_dir / "canonical_results_index.jsonl"
    lock_path = comparisons_dir / "canonical_results_index.jsonl.lock"

    # ── Critical section: lockfile-protected atomic write ─────────────────────
    lock_handle = _acquire_registry_lock(lock_path)
    try:
        # Re-check under lock: run_id must not already be registered.
        if index_path.exists():
            for raw_line in index_path.read_text(encoding="utf-8").splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if entry.get("run_id") == run_id:
                    raise ValueError(
                        f"run_id '{run_id}' is already registered in {index_path}."
                    )

        # Compile self-contained audit payload.
        audit_record: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_head": env_fields["git_head"],
            "manifest_hash": env_fields["manifest_hash"],
            "env_snapshot_hash": env_snapshot_hash,
            "recomputed_aggregates": recomputed_aggregates,
            "registry_version": REGISTRY_VERSION,
        }

        # Atomic append + hardware sync.
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        with open(index_path, "a", encoding="utf-8") as index_fh:
            index_fh.write(json.dumps(audit_record) + "\n")
            os.fsync(index_fh.fileno())

    finally:
        lock_handle.close()
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass

    return audit_record


# ── Discovery and quarantine (Section 1) ──────────────────────────────────────

def quarantine_unregistered_results(
    workspace: Path,
    index_records: list[dict[str, Any]],
) -> tuple[list[str], bool]:
    """Discovery scan and quarantine check for build_validation_state.

    Performs a RAIL-6 compliant non-recursive single-level glob:
      tar_state/experiments/*/result.json
    os.walk is explicitly not used.

    Phase-A two-tier trust model
    ----------------------------
    Tier 1 — Index-registered: experiment directory name matches a ``run_id``
        or ``logical_name`` field in any index_records entry.  Accepted without
        env_snapshot_hash verification.  Phase B will tighten this to require
        hash verification for entries that carry an env_snapshot_hash.

    Tier 2 — Archive-complete: experiment appears in experiment_archive.json
        with status ``"complete"`` but has no matching index entry.  These are
        pre-registration results that completed through the autonomous pipeline
        before the canonical index existed; accepted without hash verification.

    Quarantine trigger: an experiment that has a ``result_env.json`` sibling
        (i.e. is env-snapshot-capable) yet matches neither tier.  An env-capable
        result with no traceable provenance is the only genuinely suspicious
        case.  Experiments without result_env.json predate the env-snapshot
        system and cannot be hash-verified in any tier; they are not flagged.

    Returns (quarantine_set, is_clean_trust_state).
    is_clean_trust_state = (len(quarantine_set) == 0).

    Never raises: always completes and preserves observability (Section 1).
    """
    experiments_dir = workspace / "tar_state" / "experiments"
    if not experiments_dir.is_dir():
        return [], True

    # Tier 1: names drawn from index records — accept run_id OR logical_name.
    index_names: set[str] = set()
    for record in index_records:
        r_id = record.get("run_id")
        l_name = record.get("logical_name")
        if r_id:
            index_names.add(str(r_id))
        if l_name:
            index_names.add(str(l_name))

    # Tier 2: completed IDs from the autonomous experiment archive.
    archive_complete: set[str] = set()
    archive_path = workspace / "tar_state" / "experiment_archive.json"
    if archive_path.exists():
        try:
            archive_data = json.loads(archive_path.read_text(encoding="utf-8"))
            for exp in archive_data.get("experiments", []):
                if exp.get("status") == "complete" and exp.get("id"):
                    archive_complete.add(str(exp["id"]))
        except Exception:
            pass  # unreadable archive — no tier-2 trust granted this run

    # Non-recursive single-level glob (RAIL-6 compliant; os.walk forbidden).
    quarantine_set: list[str] = []
    for result_path in experiments_dir.glob("*/result.json"):
        run_id = result_path.parent.name

        if run_id in index_names:        # tier 1
            continue
        if run_id in archive_complete:   # tier 2
            continue

        # Neither tier matched.  Only flag if an env snapshot is present —
        # env-capable but unregistered is the suspicious case.
        env_path = result_path.with_name("result_env.json")
        if env_path.exists():
            quarantine_set.append(run_id)

    is_clean_trust_state = len(quarantine_set) == 0
    return quarantine_set, is_clean_trust_state
