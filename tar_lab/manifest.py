"""
tar_lab/manifest.py — Execution manifest schema and validation.

RAIL 3: Daemons may not start training runs without a valid manifest.
Every manifest must:
  - Live as a committed file in the git repo (C: source-of-truth).
  - Have its content_hash match the SHA-256 of the file's actual content
    (computed over the JSON payload with content_hash set to the sentinel
    value "UNSIGNED" before hashing, so the hash can be embedded without
    circular dependency).
  - Be verified clean in `git status` — i.e., committed and not dirty.
  - List the exact experiment IDs authorised to run.

Usage by execution callers:
    from tar_lab.manifest import load_and_verify_manifest, ManifestGateError

    manifest = load_and_verify_manifest(
        manifest_path=Path("manifests/run-phase16-rerun.json"),
        repo_root=_REPO,
    )
    manifest.assert_experiment_authorised("phase16-rerun-seed0")

Writing a manifest (user step, not automated):
    Produce a JSON file matching ExecutionManifest, set content_hash to
    "UNSIGNED", compute sha256 over the file content, replace "UNSIGNED"
    with the hex digest, commit to git.  Helper: compute_manifest_hash().
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_HASH_SENTINEL = "UNSIGNED"


class ManifestGateError(Exception):
    """Raised when execution cannot proceed due to a missing or invalid manifest."""


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class ManifestExperimentEntry:
    """One experiment that this manifest authorises."""
    experiment_id: str
    name: str
    allowed_datasets: list[str] = field(default_factory=list)
    allowed_methods: list[str] = field(default_factory=list)
    allowed_seeds: list[int] = field(default_factory=list)
    time_limit_h: float = 48.0
    run_limit: int = 1
    notes: str = ""


@dataclass
class ExecutionManifest:
    """
    A user-authored authorisation document for one or more experiment runs.

    Fields
    ------
    manifest_id       Unique string, e.g. "manifest-phase16-rerun-20260515".
    manifest_schema   Always "tar_execution_manifest_v1".
    created_at        ISO timestamp when the manifest was written.
    authorised_by     git config user.name of the human who wrote it.
    purpose           Human-readable description of why these runs are needed.
    experiments       List of ManifestExperimentEntry — exactly what is allowed.
    global_time_limit_h  Wall-clock budget for the whole session.
    content_hash      SHA-256 hex of the file with content_hash set to
                      "UNSIGNED" — see compute_manifest_hash().  Must be
                      present and correct for the manifest to pass verification.
    _path             Set at load time; not serialised.
    """
    manifest_id: str
    manifest_schema: str
    created_at: str
    authorised_by: str
    purpose: str
    experiments: list[ManifestExperimentEntry]
    global_time_limit_h: float = 48.0
    content_hash: str = _HASH_SENTINEL
    _path: Optional[Path] = field(default=None, repr=False, compare=False)

    def assert_experiment_authorised(self, experiment_id: str) -> ManifestExperimentEntry:
        """Raise ManifestGateError if experiment_id is not in this manifest."""
        for entry in self.experiments:
            if entry.experiment_id == experiment_id:
                return entry
        ids = [e.experiment_id for e in self.experiments]
        raise ManifestGateError(
            f"Experiment '{experiment_id}' is not listed in manifest "
            f"'{self.manifest_id}'. Authorised IDs: {ids}. "
            f"Add it to the manifest and re-commit before running."
        )

    def authorised_ids(self) -> list[str]:
        return [e.experiment_id for e in self.experiments]


# ── hash helpers ──────────────────────────────────────────────────────────────

def compute_manifest_hash(manifest_path: Path) -> str:
    """
    Compute the content hash for a manifest file.

    The hash is SHA-256 of the UTF-8-encoded JSON payload with the
    content_hash field replaced by the sentinel "UNSIGNED".  This lets the
    hash be embedded in the file without a circular dependency.

    Use this when creating a manifest: set content_hash to "UNSIGNED", call
    this function, then replace the field with the result, then commit.
    """
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["content_hash"] = _HASH_SENTINEL
    canonical = json.dumps(raw, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_hash(manifest_path: Path, claimed_hash: str) -> None:
    actual = compute_manifest_hash(manifest_path)
    if actual != claimed_hash:
        raise ManifestGateError(
            f"Manifest content hash mismatch for '{manifest_path.name}'. "
            f"Claimed: {claimed_hash[:16]}…  Actual: {actual[:16]}…  "
            f"The file has been modified since the hash was computed. "
            f"Re-compute and re-commit."
        )


def _verify_git_committed(manifest_path: Path, repo_root: Path) -> None:
    """Refuse if the manifest file is dirty or untracked in git."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", str(manifest_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        raise ManifestGateError(
            f"Cannot verify git status for manifest '{manifest_path}': {exc}"
        ) from exc

    status_line = (result.stdout or "").strip()
    if status_line:
        raise ManifestGateError(
            f"Manifest '{manifest_path.name}' is not cleanly committed in git. "
            f"git status output: {repr(status_line)}. "
            f"Commit the manifest before running."
        )

    # Also check the file is actually tracked (not just clean because unknown)
    try:
        ls = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(manifest_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        raise ManifestGateError(
            f"Cannot verify git tracking for manifest '{manifest_path}': {exc}"
        ) from exc

    if ls.returncode != 0:
        raise ManifestGateError(
            f"Manifest '{manifest_path.name}' is not tracked by git. "
            f"Add it with `git add` and commit before running."
        )


# ── load / verify ─────────────────────────────────────────────────────────────

def load_and_verify_manifest(
    manifest_path: Path,
    repo_root: Path,
    *,
    skip_git_check: bool = False,
) -> ExecutionManifest:
    """
    Load and fully verify a manifest file.

    Raises ManifestGateError with a human-readable explanation if any check
    fails.  Never returns a manifest that has not passed all checks.

    Parameters
    ----------
    manifest_path   Path to the manifest JSON file.  If relative, resolved
                    against repo_root.
    repo_root       Root of the git repository (C: drive).
    skip_git_check  Set True only in unit tests — never in production code.
    """
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path

    if not manifest_path.exists():
        raise ManifestGateError(
            f"Manifest file not found: '{manifest_path}'. "
            f"Create and commit a manifest before running."
        )

    try:
        raw: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ManifestGateError(
            f"Cannot parse manifest '{manifest_path.name}': {exc}"
        ) from exc

    schema = raw.get("manifest_schema", "")
    if schema != "tar_execution_manifest_v1":
        raise ManifestGateError(
            f"Unknown manifest schema '{schema}' in '{manifest_path.name}'. "
            f"Expected 'tar_execution_manifest_v1'."
        )

    claimed_hash = raw.get("content_hash", "")
    if not claimed_hash or claimed_hash == _HASH_SENTINEL:
        raise ManifestGateError(
            f"Manifest '{manifest_path.name}' has no content_hash (or still "
            f"contains the UNSIGNED sentinel). Compute and embed the hash, "
            f"then commit."
        )

    _verify_hash(manifest_path, claimed_hash)

    if not skip_git_check:
        _verify_git_committed(manifest_path, repo_root)

    experiments = [
        ManifestExperimentEntry(**{
            k: v for k, v in e.items()
            if k in ManifestExperimentEntry.__dataclass_fields__
        })
        for e in raw.get("experiments", [])
    ]

    manifest = ExecutionManifest(
        manifest_id=str(raw.get("manifest_id", "")),
        manifest_schema=schema,
        created_at=str(raw.get("created_at", "")),
        authorised_by=str(raw.get("authorised_by", "")),
        purpose=str(raw.get("purpose", "")),
        experiments=experiments,
        global_time_limit_h=float(raw.get("global_time_limit_h", 48.0)),
        content_hash=claimed_hash,
        _path=manifest_path,
    )

    if not manifest.manifest_id:
        raise ManifestGateError(
            f"Manifest '{manifest_path.name}' has no manifest_id field."
        )
    if not manifest.experiments:
        raise ManifestGateError(
            f"Manifest '{manifest_path.manifest_id}' authorises zero experiments. "
            f"Nothing to run."
        )

    return manifest


def write_refuse_note(
    workspace: Path,
    component: str,
    reason: str,
    *,
    experiment_id: str = "",
    manifest_path: str = "",
) -> None:
    """
    Write a structured refusal note to tar_state/stat_audit/ when a
    spawn is refused due to a missing or invalid manifest.
    """
    from tar_lab.result_artifacts import utc_now_iso, utc_stamp

    note = {
        "artifact_schema": "tar_manifest_refusal_v1",
        "refused_at": utc_now_iso(),
        "component": component,
        "reason": reason,
        "experiment_id": experiment_id,
        "manifest_path": manifest_path,
    }
    audit_dir = workspace / "tar_state" / "stat_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    note_path = audit_dir / f"manifest_refusal__{stamp}.json"
    note_path.write_text(json.dumps(note, indent=2), encoding="utf-8")
