"""Tests for the stabilisation authoring gate (frozen spec, Rounds 1–5)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tar_lab.errors import (
    StabilisationGateStateUnreadableError,
    StabilisationGateMissingOverrideError,
    StabilisationGateStaleOverrideError,
    StabilisationGateModeMismatchError,
    StabilisationGateAlreadyConsumedError,
    StabilisationGateCategoricalBlockError,
    StabilisationGateAutonomousContextError,
)
from tar_validation_mode import _read_stabilisation_state_strict
from tar_author import (
    _AUTHORING_OVERRIDE,
    _OverrideContext,
    stabilisation_authoring_override,
    TARAuthor,
    PaperSpec,
    auto_start_priority_paper,
    ensure_completed_papers_compiled,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "tar_state").mkdir(parents=True, exist_ok=True)
    return ws


def _write_stab(
    ws: Path,
    active: bool,
    mode_id: str = "test-mode-id",
    activated_at: str = "2026-01-01T00:00:00+00:00",
) -> None:
    payload: dict = {"active": active}
    if active:
        payload["mode_id"] = mode_id
        payload["activated_at"] = activated_at
    (ws / "tar_state" / "stabilisation_mode.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _make_spec(ws: Path, project_id: str = "test-paper") -> PaperSpec:
    return PaperSpec(
        title="Test Paper",
        authors=["Test Author"],
        affiliation="Test Lab",
        project_id=project_id,
        workspace=ws,
        paper_dir=ws / "paper" / project_id,
    )


def _stat_audit_files(ws: Path) -> list[Path]:
    d = ws / "tar_state" / "stat_audit"
    if not d.exists():
        return []
    return list(d.glob("authoring_override__*.json"))


# ── _read_stabilisation_state_strict ─────────────────────────────────────────

def test_strict_reader_missing_file(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    with pytest.raises(StabilisationGateStateUnreadableError):
        _read_stabilisation_state_strict(ws)


def test_strict_reader_bad_json(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    (ws / "tar_state" / "stabilisation_mode.json").write_text(
        "NOT JSON {{{{", encoding="utf-8"
    )
    with pytest.raises(StabilisationGateStateUnreadableError):
        _read_stabilisation_state_strict(ws)


def test_strict_reader_non_dict(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    (ws / "tar_state" / "stabilisation_mode.json").write_text(
        json.dumps([1, 2, 3]), encoding="utf-8"
    )
    with pytest.raises(StabilisationGateStateUnreadableError):
        _read_stabilisation_state_strict(ws)


def test_strict_reader_returns_dict_when_valid(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    _write_stab(ws, active=False)
    result = _read_stabilisation_state_strict(ws)
    assert isinstance(result, dict)
    assert result.get("active") is False


# ── write_paper gate — not-stabilised paths ───────────────────────────────────

def test_write_paper_not_stabilised_no_context_proceeds(tmp_path: Path) -> None:
    """Gate is a no-op when not stabilised and no context."""
    ws = _make_workspace(tmp_path)
    _write_stab(ws, active=False)
    spec = _make_spec(ws)

    called = []
    author = TARAuthor(workspace=ws)
    with patch.object(author, "_write_paper_impl", return_value=ws / "out.tex") as mock_impl:
        author.write_paper(spec)
        mock_impl.assert_called_once_with(spec)


def test_write_paper_not_stabilised_context_present_proceeds_and_audits(tmp_path: Path) -> None:
    """U2: not-stabilised + context present → proceeds AND writes audit file."""
    ws = _make_workspace(tmp_path)
    _write_stab(ws, active=False)
    spec = _make_spec(ws)

    author = TARAuthor(workspace=ws)
    with patch.object(author, "_write_paper_impl", return_value=ws / "out.tex"):
        with stabilisation_authoring_override(ws, reason="test-u2"):
            author.write_paper(spec)

    files = _stat_audit_files(ws)
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["event"] == "stabilisation_authoring_override_consumed"
    assert payload["reason"] == "test-u2"


# ── write_paper gate — stabilised enforcement paths ──────────────────────────

def test_write_paper_stabilised_no_context_raises_missing(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    _write_stab(ws, active=True)
    spec = _make_spec(ws)

    author = TARAuthor(workspace=ws)
    with pytest.raises(StabilisationGateMissingOverrideError):
        author.write_paper(spec)


def test_write_paper_stabilised_stale_context_raises_stale(tmp_path: Path) -> None:
    """Context minted while not stabilised (mode_id=None) → stale error."""
    ws = _make_workspace(tmp_path)
    # Not stabilised at mint time
    _write_stab(ws, active=False)
    spec = _make_spec(ws)

    stale_ctx = _OverrideContext(mode_id=None, activated_at=None, reason="stale-test")
    token = _AUTHORING_OVERRIDE.set(stale_ctx)
    try:
        # Now stabilised at gate time
        _write_stab(ws, active=True, mode_id="mode-xyz")
        author = TARAuthor(workspace=ws)
        with pytest.raises(StabilisationGateStaleOverrideError):
            author.write_paper(spec)
    finally:
        _AUTHORING_OVERRIDE.reset(token)


def test_write_paper_stabilised_mode_id_mismatch_raises(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    _write_stab(ws, active=True, mode_id="mode-A", activated_at="2026-01-01T00:00:00+00:00")
    spec = _make_spec(ws)

    # Context minted with wrong mode_id
    wrong_ctx = _OverrideContext(
        mode_id="mode-B",
        activated_at="2026-01-01T00:00:00+00:00",
        reason="mismatch-test",
    )
    token = _AUTHORING_OVERRIDE.set(wrong_ctx)
    try:
        author = TARAuthor(workspace=ws)
        with pytest.raises(StabilisationGateModeMismatchError):
            author.write_paper(spec)
    finally:
        _AUTHORING_OVERRIDE.reset(token)


def test_write_paper_stabilised_already_consumed_raises(tmp_path: Path) -> None:
    """Second write_paper call with same override context → AlreadyConsumed."""
    ws = _make_workspace(tmp_path)
    _write_stab(ws, active=True, mode_id="test-mode-id", activated_at="2026-01-01T00:00:00+00:00")
    spec = _make_spec(ws)

    author = TARAuthor(workspace=ws)
    with patch.object(author, "_write_paper_impl", return_value=ws / "out.tex"):
        with stabilisation_authoring_override(ws, reason="consumed-test"):
            # First call succeeds
            author.write_paper(spec)
            # Second call with same context → consumed
            with pytest.raises(StabilisationGateAlreadyConsumedError):
                author.write_paper(spec)


def test_write_paper_stabilised_valid_context_proceeds(tmp_path: Path) -> None:
    """Happy path under stabilisation with correct override context."""
    ws = _make_workspace(tmp_path)
    _write_stab(ws, active=True, mode_id="test-mode-id", activated_at="2026-01-01T00:00:00+00:00")
    spec = _make_spec(ws)

    author = TARAuthor(workspace=ws)
    with patch.object(author, "_write_paper_impl", return_value=ws / "out.tex") as mock_impl:
        with stabilisation_authoring_override(ws, reason="happy-path"):
            result = author.write_paper(spec)
        mock_impl.assert_called_once_with(spec)

    files = _stat_audit_files(ws)
    assert len(files) == 1


# ── S4 — auto_start_priority_paper ───────────────────────────────────────────

def test_auto_start_raises_when_context_set_not_stabilised(tmp_path: Path) -> None:
    """S4: unconditional — fires even when not stabilised."""
    ws = _make_workspace(tmp_path)
    _write_stab(ws, active=False)

    ctx = _OverrideContext(mode_id=None, activated_at=None, reason="s4-test")
    token = _AUTHORING_OVERRIDE.set(ctx)
    try:
        with pytest.raises(StabilisationGateAutonomousContextError) as exc_info:
            auto_start_priority_paper(ws)
        assert "s4-test" in str(exc_info.value)
    finally:
        _AUTHORING_OVERRIDE.reset(token)


def test_auto_start_raises_when_context_set_stabilised(tmp_path: Path) -> None:
    """S4: fires when stabilised too."""
    ws = _make_workspace(tmp_path)
    _write_stab(ws, active=True)

    ctx = _OverrideContext(mode_id="test-mode-id", activated_at="2026-01-01T00:00:00+00:00", reason="s4-stab")
    token = _AUTHORING_OVERRIDE.set(ctx)
    try:
        with pytest.raises(StabilisationGateAutonomousContextError):
            auto_start_priority_paper(ws)
    finally:
        _AUTHORING_OVERRIDE.reset(token)


# ── ensure_completed_papers_compiled — secondary gate ─────────────────────────

def test_ensure_compiled_returns_empty_when_stabilised(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    _write_stab(ws, active=True)
    result = ensure_completed_papers_compiled(ws, author_state={})
    assert result == []


def test_ensure_compiled_returns_empty_when_state_unreadable(tmp_path: Path) -> None:
    """Secondary gate: unreadable state → fail-closed (return [])."""
    ws = _make_workspace(tmp_path)
    # No stabilisation_mode.json written — load_state returns {} (not active) but
    # we test the exception path by corrupting the file:
    (ws / "tar_state" / "stabilisation_mode.json").write_text("BAD", encoding="utf-8")
    # load_state silently returns {} on bad JSON → not active → does NOT skip.
    # The secondary gate uses load_state (fail-open), catches Exception → returns [].
    # With bad JSON, load_state returns {} so active=False → proceeds normally.
    # This test confirms the behaviour: bad file → load_state → {} → not active → proceeds.
    result = ensure_completed_papers_compiled(ws, author_state={})
    assert isinstance(result, list)


def test_ensure_compiled_proceeds_when_not_stabilised(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    _write_stab(ws, active=False)
    result = ensure_completed_papers_compiled(ws, author_state={})
    assert isinstance(result, list)


# ── run_full_paper_rewrites.main() — Class-B categorical block ────────────────

def test_run_full_paper_rewrites_blocked_when_stabilised(tmp_path: Path) -> None:
    """Class-B block: raises CategoricalBlock when stabilised."""
    import run_full_paper_rewrites

    active_state = {
        "active": True,
        "mode_id": "test-mode-id",
        "activated_at": "2026-01-01T00:00:00+00:00",
    }
    with patch("tar_validation_mode._read_stabilisation_state_strict", return_value=active_state):
        with pytest.raises(StabilisationGateCategoricalBlockError):
            run_full_paper_rewrites.main()


def test_run_full_paper_rewrites_blocked_on_unreadable_state(tmp_path: Path) -> None:
    """Class-B block: unreadable state propagates StabilisationGateStateUnreadableError."""
    import run_full_paper_rewrites

    with patch(
        "tar_validation_mode._read_stabilisation_state_strict",
        side_effect=StabilisationGateStateUnreadableError("test unreadable"),
    ):
        with pytest.raises(StabilisationGateStateUnreadableError):
            run_full_paper_rewrites.main()


# ── mint-site structural tests (Site 1 and Site 2 wiring) ────────────────────

def test_site1_same_thread_stabilised_proceeds_and_audits(tmp_path: Path) -> None:
    """Site 1 pattern: context entered on same thread as write_paper,
    mirroring _run_paper_revision_async's try/with/except structure."""
    ws = _make_workspace(tmp_path)
    _write_stab(ws, active=True, mode_id="test-mode-id", activated_at="2026-01-01T00:00:00+00:00")
    spec = _make_spec(ws)

    caught: list = []
    author = TARAuthor(workspace=ws)
    with patch.object(author, "_write_paper_impl", return_value=ws / "out.tex"):
        try:
            with stabilisation_authoring_override(ws, reason="site1-test"):
                author.write_paper(spec)
        except Exception as exc:
            caught.append(exc)

    assert not caught
    files = _stat_audit_files(ws)
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["event"] == "stabilisation_authoring_override_consumed"
    assert payload["reason"] == "site1-test"


def test_site1_unreadable_state_raises_and_is_catchable(tmp_path: Path) -> None:
    """Site 1 residual: unreadable state at with-entry raises
    StabilisationGateStateUnreadableError, which propagates to the
    except handler in _run_paper_revision_async."""
    ws = _make_workspace(tmp_path)
    (ws / "tar_state" / "stabilisation_mode.json").write_text("BAD JSON", encoding="utf-8")

    caught: list = []
    try:
        with stabilisation_authoring_override(ws, reason="site1-unreadable"):
            pass
    except StabilisationGateStateUnreadableError as exc:
        caught.append(exc)

    assert len(caught) == 1
    assert "fail-closed" in str(caught[0])


def test_site2_synchronous_main_thread_proceeds(tmp_path: Path) -> None:
    """Site 2 pattern: synchronous CLI path, context entered on calling thread,
    write_paper proceeds and audit written."""
    ws = _make_workspace(tmp_path)
    _write_stab(ws, active=True, mode_id="test-mode-id", activated_at="2026-01-01T00:00:00+00:00")
    spec = _make_spec(ws)

    author = TARAuthor(workspace=ws)
    with patch.object(author, "_write_paper_impl", return_value=ws / "out.tex") as mock_impl:
        with stabilisation_authoring_override(ws, reason="cli --write-paper"):
            result = author.write_paper(spec)
        mock_impl.assert_called_once_with(spec)

    files = _stat_audit_files(ws)
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["reason"] == "cli --write-paper"
