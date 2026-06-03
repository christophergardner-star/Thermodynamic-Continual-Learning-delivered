"""Unit tests for the per-source ingestion circuit breaker."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tar_evidence_ingest import (
    _CIRCUIT_BREAKER_COOLDOWN_S,
    _CIRCUIT_BREAKER_THRESHOLD,
    _apply_circuit_breaker,
)

NOW = datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.utc)


def _entry(ok: bool) -> dict:
    return {"ok": ok, "last_error": "" if ok else "timeout", "ingested": 0}


def test_failure_increments_below_threshold_no_circuit():
    health = {"arxiv": _entry(False)}
    prior = {"arxiv": {"consecutive_failures": 1}}
    _apply_circuit_breaker(health, prior, {"arxiv"}, now=NOW)
    assert health["arxiv"]["consecutive_failures"] == 2
    # Below threshold (3): circuit stays closed.
    assert health["arxiv"]["circuit_open_until"] == ""


def test_circuit_opens_at_threshold():
    health = {"arxiv": _entry(False)}
    prior = {"arxiv": {"consecutive_failures": _CIRCUIT_BREAKER_THRESHOLD - 1}}
    _apply_circuit_breaker(health, prior, {"arxiv"}, now=NOW)
    assert health["arxiv"]["consecutive_failures"] == _CIRCUIT_BREAKER_THRESHOLD
    expected = (NOW + timedelta(seconds=_CIRCUIT_BREAKER_COOLDOWN_S)).isoformat()
    assert health["arxiv"]["circuit_open_until"] == expected


def test_success_resets_and_closes_circuit():
    health = {"arxiv": _entry(True)}
    prior = {"arxiv": {"consecutive_failures": 5, "circuit_open_until": (NOW + timedelta(hours=1)).isoformat()}}
    _apply_circuit_breaker(health, prior, {"arxiv"}, now=NOW)
    assert health["arxiv"]["consecutive_failures"] == 0
    assert health["arxiv"]["circuit_open_until"] == ""


def test_unattempted_source_preserves_open_circuit():
    open_until = (NOW + timedelta(minutes=30)).isoformat()
    health = {"arxiv": _entry(True)}  # default ok=True, but it did NOT run
    prior = {"arxiv": {"consecutive_failures": 4, "circuit_open_until": open_until}}
    _apply_circuit_breaker(health, prior, ran_sources=set(), now=NOW)
    # Not attempted -> counter and open circuit must survive (no false recovery).
    assert health["arxiv"]["consecutive_failures"] == 4
    assert health["arxiv"]["circuit_open_until"] == open_until


def test_expired_circuit_not_carried_for_unattempted_source():
    expired = (NOW - timedelta(minutes=1)).isoformat()
    health = {"arxiv": _entry(True)}
    prior = {"arxiv": {"consecutive_failures": 4, "circuit_open_until": expired}}
    _apply_circuit_breaker(health, prior, ran_sources=set(), now=NOW)
    # Expired circuit is effectively closed (half-open: source becomes eligible).
    assert health["arxiv"].get("circuit_open_until", "") == ""
    assert health["arxiv"]["consecutive_failures"] == 4


def test_open_circuit_extends_not_shrinks_on_further_failure():
    far = (NOW + timedelta(hours=5)).isoformat()
    health = {"arxiv": _entry(False)}
    prior = {"arxiv": {"consecutive_failures": 9, "circuit_open_until": far}}
    _apply_circuit_breaker(health, prior, {"arxiv"}, now=NOW)
    # New default window is +1h; a later existing window must not be shortened.
    assert health["arxiv"]["circuit_open_until"] == far
