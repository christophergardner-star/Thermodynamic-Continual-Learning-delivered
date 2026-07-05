"""WS3.3a — the fast ingest cycle must honour per-source cooldowns.

Regression: _run_fast_cycle called arxiv.latest()/openalex.latest() every cycle
regardless of an open circuit / rate-limit cooldown, re-hitting a throttled feed
and pinning its consecutive_failures ever higher (observed live: arxiv at 119).
The fix reuses the same cooldown check the daily cycle uses and SKIPS a cooled-
down source WITHOUT recording a run (so it stays out of ran_sources and its
cooldown is preserved by _apply_circuit_breaker).
"""
from __future__ import annotations

import types
from datetime import datetime, timedelta, timezone

from tar_evidence_ingest import ExternalEvidenceIngestor


def _ingestor(source_health: dict) -> ExternalEvidenceIngestor:
    # Bypass the heavy constructor; we only exercise cooldown routing.
    ing = ExternalEvidenceIngestor.__new__(ExternalEvidenceIngestor)
    ing._prior_state = {"source_health": source_health}
    return ing


def _future() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()


def _past() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()


class _BoomClient:
    """A client whose latest() must never be called."""
    def latest(self, **_kw):
        raise AssertionError("latest() called on a source that is in cooldown")


class _OKClient:
    def __init__(self):
        self.called = False

    def latest(self, **_kw):
        self.called = True
        return types.SimpleNamespace(ok=True, items=[{"paper_id": "p1"}],
                                     error="", rate_limited=False)


# --------------------------------------------------- _is_source_in_cooldown ---

def test_cooldown_detects_open_circuit_and_rate_limit():
    ing = _ingestor({
        "arxiv": {"circuit_open_until": _future()},
        "openalex": {"rate_limited_until": _future()},
        "crossref": {"rate_limited_until": _past()},   # expired -> not in cooldown
        "semantic_scholar": {},                        # clean
    })
    assert ing._is_source_in_cooldown("arxiv") is True
    assert ing._is_source_in_cooldown("openalex") is True
    assert ing._is_source_in_cooldown("crossref") is False
    assert ing._is_source_in_cooldown("semantic_scholar") is False
    assert ing._is_source_in_cooldown("unknown_source") is False


# ------------------------------------------------------------ _run_fast_cycle -

def test_fast_cycle_skips_cooled_source_without_recording_a_run():
    ing = _ingestor({"arxiv": {"circuit_open_until": _future()}})  # arxiv cooled
    ing.arxiv = _BoomClient()          # would raise if called
    ing.openalex = _OKClient()
    ing._ingest_papers = lambda items: (len(items), len(items), 0)

    cr = types.SimpleNamespace(source_runs=[], errors=[])
    ing._run_fast_cycle(cr, {"categories": ["cs.LG"], "queries": ["q"],
                             "connected_queries": []})

    # arxiv skipped: NOT in the runs (so it stays out of ran_sources); only
    # openalex actually fetched + recorded.
    assert [r.source for r in cr.source_runs] == ["openalex"]
    assert ing.openalex.called is True


def test_fast_cycle_all_cooled_records_nothing():
    ing = _ingestor({
        "arxiv": {"circuit_open_until": _future()},
        "openalex": {"rate_limited_until": _future()},
    })
    ing.arxiv = _BoomClient()
    ing.openalex = _BoomClient()
    ing._ingest_papers = lambda items: (len(items), len(items), 0)

    cr = types.SimpleNamespace(source_runs=[], errors=[])
    ing._run_fast_cycle(cr, {"categories": ["cs.LG"], "queries": ["q"],
                             "connected_queries": []})
    assert cr.source_runs == []
    assert cr.errors == []  # a skip is not a failure


def test_fast_cycle_runs_healthy_sources_normally():
    ing = _ingestor({})  # nothing cooled
    ing.arxiv = _OKClient()
    ing.openalex = _OKClient()
    ing._ingest_papers = lambda items: (len(items), len(items), 0)

    cr = types.SimpleNamespace(source_runs=[], errors=[])
    ing._run_fast_cycle(cr, {"categories": ["cs.LG"], "queries": ["q"],
                             "connected_queries": []})
    assert sorted(r.source for r in cr.source_runs) == ["arxiv", "openalex"]
    assert ing.arxiv.called and ing.openalex.called
