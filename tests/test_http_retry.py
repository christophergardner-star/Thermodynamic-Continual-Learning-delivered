"""Unit tests for the literature source retry/backoff helper."""
from __future__ import annotations

from literature._http_retry import (
    fetch_with_retry,
    is_retryable,
    is_retryable_mapping,
)
from literature.schemas import FetchResult


def _ok() -> FetchResult:
    return FetchResult(ok=True, source="arxiv", items=[{"paper_id": "x"}])


def _timeout() -> FetchResult:
    return FetchResult(ok=False, source="arxiv", error="unexpected: The read operation timed out")


def _url_error() -> FetchResult:
    return FetchResult(ok=False, source="arxiv", error="url_error: [Errno 111] Connection refused")


def _http_429() -> FetchResult:
    return FetchResult(ok=False, source="arxiv", error="http_429: Too Many Requests", rate_limited=True)


def _http_503() -> FetchResult:
    return FetchResult(ok=False, source="openalex", error="http_503: Service Unavailable")


def _http_404() -> FetchResult:
    return FetchResult(ok=False, source="openalex", error="http_404: Not Found")


# --------------------------------------------------------------------------- #
# is_retryable classification
# --------------------------------------------------------------------------- #

def test_is_retryable_transient_errors():
    assert is_retryable(_timeout()) is True
    assert is_retryable(_url_error()) is True
    assert is_retryable(_http_503()) is True


def test_is_retryable_never_for_429():
    assert is_retryable(_http_429()) is False


def test_is_retryable_never_for_success_or_4xx():
    assert is_retryable(_ok()) is False
    assert is_retryable(_http_404()) is False


# --------------------------------------------------------------------------- #
# fetch_with_retry behaviour
# --------------------------------------------------------------------------- #

def _counting_source(results):
    """Return a fetch callable that yields the given results in order, and a
    call counter list (single-element, mutated in place)."""
    calls = {"n": 0}

    def fetch() -> FetchResult:
        idx = min(calls["n"], len(results) - 1)
        calls["n"] += 1
        return results[idx]

    return fetch, calls


def test_retries_on_timeout_then_succeeds():
    slept: list[float] = []
    fetch, calls = _counting_source([_timeout(), _timeout(), _ok()])
    result = fetch_with_retry(fetch, attempts=3, sleep=slept.append)
    assert result.ok is True
    assert calls["n"] == 3
    # Two backoffs before the third attempt: 1s then 2s.
    assert slept == [1.0, 2.0]


def test_gives_up_after_n_attempts():
    slept: list[float] = []
    fetch, calls = _counting_source([_timeout()])
    result = fetch_with_retry(fetch, attempts=3, sleep=slept.append)
    assert result.ok is False
    assert calls["n"] == 3  # initial + 2 retries
    assert slept == [1.0, 2.0]


def test_does_not_retry_on_429():
    slept: list[float] = []
    fetch, calls = _counting_source([_http_429(), _ok()])
    result = fetch_with_retry(fetch, attempts=3, sleep=slept.append)
    assert result.rate_limited is True
    assert calls["n"] == 1  # returned immediately, no retry
    assert slept == []


def test_does_not_retry_on_success():
    slept: list[float] = []
    fetch, calls = _counting_source([_ok()])
    result = fetch_with_retry(fetch, attempts=3, sleep=slept.append)
    assert result.ok is True
    assert calls["n"] == 1
    assert slept == []


# --------------------------------------------------------------------------- #
# Mapping (dict) variant — Semantic Scholar's _get/_post shape
# --------------------------------------------------------------------------- #

def test_is_retryable_mapping_classification():
    assert is_retryable_mapping({"ok": False, "error": "url_error: timed out"}) is True
    assert is_retryable_mapping({"ok": False, "error": "http_502: Bad Gateway"}) is True
    # 429 surfaces as rate_limited in SS — must not retry.
    assert is_retryable_mapping({"ok": False, "error": "rate_limited", "rate_limited": True}) is False
    assert is_retryable_mapping({"ok": True, "data": {}}) is False
    assert is_retryable_mapping({"ok": False, "error": "http_404: Not Found"}) is False


def test_fetch_with_retry_mapping_retries_then_succeeds():
    slept: list[float] = []
    results = [
        {"ok": False, "error": "unexpected: The read operation timed out"},
        {"ok": True, "data": {"x": 1}},
    ]
    fetch, calls = _counting_source(results)
    out = fetch_with_retry(fetch, attempts=3, sleep=slept.append, retryable=is_retryable_mapping)
    assert out["ok"] is True
    assert calls["n"] == 2
    assert slept == [1.0]


def test_fetch_with_retry_mapping_no_retry_on_rate_limited():
    slept: list[float] = []
    fetch, calls = _counting_source([{"ok": False, "error": "rate_limited", "rate_limited": True}])
    out = fetch_with_retry(fetch, attempts=3, sleep=slept.append, retryable=is_retryable_mapping)
    assert out["rate_limited"] is True
    assert calls["n"] == 1
    assert slept == []
