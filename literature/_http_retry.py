"""
Shared retry/backoff helper for literature source fetchers.

Source clients (arxiv, openalex, crossref, semantic_scholar, pwc) each expose a
``_fetch``-style method that performs one network round-trip and returns a
``FetchResult``. Transient transport failures (read timeouts, connection
resets, transient 5xx) should not be surfaced as a hard source failure on the
first hiccup — they should be retried a few times with exponential backoff.

Rate limiting (HTTP 429) is deliberately NOT retried here: the clients already
honour a rate-limit cooldown, and hammering a throttled endpoint only makes it
worse. A ``FetchResult`` flagged ``rate_limited`` is returned immediately.

The public return shape is unchanged: callers still get a single
``FetchResult``. Only the number of attempts behind the scenes differs.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable, Mapping

from literature.schemas import FetchResult


# Substrings (lower-cased) that mark a transient transport error worth retrying.
_TRANSIENT_MARKERS = (
    "timed out",
    "timeout",
    "url_error",
    "urlerror",
    "connection reset",
    "connection refused",
    "connection aborted",
    "temporarily unavailable",
    "broken pipe",
)


def _error_is_retryable(ok: bool, rate_limited: bool, error: str | None) -> bool:
    """Core classification shared by the FetchResult and mapping variants.

    Retry on transient transport errors and 5xx responses. Never retry a
    success, a rate-limited result, or an explicit HTTP 429.
    """
    if ok or rate_limited:
        return False

    err = (error or "").lower()
    if not err:
        return False

    # Explicit rate limiting is never retried (respect the cooldown).
    if "429" in err or "rate_limited" in err:
        return False

    if any(marker in err for marker in _TRANSIENT_MARKERS):
        return True

    # Retry server-side 5xx, but not 4xx (client errors won't fix themselves).
    match = re.search(r"http_(\d{3})", err)
    if match and 500 <= int(match.group(1)) <= 599:
        return True

    return False


def is_retryable(result: FetchResult) -> bool:
    """Whether a failed ``FetchResult`` should be retried."""
    return _error_is_retryable(result.ok, result.rate_limited, result.error)


def is_retryable_mapping(result: Mapping[str, Any]) -> bool:
    """Whether a failed dict-style result (e.g. Semantic Scholar's _get/_post,
    shaped ``{"ok": bool, "error": str, "rate_limited": bool}``) should be
    retried."""
    return _error_is_retryable(
        bool(result.get("ok", False)),
        bool(result.get("rate_limited", False)),
        result.get("error"),
    )


def fetch_with_retry(
    fetch: Callable[[], Any],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    retryable: Callable[[Any], bool] = is_retryable,
) -> Any:
    """Call ``fetch`` up to ``attempts`` times, backing off on transient errors.

    Backoff is exponential: ``base_delay`` * 2**(n-1) seconds before the n-th
    retry (1s / 2s / 4s with the defaults). Returns the last result obtained —
    the public shape is unchanged.

    ``retryable`` selects the classifier: the default handles ``FetchResult``;
    pass ``is_retryable_mapping`` for dict-returning fetchers. ``sleep`` is
    injectable so unit tests can run without real delays.
    """
    if attempts < 1:
        attempts = 1
    result = fetch()
    for attempt in range(1, attempts):
        if not retryable(result):
            return result
        sleep(base_delay * (2 ** (attempt - 1)))
        result = fetch()
    return result
