"""
Console/log tee helpers for long-running benchmark suites.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterator, TextIO


class _TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            try:
                stream.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass


@contextmanager
def tee_console(log_path: str | Path) -> Iterator[None]:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as fh:
        out = _TeeStream(sys.stdout, fh)
        err = _TeeStream(sys.stderr, fh)
        with redirect_stdout(out), redirect_stderr(err):
            yield
