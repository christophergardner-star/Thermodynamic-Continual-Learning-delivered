"""
Quarantined legacy ASC entrypoint.

This historical script is intentionally disabled because it couples the warp
network to the same minimizing optimizer as the online model. That is not a
scientifically valid ASC training procedure.
"""

from __future__ import annotations


class ScientificValidityError(RuntimeError):
    """Raised when a quarantined training path is invoked."""


MESSAGE = (
    "asc_train.py is quarantined. This legacy script places the ASC warp "
    "network on a minimizing optimizer, so it is not a scientifically valid "
    "adversarial ASC trainer. Use `python asc_train_full.py` for the "
    "canonical min-max ASC path."
)


def main() -> int:
    raise ScientificValidityError(MESSAGE)


if __name__ == "__main__":
    raise SystemExit(main())
