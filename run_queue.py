from __future__ import annotations


def main() -> int:
    print(
        "run_queue.py has been retired and moved to legacy_quarantine/run_queue.py.\n"
        "Broad unattended queue execution is no longer allowed.\n"
        "Use the bounded manifest + orchestrator flow instead.",
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
