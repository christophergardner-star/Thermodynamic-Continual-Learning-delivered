"""B5 launcher — drive the method-synthesis loop from variant proposals (operator-invoked).

This is the deliberate, occasional entry point for TAR's boldest self-improvement step:
synthesise + validate a NEW continual-learning method from the top method_refinement variant
proposal(s). It is multiply gated and NEVER auto-adopts:
  * OFF unless tar_state/method_synthesis.enabled exists (RAIL-3 human opt-in);
  * each validated candidate is QUARANTINED (tar_state/synthesized_methods_pending/) — never
    auto-loaded into the registry;
  * adoption is a separate explicit step (--approve), append-only-guarded.

Synthesis calls an LLM (ANTHROPIC_API_KEY) + a CPU sandbox + minibench — it can take a while
and costs API tokens. Run it deliberately, not in a loop.

Usage:
  python scripts/run_method_synthesis.py --run [--max 1]      # synthesise -> pending
  python scripts/run_method_synthesis.py --list               # show pending candidates
  python scripts/run_method_synthesis.py --approve <id> --by "Your Name"   # human adoption
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from tar_storage import ensure_workspace_layout  # noqa: E402
from tar_lab import synthesis_loop as sl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=None, help="workspace root (default: resolved)")
    ap.add_argument("--run", action="store_true", help="synthesise from the top variant proposals")
    ap.add_argument("--max", type=int, default=1, help="max candidates to synthesise (default 1)")
    ap.add_argument("--list", action="store_true", help="list pending candidates")
    ap.add_argument("--approve", default=None, help="candidate_id to adopt (human approval)")
    ap.add_argument("--by", default=None, help="human approver name (required with --approve)")
    args = ap.parse_args(argv)

    ws = ensure_workspace_layout(Path(args.workspace).resolve() if args.workspace else None, repo_root=_REPO)
    print(f"[method_synthesis] workspace={ws}")
    print(f"[method_synthesis] enabled={sl.is_enabled(ws)} (flag: tar_state/{sl.ENABLE_FLAG})")

    if args.approve:
        if not args.by:
            print("[method_synthesis] --approve requires --by \"<name>\"", file=sys.stderr)
            return 2
        res = sl.approve_pending_candidate(ws, args.approve, approved_by=args.by)
        print(json.dumps(res, indent=2))
        return 0 if res.get("approved") else 1

    if args.list:
        print(json.dumps(sl.load_pending_candidates(ws), indent=2))
        return 0

    if args.run:
        summary = sl.run_synthesis_from_proposals(ws, max_candidates=args.max)
        # don't dump the full candidate bodies; show the summary view
        view = {k: v for k, v in summary.items() if k != "candidates"}
        view["candidates"] = [
            {"candidate_id": c.get("candidate_id"), "status": c.get("status"),
             "method_key": c.get("method_key"), "pending_path": c.get("pending_path")}
            for c in summary.get("candidates", [])
        ]
        print(json.dumps(view, indent=2))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
