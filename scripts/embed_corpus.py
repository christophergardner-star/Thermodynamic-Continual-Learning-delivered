"""Backfill the literature corpus embeddings onto a single semantic model.

This is the explicit, operator-run activation half of Workstream A1. By default
NoveltyGate still loads allenai-specter and nothing is embedded; this script is
what re-embeds the whole corpus into one coherent space once an operator has
chosen to unify on the VectorVault's bge-small model.

Activation recipe (deliberate, reversible):
  1. add  "TAR_NOVELTY_EMBEDDER_MODEL": "BAAI/bge-small-en-v1.5"  to
     tar_state/api_secrets.json   (ensure_workspace_layout merges it into env)
  2. python scripts/embed_corpus.py            # re-embeds the whole corpus
  3. PID-targeted dashboard/daemon restart so NoveltyGate picks up the new model

Read-only preview (no model load, no writes):
  python scripts/embed_corpus.py --dry-run

Useful flags:
  --workspace PATH   target a specific workspace (default: TAR_WORKSPACE / repo)
  --model NAME       override the model (default: TAR_NOVELTY_EMBEDDER_MODEL)
  --only-missing     fill only NULL embeddings (incremental; coherent only AFTER
                     a full backfill on the same model)
  --limit N          cap the number of papers (smoke test)
  --allow-download   permit the one-time model download (else local files only)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from tar_storage import ensure_workspace_layout  # noqa: E402
from literature.knowledge_graph import LiteratureKnowledgeGraph  # noqa: E402
from literature.corpus_embedder import (  # noqa: E402
    embed_corpus,
    build_corpus_embedder,
    novelty_embedder_model,
)


def _db_path(workspace: Path) -> Path:
    return workspace / "tar_state" / "literature" / "literature_graph.db"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=None, help="workspace root (default: resolved from TAR_WORKSPACE / repo)")
    parser.add_argument("--model", default=None, help="embedding model name (default: TAR_NOVELTY_EMBEDDER_MODEL)")
    parser.add_argument("--only-missing", action="store_true", help="embed only papers with no embedding yet")
    parser.add_argument("--limit", type=int, default=None, help="cap number of papers processed")
    parser.add_argument("--allow-download", action="store_true", help="permit one-time model download")
    parser.add_argument("--dry-run", action="store_true", help="report counts only; load no model, write nothing")
    args = parser.parse_args(argv)

    ws = Path(args.workspace).resolve() if args.workspace else None
    # ensure_workspace_layout also merges tar_state/api_secrets.json into the
    # environment, so TAR_NOVELTY_EMBEDDER_MODEL / TAR_ALLOW_MODEL_DOWNLOAD set
    # there are honoured here.
    workspace = ensure_workspace_layout(ws, repo_root=_REPO)
    db = _db_path(workspace)
    if not db.exists():
        print(f"[embed_corpus] no literature graph at {db}", file=sys.stderr)
        return 2

    graph = LiteratureKnowledgeGraph(str(db))
    try:
        total = graph.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        missing = graph.conn.execute(
            "SELECT COUNT(*) FROM papers WHERE embedding IS NULL"
        ).fetchone()[0]
        model = args.model or novelty_embedder_model()
        print(f"[embed_corpus] workspace={workspace}")
        print(f"[embed_corpus] db={db}")
        print(f"[embed_corpus] papers={total}  without_embedding={missing}  model={model}")

        if args.dry_run:
            scope = "missing-only" if args.only_missing else "ALL (re-embed)"
            n = missing if args.only_missing else total
            if args.limit is not None:
                n = min(n, args.limit)
            print(f"[embed_corpus] DRY-RUN: would embed {n} papers ({scope}); no model loaded, nothing written.")
            return 0

        try:
            embedder = build_corpus_embedder(args.model, allow_download=args.allow_download or None)
        except Exception as exc:
            print(f"[embed_corpus] could not load embedder '{model}': {exc}", file=sys.stderr)
            print("[embed_corpus] hint: --allow-download for the one-time fetch, or check sentence-transformers.", file=sys.stderr)
            return 3

        def _progress(done: int, tot: int) -> None:
            if tot and (done % 100 == 0 or done == tot):
                print(f"[embed_corpus]   {done}/{tot}")

        report = embed_corpus(
            graph,
            embedder,
            only_missing=args.only_missing,
            limit=args.limit,
            progress=_progress,
        )
        print(f"[embed_corpus] DONE: {report}")
        return 0
    finally:
        graph.close()


if __name__ == "__main__":
    raise SystemExit(main())
