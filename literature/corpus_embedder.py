"""Backfill + unify the paper-corpus embeddings on a single semantic model.

Context (2026-06-06): only ~115/1596 papers in the literature graph carried an
embedding, and those came from Semantic Scholar's SPECTER2 passthrough — a
DIFFERENT vector space than the bge-small embedder now engaged in TAR's
VectorVault. ``NoveltyGate`` compares a query embedding against the stored
``papers.embedding`` column, so for >92% of the corpus novelty fell back to
lexical Jaccard, and the few embedded papers lived in an incompatible space (a
dimension mismatch silently scores 0.0 in cosine similarity). Re-embedding the
corpus into ONE coherent space is what makes corpus-level novelty semantic.

Everything here is INERT by default:
  * ``NoveltyGate`` keeps loading allenai-specter unless ``TAR_NOVELTY_EMBEDDER_MODEL``
    is set (then both the gate AND this backfill use that model — one knob).
  * No corpus embedding is written unless the explicit ``scripts/embed_corpus.py``
    backfill is run, or the env is set and the live ingestor embeds on ingest.

The embedder is duck-typed: any object exposing ``embed(text) -> list[float]``
works, so tests inject a deterministic stub and stay offline.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional, Protocol

from literature.knowledge_graph import LiteratureKnowledgeGraph


# The single opt-in knob. Unset -> legacy behaviour (no live change). Set to a
# sentence-transformers model name (e.g. BAAI/bge-small-en-v1.5) to unify corpus
# novelty on the same space the VectorVault uses.
NOVELTY_EMBEDDER_ENV = "TAR_NOVELTY_EMBEDDER_MODEL"
# Default when the env is unset — preserves the historical NoveltyGate model
# exactly so there is no behaviour change until an operator opts in.
LEGACY_NOVELTY_MODEL = "allenai-specter"


class _Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


def novelty_embedder_model(default: str = LEGACY_NOVELTY_MODEL) -> str:
    """Resolve the model NoveltyGate + the corpus backfill should use.

    Returns ``TAR_NOVELTY_EMBEDDER_MODEL`` when set (the opt-in to unify on
    bge-small), else ``allenai-specter`` so default behaviour is unchanged.
    """
    value = os.environ.get(NOVELTY_EMBEDDER_ENV, "").strip()
    return value or default


def novelty_embedder_configured() -> bool:
    """True only when an operator has explicitly set the unify-on env."""
    return bool(os.environ.get(NOVELTY_EMBEDDER_ENV, "").strip())


def build_corpus_embedder(
    model_name: Optional[str] = None,
    *,
    allow_download: Optional[bool] = None,
) -> _Embedder:
    """Construct the shared semantic embedder (the VectorVault's SemanticEmbedder).

    Raises ``RuntimeError`` if sentence-transformers is unavailable or the model
    cannot be loaded. Callers in the live path MUST catch and degrade gracefully.
    """
    from tar_lab.memory.vault import SemanticEmbedder

    if model_name is None:
        model_name = novelty_embedder_model()
    if allow_download is None:
        allow_download = os.environ.get("TAR_ALLOW_MODEL_DOWNLOAD", "").strip() == "1"
    return SemanticEmbedder(model_name, allow_download=allow_download)


def paper_embed_text(title: Optional[str], abstract: Optional[str]) -> str:
    """Build the text a paper is embedded from (title + abstract)."""
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if title and abstract:
        return f"{title}\n{abstract}"
    return title or abstract


def embed_corpus(
    graph: LiteratureKnowledgeGraph,
    embedder: _Embedder,
    *,
    only_missing: bool = False,
    limit: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    """Embed papers in ``graph`` with ``embedder``, writing the embedding column.

    ``only_missing=False`` (default for a fresh backfill) re-embeds EVERY paper so
    the whole corpus shares one coherent vector space. ``only_missing=True`` fills
    only the NULL embeddings — an incremental top-up that is ONLY coherent after a
    full backfill on the same model.

    Returns a report dict: total / embedded / skipped_empty / errors / dimension.
    Idempotent: re-running reproduces the same stored vectors.
    """
    where = "WHERE embedding IS NULL" if only_missing else ""
    sql = f"SELECT paper_id, title, abstract FROM papers {where} ORDER BY paper_id"
    rows = graph.conn.execute(sql).fetchall()
    if limit is not None:
        rows = rows[: max(0, int(limit))]

    total = len(rows)
    embedded = skipped_empty = errors = 0
    dimension: Optional[int] = None
    for idx, row in enumerate(rows):
        text = paper_embed_text(row["title"], row["abstract"])
        if not text:
            skipped_empty += 1
            if progress is not None:
                progress(idx + 1, total)
            continue
        try:
            vec = embedder.embed(text)
        except Exception:
            errors += 1
            if progress is not None:
                progress(idx + 1, total)
            continue
        if not vec:
            errors += 1
            if progress is not None:
                progress(idx + 1, total)
            continue
        if dimension is None:
            dimension = len(vec)
        if graph.update_paper_embedding(row["paper_id"], vec):
            embedded += 1
        if progress is not None:
            progress(idx + 1, total)

    return {
        "total": total,
        "embedded": embedded,
        "skipped_empty": skipped_empty,
        "errors": errors,
        "dimension": dimension,
        "only_missing": only_missing,
        "model": getattr(embedder, "model_name", None),
    }
