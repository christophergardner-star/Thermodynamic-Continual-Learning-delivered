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


# Override knob. Set to a sentence-transformers model name to force a specific
# space; unset uses the default below.
NOVELTY_EMBEDDER_ENV = "TAR_NOVELTY_EMBEDDER_MODEL"
# Default embedder. The corpus was fully re-embedded into bge-small space
# (1923/1923 papers, 384-dim) on 2026-06-06, and the VectorVault uses the same
# model — so bge-small is now the CORRECT default. (The historical
# allenai-specter default is 768-dim and mismatches the stored corpus, which
# silently scores cosine 0.0; see the module docstring.) Kept as a named
# constant so a future re-embed can retune it in one place.
LEGACY_NOVELTY_MODEL = "BAAI/bge-small-en-v1.5"


class _Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


def novelty_embedder_model(default: str = LEGACY_NOVELTY_MODEL) -> str:
    """Resolve the model NoveltyGate + the corpus backfill should use.

    Returns ``TAR_NOVELTY_EMBEDDER_MODEL`` when set, else bge-small — the space
    the corpus is actually embedded in (query and corpus vectors must match or
    cosine similarity silently degrades to 0.0).
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


def embed_paper_if_configured(
    graph: LiteratureKnowledgeGraph,
    paper: Any,
    embedder: Optional[_Embedder],
) -> bool:
    """Embed a single freshly-ingested paper IF an embedder is provided.

    No-op (returns False) when ``embedder`` is None — which is the default, since
    the live ingestor only builds an embedder when TAR_NOVELTY_EMBEDDER_MODEL is
    set. Never raises: an embedding failure must not break ingestion.
    """
    if embedder is None:
        return False
    text = paper_embed_text(getattr(paper, "title", None), getattr(paper, "abstract", None))
    if not text:
        return False
    try:
        vec = embedder.embed(text)
    except Exception:
        return False
    if not vec:
        return False
    try:
        return graph.update_paper_embedding(paper.paper_id, vec)
    except Exception:
        return False


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
