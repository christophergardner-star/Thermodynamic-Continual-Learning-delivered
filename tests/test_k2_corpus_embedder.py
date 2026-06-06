"""Workstream A1 — embed the full corpus + unify NoveltyGate on one model.

Guarantees:
  (A) env resolution is inert by default: novelty_embedder_model() == allenai-specter
      unless TAR_NOVELTY_EMBEDDER_MODEL is set (then both the gate and the backfill
      use that one model).
  (B) embed_corpus re-embeds the whole corpus into a single coherent space,
      OVERWRITING stale embeddings from a different model (the dim-mismatch footgun).
  (C) only_missing fills just the NULLs; limit is respected.
  (D) end-to-end: after the backfill, NoveltyGate finds a SEMANTIC match across the
      corpus via embedding_similarity (the whole point of A1).

A deterministic stub embedder keeps the suite offline + fast.
"""
import json
import math
import re

from literature.knowledge_graph import LiteratureKnowledgeGraph
from literature.schemas import Paper
from literature import corpus_embedder as ce


class _Arr(list):
    """List that also answers .tolist() — matches what SentenceTransformer.encode returns."""

    def tolist(self):
        return list(self)


class _StubEmbedder:
    """Deterministic bag-of-words embedder: similar text -> similar vector. Offline."""

    model_name = "stub-embedder"
    _VOCAB = [
        "continual", "learning", "forgetting", "quantum",
        "graph", "thermodynamic", "replay", "regularization",
    ]

    def embed(self, text: str) -> list[float]:
        toks = set(re.sub(r"[^a-z ]", " ", text.lower()).split())
        v = [1.0 if w in toks else 0.0 for w in self._VOCAB]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    # NoveltyGate._embedding_search calls .encode(text, convert_to_tensor=False).tolist()
    def encode(self, text: str, convert_to_tensor: bool = False):
        return _Arr(self.embed(text))


def _paper(pid: str, title: str, abstract: str | None = None) -> Paper:
    return Paper(paper_id=pid, title=title, abstract=abstract)


def _graph(tmp_path) -> LiteratureKnowledgeGraph:
    g = LiteratureKnowledgeGraph(str(tmp_path / "lit.db"))
    g.upsert_paper(_paper("p1", "Continual learning without forgetting",
                          "A method to reduce catastrophic forgetting in continual learning."))
    g.upsert_paper(_paper("p2", "Quantum graph neural networks",
                          "Quantum circuits for graph representation."))
    g.upsert_paper(_paper("p3", "Replay and regularization", None))  # title-only
    return g


# ---- (A) env resolution is inert by default ------------------------------------

def test_env_resolution_default_unchanged(monkeypatch):
    monkeypatch.delenv(ce.NOVELTY_EMBEDDER_ENV, raising=False)
    assert ce.novelty_embedder_model() == ce.LEGACY_NOVELTY_MODEL == "allenai-specter"
    assert ce.novelty_embedder_configured() is False


def test_env_resolution_opt_in(monkeypatch):
    monkeypatch.setenv(ce.NOVELTY_EMBEDDER_ENV, "BAAI/bge-small-en-v1.5")
    assert ce.novelty_embedder_model() == "BAAI/bge-small-en-v1.5"
    assert ce.novelty_embedder_configured() is True


def test_novelty_gate_uses_resolved_model(monkeypatch):
    """The gate must load the env-resolved model name, not a hardcoded one."""
    import sys
    import types

    captured = {}

    def _fake_ctor(name, *args, **kwargs):
        captured["name"] = name
        raise RuntimeError("offline stub")  # we only care which name was requested

    # Inject a fake sentence_transformers so the gate's lazy import resolves to it
    # without importing the real (heavy, env-fragile) package.
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _fake_ctor
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setenv(ce.NOVELTY_EMBEDDER_ENV, "BAAI/bge-small-en-v1.5")
    from literature.novelty_gate import _try_load_sentence_transformer

    assert _try_load_sentence_transformer() is None  # ctor raised -> None (graceful)
    assert captured["name"] == "BAAI/bge-small-en-v1.5"


# ---- (B) full backfill overwrites a stale, different-space embedding ------------

def test_full_backfill_overwrites_stale_embedding(tmp_path):
    g = _graph(tmp_path)
    # p1 carries a stale 5-dim embedding from a "different model"
    g.update_paper_embedding("p1", [0.1, 0.2, 0.3, 0.4, 0.5])
    stub = _StubEmbedder()

    report = ce.embed_corpus(g, stub, only_missing=False)
    assert report["embedded"] == 3
    assert report["dimension"] == len(_StubEmbedder._VOCAB)  # coherent new space
    assert report["errors"] == 0

    rows = g.conn.execute("SELECT paper_id, embedding FROM papers ORDER BY paper_id").fetchall()
    dims = set()
    for r in rows:
        vec = json.loads(r["embedding"])
        dims.add(len(vec))
    assert dims == {len(_StubEmbedder._VOCAB)}  # p1's stale 5-dim vector was replaced
    g.close()


# ---- (C) only_missing + limit semantics ----------------------------------------

def test_only_missing_fills_just_nulls(tmp_path):
    g = _graph(tmp_path)
    stub = _StubEmbedder()
    g.update_paper_embedding("p1", stub.embed("Continual learning without forgetting"))
    g.update_paper_embedding("p2", stub.embed("Quantum graph neural networks"))
    # p3 has no embedding

    report = ce.embed_corpus(g, stub, only_missing=True)
    assert report["total"] == 1 and report["embedded"] == 1  # only p3
    assert g.papers_without_embeddings(limit=10) == []
    g.close()


def test_limit_is_respected(tmp_path):
    g = _graph(tmp_path)
    report = ce.embed_corpus(g, _StubEmbedder(), only_missing=False, limit=2)
    assert report["total"] == 2 and report["embedded"] == 2
    assert len(g.papers_without_embeddings(limit=10)) == 1  # one left untouched
    g.close()


# ---- (D) end-to-end: NoveltyGate finds a semantic match across the corpus -------

def test_novelty_gate_finds_semantic_match_after_backfill(tmp_path):
    from literature.novelty_gate import NoveltyGate

    g = _graph(tmp_path)
    stub = _StubEmbedder()
    ce.embed_corpus(g, stub, only_missing=False)

    gate = NoveltyGate(g, load_embedding_model=False)
    gate._embedder = stub  # same space as the backfill (the unification A1 enforces)

    similar = gate._find_similar_papers("continual learning forgetting", max_results=5)
    ids = {s.paper_id for s in similar}
    assert "p1" in ids  # the CL paper matches semantically
    assert "p2" not in ids  # the quantum/graph paper does not
    p1 = next(s for s in similar if s.paper_id == "p1")
    assert p1.similarity_reason == "embedding_similarity"
    assert p1.similarity_score >= 0.60
    g.close()
