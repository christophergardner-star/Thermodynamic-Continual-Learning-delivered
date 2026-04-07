from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from hashlib import blake2b
from threading import Event, Thread
from time import sleep
from typing import Any, Iterable, Optional

from tar_lab.errors import MemoryIntegrityError, MemoryRebuildRequiredError, ScientificValidityError
from tar_lab.schemas import (
    BibliographyEntry,
    BreakthroughReport,
    ClaimCluster,
    ClaimConflict,
    EvidenceTrace,
    GovernorMetrics,
    KnowledgeGraphEntry,
    KnowledgeGraphState,
    LiteratureCapabilityReport,
    MemoryStoreManifest,
    MemorySearchHit,
    PaperArtifact,
    ProblemExecutionReport,
    ProblemStudyReport,
    ResearchDocument,
    SelfCorrectionNote,
    VerificationReport,
    utc_now_iso,
)
from tar_lab.state import TARStateStore

try:
    import chromadb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]


DEFAULT_EMBEDDER_MODEL = "BAAI/bge-small-en-v1.5"
MEMORY_SCHEMA_VERSION = 1
LEGACY_COLLECTION_NAME = "lab_history"


class LexicalProjectionEmbedder:
    """Deterministic non-hash fallback when a sentence-transformer model is unavailable locally."""

    _alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 "

    def __init__(self):
        self.model_name = "lexical-semantic-fallback"
        self._index = {char: idx for idx, char in enumerate(self._alphabet)}
        self._dim = len(self._alphabet) * len(self._alphabet)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
        padded = f" {normalized} "
        values = [0.0] * self._dim
        for left, right in zip(padded, padded[1:]):
            left_idx = self._index.get(left, self._index[" "])
            right_idx = self._index.get(right, self._index[" "])
            values[left_idx * len(self._alphabet) + right_idx] += 1.0
        norm = math.sqrt(sum(item * item for item in values))
        if norm <= 1e-12:
            return values
        return [item / norm for item in values]


class SemanticEmbedder:
    def __init__(self, model_name: str, *, allow_download: bool = False):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        self.model_name = model_name
        self.model = self._load_model(model_name, allow_download=allow_download)
        self._dimension: Optional[int] = None

    def _load_model(self, model_name: str, *, allow_download: bool):
        kwargs = {"device": "cpu"}
        try:
            return SentenceTransformer(model_name, local_files_only=not allow_download, **kwargs)
        except TypeError:
            if not allow_download:
                raise RuntimeError("The installed sentence-transformers version does not support local_files_only")
            return SentenceTransformer(model_name, **kwargs)

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(item) for item in vector]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed("vector dimension probe"))
        return self._dimension


class ScientificReranker:
    def __init__(self):
        self.name = "scientific-hybrid-reranker"

    def rerank(
        self,
        query: str,
        query_embedding: list[float],
        candidates: list["RetrievedDocument"],
        *,
        top_k: int,
    ) -> list[tuple[float, "RetrievedDocument"]]:
        scored: list[tuple[float, RetrievedDocument]] = []
        for candidate in candidates:
            dense = VectorVault._cosine(query_embedding, candidate.embedding)
            lexical = VectorVault._lexical_score(query, candidate.document)
            coverage = VectorVault._coverage_score(query, candidate.document)
            kind_bonus = self._kind_bonus(candidate.metadata)
            evidence_bonus = self._evidence_bonus(candidate.metadata)
            score = (0.52 * dense) + (0.20 * lexical) + (0.18 * coverage) + kind_bonus + evidence_bonus
            scored.append((score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _kind_bonus(metadata: dict[str, Any]) -> float:
        kind = str(metadata.get("kind", ""))
        if kind == "paper_claim":
            return 0.08
        if kind in {"paper_section", "paper_table", "paper_figure", "paper_bibliography"}:
            return 0.04
        return 0.0

    @staticmethod
    def _evidence_bonus(metadata: dict[str, Any]) -> float:
        bonus = 0.0
        if metadata.get("page_number", -1) not in {-1, None, ""}:
            bonus += 0.02
        if metadata.get("bibliography_entry_id") or metadata.get("citation_entry_ids"):
            bonus += 0.03
        if metadata.get("source_excerpt"):
            bonus += 0.02
        return bonus


@dataclass
class RetrievedDocument:
    document_id: str
    document: str
    metadata: dict[str, Any]
    embedding: list[float]


class VectorVault:
    def __init__(self, workspace: str = "."):
        if chromadb is None:
            raise RuntimeError("chromadb is not installed; install it to enable TAR vector memory")
        self.store = TARStateStore(workspace)
        self.db_dir = self.store.state_dir / "memory"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.claim_clusters_path = self.db_dir / "claim_clusters.json"
        self.claim_conflicts_path = self.db_dir / "claim_conflicts.json"
        self.embedder = self._resolve_embedder()
        self.embedder_name = getattr(self.embedder, "model_name", DEFAULT_EMBEDDER_MODEL)
        self.embedding_dim = self._embedding_dimension(self.embedder)
        self.reranker = ScientificReranker()
        self.reranker_name = self.reranker.name
        self.memory_manifest = self._resolve_memory_manifest()
        self.collection = self._bind_collection(self.memory_manifest.collection_name)

    def stats(self) -> dict[str, Any]:
        manifest = self.memory_manifest
        return {
            "documents": self.collection.count(),
            "path": str(self.db_dir),
            "collection_name": manifest.collection_name,
            "schema_version": manifest.schema_version,
            "fingerprint": manifest.fingerprint,
            "embedder": self.embedder_name,
            "embedding_dim": self.embedding_dim,
            "semantic_research_ready": getattr(self, "semantic_ready", False),
            "reranker": self.reranker_name,
            "reranker_ready": True,
            "claim_clusters": len(self._load_claim_clusters()),
            "claim_conflicts": len(self._load_claim_conflicts()),
            "state": manifest.state,
            "rebuild_required": manifest.state in {"rebuild_required", "rebuilding"},
            "last_rebuild_at": manifest.last_rebuild_at,
            "last_error": manifest.last_error,
            "retired_collections": list(manifest.retired_collection_names),
        }

    def _list_collection_names(self) -> list[str]:
        rows = self.client.list_collections()
        names: list[str] = []
        for item in rows:
            if isinstance(item, str):
                names.append(item)
            else:
                name = getattr(item, "name", None)
                if name:
                    names.append(str(name))
        return sorted(set(names))

    def _embedding_dimension(self, embedder: Any) -> int:
        dimension = getattr(embedder, "dimension", None)
        if isinstance(dimension, int) and dimension > 0:
            return dimension
        vector = embedder.embed("vector dimension probe")
        return max(1, len(vector))

    def _collection_fingerprint(self) -> str:
        payload = f"{MEMORY_SCHEMA_VERSION}|{self.embedder_name}|{self.embedding_dim}"
        return blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()

    def _collection_name(self, fingerprint: str) -> str:
        return f"lab_history__{fingerprint}"

    def _bind_collection(self, collection_name: str):
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"space": "cosine"},
        )

    def _save_memory_manifest(self, manifest: MemoryStoreManifest) -> MemoryStoreManifest:
        try:
            self.store.save_memory_manifest(manifest)
        except PermissionError:
            stored = self.store.load_memory_manifest()
            if stored is not None:
                self.memory_manifest = stored
                return stored
            raise
        stored = self.store.load_memory_manifest()
        self.memory_manifest = stored if stored is not None else manifest
        return self.memory_manifest

    def _resolve_memory_manifest(self) -> MemoryStoreManifest:
        fingerprint = self._collection_fingerprint()
        collection_name = self._collection_name(fingerprint)
        existing = self.store.load_memory_manifest()
        collection_names = self._list_collection_names()
        retired = set(existing.retired_collection_names if existing is not None else [])

        if existing is not None and existing.fingerprint == fingerprint:
            state = existing.state
            if existing.collection_name not in collection_names:
                state = "rebuild_required"
            manifest = existing.model_copy(
                update={
                    "collection_name": existing.collection_name,
                    "embedder_name": self.embedder_name,
                    "embedding_dim": self.embedding_dim,
                    "semantic_research_ready": getattr(self, "semantic_ready", False),
                    "state": state,
                }
            )
            return self._save_memory_manifest(manifest)

        if existing is not None:
            retired.add(existing.collection_name)
        if LEGACY_COLLECTION_NAME in collection_names and LEGACY_COLLECTION_NAME != collection_name:
            retired.add(LEGACY_COLLECTION_NAME)

        manifest = MemoryStoreManifest(
            fingerprint=fingerprint,
            collection_name=collection_name,
            embedder_name=self.embedder_name,
            embedding_dim=self.embedding_dim,
            semantic_research_ready=getattr(self, "semantic_ready", False),
            state="rebuild_required",
            retired_collection_names=sorted(retired),
        )
        return self._save_memory_manifest(manifest)

    def begin_rebuild(self) -> None:
        if self.memory_manifest.state == "rebuilding":
            return
        self._save_memory_manifest(
            self.memory_manifest.model_copy(
                update={
                    "state": "rebuilding",
                    "last_error": None,
                }
            )
        )

    def complete_rebuild(self) -> None:
        self._save_memory_manifest(
            self.memory_manifest.model_copy(
                update={
                    "state": "healthy",
                    "last_rebuild_at": utc_now_iso(),
                    "last_error": None,
                }
            )
        )

    def mark_degraded(self, message: str) -> None:
        self._save_memory_manifest(
            self.memory_manifest.model_copy(
                update={
                    "state": "degraded",
                    "last_error": message,
                }
            )
        )

    def index_metric(self, metric: GovernorMetrics) -> None:
        document_id = f"metric:{metric.trial_id}:{metric.step}"
        text = self.metric_text(metric)
        metadata = {
            "kind": "metric",
            "trial_id": metric.trial_id,
            "step": metric.step,
            "energy_e": metric.energy_e,
            "entropy_sigma": metric.entropy_sigma,
            "drift_rho": metric.drift_rho,
            "grad_norm": metric.grad_norm,
            "effective_dimensionality": metric.effective_dimensionality,
            "equilibrium_fraction": metric.equilibrium_fraction,
        }
        self._upsert(document_id, text, metadata)

    def index_knowledge_entry(self, entry: KnowledgeGraphEntry) -> None:
        alpha = entry.hyperparameters.get("alpha")
        eta = entry.hyperparameters.get("eta")
        document_id = f"knowledge:{entry.trial_id}"
        text = (
            f"Trial {entry.trial_id} strategy {entry.strategy_family} outcome {entry.outcome}. "
            f"alpha={alpha} eta={eta} fail_reason={entry.fail_reason or 'none'}."
        )
        metadata = {
            "kind": "knowledge",
            "trial_id": entry.trial_id,
            "strategy_family": entry.strategy_family,
            "outcome": entry.outcome,
            "alpha": float(alpha) if alpha is not None else -1.0,
            "eta": float(eta) if eta is not None else -1.0,
            "fail_reason": entry.fail_reason or "",
        }
        self._upsert(document_id, text, metadata)

    def index_knowledge_graph(self, graph: KnowledgeGraphState) -> None:
        for entry in graph.entries:
            self.index_knowledge_entry(entry)

    def index_self_correction(self, note: SelfCorrectionNote) -> None:
        document_id = f"self_correction:{note.trial_id}"
        text = (
            f"Self-correction for {note.trial_id}. outcome={note.outcome}. "
            f"E={note.energy_e:.6f} sigma={note.entropy_sigma:.6f} rho={note.drift_rho:.6f} "
            f"D_PR={note.effective_dimensionality:.4f} eq={note.equilibrium_fraction:.2%}. "
            f"Explanation: {note.explanation} Corrective action: {note.corrective_action}."
        )
        metadata = {
            "kind": "self_correction",
            "trial_id": note.trial_id,
            "outcome": note.outcome,
            "energy_e": note.energy_e,
            "entropy_sigma": note.entropy_sigma,
            "drift_rho": note.drift_rho,
            "effective_dimensionality": note.effective_dimensionality,
            "equilibrium_fraction": note.equilibrium_fraction,
        }
        self._upsert(document_id, text, metadata)

    def index_research_document(self, document: ResearchDocument) -> None:
        text = (
            f"{document.title}. {document.summary} "
            f"Problems: {'; '.join(document.problem_statements)} "
            f"Tags: {' '.join(document.tags)}"
        ).strip()
        metadata = {
            "kind": "research",
            "document_id": document.document_id,
            "source_kind": document.source_kind,
            "source_name": document.source_name,
            "published_at": document.published_at or "",
            "url": document.url,
        }
        self._upsert(f"research:{document.document_id}", text, metadata)

    def index_paper_artifact(self, artifact: PaperArtifact) -> None:
        for claim in artifact.claims:
            text = f"{artifact.title}. [{claim.label}] {claim.text}"
            metadata = {
                "kind": "paper_claim",
                "paper_id": artifact.paper_id,
                "paper_title": artifact.title,
                "claim_id": claim.claim_id,
                "label": claim.label,
                "polarity": claim.polarity,
                "page_number": claim.page_number or -1,
                "section_id": claim.section_id,
                "source_path": artifact.source_path,
                "source_excerpt": claim.source_excerpt or "",
                "citation_entry_ids": json.dumps(claim.citation_entry_ids),
                "span_start": claim.span_start if claim.span_start is not None else -1,
                "span_end": claim.span_end if claim.span_end is not None else -1,
                "evidence_kind": claim.evidence_kind,
            }
            self._upsert(f"paper_claim:{claim.claim_id}", text, metadata)
        for section in artifact.sections:
            text = f"{artifact.title}. [{section.heading}] {section.text}"
            metadata = {
                "kind": "paper_section",
                "paper_id": artifact.paper_id,
                "paper_title": artifact.title,
                "section_id": section.section_id,
                "heading": section.heading,
                "page_start": section.page_start or -1,
                "page_end": section.page_end or -1,
                "source_path": artifact.source_path,
                "source_excerpt": section.text[:320],
            }
            self._upsert(f"paper_section:{artifact.paper_id}:{section.section_id}", text, metadata)
        for entry in artifact.bibliography:
            text = f"{artifact.title}. [reference] {entry.raw_text}"
            metadata = {
                "kind": "paper_bibliography",
                "paper_id": artifact.paper_id,
                "paper_title": artifact.title,
                "entry_id": entry.entry_id,
                "citation_key": entry.citation_key,
                "page_number": entry.page_number or -1,
                "source_path": artifact.source_path,
                "source_excerpt": entry.source_excerpt or entry.raw_text[:320],
            }
            self._upsert(f"paper_bibliography:{entry.entry_id}", text, metadata)
        for table in artifact.tables:
            text = f"{artifact.title}. [table] {table.caption}. {table.raw_text}"
            metadata = {
                "kind": "paper_table",
                "paper_id": artifact.paper_id,
                "paper_title": artifact.title,
                "table_id": table.table_id,
                "section_id": table.section_id or "",
                "page_number": table.page_number or -1,
                "source_path": artifact.source_path,
                "source_excerpt": table.context_excerpt or table.raw_text[:320],
            }
            self._upsert(f"paper_table:{table.table_id}", text, metadata)
        for figure in artifact.figures:
            text = f"{artifact.title}. [figure] {figure.caption}. {figure.raw_text}"
            metadata = {
                "kind": "paper_figure",
                "paper_id": artifact.paper_id,
                "paper_title": artifact.title,
                "figure_id": figure.figure_id,
                "section_id": figure.section_id or "",
                "page_number": figure.page_number or -1,
                "source": figure.source,
                "source_path": artifact.source_path,
                "source_excerpt": figure.context_excerpt or figure.raw_text[:320],
            }
            self._upsert(f"paper_figure:{figure.figure_id}", text, metadata)
        summary = f"{artifact.title}. {artifact.abstract} Sections: {'; '.join(section.heading for section in artifact.sections[:6])}"
        self._upsert(
            f"paper:{artifact.paper_id}",
            summary,
            {
                "kind": "paper",
                "paper_id": artifact.paper_id,
                "paper_title": artifact.title,
                "source_path": artifact.source_path,
                "ocr_used": artifact.ocr_used,
                "parser_used": artifact.parser_used or "",
            },
        )
        self._refresh_claim_clusters()

    def index_verification_report(self, report: VerificationReport) -> None:
        text = (
            f"Verification for {report.trial_id}. verdict={report.verdict}. "
            f"control_score={report.control_score:.4f} "
            f"loss_std={report.seed_variance.loss_std:.4f} "
            f"d_std={report.seed_variance.dimensionality_std:.4f} "
            f"ece={report.calibration.ece:.4f}. "
            f"Recommendations: {'; '.join(report.recommendations)}."
        )
        metadata = {
            "kind": "verification",
            "trial_id": report.trial_id,
            "verdict": report.verdict,
            "control_score": report.control_score,
            "ece": report.calibration.ece,
        }
        self._upsert(f"verification:{report.trial_id}", text, metadata)

    def index_breakthrough_report(self, report: BreakthroughReport) -> None:
        text = (
            f"Breakthrough report for {report.trial_id}. status={report.status}. "
            f"novelty={report.novelty_score:.4f} stability={report.stability_score:.4f} "
            f"calibration={report.calibration_score:.4f}. "
            f"Summary: {report.summary}. "
            f"Rationale: {'; '.join(report.rationale)}."
        )
        metadata = {
            "kind": "breakthrough",
            "trial_id": report.trial_id,
            "status": report.status,
            "novelty_score": report.novelty_score,
            "stability_score": report.stability_score,
            "calibration_score": report.calibration_score,
        }
        self._upsert(f"breakthrough:{report.trial_id}", text, metadata)

    def index_problem_study(self, report: ProblemStudyReport) -> None:
        hypothesis_summaries = [item.hypothesis for item in report.hypotheses[:4]]
        text = (
            f"Problem study {report.problem_id} for domain {report.domain}. "
            f"Problem: {report.problem}. "
            f"Hypotheses: {'; '.join(hypothesis_summaries)}. "
            f"Benchmarks: {'; '.join(report.benchmark_targets)}. "
            f"Metrics: {'; '.join(report.metric_hooks)}. "
            f"Next action: {report.next_action}."
        )
        metadata = {
            "kind": "problem_study",
            "problem_id": report.problem_id,
            "domain": report.domain,
            "profile_id": report.profile_id,
            "status": report.status,
            "hypothesis_ids": ",".join(item.hypothesis_id for item in report.hypotheses[:4]),
        }
        self._upsert(f"problem_study:{report.problem_id}", text, metadata)

    def index_problem_execution(self, report: ProblemExecutionReport) -> None:
        experiment_summaries = [
            f"{item.name}:{item.status}:{','.join(f'{k}={v:.4f}' for k, v in item.metrics.items())}"
            for item in report.experiments[:3]
        ]
        text = (
            f"Problem execution {report.problem_id} for domain {report.domain}. "
            f"Status={report.status}. Summary: {report.summary}. "
            f"Next step: {report.recommended_next_step}. "
            f"Experiments: {'; '.join(experiment_summaries)}."
        )
        metadata = {
            "kind": "problem_execution",
            "problem_id": report.problem_id,
            "domain": report.domain,
            "profile_id": report.profile_id,
            "status": report.status,
        }
        self._upsert(f"problem_execution:{report.problem_id}", text, metadata)

    def search(
        self,
        query: str,
        n_results: int = 3,
        kind: Optional[str] = None,
        *,
        require_research_grade: bool = False,
    ) -> list[MemorySearchHit]:
        if require_research_grade:
            self.ensure_research_ready()
        if self.collection.count() == 0:
            return []
        candidates = self._load_documents(kind=kind)
        if not candidates:
            return []
        query_embedding = self.embedder.embed(query)
        dense_ranked = sorted(
            ((self._cosine(query_embedding, candidate.embedding), candidate) for candidate in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        lexical_ranked = sorted(
            ((self._lexical_score(query, candidate.document), candidate) for candidate in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        candidate_pool: dict[str, RetrievedDocument] = {}
        for _, candidate in dense_ranked[: max(n_results * 4, 8)]:
            candidate_pool[candidate.document_id] = candidate
        for _, candidate in lexical_ranked[: max(n_results * 4, 8)]:
            candidate_pool[candidate.document_id] = candidate
        scored = self.reranker.rerank(
            query,
            query_embedding,
            list(candidate_pool.values()),
            top_k=n_results,
        )
        hits = [
            MemorySearchHit(
                document_id=item.document_id,
                score=round(score, 6),
                document=item.document,
                metadata=dict(item.metadata),
            )
            for score, item in scored[:n_results]
        ]
        return self._annotate_hits_with_contradictions(hits)

    def search_similar_trials(self, metrics: list[GovernorMetrics], n_results: int = 3) -> list[MemorySearchHit]:
        if not metrics:
            return []
        latest = metrics[-1]
        query = (
            f"Find experiments with similar loss spikes. "
            f"E={latest.energy_e:.6f} sigma={latest.entropy_sigma:.6f} rho={latest.drift_rho:.6f} "
            f"grad={latest.grad_norm:.6f} D_PR={latest.effective_dimensionality:.4f} "
            f"eq={latest.equilibrium_fraction:.2%} outcome recovery."
        )
        return self.search(query, n_results=n_results)

    @staticmethod
    def metric_text(metric: GovernorMetrics) -> str:
        return (
            f"Trial {metric.trial_id} step {metric.step} "
            f"E={metric.energy_e:.6f} sigma={metric.entropy_sigma:.6f} "
            f"rho={metric.drift_rho:.6f} drift_l2={metric.drift_l2:.6f} grad={metric.grad_norm:.6f} "
            f"D_PR={metric.effective_dimensionality:.4f} eq={metric.equilibrium_fraction:.2%}."
        )

    def _upsert(self, document_id: str, document: str, metadata: dict[str, Any]) -> None:
        try:
            self.collection.upsert(
                ids=[document_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[self.embedder.embed(document)],
            )
        except Exception as exc:  # pragma: no cover - backend-specific failure path
            message = (
                "TAR vector memory collection is incompatible with the active embedder. "
                f"collection={self.memory_manifest.collection_name} "
                f"embedder={self.embedder_name} dim={self.embedding_dim}: {exc}"
            )
            self.mark_degraded(message)
            raise MemoryIntegrityError(message) from exc

    def _resolve_embedder(self) -> Any:
        model_name = os.environ.get("TAR_EMBEDDER_MODEL", DEFAULT_EMBEDDER_MODEL).strip() or DEFAULT_EMBEDDER_MODEL
        allow_download = os.environ.get("TAR_ALLOW_MODEL_DOWNLOAD", "").strip() == "1"
        try:
            embedder = SemanticEmbedder(model_name, allow_download=allow_download)
            self.embedder_name = model_name
            self.semantic_ready = True
            return embedder
        except Exception:
            fallback = LexicalProjectionEmbedder()
            self.embedder_name = fallback.model_name
            self.semantic_ready = False
            return fallback

    def ensure_research_ready(self) -> None:
        if not getattr(self, "semantic_ready", False):
            raise ScientificValidityError(
                "Research-grade literature retrieval requires a semantic embedding model. "
                "Configure sentence-transformers with BAAI/bge-small-en-v1.5 available locally "
                "or enable TAR_ALLOW_MODEL_DOWNLOAD=1 before running literature-grounded research."
            )

    def _load_documents(self, *, kind: Optional[str] = None) -> list[RetrievedDocument]:
        where = {"kind": kind} if kind else None
        payload = self.collection.get(where=where, include=["documents", "metadatas", "embeddings"])
        ids = payload.get("ids", [])
        documents = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])
        embeddings = payload.get("embeddings", [])
        rows: list[RetrievedDocument] = []
        for idx, document_id in enumerate(ids):
            embedding = embeddings[idx] if idx < len(embeddings) else []
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            rows.append(
                RetrievedDocument(
                    document_id=document_id,
                    document=documents[idx] if idx < len(documents) else "",
                    metadata=metadatas[idx] if idx < len(metadatas) and metadatas[idx] is not None else {},
                    embedding=[float(item) for item in embedding] if embedding is not None else [],
                )
            )
        return rows

    def _refresh_claim_clusters(self) -> None:
        claims = [item for item in self._load_documents(kind="paper_claim") if item.metadata.get("claim_id")]
        if not claims:
            self.claim_clusters_path.write_text("[]", encoding="utf-8")
            self.claim_conflicts_path.write_text("[]", encoding="utf-8")
            return

        parent = list(range(len(claims)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for idx, left in enumerate(claims):
            for jdx in range(idx + 1, len(claims)):
                right = claims[jdx]
                dense = self._cosine(left.embedding, right.embedding)
                lexical = self._lexical_score(left.document, right.document)
                if dense >= 0.80 or (dense >= 0.60 and lexical >= 0.35):
                    union(idx, jdx)

        groups: dict[int, list[int]] = {}
        for idx in range(len(claims)):
            groups.setdefault(find(idx), []).append(idx)

        clusters: list[ClaimCluster] = []
        conflicts: list[ClaimConflict] = []
        for indices in groups.values():
            if len(indices) < 2:
                continue
            topic_terms = self._cluster_topic_terms([claims[idx].document for idx in indices])
            contradiction_pairs: list[list[str]] = []
            for offset, idx in enumerate(indices):
                left = claims[idx]
                for jdx in indices[offset + 1 :]:
                    right = claims[jdx]
                    if left.metadata.get("paper_id") == right.metadata.get("paper_id"):
                        continue
                    if left.metadata.get("polarity") == right.metadata.get("polarity"):
                        continue
                    if "neutral" in {left.metadata.get("polarity"), right.metadata.get("polarity")}:
                        continue
                    contradiction_pairs.append([str(left.metadata["claim_id"]), str(right.metadata["claim_id"])])
                    conflicts.append(
                        ClaimConflict(
                            left_claim_id=str(left.metadata["claim_id"]),
                            right_claim_id=str(right.metadata["claim_id"]),
                            reason=f"Semantically similar claims with opposite polarity: {', '.join(topic_terms[:6])}",
                            score=round(self._contradiction_score(left, right), 6),
                            left_paper_id=str(left.metadata.get("paper_id", "")) or None,
                            right_paper_id=str(right.metadata.get("paper_id", "")) or None,
                            left_page_number=self._optional_int(left.metadata.get("page_number")),
                            right_page_number=self._optional_int(right.metadata.get("page_number")),
                            topic_terms=topic_terms,
                        )
                    )
            clusters.append(
                ClaimCluster(
                    cluster_id=self._stable_id("claim_cluster", "|".join(sorted(str(claims[idx].metadata["claim_id"]) for idx in indices))),
                    claim_ids=sorted(str(claims[idx].metadata["claim_id"]) for idx in indices),
                    topic_terms=topic_terms,
                    contradiction_pairs=contradiction_pairs,
                    evidence_count=len(indices),
                )
            )

        self.claim_clusters_path.write_text(
            json.dumps([item.model_dump(mode="json") for item in clusters], indent=2),
            encoding="utf-8",
        )
        self.claim_conflicts_path.write_text(
            json.dumps([item.model_dump(mode="json") for item in conflicts], indent=2),
            encoding="utf-8",
        )

    def _annotate_hits_with_contradictions(self, hits: list[MemorySearchHit]) -> list[MemorySearchHit]:
        clusters = self._load_claim_clusters()
        conflicts = self._load_claim_conflicts()
        claim_to_cluster: dict[str, ClaimCluster] = {}
        for cluster in clusters:
            for claim_id in cluster.claim_ids:
                claim_to_cluster[claim_id] = cluster
        claim_to_conflicts: dict[str, list[ClaimConflict]] = {}
        for conflict in conflicts:
            claim_to_conflicts.setdefault(conflict.left_claim_id, []).append(conflict)
            claim_to_conflicts.setdefault(conflict.right_claim_id, []).append(conflict)

        annotated: list[MemorySearchHit] = []
        for hit in hits:
            metadata = dict(hit.metadata)
            if metadata.get("kind") == "paper_claim":
                claim_id = str(metadata.get("claim_id", ""))
                cluster = claim_to_cluster.get(claim_id)
                if cluster is not None:
                    metadata["claim_cluster_id"] = cluster.cluster_id
                    metadata["cluster_topic_terms"] = cluster.topic_terms
                    related_conflicts = claim_to_conflicts.get(claim_id, [])
                    if related_conflicts:
                        metadata["contradictory_claims"] = [item.model_dump(mode="json") for item in related_conflicts]
                metadata["evidence_trace"] = self._build_evidence_trace(hit).model_dump(mode="json")
            annotated.append(hit.model_copy(update={"metadata": metadata}))
        return annotated

    def build_evidence_traces(self, hits: list[MemorySearchHit]) -> list[EvidenceTrace]:
        return [self._build_evidence_trace(hit) for hit in hits]

    def _load_claim_clusters(self) -> list[ClaimCluster]:
        if not self.claim_clusters_path.exists():
            return []
        payload = json.loads(self.claim_clusters_path.read_text(encoding="utf-8"))
        return [ClaimCluster.model_validate(item) for item in payload]

    def _load_claim_conflicts(self) -> list[ClaimConflict]:
        if not self.claim_conflicts_path.exists():
            return []
        payload = json.loads(self.claim_conflicts_path.read_text(encoding="utf-8"))
        return [ClaimConflict.model_validate(item) for item in payload]

    def capability_report(self) -> LiteratureCapabilityReport:
        return LiteratureCapabilityReport(
            semantic_model=DEFAULT_EMBEDDER_MODEL,
            semantic_ready=getattr(self, "semantic_ready", False),
            reranker=self.reranker_name,
            reranker_ready=True,
            notes=[] if getattr(self, "semantic_ready", False) else ["semantic_embedder_unavailable"],
        )

    @staticmethod
    def _cluster_topic_terms(texts: list[str]) -> list[str]:
        counts: dict[str, int] = {}
        stop = {"the", "and", "for", "with", "that", "this", "from", "into", "than", "were", "have", "has"}
        for text in texts:
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                if len(token) <= 3 or token in stop:
                    continue
                counts[token] = counts.get(token, 0) + 1
        return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(item * item for item in left))
        right_norm = math.sqrt(sum(item * item for item in right))
        if left_norm <= 1e-12 or right_norm <= 1e-12:
            return 0.0
        return max(-1.0, min(1.0, dot / (left_norm * right_norm)))

    @staticmethod
    def _lexical_score(query: str, document: str) -> float:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        doc_tokens = set(re.findall(r"[a-z0-9]+", document.lower()))
        if not query_tokens or not doc_tokens:
            return 0.0
        overlap = len(query_tokens & doc_tokens)
        return overlap / math.sqrt(len(query_tokens) * len(doc_tokens))

    @staticmethod
    def _coverage_score(query: str, document: str) -> float:
        query_tokens = [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 2]
        if not query_tokens:
            return 0.0
        doc_tokens = set(re.findall(r"[a-z0-9]+", document.lower()))
        covered = sum(1 for token in query_tokens if token in doc_tokens)
        return covered / len(query_tokens)

    def _contradiction_score(self, left: RetrievedDocument, right: RetrievedDocument) -> float:
        dense = self._cosine(left.embedding, right.embedding)
        lexical = self._lexical_score(left.document, right.document)
        coverage = self._coverage_score(left.document, right.document)
        return max(0.5, min(0.99, 0.60 * dense + 0.20 * lexical + 0.20 * coverage))

    def _build_evidence_trace(self, hit: MemorySearchHit) -> EvidenceTrace:
        metadata = hit.metadata or {}
        contradictions = metadata.get("contradictory_claims") or []
        contradiction_summary = [
            f"{item.get('left_claim_id')} vs {item.get('right_claim_id')}: {item.get('reason')}"
            for item in contradictions[:3]
            if isinstance(item, dict)
        ]
        bibliography_ids = self._decode_string_list(metadata.get("citation_entry_ids"))
        bibliography_entry_id = metadata.get("bibliography_entry_id")
        if bibliography_entry_id and bibliography_entry_id not in bibliography_ids:
            bibliography_ids = [bibliography_entry_id, *bibliography_ids]
        return EvidenceTrace(
            document_id=hit.document_id,
            kind=str(metadata.get("kind", "memory")),
            paper_id=metadata.get("paper_id"),
            paper_title=metadata.get("paper_title"),
            claim_id=metadata.get("claim_id"),
            section_id=metadata.get("section_id"),
            page_number=self._optional_int(metadata.get("page_number")),
            score=hit.score,
            source_path=metadata.get("source_path"),
            excerpt=str(metadata.get("source_excerpt") or hit.document[:320]),
            bibliography_entry_ids=[str(item) for item in bibliography_ids if item],
            contradiction_count=len(contradictions),
            contradiction_summary=contradiction_summary,
        )

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value in {None, "", -1}:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _decode_string_list(value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item]
        if isinstance(value, str):
            try:
                payload = json.loads(value)
            except json.JSONDecodeError:
                return [value]
            if isinstance(payload, list):
                return [str(item) for item in payload if item]
            if payload:
                return [str(payload)]
        return [str(value)]

    @staticmethod
    def _stable_id(prefix: str, value: str) -> str:
        return f"{prefix}:{blake2b(value.encode('utf-8'), digest_size=8).hexdigest()}"

    def close(self) -> None:
        system = getattr(self.client, "_system", None)
        if system is not None:
            try:
                system.stop()
            except Exception:
                pass
        clear_cache = getattr(self.client, "clear_system_cache", None)
        if callable(clear_cache):
            try:
                clear_cache()
            except Exception:
                pass


class MemoryIndexer:
    def __init__(self, store: TARStateStore, vault: VectorVault, poll_interval_s: float = 0.5):
        self.store = store
        self.vault = vault
        self.poll_interval_s = poll_interval_s
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._metrics_mtime: Optional[float] = None
        self._graph_mtime: Optional[float] = None
        self._research_mtime: Optional[float] = None
        self._verification_mtime: Optional[float] = None
        self._breakthrough_mtime: Optional[float] = None
        self._problem_studies_mtime: Optional[float] = None
        self._problem_executions_mtime: Optional[float] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def sync_once(self) -> None:
        if self.vault.memory_manifest.state in {"rebuild_required", "degraded"}:
            self.vault.begin_rebuild()
        try:
            metrics_path = self.store.metrics_log_path
            if metrics_path.exists():
                current_mtime = metrics_path.stat().st_mtime
                if self._metrics_mtime != current_mtime:
                    for metric in self.store.iter_metrics():
                        self.vault.index_metric(metric)
                    self._metrics_mtime = current_mtime

            graph_path = self.store.knowledge_graph_path
            if graph_path.exists():
                current_mtime = graph_path.stat().st_mtime
                if self._graph_mtime != current_mtime:
                    self.vault.index_knowledge_graph(self.store.load_knowledge_graph())
                    self._graph_mtime = current_mtime

            research_path = self.store.research_intel_path
            if research_path.exists():
                current_mtime = research_path.stat().st_mtime
                if self._research_mtime != current_mtime:
                    for document in self.store.iter_research_documents():
                        self.vault.index_research_document(document)
                    self._research_mtime = current_mtime

            verification_path = self.store.verification_reports_path
            if verification_path.exists():
                current_mtime = verification_path.stat().st_mtime
                if self._verification_mtime != current_mtime:
                    for report in self.store.iter_verification_reports():
                        self.vault.index_verification_report(report)
                    self._verification_mtime = current_mtime

            breakthrough_path = self.store.breakthrough_reports_path
            if breakthrough_path.exists():
                current_mtime = breakthrough_path.stat().st_mtime
                if self._breakthrough_mtime != current_mtime:
                    for report in self.store.iter_breakthrough_reports():
                        self.vault.index_breakthrough_report(report)
                    self._breakthrough_mtime = current_mtime

            problem_studies_path = self.store.problem_studies_path
            if problem_studies_path.exists():
                current_mtime = problem_studies_path.stat().st_mtime
                if self._problem_studies_mtime != current_mtime:
                    for report in self.store.iter_problem_studies():
                        self.vault.index_problem_study(report)
                    self._problem_studies_mtime = current_mtime

            problem_executions_path = self.store.problem_executions_path
            if problem_executions_path.exists():
                current_mtime = problem_executions_path.stat().st_mtime
                if self._problem_executions_mtime != current_mtime:
                    for report in self.store.iter_problem_executions():
                        self.vault.index_problem_execution(report)
                    self._problem_executions_mtime = current_mtime
        except MemoryIntegrityError:
            raise
        except Exception as exc:  # pragma: no cover - defensive sync boundary
            message = f"Vector memory sync failed: {exc}"
            self.vault.mark_degraded(message)
            raise MemoryIntegrityError(message) from exc
        else:
            self.vault.complete_rebuild()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                self.sync_once()
            except Exception:
                pass
            sleep(self.poll_interval_s)
