from __future__ import annotations

import re
from hashlib import blake2b
from threading import Event, Thread
from time import sleep
from typing import Any, Optional

from tar_lab.schemas import (
    BreakthroughReport,
    GovernorMetrics,
    KnowledgeGraphEntry,
    KnowledgeGraphState,
    MemorySearchHit,
    ResearchDocument,
    SelfCorrectionNote,
    VerificationReport,
)
from tar_lab.state import TARStateStore

try:
    import chromadb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None  # type: ignore[assignment]


class HashEmbedder:
    def __init__(self, dim: int = 96):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        values = [0.0] * self.dim
        for token in re.findall(r"\w+|[^\w\s]", text.lower()):
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % self.dim
            sign = 1.0 if (raw >> 1) & 1 else -1.0
            values[index] += sign
        norm = sum(item * item for item in values) ** 0.5
        if norm <= 1e-12:
            return values
        return [item / norm for item in values]


class VectorVault:
    def __init__(self, workspace: str = "."):
        if chromadb is None:
            raise RuntimeError("chromadb is not installed; install it to enable TAR vector memory")
        self.store = TARStateStore(workspace)
        self.db_dir = self.store.state_dir / "memory"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.client.get_or_create_collection(
            name="lab_history",
            metadata={"space": "cosine"},
        )
        self.embedder = HashEmbedder()

    def stats(self) -> dict[str, Any]:
        return {"documents": self.collection.count(), "path": str(self.db_dir)}

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

    def search(self, query: str, n_results: int = 3, kind: Optional[str] = None) -> list[MemorySearchHit]:
        if self.collection.count() == 0:
            return []
        where = {"kind": kind} if kind else None
        result = self.collection.query(
            query_embeddings=[self.embedder.embed(query)],
            n_results=n_results,
            where=where,
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[MemorySearchHit] = []
        for idx, document_id in enumerate(ids):
            distance = float(distances[idx]) if idx < len(distances) and distances[idx] is not None else 1.0
            hits.append(
                MemorySearchHit(
                    document_id=document_id,
                    score=max(0.0, 1.0 - distance),
                    document=docs[idx] if idx < len(docs) else "",
                    metadata=metas[idx] if idx < len(metas) and metas[idx] is not None else {},
                )
            )
        return hits

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
        self.collection.upsert(
            ids=[document_id],
            documents=[document],
            metadatas=[metadata],
            embeddings=[self.embedder.embed(document)],
        )

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

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                self.sync_once()
            except Exception:
                pass
            sleep(self.poll_interval_s)
