import json
import tempfile
from pathlib import Path
from types import MethodType

from tar_lab.errors import MemoryIntegrityError
from tar_lab.memory.vault import LEGACY_COLLECTION_NAME, MemoryIndexer, VectorVault
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import GovernorMetrics
from tar_lab.state import TARStateStore


class TinySentenceTransformer:
    def __init__(self, model_name: str, local_files_only: bool = True, device: str = "cpu"):
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.device = device

    def encode(self, text: str, normalize_embeddings: bool = True):
        lowered = text.lower()
        vector = [
            float("trial" in lowered),
            float("entropy" in lowered),
            float("drift" in lowered),
            float("research" in lowered),
            float("verification" in lowered),
            float("breakthrough" in lowered),
        ]
        norm = sum(item * item for item in vector) ** 0.5
        if norm and normalize_embeddings:
            return [item / norm for item in vector]
        return vector


def _metric() -> GovernorMetrics:
    return GovernorMetrics(
        trial_id="trial-memory",
        step=1,
        energy_e=0.1,
        entropy_sigma=0.2,
        drift_l2=0.3,
        drift_rho=0.04,
        grad_norm=0.5,
        effective_dimensionality=2.0,
        equilibrium_fraction=0.0,
    )


def test_vector_vault_rotates_collection_when_embedder_changes(monkeypatch):
    import tar_lab.memory.vault as vault_module

    monkeypatch.setattr(vault_module, "SentenceTransformer", TinySentenceTransformer)
    with tempfile.TemporaryDirectory() as tmp:
        store = TARStateStore(tmp)
        store.append_metric(_metric())
        vault = VectorVault(tmp)
        try:
            MemoryIndexer(store, vault).sync_once()
            original_stats = vault.stats()
        finally:
            vault.close()

        monkeypatch.setattr(vault_module, "SentenceTransformer", None)
        migrated_vault = VectorVault(tmp)
        try:
            migrated_stats = migrated_vault.stats()
            assert migrated_stats["collection_name"] != original_stats["collection_name"]
            assert original_stats["collection_name"] in migrated_stats["retired_collections"]
            assert migrated_stats["state"] == "rebuild_required"

            MemoryIndexer(store, migrated_vault).sync_once()
            rebuilt_stats = migrated_vault.stats()
            assert rebuilt_stats["state"] == "healthy"
            assert rebuilt_stats["documents"] >= 1

            manifest = store.load_memory_manifest()
            assert manifest is not None
            assert manifest.collection_name == rebuilt_stats["collection_name"]
            assert manifest.embedding_dim == rebuilt_stats["embedding_dim"]
        finally:
            migrated_vault.close()


def test_vector_vault_retires_legacy_collection_name(monkeypatch):
    import tar_lab.memory.vault as vault_module

    monkeypatch.setattr(vault_module, "SentenceTransformer", None)
    with tempfile.TemporaryDirectory() as tmp:
        db_dir = Path(tmp) / "tar_state" / "memory"
        db_dir.mkdir(parents=True, exist_ok=True)
        client = vault_module.chromadb.PersistentClient(path=str(db_dir))
        client.get_or_create_collection(name=LEGACY_COLLECTION_NAME, metadata={"space": "cosine"})

        vault = VectorVault(tmp)
        try:
            stats = vault.stats()
            assert LEGACY_COLLECTION_NAME in stats["retired_collections"]
            assert stats["collection_name"] != LEGACY_COLLECTION_NAME
        finally:
            vault.close()


def test_orchestrator_status_reports_memory_warning_without_crashing():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            assert orchestrator.memory_indexer is not None

            def broken_sync_once(self):
                raise MemoryIntegrityError("memory manifest mismatch")

            orchestrator.memory_indexer.sync_once = MethodType(broken_sync_once, orchestrator.memory_indexer)
            payload = orchestrator.status()
            assert payload["memory_warning"] == "memory manifest mismatch"
            assert payload["memory"]["error"] == "memory manifest mismatch"
        finally:
            orchestrator.shutdown()


def test_memory_degradation_does_not_break_dry_run_or_study_problem():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            orchestrator._sync_memory = lambda: False  # type: ignore[method-assign]
            dry_run = orchestrator.run_dry_run()
            assert dry_run.trial_id

            study = orchestrator.study_problem(
                "Investigate optimization stability in deep learning",
                build_env=False,
                max_results=0,
                benchmark_tier="validation",
            )
            assert study.problem_id
        finally:
            orchestrator.shutdown()


def test_state_store_migrates_legacy_problem_study_hypotheses():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            report = orchestrator.study_problem(
                "Investigate optimization stability in deep learning",
                build_env=False,
                max_results=0,
                benchmark_tier="validation",
            )
            payloads = [
                json.loads(line)
                for line in orchestrator.store.problem_studies_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            payloads[-1]["hypotheses"] = ["Legacy migrated hypothesis."]
            orchestrator.store.problem_studies_path.write_text(
                "\n".join(json.dumps(item) for item in payloads) + "\n",
                encoding="utf-8",
            )
            migrated = orchestrator.store.latest_problem_study(report.problem_id)
            assert migrated is not None
            assert migrated.hypotheses
            assert migrated.hypotheses[0].hypothesis == "Legacy migrated hypothesis."
        finally:
            orchestrator.shutdown()
