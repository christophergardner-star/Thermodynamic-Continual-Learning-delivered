import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tar_lab.data_manager import DataManager, DataStarvationError
from tar_lab.errors import ScientificValidityError
from tar_lab.experiment_backends import ExperimentBackendRegistry
from tar_lab.orchestrator import TAROrchestrator
from tar_lab.schemas import (
    DataBundleProvenance,
    DataProvenance,
    DatasetManifest,
    DatasetSourceConfig,
    PreparedDataBundle,
    TokenizerProvenance,
)


def _research_safe_bundle(tmp: str) -> PreparedDataBundle:
    tokenizer = TokenizerProvenance(
        stream_name="anchor",
        tokenizer_id="unit-test-tokenizer",
        tokenizer_class="UnitTokenizer",
        tokenizer_hash="tok-hash",
        tokenizer_vocab_size=1024,
        integrity_check=True,
    )
    anchor = DataProvenance(
        stream_name="anchor",
        dataset_name="local-anchor",
        dataset_split="train",
        data_mode="CACHED_REAL",
        data_purity="local_real",
        source_kind="local_file",
        dataset_identifier="local-anchor",
        local_path=str(Path(tmp) / "anchor.txt"),
        sampling_strategy="deterministic_sharding",
        dataset_fingerprint="anchor-fingerprint",
        tokenizer_id="unit-test-tokenizer",
        tokenizer_class="UnitTokenizer",
        tokenizer_hash="tok-hash",
        tokenizer_vocab_size=1024,
        integrity_check=True,
        is_real_data=True,
        provenance_complete=True,
        research_safe=True,
        tokenizer_provenance=tokenizer,
    )
    research_tokenizer = tokenizer.model_copy(update={"stream_name": "research"})
    research = anchor.model_copy(
        update={
            "stream_name": "research",
            "dataset_name": "local-research",
            "dataset_identifier": "local-research",
            "local_path": str(Path(tmp) / "research.txt"),
            "dataset_fingerprint": "research-fingerprint",
            "tokenizer_provenance": research_tokenizer,
        }
    )
    bundle_provenance = DataBundleProvenance(
        anchor=anchor,
        research=research,
        run_intent="research",
        data_purity="local_real",
        integrity_check=True,
        tokenizer_provenance={"anchor": tokenizer, "research": research_tokenizer},
        provenance_complete=True,
        research_grade=True,
        has_fallback=False,
    )
    source = DatasetSourceConfig(name="local-anchor", mode="CACHED_REAL")
    manifest_anchor = DatasetManifest(
        stream_name="anchor",
        tokenizer_id="unit-test-tokenizer",
        records=8,
        source=source,
        provenance=anchor,
        run_intent="research",
        provenance_complete=True,
        research_grade=True,
    )
    manifest_research = DatasetManifest(
        stream_name="research",
        tokenizer_id="unit-test-tokenizer",
        records=8,
        source=source.model_copy(update={"name": "local-research"}),
        provenance=research,
        run_intent="research",
        provenance_complete=True,
        research_grade=True,
    )
    return PreparedDataBundle(
        anchor_manifest_path="/data/anchor/manifest.json",
        research_manifest_path="/data/research/manifest.json",
        anchor_manifest=manifest_anchor,
        research_manifest=manifest_research,
        data_provenance=bundle_provenance,
        run_intent="research",
        provenance_complete=True,
        research_grade=True,
    )


def test_research_run_without_dataset_raises_data_starvation_error():
    with tempfile.TemporaryDirectory() as tmp:
        manager = DataManager(tmp)
        source = DatasetSourceConfig(
            name="definitely-missing-research-corpus",
            subset="missing",
            split="train",
            mode="DOWNLOAD_REAL",
        )
        with pytest.raises(DataStarvationError, match="Scientific fallback is disabled"):
            manager.prepare_stream("research", source, run_intent="research")


def test_research_run_without_tokenizer_raises_data_starvation_error():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        local_text = root / "local_corpus.txt"
        local_text.write_text("real data but missing tokenizer\n", encoding="utf-8")
        manager = DataManager(tmp)
        with pytest.raises(DataStarvationError, match="explicit tokenizer_id"):
            manager.prepare_stream(
                "research",
                DatasetSourceConfig(name=str(local_text), mode="OFFLINE_FALLBACK"),
                tokenizer_id=None,
                run_intent="research",
            )


def test_manifest_logs_tokenizer_vocab_and_provenance_completeness():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        local_text = root / "local_corpus.txt"
        local_text.write_text("continual learning requires rigorous provenance\n", encoding="utf-8")
        manager = DataManager(tmp)
        manifest = manager.prepare_stream(
            "anchor",
            DatasetSourceConfig(name=str(local_text), mode="OFFLINE_FALLBACK"),
        )
        assert manifest.provenance is not None
        assert manifest.provenance.tokenizer_hash
        assert manifest.provenance.tokenizer_vocab_size == 8192
        assert manifest.provenance.dataset_fingerprint
        assert manifest.provenance.data_purity == "local_real"
        assert manifest.provenance_complete
        assert manifest.provenance.provenance_complete
        assert manifest.provenance.tokenizer_provenance is not None


def test_status_reports_data_purity_and_research_grade(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("TAR_DATA_MODE", "OFFLINE_FALLBACK")
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            orchestrator.plan_trial(dry_run=True)
            status = orchestrator.status()
            assert status["data_purity"] in {"fallback", "cached_real", "local_real", "mixed", "download_real"}
            assert status["data_provenance"] is not None
            assert status["run_intent"] == "control"
            assert status["backend_readiness"] == "executable"
            assert status["research_grade"] is False
        finally:
            orchestrator.shutdown()


def test_scaffold_backend_rejected_for_research_mode():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            bundle = _research_safe_bundle(tmp)
            plan = SimpleNamespace(
                hyperparameters={"backend_id": "coding_asc"},
                trial_id="trial-research",
                strategy_family="elastic_anchor",
                anchor_path="anchors/anchor.pt",
                fim_lambda=0.1,
                bregman_budget=0.2,
                drift_budget=0.05,
                governor_thresholds=orchestrator.governor.thresholds,
                protected_layers=["layer.a"],
                mutable_layers=["layer.b"],
            )
            with pytest.raises(ScientificValidityError, match="scaffold-only"):
                orchestrator._build_payload_config(plan, bundle, run_intent="research")
        finally:
            orchestrator.shutdown()


def test_executable_backend_accepted_for_research_mode():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = TAROrchestrator(workspace=tmp)
        try:
            bundle = _research_safe_bundle(tmp)
            plan = SimpleNamespace(
                hyperparameters={"backend_id": "asc_text"},
                trial_id="trial-research",
                strategy_family="elastic_anchor",
                anchor_path="anchors/anchor.pt",
                fim_lambda=0.1,
                bregman_budget=0.2,
                drift_budget=0.05,
                governor_thresholds=orchestrator.governor.thresholds,
                protected_layers=["layer.a"],
                mutable_layers=["layer.b"],
            )
            payload = orchestrator._build_payload_config(plan, bundle, run_intent="research")
            assert payload.research_grade is True
            assert payload.provenance_complete is True
            assert payload.backend_provenance is not None
            assert payload.backend_provenance.status == "executable"
        finally:
            orchestrator.shutdown()


def test_backend_registry_labels_executable_and_scaffold_backends():
    with tempfile.TemporaryDirectory() as tmp:
        registry = ExperimentBackendRegistry(tmp)
        backends = {item.backend_id: item for item in registry.list_backends()}
        assert backends["asc_text"].status == "executable"
        assert backends["asc_cv"].status == "executable"
        assert backends["asc_rl"].status == "executable"
        assert backends["asc_qml"].status == "executable"
        assert backends["coding_asc"].status == "scaffold"
