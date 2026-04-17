import json
from pathlib import Path

import pytest

from tar_lab.self_improvement import SelfImprovementEngine
from tar_lab.schemas import TrainingSignalRecord
from tar_lab.state import TARStateStore


def _write_synthetic_pack(workspace: Path) -> tuple[str, str, list[str]]:
    pack_dir = workspace / "eval_artifacts" / "external_validation"
    pack_dir.mkdir(parents=True, exist_ok=True)
    item_ids = [
        "anchor-item-001",
        "anchor-item-002",
    ]
    (pack_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "pack_manifest_sha256": "sealed-pack-hash",
                "item_count": len(item_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (pack_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps({"item_id": item_id}) for item_id in item_ids) + "\n",
        encoding="utf-8",
    )
    return (
        "eval_artifacts/external_validation",
        str(pack_dir / "run_manifest.json"),
        item_ids,
    )


def _signal(signal_id: str, source_id: str, **updates: object) -> TrainingSignalRecord:
    payload = {
        "signal_id": signal_id,
        "kind": "problem_study",
        "source_id": source_id,
        "project_id": "proj-1",
        "messages": [{"role": "user", "content": "study this"}],
        "gold_response": "{\"decision\": \"continue\"}",
        "quality_score": 0.8,
    }
    payload.update(updates)
    return TrainingSignalRecord.model_validate(payload)


def test_anchor_pack_initialized_and_loadable(tmp_path: Path):
    pack_path, run_manifest_path, _ = _write_synthetic_pack(tmp_path)
    engine = SelfImprovementEngine(str(tmp_path))
    manifest = engine.initialize_anchor_pack(
        pack_path=pack_path,
        run_manifest_path=run_manifest_path,
        baseline_mean_score=0.4625,
        baseline_overclaim_rate=0.0,
    )
    store = TARStateStore(str(tmp_path))
    loaded = store.load_anchor_manifest()
    assert loaded is not None
    assert store.self_improvement_initialized() is True
    assert manifest.run_manifest_hash_sha256 == loaded.run_manifest_hash_sha256
    assert len(manifest.run_manifest_hash_sha256) == 64


def test_anchor_pack_cannot_be_initialized_twice(tmp_path: Path):
    pack_path, run_manifest_path, _ = _write_synthetic_pack(tmp_path)
    engine = SelfImprovementEngine(str(tmp_path))
    engine.initialize_anchor_pack(pack_path, run_manifest_path, 0.4625, 0.0)
    with pytest.raises(RuntimeError):
        engine.initialize_anchor_pack(pack_path, run_manifest_path, 0.4625, 0.0)


def test_anchor_integrity_check_passes_on_unmodified(tmp_path: Path):
    pack_path, run_manifest_path, _ = _write_synthetic_pack(tmp_path)
    engine = SelfImprovementEngine(str(tmp_path))
    engine.initialize_anchor_pack(pack_path, run_manifest_path, 0.4625, 0.0)
    assert engine.verify_anchor_integrity() is True


def test_curate_signal_rejects_overclaim(tmp_path: Path):
    pack_path, run_manifest_path, _ = _write_synthetic_pack(tmp_path)
    engine = SelfImprovementEngine(str(tmp_path))
    engine.initialize_anchor_pack(pack_path, run_manifest_path, 0.4625, 0.0)
    accepted = engine.curate_signal(
        _signal("sig-overclaim", "local-source-1", overclaim_present=True)
    )
    assert accepted is False
    assert engine.list_signals() == []


def test_curate_signal_rejects_anchor_overlap(tmp_path: Path):
    pack_path, run_manifest_path, item_ids = _write_synthetic_pack(tmp_path)
    engine = SelfImprovementEngine(str(tmp_path))
    engine.initialize_anchor_pack(pack_path, run_manifest_path, 0.4625, 0.0)
    accepted = engine.curate_signal(_signal("sig-overlap", item_ids[0]))
    assert accepted is False
    assert engine.list_signals() == []


def test_assemble_delta_excludes_anchor_overlaps(tmp_path: Path):
    pack_path, run_manifest_path, item_ids = _write_synthetic_pack(tmp_path)
    engine = SelfImprovementEngine(str(tmp_path))
    engine.initialize_anchor_pack(pack_path, run_manifest_path, 0.4625, 0.0)
    assert engine.curate_signal(_signal("sig-clean-1", "local-source-1")) is True
    assert engine.curate_signal(
        _signal("sig-clean-2", "local-source-2", kind="research_decision")
    ) is True
    overlap_signal = _signal(
        "sig-overlap-persisted",
        item_ids[0],
        anchor_pack_overlap=True,
    )
    signals_dir = tmp_path / "tar_state" / "self_improvement" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    (signals_dir / f"{overlap_signal.signal_id}.json").write_text(
        overlap_signal.model_dump_json(indent=2),
        encoding="utf-8",
    )
    cycle = engine.start_cycle()
    delta = engine.assemble_delta(cycle.cycle_id)
    assert delta.anchor_overlaps_excluded >= 1
    assert "sig-overlap-persisted" not in delta.signal_ids


def test_gate_fails_on_anchor_integrity_failure(tmp_path: Path):
    pack_path, run_manifest_path, _ = _write_synthetic_pack(tmp_path)
    engine = SelfImprovementEngine(str(tmp_path))
    engine.initialize_anchor_pack(pack_path, run_manifest_path, 0.4625, 0.0)
    gate_passed, reason = engine.evaluate_gate(
        probe_mean_score=0.4625,
        probe_overclaim_rate=0.0,
        anchor_hash_verified=False,
    )
    assert gate_passed is False
    assert "anchor_integrity" in reason


def test_consecutive_gate_failures_pause_cycle(tmp_path: Path):
    pack_path, run_manifest_path, _ = _write_synthetic_pack(tmp_path)
    engine = SelfImprovementEngine(str(tmp_path))
    engine.initialize_anchor_pack(pack_path, run_manifest_path, 0.4625, 0.0)
    cycle = engine.start_cycle()
    cycle = engine.record_gate_failure(cycle, "failure-1")
    cycle = engine.record_gate_failure(cycle, "failure-2")
    cycle = engine.record_gate_failure(cycle, "failure-3")
    assert cycle.status == "paused_consecutive_failures"
    assert cycle.human_resume_required is True
