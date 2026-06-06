"""
run_self_improvement_cycle2.py
TAR Self-Improvement Cycle 2 launcher.

Runs the full cycle:
  harvest_human_review_signals()   — pull real human-review decisions into signals store
  run1()                           — LoRA fine-tune on the curated delta
  probe_adapter()                  — evaluate adapter on anchor pack, run safety gate
  deploy()                         — deploy ONLY if gate PASSED (separate guarded step)

Hardware requirement: GPU with >=14GB VRAM (Qwen2.5-7B-Instruct in fp16).
  - RunPod A40 / 3090 / 4090 or equivalent.
  - Set TAR_WS38_BASE_MODEL=/workspace/models/Qwen2.5-7B-Instruct if the model
    is pre-cached in the RunPod volume; otherwise HF Hub download is used.
  - Set TAR_ALLOW_MODEL_DOWNLOAD=1 to permit HF Hub download inside probe_adapter.

Usage:
    python run_self_improvement_cycle2.py
    python run_self_improvement_cycle2.py --dry-run   # validate only, no training
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the delivered directory root or C:\Users\cgard\TAR
DELIVERED = Path(__file__).resolve().parent
SOURCE = Path("C:/Users/cgard/TAR/Thermodynamic-Continual-Learning-delivered")
for p in (DELIVERED, SOURCE):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

WORKSPACE = DELIVERED  # E:/TAR/Thermodynamic-Continual-Learning-delivered
CYCLE_ID = "cycle-202e9f28"
DELTA_ID = "delta-24de4a79"


def check_hardware() -> dict:
    import torch
    ok = torch.cuda.is_available()
    if ok:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
    else:
        vram_gb = 0.0
        gpu_name = "none"
    return {
        "cuda_available": ok,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "sufficient": ok and vram_gb >= 14.0,
    }


def main(dry_run: bool = False) -> None:
    from tar_lab.self_improvement import SelfImprovementEngine

    print("=== TAR Self-Improvement Cycle 2 ===")
    print(f"workspace : {WORKSPACE}")
    print(f"cycle_id  : {CYCLE_ID}")
    print(f"delta_id  : {DELTA_ID}")
    print()

    hw = check_hardware()
    print("Hardware check:")
    for k, v in hw.items():
        print(f"  {k}: {v}")
    print()

    if not hw["sufficient"] and not dry_run:
        print("ERROR: Insufficient GPU VRAM for Qwen2.5-7B-Instruct.")
        print(f"  Found {hw['vram_gb']}GB on {hw['gpu_name']} — need >=14GB.")
        print("  Run on RunPod (A40/3090/4090) or set TAR_WS38_BASE_MODEL to a smaller model.")
        sys.exit(1)

    engine = SelfImprovementEngine(str(WORKSPACE))

    # Verify cycle and delta still exist
    cycle = engine.load_cycle(CYCLE_ID)
    if cycle is None:
        print(f"ERROR: Cycle {CYCLE_ID} not found in cycles directory.")
        sys.exit(1)

    delta = engine.load_delta(DELTA_ID)
    if delta is None:
        print(f"ERROR: Delta {DELTA_ID} not found.")
        sys.exit(1)

    if not delta.ready:
        print(f"ERROR: Delta {DELTA_ID} not ready: signal_count={delta.signal_count}, diversity={delta.diversity_score:.3f}")
        sys.exit(1)

    print(f"Delta verified: {delta.signal_count} signals, diversity={delta.diversity_score:.3f}, ready={delta.ready}")
    print(f"Signal kinds: {delta.kind_distribution}")
    print()

    # Verify anchor integrity
    anchor_ok = engine.verify_anchor_integrity()
    print(f"Anchor integrity: {'OK' if anchor_ok else 'FAILED'}")
    if not anchor_ok:
        print("ERROR: Anchor integrity check failed. Aborting.")
        sys.exit(1)

    # --- T2: Harvest human-review signals (B7) ---
    # Idempotent: stable signal_id per review_id; re-harvesting is safe.
    # Fail-safe: errors are logged but do NOT abort the cycle.
    print()
    print("=== Pre-training: Harvesting human-review signals ===")
    try:
        harvest = engine.harvest_human_review_signals()
        if "error" in harvest:
            print(f"  WARNING: harvest returned error={harvest['error']} — continuing with existing signals.")
        else:
            print(f"  scanned          : {harvest.get('scanned', 0)}")
            print(f"  harvested        : {harvest.get('harvested', 0)}")
            print(f"  skipped_no_decision: {harvest.get('skipped_no_decision', 0)}")
            print(f"  rejected         : {harvest.get('rejected', 0)}")
            print(f"  by_kind          : {harvest.get('by_kind', {})}")
    except Exception as exc:
        print(f"  WARNING: harvest_human_review_signals() raised {type(exc).__name__}: {exc}")
        print("  Continuing cycle with existing signals (harvest error is non-fatal).")
    print()

    if dry_run:
        print("DRY RUN complete — all checks passed. Re-run without --dry-run to train.")
        return

    # --- Phase 1: LoRA fine-tune ---
    print("=== Phase 1: LoRA fine-tuning (run1) ===")
    retrain = engine.run1(CYCLE_ID, DELTA_ID)
    print(f"  retrain_id       : {retrain.retrain_id}")
    print(f"  adapter_output   : {retrain.adapter_output_path}")
    print(f"  anchor_verified  : {retrain.anchor_hash_verified}")
    print(f"  probe_mean_score : {retrain.probe_mean_score}  (None = not yet evaluated)")
    print(f"  run_kind         : {retrain.run_kind}")
    print()

    # Free the Phase-1 training model's GPU memory before the probe reloads the model.
    # Otherwise the trained model stays resident and probe_adapter() loading a fresh
    # base+adapter = two 7B models on one GPU -> CUDA OOM on a 24GB card (observed on
    # RTX 3090). run1()'s model/trainer are local and GC-eligible once it returns;
    # gc.collect() + empty_cache() reclaims the VRAM so the probe's single model fits.
    try:
        import gc as _gc
        import torch as _torch
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            _torch.cuda.ipc_collect()
        print("  [mem] freed Phase-1 GPU memory before probe")
        print()
    except Exception as _mem_exc:
        print(f"  [mem] cache-free skipped: {_mem_exc}")

    # --- Phase 2: Probe adapter + gate evaluation (T1 — closes GAP-1) ---
    # probe_adapter() evaluates the trained adapter on the frozen anchor pack,
    # sets probe_mean_score + probe_overclaim_rate, runs evaluate_gate(), records
    # gate_passed / gate_failure_reason on the RetrainRecord, and (on failure)
    # calls record_gate_failure() on the cycle. It does NOT deploy.
    # The predictor is injectable for CPU/offline testing; by default it constructs
    # HFCausalLMPredictor (base model + adapter) — requires GPU for Qwen2.5-7B.
    print("=== Phase 2: Probe adapter + gate evaluation ===")
    try:
        retrain = engine.probe_adapter(retrain.retrain_id)
    except Exception as exc:
        print(f"ERROR: probe_adapter() raised {type(exc).__name__}: {exc}")
        print("Gate cannot be evaluated — recording cycle failure and aborting.")
        engine.record_gate_failure(cycle, f"probe_adapter_exception: {exc}")
        sys.exit(2)

    print(f"  probe_mean_score    : {retrain.probe_mean_score}")
    print(f"  probe_overclaim_rate: {retrain.probe_overclaim_rate}")
    print(f"  anchor_verified     : {retrain.anchor_hash_verified}")
    print(f"  gate_passed         : {retrain.gate_passed}")
    print(f"  gate_failure_reason : {retrain.gate_failure_reason}")
    print()

    if not retrain.gate_passed:
        print("Gate FAILED — adapter NOT deployed.")
        print(f"  reason: {retrain.gate_failure_reason}")
        print("  (cycle gate-failure counter incremented by probe_adapter; check cycle record.)")
        sys.exit(2)

    print("Gate PASSED.")
    print()

    # --- Phase 3: Deploy (only fires on genuine gate PASS) ---
    # gate_passed=True is already set on the retrain record by probe_adapter().
    # deploy() re-checks gate_passed and raises RuntimeError if it is not True,
    # so this step is doubly guarded.
    print("=== Phase 3: Deploying adapter ===")
    engine.deploy(retrain_id=retrain.retrain_id, cycle_id=CYCLE_ID)

    # Update serving mode
    import os
    from datetime import datetime, timezone
    serving_path = WORKSPACE / "tar_state/operator_serving.json"
    serving = json.loads(serving_path.read_text(encoding="utf-8"))
    serving["mode"] = "tuned_local"
    serving.pop("_note", None)
    serving["selected_at"] = datetime.now(timezone.utc).isoformat()
    serving_path.write_text(json.dumps(serving, indent=2), encoding="utf-8")

    print("  Adapter deployed. operator_serving.json updated to mode=tuned_local.")
    print()

    active = json.loads((WORKSPACE / "tar_state/serving/active_adapter.json").read_text(encoding="utf-8"))
    print("  active_adapter.json:")
    for k, v in active.items():
        print(f"    {k}: {v}")

    print()
    print("=== Cycle 2 COMPLETE ===")
    print(f"  probe_mean_score : {retrain.probe_mean_score}")
    print(f"  gate             : PASSED")
    print(f"  adapter          : {retrain.adapter_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAR Self-Improvement Cycle 2")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no training")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
