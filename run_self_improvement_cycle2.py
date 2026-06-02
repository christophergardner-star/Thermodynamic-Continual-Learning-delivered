"""
run_self_improvement_cycle2.py
TAR Self-Improvement Cycle 2 launcher.

Runs the full cycle: run1() -> gate_eval() -> deploy()
using the pre-assembled delta-24de4a79 (29 signals, 4 kinds, diversity=0.8).

Hardware requirement: GPU with >=14GB VRAM (Qwen2.5-7B-Instruct in fp16).
  - RunPod A40/3090/4090 or equivalent.
  - Set TAR_WS38_BASE_MODEL=/workspace/models/Qwen2.5-7B-Instruct if model
    is pre-cached in the RunPod volume; otherwise HF download will be used.

NOTE (GAP 1 — pre-RunPod TODO): run1() trains the adapter but does not evaluate
it.  probe_mean_score on the RetrainRecord will be None after run1().  Before
calling evaluate_gate(), you must run the eval harness against baseline_eval_v1
using the trained adapter, read overall.mean_score from results.json, and set
probe_mean_score on the retrain record:

    from tar_lab.eval_harness import run_eval_suite  # or equivalent
    eval_result = run_eval_suite(adapter_path, anchor_pack_path)
    retrain = retrain.model_copy(update={
        "probe_mean_score": eval_result["overall"]["mean_score"],
        "probe_overclaim_rate": eval_result["overall"].get("overclaim_rate", 0.0),
    })
    engine.save_retrain(retrain)

Without this step, probe_mean_score defaults to 0.0 and the gate always fails.

Usage:
    python run_self_improvement_cycle2.py
    python run_self_improvement_cycle2.py --dry-run   # validate only, no training
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the delivered directory root or E:\TAR
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

    if dry_run:
        print()
        print("DRY RUN complete — all checks passed. Re-run without --dry-run to train.")
        return

    # Run1: LoRA fine-tune
    print()
    print("=== Phase 1: LoRA fine-tuning (run1) ===")
    retrain = engine.run1(CYCLE_ID, DELTA_ID)
    print(f"retrain_id       : {retrain.retrain_id}")
    print(f"adapter_output   : {retrain.adapter_output_path}")
    print(f"anchor_verified  : {retrain.anchor_hash_verified}")
    print(f"probe_mean_score : {retrain.probe_mean_score}")
    print(f"probe_overclaim  : {retrain.probe_overclaim_rate}")
    print(f"run_kind         : {retrain.run_kind}")

    # --- GAP 1 placeholder ---
    # probe_mean_score is None here because run1() does not evaluate the adapter.
    # Before this script can successfully pass the gate, insert an eval step here:
    #   eval_result = run_eval_suite(retrain.adapter_output_path, anchor_pack_path)
    #   retrain = retrain.model_copy(update={"probe_mean_score": ..., "probe_overclaim_rate": ...})
    #   engine.save_retrain(retrain)
    # See module docstring for the full pattern.
    if retrain.probe_mean_score is None:
        print()
        print("WARNING: probe_mean_score is None — eval step not yet implemented (GAP 1).")
        print("         Gate will fail with mean_score=0.0 < floor=0.40.")
        print("         Implement eval harness call before RunPod session.")

    # Gate evaluation
    print()
    print("=== Phase 2: Gate evaluation ===")
    gate_passed, reason = engine.evaluate_gate(
        probe_mean_score=retrain.probe_mean_score or 0.0,
        probe_overclaim_rate=retrain.probe_overclaim_rate or 0.0,
        anchor_hash_verified=retrain.anchor_hash_verified or False,
    )
    print(f"gate_passed : {gate_passed}")
    print(f"reason      : {reason}")

    if not gate_passed:
        print()
        print("Gate FAILED — recording failure, adapter NOT deployed.")
        updated_cycle = engine.record_gate_failure(cycle, reason)
        print(f"cycle status updated to: {updated_cycle.status}")
        print(f"consecutive_failures   : {updated_cycle.consecutive_gate_failures}")
        sys.exit(2)

    # Save gate_passed=True to retrain record before deploy (required by deploy() guard)
    retrain = retrain.model_copy(update={"gate_passed": True})
    engine.save_retrain(retrain)

    # Deploy
    print()
    print("=== Phase 3: Deploying adapter ===")
    engine.deploy(retrain_id=retrain.retrain_id, cycle_id=CYCLE_ID)

    # Update serving mode
    serving_path = WORKSPACE / "tar_state/operator_serving.json"
    serving = json.loads(serving_path.read_text(encoding="utf-8"))
    serving["mode"] = "tuned_local"
    serving.pop("_note", None)
    import os
    from datetime import datetime, timezone
    serving["selected_at"] = datetime.now(timezone.utc).isoformat()
    serving_path.write_text(json.dumps(serving, indent=2), encoding="utf-8")

    print("Adapter deployed. operator_serving.json updated to mode=tuned_local.")
    print()

    active = json.loads((WORKSPACE / "tar_state/serving/active_adapter.json").read_text(encoding="utf-8"))
    print("active_adapter.json:")
    for k, v in active.items():
        print(f"  {k}: {v}")

    print()
    print("=== Cycle 2 COMPLETE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAR Self-Improvement Cycle 2")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no training")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
