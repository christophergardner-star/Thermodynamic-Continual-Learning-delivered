"""
Read hyperparameter_selection.json (output of Step 1) and patch
phase16_cifar100_rerun.py and phase17_tinyimagenet_rerun.py with the
locked best_configs for each method.

Usage:
    python apply_locked_hyperparameters.py [--dry-run]

Run this after run_hyperparameter_selection.py completes.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_TAR_STATE = Path(r"E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state")
HP_FILE = _TAR_STATE / "hyperparameter_selection.json"

SCRIPTS = [
    _REPO / "phase16_cifar100_rerun.py",
    _REPO / "phase17_tinyimagenet_rerun.py",
]

# Only params that have the "# Update from hyperparameter_selection.json" marker.
# ewc_lambda and si_c are already set from Phase 12/13 results and not re-swept.
METHOD_PATCH_MAP = {
    "der_plus_plus": {
        "param": "der_mem_size",
        "pattern": r'("der_mem_size":\s*)[\d]+(\s*,\s*#\s*Update from hyperparameter_selection\.json.*)',
    },
    "lwf": {
        "param": "lwf_alpha",
        "pattern": r'("lwf_alpha":\s*)[\d.]+(\s*,\s*#\s*Update from hyperparameter_selection\.json.*)',
    },
    "lwf_temperature": {  # companion param swept together with lwf_alpha
        "method": "lwf",
        "param": "lwf_temperature",
        "pattern": r'("lwf_temperature":\s*)[\d.]+(\s*,?\s*)',
    },
}


def main(dry_run: bool = False) -> None:
    if not HP_FILE.exists():
        print(f"ERROR: {HP_FILE} not found. Run run_hyperparameter_selection.py first.")
        sys.exit(1)

    hp = json.loads(HP_FILE.read_text())
    selected: dict = hp.get("selected", {})

    if not selected:
        print("ERROR: hyperparameter_selection.json has no 'selected' entry.")
        sys.exit(1)

    # Extract {method: config_dict} for non-failed entries
    best_configs: dict = {}
    print(f"Locked hyperparameters from {HP_FILE.name}:")
    for method, entry in selected.items():
        cfg = entry.get("config", {})
        status = entry.get("status", "ok")
        if status in ("all_failed", "failed", "collapsed"):
            print(f"  {method}: SKIPPED (status={status})")
        else:
            best_configs[method] = cfg
            print(f"  {method}: {cfg}  forgetting={entry.get('mean_forgetting','?')}")

    for script_path in SCRIPTS:
        if not script_path.exists():
            print(f"\nWARNING: {script_path.name} not found, skipping")
            continue

        content = script_path.read_text(encoding="utf-8")
        original = content
        changes: list[str] = []

        for patch_key, spec in METHOD_PATCH_MAP.items():
            method = spec.get("method", patch_key)  # lwf_temperature uses method="lwf"
            if method not in best_configs:
                continue
            cfg = best_configs[method]
            param = spec["param"]
            if param not in cfg:
                continue
            new_val = cfg[param]
            pattern = spec["pattern"]
            # Replace the value while keeping surrounding context
            new_val_str = str(int(new_val)) if isinstance(new_val, float) and new_val == int(new_val) else str(new_val)
            replacement = rf'\g<1>{new_val_str}\2'
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                changes.append(f"  {method}.{param} = {new_val_str}")
                content = new_content

        if content != original:
            print(f"\n{script_path.name}: {len(changes)} change(s):")
            for c in changes:
                print(c)
            if not dry_run:
                script_path.write_text(content, encoding="utf-8")
                print(f"  Written.")
            else:
                print(f"  [DRY RUN — not written]")
        else:
            print(f"\n{script_path.name}: no matching placeholders found (check patterns)")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
