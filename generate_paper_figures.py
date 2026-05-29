"""
Generate conference-quality figures for the TCL paper.
Run once from the repo root: python generate_paper_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER_FIGURES = Path(__file__).parent / "paper" / "figures"
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

COMP_DIR = Path(__file__).parent / "tar_state" / "comparisons"

P10_PATH = COMP_DIR / "phase10_controlled_rerun_20260509T132155Z.json"
P11_PATH = COMP_DIR / "phase11_ablation__20260511T113318Z.json"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

METHOD_COLORS = {
    "TCL": "#2a9d8f",
    "EWC (λ=100)": "#e76f51",
    "SI (c=0.1)": "#457b9d",
    "SGD": "#6d6875",
}


# ── Figure 1: JAF scatter ────────────────────────────────────────────────────

def generate_jaf_scatter(p10: dict) -> Path:
    out = PAPER_FIGURES / "jaf_scatter.pdf"
    per_seed = p10.get("per_seed", [])
    agg = p10.get("aggregate", {})

    METHODS = [
        ("tcl",          "TCL",          "#2a9d8f", "D"),
        ("ewc",          "EWC (λ=100)",  "#e76f51", "s"),
        ("si",           "SI (c=0.1)",   "#457b9d", "^"),
        ("sgd_baseline", "SGD",          "#6d6875", "o"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    for key, label, color, marker in METHODS:
        f_seeds = [row[f"{key}_forgetting"] for row in per_seed if f"{key}_forgetting" in row]
        a_seeds = [row[f"{key}_acc"] for row in per_seed if f"{key}_acc" in row]
        if f_seeds:
            ax.scatter(f_seeds, a_seeds, color=color, s=50, alpha=0.55,
                       zorder=3, edgecolors="none")
        f_mean = agg.get(key, {}).get("forgetting_mean")
        a_mean = agg.get(key, {}).get("acc_mean")
        if f_mean is not None and a_mean is not None:
            ax.scatter([f_mean], [a_mean], color=color, s=160, marker=marker,
                       edgecolors="white", linewidths=1.2, zorder=4, label=label)

    ax.axhline(0.55, color="#cc0000", linewidth=0.9, linestyle="--", alpha=0.6,
               label="Collapse threshold (acc ≤ 0.55)")
    ax.set_xlabel("Mean Forgetting ↓")
    ax.set_ylabel("Final Accuracy ↑")
    ax.set_title("Joint Accuracy-Forgetting: per-seed and mean by method\n"
                 "(diamonds/squares/triangles/circles = means; small dots = individual seeds)",
                 fontsize=9)
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Written: {out}")
    return out


# ── Figure 2: Ablation bar chart ─────────────────────────────────────────────

def generate_ablation_bars(p11: dict) -> Path:
    out = PAPER_FIGURES / "ablation_bars.pdf"
    agg = p11.get("aggregate", {})

    conditions = [
        ("sgd",          "SGD",           "#6d6875"),
        ("governor_only","Governor-only", "#f4a261"),
        ("penalty_only", "Penalty-only",  "#264653"),
        ("full_tcl",     "Full TCL",      "#2a9d8f"),
    ]

    labels = [c[1] for c in conditions]
    means  = [agg.get(c[0], {}).get("forgetting_mean", 0) for c in conditions]
    stds   = [agg.get(c[0], {}).get("forgetting_std", 0)  for c in conditions]
    colors = [c[2] for c in conditions]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=6,
                  width=0.55, edgecolor="none", error_kw={"linewidth": 1.2})
    for bar, val, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.004,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Forgetting ↓")
    ax.set_title("Component ablation: mean forgetting ± std (5 seeds)\n"
                 "Full TCL vs. isolated governor and penalty components", fontsize=9)
    ax.set_ylim(0, max(means) + max(stds) + 0.05)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Written: {out}")
    return out


# ── Figure 3: Baseline comparison grouped bars ───────────────────────────────

def generate_baseline_comparison(p10: dict) -> Path:
    out = PAPER_FIGURES / "baseline_comparison.pdf"
    agg = p10.get("aggregate", {})

    # Primary + best-tuned baselines (best-tuned from phase12/13 data hardcoded)
    ENTRIES = [
        ("tcl",          "TCL\n(no tuning)", "#2a9d8f",
         agg.get("tcl", {}).get("forgetting_mean", 0.1161),
         agg.get("tcl", {}).get("forgetting_std",  0.029)),
        ("ewc_default",  "EWC\n(λ=100)",     "#e76f51",
         agg.get("ewc", {}).get("forgetting_mean", 0.1909),
         agg.get("ewc", {}).get("forgetting_std",  0.023)),
        ("ewc_tuned",    "EWC\n(λ=1000)†",   "#f4a261", 0.1602, 0.089),
        ("si_default",   "SI\n(c=0.1) [coll]","#457b9d", 0.0920, 0.0007),
        ("si_tuned",     "SI\n(c=0.01)†",     "#264653", 0.0488, 0.009),
        ("sgd",          "SGD",               "#6d6875",
         agg.get("sgd_baseline", {}).get("forgetting_mean", 0.2331),
         agg.get("sgd_baseline", {}).get("forgetting_std",  0.003)),
    ]

    labels = [e[1] for e in ENTRIES]
    means  = [e[3] for e in ENTRIES]
    stds   = [e[4] for e in ENTRIES]
    colors = [e[2] for e in ENTRIES]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5,
                  width=0.60, edgecolor="none", error_kw={"linewidth": 1.2})
    # Highlight TCL with white edge
    bars[0].set_edgecolor("#ffffff")
    bars[0].set_linewidth(1.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Forgetting ↓")
    ax.set_title("Primary baseline comparison: mean forgetting ± std\n"
                 "†~swept result (see Appendices); [coll] = accuracy collapsed to chance", fontsize=9)
    ax.set_ylim(0, 0.30)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Written: {out}")
    return out


if __name__ == "__main__":
    p10 = json.loads(P10_PATH.read_text(encoding="utf-8"))
    p11 = json.loads(P11_PATH.read_text(encoding="utf-8"))

    generate_jaf_scatter(p10)
    generate_ablation_bars(p11)
    generate_baseline_comparison(p10)
    print("[figures] All done.")
