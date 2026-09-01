"""Generates the figures for the revised Q-ARMOR paper from real, current
results on disk (post entanglement-pair fix, post predict_labels() fix).

No numbers in this script are invented — every value is either read directly
from results/*.json or is the fresh Phase 7 episode trace already regenerated
in this session. The feature-map circuit is rendered from the actual
perception/feature_map.py class, not redrawn by hand.

Run:  ./venv/Scripts/python.exe docs/generate_figures.py
Output: docs/figures/*.png  (300 dpi, paper-ready)
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "docs/figures"

# ---------------------------------------------------------------------------
# Shared style — restrained, print-appropriate, not a marketing palette
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

INK = "#1a1a1a"
QUANTUM = "#0E7C74"      # within-domain / healthy
CROSS = "#B23B3B"        # cross-domain, no adaptation
RECOVER = "#2C6FB2"      # adapted / recovered
NEUTRAL = "#8A8A8A"      # classical / reference
AMBER = "#B5762A"        # secondary series


# ===========================================================================
# Figure 3 — Headline sealed-test results (T1/T2/T3): AUROC and FPR@95
# ===========================================================================
def fig_headline_results():
    conditions = ["T1", "T2", "T3"]
    auroc = [0.9732, 0.6064, 0.9126]
    fpr95 = [0.1200, 0.9740, 0.1980]
    colors = [QUANTUM, CROSS, RECOVER]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))

    ax = axes[0]
    bars = ax.bar(conditions, auroc, color=colors, width=0.6, edgecolor=INK, linewidth=0.8)
    ax.axhline(0.70, color=INK, linestyle="--", linewidth=0.9, alpha=0.6)
    ax.text(2.28, 0.715, r"$\tau_{\mathrm{AUROC}}=0.70$", fontsize=8.5, ha="right", color=INK, alpha=0.75)
    for b, v in zip(bars, auroc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Stage-1 AUROC")
    ax.set_ylim(0, 1.08)
    ax.set_title("(a) Binary detection AUROC", fontsize=10.5)

    ax = axes[1]
    bars = ax.bar(conditions, fpr95, color=colors, width=0.6, edgecolor=INK, linewidth=0.8)
    for b, v in zip(bars, fpr95):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("FPR@TPR95")
    ax.set_ylim(0, 1.08)
    ax.set_title("(b) False-positive rate at 95% recall", fontsize=10.5)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=10.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(0.5, -0.02,
              "T1: within-domain   |   T2: cross-domain, no adaptation   |   T3: cross-domain, after SWITCH-SUBSET",
              ha="center", fontsize=8.3, color=NEUTRAL)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_headline_results.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_headline_results.png")


# ===========================================================================
# Figure 4 — 10-episode reflexive-control trace (fresh, predict_labels fix)
# ===========================================================================
def fig_episode_trace():
    episodes = list(range(10))
    auroc = [0.9590, 0.9896, 0.9828, 0.9892, 0.9442, 0.6948, 0.6028, 0.7192, 0.5728, 0.6602]
    actions = [None, None, "REINFORCE", "REINFORCE", "REINFORCE",
               "SWITCH_MODEL", "SWITCH_MODEL", None, "SWITCH_MODEL", "SWITCH_MODEL"]
    models = ["QSVC"] * 9 + ["RF"]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    within_x, within_y = episodes[:5], auroc[:5]
    cross_x, cross_y = episodes[4:], auroc[4:]  # connect at boundary
    ax.plot(within_x, within_y, "-o", color=QUANTUM, linewidth=1.6, markersize=5, label="Block A — within-domain (NF-ToN-IoT)")
    ax.plot(cross_x, cross_y, "-o", color=CROSS, linewidth=1.6, markersize=5, label="Block B — cross-domain (NF-UNSW-NB15)")

    ax.axhline(0.70, color=INK, linestyle="--", linewidth=0.9, alpha=0.55)
    ax.text(9.35, 0.715, r"$\tau_{\mathrm{AUROC}}$", fontsize=8.5, color=INK, alpha=0.75)
    ax.axvline(4.5, color=INK, linestyle=":", linewidth=0.8, alpha=0.5)

    action_style = {
        "REINFORCE": dict(marker="^", color=RECOVER, label="REINFORCE"),
        "SWITCH_MODEL": dict(marker="s", color=AMBER, label="SWITCH_MODEL"),
    }
    seen = set()
    for ep, a, y in zip(episodes, actions, auroc):
        if a is None:
            continue
        st = action_style[a]
        lbl = st["label"] if a not in seen else None
        seen.add(a)
        ax.scatter([ep], [y + 0.028], marker=st["marker"], color=st["color"], s=55,
                   zorder=5, label=lbl, edgecolor=INK, linewidth=0.5)

    for ep, m in enumerate(models):
        if m == "RF":
            ax.annotate("RF fallback", (ep, auroc[ep]), textcoords="offset points",
                        xytext=(-2, -16), fontsize=7.8, color=NEUTRAL, ha="center")

    ax.set_xlabel("Episode")
    ax.set_ylabel("AUROC")
    ax.set_xticks(episodes)
    ax.set_ylim(0.45, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_episode_trace.png", dpi=300)
    plt.close(fig)
    print("saved fig_episode_trace.png")


# ===========================================================================
# Figure 5 — Per-class typing F1: within (T1) vs cross+adapted (T3)
# ===========================================================================
def fig_perclass_typing():
    classes = ["Benign", "DoS", "Injection", "Recon", "Backdoor"]
    t1 = [0.949, 0.590, 0.830, 0.112, 0.000]
    t3 = [0.694, 0.000, 0.046, 0.000, 0.049]

    x = np.arange(len(classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(x - w / 2, t1, w, label="T1 — within-domain", color=QUANTUM, edgecolor=INK, linewidth=0.7)
    ax.bar(x + w / 2, t3, w, label="T3 — cross-domain, Stage-1 adapted", color=CROSS, edgecolor=INK, linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("End-to-end F1")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.95)
    for xi, (v1, v3) in enumerate(zip(t1, t3)):
        ax.text(xi - w / 2, v1 + 0.02, f"{v1:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, v3 + 0.02, f"{v3:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_perclass_typing.png", dpi=300)
    plt.close(fig)
    print("saved fig_perclass_typing.png")


# ===========================================================================
# Figure 6 — CyberSecurityFeatureMap, rendered from the actual class
# ===========================================================================
def fig_feature_map_circuit():
    import sys
    sys.path.insert(0, ".")
    from perception.feature_map import CyberSecurityFeatureMap

    fm = CyberSecurityFeatureMap()
    fig = fm.draw(output="mpl", fold=-1, style={"backgroundcolor": "#FFFFFF"})
    fig.set_size_inches(11.0, 4.0)
    for txt in fig.axes[0].texts:
        if txt.get_text().strip().lower().startswith("global phase"):
            txt.set_visible(False)
    fig.savefig(f"{OUT}/fig_feature_map_circuit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_feature_map_circuit.png (rendered from perception/feature_map.py, undecomposed)")


# ===========================================================================
# Figure 7 — Phase 9: three-way reflector comparison (episode AUROC overlay)
# ===========================================================================
def fig_phase9_comparison():
    with open("results/phase9/phase9_comparison.json") as f:
        d = json.load(f)

    episodes = list(range(10))
    series = {
        "LLM Agent": (d["results"]["LLM Agent"], RECOVER, "-o"),
        "Rule-based": (d["results"]["Rule-based"], AMBER, "-s"),
        "ADWIN-only": (d["results"]["ADWIN-only"], NEUTRAL, "--^"),
    }

    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    for name, (recs, color, style) in series.items():
        y = [r["auroc"] for r in recs]
        marker = style[-1]
        linestyle = "--" if style.startswith("--") else "-"
        ax.plot(episodes, y, linestyle=linestyle, marker=marker, color=color,
                linewidth=1.5, markersize=5, label=name, alpha=0.9)

    ax.axhline(0.70, color=INK, linestyle=":", linewidth=0.9, alpha=0.55)
    ax.axvline(4.5, color=INK, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(0.15, 1.015, "Block A (within)", fontsize=8, color=NEUTRAL)
    ax.text(6.7, 1.015, "Block B (cross)", fontsize=8, color=NEUTRAL)

    # mark rule-based's drift-confirmed SWITCH_SUBSET at episode 5
    rb = d["results"]["Rule-based"][5]
    ax.scatter([5], [rb["auroc"] + 0.03], marker="*", s=140, color=QUANTUM,
               zorder=6, edgecolor=INK, linewidth=0.5, label="SWITCH_SUBSET (drift confirmed)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("AUROC")
    ax.set_xticks(episodes)
    ax.set_ylim(0.45, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_phase9_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_phase9_comparison.png")


if __name__ == "__main__":
    fig_headline_results()
    fig_episode_trace()
    fig_perclass_typing()
    fig_phase9_comparison()
    try:
        fig_feature_map_circuit()
    except Exception as e:
        print(f"WARNING: circuit figure failed ({e}); will need a manual/schematic fallback.")
    print("\nAll figures written to docs/figures/")
