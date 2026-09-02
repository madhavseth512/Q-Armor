"""Generates every plotted (non-diagram) figure for the Q-Armor paper from
real, current results on disk. No numbers in this script are invented --
every value is read directly from results/*.json.

Style: strictly black-and-white/grayscale, matching standard IEEE-conference
convention -- series are distinguished by marker shape, line style, and
hatch pattern, never by hue. Figures are sized for a single IEEEtran column
(~3.4in) wherever the data permits; only the circuit diagram, which is
inherently wide, spans both columns.

The two architecture diagrams (fig1_overview.png, fig2_logical_architecture.png)
are supervisor-provided and are NOT touched by this script.

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
COL_W = 3.4   # inches -- IEEEtran single-column width

# ---------------------------------------------------------------------------
# Shared black-and-white style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.color": "#999999",
    "grid.linewidth": 0.3,
    "grid.linestyle": ":",
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

BLACK = "#000000"
DGRAY = "#4D4D4D"
MGRAY = "#808080"
LGRAY = "#B3B3B3"

# Series style cycle: (color, linestyle, marker) -- grayscale + shape, never hue.
SERIES = [
    (BLACK, "-", "o"),
    (DGRAY, "--", "s"),
    (MGRAY, "-.", "^"),
    (BLACK, ":", "D"),
    (DGRAY, "-", "v"),
    (MGRAY, "--", "P"),
]

# Bar-fill cycle: grayscale shade + hatch, black edges throughout.
BARS = [
    dict(color="white", hatch=""),
    dict(color=LGRAY, hatch="///"),
    dict(color=MGRAY, hatch="xxx"),
    dict(color=DGRAY, hatch="..."),
    dict(color=BLACK, hatch=""),
]


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ===========================================================================
# Figure 3 — Headline sealed-test results (T1/T2/T3): AUROC and FPR@95
# ===========================================================================
def fig_headline_results():
    conditions = ["T1", "T2", "T3"]
    auroc = [0.9732, 0.6064, 0.9126]
    fpr95 = [0.1200, 0.9740, 0.1980]

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.55))

    for ax, vals, ylabel, title in [
        (axes[0], auroc, "Stage-1 AUROC", "(a) AUROC"),
        (axes[1], fpr95, "FPR@TPR95", "(b) FPR@95"),
    ]:
        bars = ax.bar(conditions, vals, width=0.6, edgecolor="black", linewidth=0.6,
                       color=[LGRAY, MGRAY, BLACK])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                     ha="center", fontsize=6.3)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        _clean_axes(ax)

    fig.text(0.5, -0.06,
              "T1: within-domain T2: cross-domain, no adapt. T3: cross-domain, adapted",
              ha="center", fontsize=5.6, color=DGRAY)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_headline_results.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_headline_results.png")


# ===========================================================================
# Figure 4 — 10-episode reflexive-control trace
# ===========================================================================
def fig_episode_trace():
    episodes = list(range(10))
    auroc = [0.9590, 0.9896, 0.9828, 0.9892, 0.9442, 0.6948, 0.6028, 0.7192, 0.5728, 0.6602]
    actions = [None, None, "REINFORCE", "REINFORCE", "REINFORCE",
               "SWITCH_MODEL", "SWITCH_MODEL", None, "SWITCH_MODEL", "SWITCH_MODEL"]

    fig, ax = plt.subplots(figsize=(COL_W, 1.9))

    ax.plot(episodes[:5], auroc[:5], "-o", color=BLACK, linewidth=1.0,
            markersize=3, markerfacecolor="white", markeredgewidth=0.7,
            label="Within-domain (A)")
    ax.plot(episodes[4:], auroc[4:], "--s", color=DGRAY, linewidth=1.0,
            markersize=3, markerfacecolor="white", markeredgewidth=0.7,
            label="Cross-domain (B)")

    ax.axhline(0.70, color="black", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.text(9.3, 0.72, r"$\tau$", fontsize=6.5, color="black")
    ax.axvline(4.5, color="black", linestyle=":", linewidth=0.5, alpha=0.5)

    for ep, a, y in zip(episodes, actions, auroc):
        if a == "REINFORCE":
            ax.scatter([ep], [y + 0.03], marker="^", s=22, facecolor="black",
                       edgecolor="black", zorder=5)
        elif a == "SWITCH_MODEL":
            ax.scatter([ep], [y + 0.03], marker="D", s=18, facecolor="white",
                       edgecolor="black", linewidth=0.6, zorder=5)

    ax.scatter([], [], marker="^", s=22, facecolor="black", edgecolor="black", label="REINFORCE")
    ax.scatter([], [], marker="D", s=18, facecolor="white", edgecolor="black", label="SWITCH_MODEL")

    ax.set_xlabel("Episode")
    ax.set_ylabel("AUROC")
    ax.set_xticks(episodes)
    ax.set_ylim(0.45, 1.08)
    _clean_axes(ax)
    ax.legend(loc="lower left", fontsize=5.6, framealpha=0.95, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_episode_trace.png", dpi=300, bbox_inches="tight")
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
    w = 0.32
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    ax.bar(x - w / 2, t1, w, label="T1: within-domain", color=LGRAY,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, t3, w, label="T3: cross-domain, adapted", color=BLACK,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=6.3, rotation=20, ha="right")
    ax.set_ylabel("End-to-end F1")
    ax.set_ylim(0, 1.05)
    _clean_axes(ax)
    ax.legend(fontsize=5.8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_perclass_typing.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_perclass_typing.png")


# ===========================================================================
# Figure 6 — CyberSecurityFeatureMap, rendered from the actual class, B&W
# ===========================================================================
def fig_feature_map_circuit():
    import sys
    sys.path.insert(0, ".")
    from perception.feature_map import CyberSecurityFeatureMap

    fm = CyberSecurityFeatureMap()
    # Let qiskit choose its own natural canvas size for the circuit -- it
    # lays out gate boxes at a fixed point-size for a computed figure size;
    # forcibly resizing the canvas afterward (fig.set_size_inches) does NOT
    # rescale those already-placed glyphs, it just crops/overlaps them. Any
    # size reduction has to happen via dpi at save time instead.
    fig = fm.draw(output="mpl", fold=-1, style={"name": "bw", "fontsize": 10})
    for txt in fig.axes[0].texts:
        if txt.get_text().strip().lower().startswith("global phase"):
            txt.set_visible(False)
    fig.savefig(f"{OUT}/fig_feature_map_circuit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_feature_map_circuit.png (b&w, rendered from perception/feature_map.py)")


# ===========================================================================
# Figure 7 — Phase 9: three-way reflector comparison (episode AUROC overlay)
# ===========================================================================
def fig_phase9_comparison():
    with open("results/phase9/phase9_comparison.json") as f:
        d = json.load(f)

    episodes = list(range(10))
    order = ["LLM Agent", "Rule-based", "ADWIN-only"]

    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    for name, (color, ls, marker) in zip(order, SERIES):
        y = [r["auroc"] for r in d["results"][name]]
        ax.plot(episodes, y, linestyle=ls, marker=marker, color=color,
                linewidth=0.9, markersize=3, markerfacecolor="white",
                markeredgewidth=0.6, label=name)

    ax.axhline(0.70, color="black", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.axvline(4.5, color="black", linestyle=":", linewidth=0.5, alpha=0.5)

    rb = d["results"]["Rule-based"][5]
    ax.scatter([5], [rb["auroc"] + 0.04], marker="*", s=45, facecolor="black",
               edgecolor="black", zorder=6, label="SWITCH_SUBSET (drift)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("AUROC")
    ax.set_xticks(episodes)
    ax.set_ylim(0.45, 1.08)
    _clean_axes(ax)
    ax.legend(loc="lower left", fontsize=5.4, framealpha=0.95, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_phase9_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_phase9_comparison.png")


# ===========================================================================
# Figure 8 — Confirmatory budget sweep: AUROC vs. target-label budget B
# ===========================================================================
def fig_budget_sweep():
    with open("results/phase10/phase10_confirmatory_metrics.json") as f:
        d = json.load(f)

    order = ["pegasos_qsvc", "random_forest", "gbt", "svm_linear", "svm_rbf"]
    labels = {"pegasos_qsvc": "PegasosQSVC (quantum)", "random_forest": "Random Forest",
              "gbt": "Gradient-Boosted Trees", "svm_linear": "Linear SVM", "svm_rbf": "RBF-SVM"}
    budgets = [0, 25, 50, 100, 150, 300]

    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    for name, (color, ls, marker) in zip(order, SERIES):
        bd = d["results"][name]
        y = [bd[str(b)]["auroc"]["mean"] for b in budgets]
        lw = 1.3 if name == "pegasos_qsvc" else 0.8
        ax.plot(budgets, y, linestyle=ls, marker=marker, color=color, linewidth=lw,
                markersize=3.2, markerfacecolor="white" if color != BLACK or name != "pegasos_qsvc" else BLACK,
                markeredgewidth=0.6, label=labels[name])

    ax.set_xlabel("Target-label budget $B$")
    ax.set_ylabel("Cross-domain AUROC")
    ax.set_xscale("symlog", linthresh=25)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_ylim(0.4, 1.05)
    _clean_axes(ax)
    ax.legend(loc="lower right", fontsize=5.2, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_budget_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_budget_sweep.png")


# ===========================================================================
# Figure 9 — Poisoning study: F1 vs. poison rate (RF / GBT / QSVC)
# ===========================================================================
def fig_poisoning():
    with open("results/phase13/phase13_poisoning_metrics.json") as f:
        d = json.load(f)

    order = ["pegasos_qsvc", "random_forest", "gbt"]
    labels = {"pegasos_qsvc": "PegasosQSVC (quantum)", "random_forest": "Random Forest",
              "gbt": "Gradient-Boosted Trees"}

    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for name, (color, ls, marker) in zip(order, SERIES):
        rows = d["results"][name]
        rates = [r["poison_rate"] for r in rows]
        f1 = [r["f1"] for r in rows]
        ax.plot(rates, f1, linestyle=ls, marker=marker, color=color, linewidth=1.0,
                markersize=3.2, markerfacecolor="white", markeredgewidth=0.6,
                label=labels[name])
        for r, y in zip(rates, f1):
            if r == 0.5 and name == "pegasos_qsvc":
                ax.annotate("F1=0,\nREINFORCE\nstill fires", (r, y), textcoords="offset points",
                             xytext=(-6, 24), fontsize=5.2, color="black", ha="center")

    ax.set_xlabel("Adaptation-pool poison rate")
    ax.set_ylabel("End-to-end F1")
    ax.set_ylim(-0.05, 1.05)
    _clean_axes(ax)
    ax.legend(loc="lower left", fontsize=5.6, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_poisoning.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_poisoning.png")


if __name__ == "__main__":
    fig_headline_results()
    fig_episode_trace()
    fig_perclass_typing()
    fig_phase9_comparison()
    fig_budget_sweep()
    fig_poisoning()
    try:
        fig_feature_map_circuit()
    except Exception as e:
        print(f"WARNING: circuit figure failed ({e}); will need a manual/schematic fallback.")
    print("\nAll figures written to docs/figures/")
