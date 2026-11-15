"""Static figures for the README (matplotlib, PNG)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import load_json, rpath  # noqa: E402
from src.run import METHODS  # noqa: E402

BLUE, ACCENT = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})


def trace_fig():
    tr = load_json(rpath("causal_trace.json"))
    y = np.array(tr["avg_norm_mlp_window_ie"])
    fig, ax = plt.subplots(figsize=(7, 3), dpi=160)
    ax.plot(range(len(y)), y, color=BLUE, lw=2, marker="o", ms=4)
    l = tr["chosen_layer"]
    ax.plot([l], [y[l]], "o", ms=9, color=ACCENT, mec=SURF, mew=2)
    ax.annotate(f"edit layer {l}", (l, y[l]), xytext=(10, 0), textcoords="offset points", color=INK, va="center")
    ax.set_xlabel("decoder layer (centre of 5-layer window)")
    ax.set_ylabel("recovered fraction\nof P(answer)")
    ax.set_title("Restoring clean MLP outputs at the subject's last token", loc="left", color=INK, fontsize=11, pad=12)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_xticks(range(len(y)))
    fig.tight_layout()
    fig.savefig(rpath("fig_causal_trace.png"))


def methods_fig():
    s = load_json(rpath("summary.json"))["methods"]
    ms = [m for m in METHODS if m in s]
    panels = [("rewrite", "rewrite prompt ok", True), ("para", "paraphrases ok", True),
              ("neigh", "neighbourhood kept", True), ("kl", "mean KL on unrelated prompts", False)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), dpi=160, sharey=True)
    for ax, (k, title, frac) in zip(axes, panels):
        v = [s[m][k] for m in ms]
        cols = [ACCENT if m == "rome" else BLUE for m in ms]
        ax.barh(range(len(ms)), v, color=cols, height=0.6)
        for i, x in enumerate(v):
            ax.text(x, i, f" {x:.0%}" if frac else f" {x:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlim(0, 1.18 if frac else max(v) * 1.35 + 1e-9)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        if frac:
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_yticks(range(len(ms)))
    axes[0].set_yticklabels(ms)
    axes[0].invert_yaxis()
    fig.suptitle("Mean over 4 edits (ROME highlighted)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(rpath("fig_methods.png"))


if __name__ == "__main__":
    trace_fig()
    methods_fig()
