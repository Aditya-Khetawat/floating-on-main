"""Step 6: draw the data figures of the paper (Figs. 3 and 4) from output/summary.json.

Figures 1, 2 and 5 are TikZ diagrams inside the LaTeX source of the paper.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import out_path

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.size": 7.5})
ORDER = ["absent", "indirect", "named_ref", "hash"]
LABELS = ["None", "Configurable", "Named ref", "Commit hash"]
COLORS = ["#c44e52", "#dd8452", "#ccb974", "#55a868"]


def fig_status(S):
    groups = [("All call sites", S["call_revision"]),
              ("Hub-literal sites", S["hub_literal_sites"]["revision"]),
              ("Remote-code sites", S["remote_code"]["revision"])]
    data = np.array([[g[1].get(k, 0) for k in ORDER] for g in groups], float)
    pct = data / data.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(3.45, 1.35))
    left = np.zeros(len(groups))
    for j in range(4):
        ax.barh(range(3), pct[:, j], left=left, color=COLORS[j], label=LABELS[j], height=0.6,
                edgecolor="white", linewidth=0.4)
        for i in range(3):
            if pct[i, j] >= 5:
                ax.text(left[i] + pct[i, j] / 2, i, f"{pct[i, j]:.1f}", ha="center", va="center",
                        fontsize=6.3, color="white")
        left += pct[:, j]
    for i in range(3):
        ax.text(101, i, f"{100 - pct[i, 0]:.1f}% pinned", va="center", fontsize=6.2)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{g[0]}\n(n={int(d.sum()):,})" for g, d in zip(groups, data)], fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 134)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of call sites (%)", fontsize=6.8)
    ax.legend(ncol=4, fontsize=5.9, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0),
              handlelength=1, columnspacing=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.3)
    plt.tight_layout(pad=0.2)
    plt.savefig(out_path("fig_status.pdf"))
    plt.savefig(out_path("fig_status.png"), dpi=600)
    plt.close(fig)


def fig_pin(S):
    api = S["by_interface"]
    order = sorted(api, key=lambda a: api[a]["repo_pin_pct"])
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.75), gridspec_kw={"width_ratios": [1.55, 1]})
    y = range(len(order))
    ax[0].barh(y, [api[a]["repo_pin_pct"] for a in order], color="#4c72b0", height=0.62)
    for i, a in enumerate(order):
        ax[0].text(api[a]["repo_pin_pct"] + 0.8, i, f"{api[a]['repo_pin_pct']:.1f}", va="center", fontsize=6.3)
    ax[0].set_yticks(list(y))
    ax[0].set_yticklabels([o.replace("HuggingFaceEmbeddings", "HFEmbeddings") for o in order],
                          fontfamily="monospace", fontsize=6.2)
    ax[0].set_xlim(0, 42)
    ax[0].set_xlabel("Repos pinning (%)", fontsize=7)
    ax[0].set_title("(a) By interface", fontsize=7.5)
    r = np.array(S["maturity"]["pinning"]) / np.array(S["maturity"]["n"]) * 100
    ax[1].bar(range(5), r, color="#55a868", width=0.65)
    for i in range(5):
        ax[1].text(i, r[i] + 0.7, f"{r[i]:.0f}", ha="center", fontsize=6.3)
    ax[1].set_xticks(range(5))
    ax[1].set_xlabel("Maturity score $M$", fontsize=7)
    ax[1].set_ylim(0, 27)
    ax[1].set_title("(b) By maturity", fontsize=7.5)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=6.3)
    plt.tight_layout(pad=0.3, w_pad=0.6)
    plt.savefig(out_path("fig_pin.pdf"))
    plt.savefig(out_path("fig_pin.png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    summary = json.load(open(out_path("summary.json")))
    fig_status(summary)
    fig_pin(summary)
    print("wrote output/fig_status.{pdf,png} and output/fig_pin.{pdf,png}")
