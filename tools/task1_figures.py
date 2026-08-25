"""Build the Task 1 figures from the tables written by task1_feature_importance_cv.ipynb.

Reads ``data/task1_gene_importance.csv``, ``data/task1_fold_importance_long.csv``,
``data/task1_accuracy_table.csv``, ``data/task1_cv_results.json`` and
``data/task1_hvg_leakage_fold0.json``; writes 300-dpi PNGs into ``figures/``.

Colour is bound once here and reused across every figure: one hue per model
family (gradient boosting / neural network) and one hue per treatment condition.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = Path(tempfile.gettempdir()) / "sc-course-2026-cache"
for sub in ("matplotlib", "numba"):
    (CACHE / sub).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

sys.path.insert(0, str(REPO / "src"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from figstyle import META_GREY, apply_figure_style, panel_letter, set_frame  # noqa: E402

DATA = REPO / "data"
FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)

# One hue per model family, reused in every figure (figure-style §4.1).
COL = {
    "xgb_hi": "#1B6CA8",   # gradient boosting, 400 rounds
    "xgb6": "#8EC0DE",     # gradient boosting, 6 rounds (same family, lighter)
    "mlp": "#C2571A",      # neural network
}
MODEL_LABEL = {
    "xgb6": "gradient boosting, 6 rounds",
    "xgb_hi": "gradient boosting, 400 rounds",
    "mlp": "neural network (1011-128-64-3)",
}
# One hue per treatment condition.
COND = {"Co-culture": "#4C72A8", "Control": "#8A8F98", "IFNγ": "#B5651D"}

TOP_N = 25


def gene_label(g: str, marker: bool) -> str:
    """Gene symbol, with a bullet if it is a marker from the paper.

    Italics are applied through ``fontstyle="italic"`` rather than mathtext, so
    that hyphenated symbols such as HLA-B keep their hyphen instead of having it
    re-rendered as a minus sign.
    """
    return f"{g}  •" if marker else g


def load():
    imp = pd.read_csv(DATA / "task1_gene_importance.csv")
    long = pd.read_csv(DATA / "task1_fold_importance_long.csv")
    acc = pd.read_csv(DATA / "task1_accuracy_table.csv")
    res = json.loads((DATA / "task1_cv_results.json").read_text())
    leak = json.loads((DATA / "task1_hvg_leakage_fold0.json").read_text())
    return imp, long, acc, res, leak


# --------------------------------------------------------------------------- #
# Figure 1 -- per-gene importance with across-fold error bars
# --------------------------------------------------------------------------- #


def fig_importance(imp, long, res):
    models = ["xgb_hi", "mlp"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.4))

    for ax, m, letter in zip(axes, models, "ab"):
        sub = imp[imp.model == m].nlargest(TOP_N, "mean_importance").iloc[::-1]
        pos = np.arange(len(sub))
        stable = sub["n_folds_in_top20"].to_numpy() == 5

        # thin stem to zero -- importance is a non-negative magnitude
        ax.hlines(pos, 0, sub["mean_importance"], color=META_GREY, lw=0.5, zorder=1)
        ax.errorbar(
            sub["mean_importance"], pos, xerr=sub["sd_importance"],
            fmt="none", ecolor=COL[m], elinewidth=0.9, capsize=1.8, capthick=0.9, zorder=2,
        )
        ax.scatter(
            sub["mean_importance"][stable], pos[stable], s=17, color=COL[m],
            zorder=3, label="top 20 in all 5 folds",
        )
        ax.scatter(
            sub["mean_importance"][~stable], pos[~stable], s=17, facecolors="white",
            edgecolors=COL[m], linewidths=0.9, zorder=3, label="top 20 in fewer folds",
        )

        ax.set_yticks(pos)
        ax.set_yticklabels(
            [gene_label(g, mk) for g, mk in zip(sub["gene"], sub["is_paper_marker"])],
            fontstyle="italic",
        )
        ax.set_xlabel("mean |SHAP| across folds" if m.startswith("xgb")
                      else "mean |gradient × input| across folds")
        ax.set_title(MODEL_LABEL[m], pad=6)
        ax.set_xlim(left=0)
        ax.margins(y=0.02)
        set_frame(ax, "open")
        panel_letter(ax, letter, dx=-0.42, dy=1.03)

    handles = [
        Line2D([], [], marker="o", ls="", ms=4.2, color=META_GREY, label="top 20 in all 5 folds"),
        Line2D([], [], marker="o", ls="", ms=4.2, mfc="white", mec=META_GREY,
               label="top 20 in fewer folds"),
        Line2D([], [], marker="$•$", ls="", ms=4.5, color="black",
               label="marker gene from Frangieh et al."),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))
    shared10 = [g for g in imp[imp.model == "xgb_hi"].nlargest(10, "mean_importance")["gene"]
                if g in set(imp[imp.model == "mlp"].nlargest(10, "mean_importance")["gene"])]
    fig.suptitle(
        f"{len(shared10)} genes rank in the top 10 of both model types: "
        + ", ".join(shared10),
        x=0.015, ha="left", y=0.995, fontsize=mpl.rcParams["font.size"],
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.965), w_pad=3.0)
    out = FIG / "task1_fig1_importance_stability.png"
    fig.savefig(out)
    return fig, out


# --------------------------------------------------------------------------- #
# Figure 2 -- rank concordance between the two required model types
# --------------------------------------------------------------------------- #


def fig_concordance(imp, res):
    conc = {c["comparison"]: c for c in res["concordance"]}
    wide = imp.pivot(index="gene", columns="model", values="mean_rank")
    marker = imp.drop_duplicates("gene").set_index("gene")["is_paper_marker"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # (a) xgb_hi vs mlp, all 1,011 genes
    ax = axes[0]
    x, y = wide["xgb_hi"], wide["mlp"]
    ax.plot([1, 1011], [1, 1011], color=META_GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.scatter(x, y, s=5, color=META_GREY, alpha=0.35, lw=0, zorder=2)
    shared = res["shared_top_genes_xgbhi_mlp"]
    ax.scatter(x[shared], y[shared], s=16, color=COL["xgb_hi"], zorder=3, lw=0)

    # The shared genes sit in a dense diagonal band, so leader lines would cross
    # the cloud.  They are named in a block in the empty upper-left instead.
    ordered = sorted(shared, key=lambda g: x[g])
    ax.text(0.03, 0.965,
            f"the {len(ordered)} genes in the top 20\nof both models:",
            transform=ax.transAxes, va="top", ha="left", fontsize=6, color=COL["xgb_hi"])
    for row, g in enumerate(ordered):
        ax.text(0.03, 0.885 - 0.055 * row, gene_label(g, bool(marker.get(g, False))),
                transform=ax.transAxes, va="top", ha="left", fontsize=6,
                fontstyle="italic", color=COL["xgb_hi"])

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.1, 1400); ax.set_ylim(1.1, 1400)
    ax.set_xlabel("mean rank, gradient boosting (400 rounds)", labelpad=3)
    ax.set_ylabel("mean rank, neural network", labelpad=3)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda v, _: {1: "1", 10: "10", 100: "100", 1000: "1000"}.get(int(round(v)), "")))
    c = conc["xgb_hi vs mlp"]
    ax.set_title("Importance ranks of the two model types,\nall 1,011 genes", pad=6)
    ax.text(
        0.97, 0.235,
        f"Spearman ρ = {c['spearman_all_genes']:.2f}\nover all 1,011 genes",
        transform=ax.transAxes, va="top", ha="right", fontsize=6,
    )
    ax.text(0.97, 0.03, "rank 1 = most important", transform=ax.transAxes,
            ha="right", fontsize=6, color=META_GREY)
    set_frame(ax, "open")
    panel_letter(ax, "a", dx=-0.19)

    # (b) top-50 overlap across the three comparisons
    ax = axes[1]
    order = ["xgb_hi vs mlp", "xgb6 vs xgb_hi", "xgb6 vs mlp"]
    labels = ["400-round boosting\nvs neural net", "6-round\nvs 400-round boosting",
              "6-round boosting\nvs neural net"]
    vals = [conc[o]["overlap_top50"] for o in order]
    colors = [COL["mlp"], COL["xgb_hi"], COL["xgb6"]]
    pos = np.arange(len(order))
    ax.hlines(pos, 0, vals, color=META_GREY, lw=0.5)
    ax.scatter(vals, pos, s=34, color=colors, zorder=3)
    for p, v in zip(pos, vals):
        ax.annotate(f"{v}", (v, p), textcoords="offset points", xytext=(7, -2), fontsize=7)
    ax.set_yticks(pos); ax.set_yticklabels(labels)
    ax.set_xlim(0, 50)
    ax.set_xlabel("genes shared between the two top-50 sets (of 50)")
    ax.set_title("Top-50 overlap: capacity matters as much\nas model family", pad=6)
    ax.margins(y=0.25)
    set_frame(ax, "open")
    panel_letter(ax, "b", dx=-0.40)

    fig.tight_layout()
    out = FIG / "task1_fig2_rank_concordance.png"
    fig.savefig(out)
    return fig, out


# --------------------------------------------------------------------------- #
# Figure 3 -- confusion matrices
# --------------------------------------------------------------------------- #


def fig_confusion(res):
    classes = res["class_names"]
    models = ["xgb6", "xgb_hi", "mlp"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4))

    for ax, m, letter in zip(axes, models, "abc"):
        cm = np.array(res["confusion"][m], dtype=float)
        row_pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j, i, f"{row_pct[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if row_pct[i, j] > 55 else "black",
                )
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
        ax.set_yticks(range(len(classes)), classes if letter == "a" else [""] * len(classes))
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{MODEL_LABEL[m].split(' (')[0]}\n{acc * 100:.1f} % overall", pad=5,
                     fontsize=mpl.rcParams["font.size"] - 0.5)
        set_frame(ax, "none")

    # The headline is computed from the matrices themselves rather than asserted:
    # the share of all errors that fall on the Control/IFNγ pair, and whether the
    # single largest off-diagonal cell is the same in every model.
    ci, ii = classes.index("Control"), classes.index("IFNγ")
    shares, largest = [], set()
    for m in models:
        cm = np.array(res["confusion"][m], dtype=float)
        off = cm.copy()
        np.fill_diagonal(off, 0.0)
        shares.append(100 * (off[ci, ii] + off[ii, ci]) / off.sum())
        i, j = np.unravel_index(np.argmax(off), off.shape)
        largest.add((classes[i], classes[j]))
    assert len(largest) == 1, largest
    (true_lab, pred_lab), = largest
    # Wrapped explicitly: at this figure width an unwrapped suptitle runs off the
    # right edge, and the wrapped block needs enough headroom not to collide with
    # the panel letters.
    headline = textwrap.fill(
        f"True {true_lab} predicted as {pred_lab} is the largest single error cell "
        f"by cell count in all three models; the {classes[ci]}/{classes[ii]} pair "
        f"carries {min(shares):.0f}–{max(shares):.0f} % of all errors",
        width=95,
    )
    subtitle = textwrap.fill(
        "rows = true condition, columns = predicted condition, cells = row % "
        "pooled over the 5 folds; because the classes differ in size, the largest "
        "cell by count need not be the largest row % (panel a)",
        width=105,
    )
    fig.suptitle(headline, x=0.015, ha="left", y=0.995, va="top",
                 fontsize=mpl.rcParams["font.size"])
    fig.text(0.015, 0.995 - 0.055 * (headline.count("\n") + 1), subtitle,
             ha="left", va="top", fontsize=mpl.rcParams["font.size"] - 1.5,
             color=META_GREY)
    fig.tight_layout(rect=(0, 0.06, 1, 0.74))
    # Panel letters are placed in FIGURE coordinates after tight_layout, at the
    # left edge of each axes and above its two-line title, so they cannot collide
    # with the title text the way axes-relative placement did.
    for ax, letter in zip(axes, "abc"):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.035, bb.y1 + 0.105, letter, fontweight="bold",
                 fontsize=mpl.rcParams["font.size"] + 1, va="bottom", ha="left")
    out = FIG / "task1_fig3_confusion.png"
    fig.savefig(out)
    return fig, out


# --------------------------------------------------------------------------- #
# Figure 4 -- accuracy vs baselines, capacity, and the HVG-leakage check
# --------------------------------------------------------------------------- #


def fig_accuracy(acc, res, leak, imp):
    models = ["xgb6", "xgb_hi", "mlp"]
    labels = ["boosting\n6 rounds", "boosting\n400 rounds", "neural\nnetwork"]

    fig = plt.figure(figsize=(7.2, 4.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # (a) per-fold balanced accuracy against the chance floor
    ax = ax_a
    floor = res["baselines"]["chance_1_over_k"]
    ax.axhline(floor, color=META_GREY, lw=0.7, ls=(0, (3, 2)))
    ax.annotate(
        f"chance floor = {floor:.3f}  (balanced accuracy of always predicting the "
        f"largest class, and of stratified random guessing)",
        (0.015, floor), xycoords=("axes fraction", "data"), xytext=(0, 5),
        textcoords="offset points", fontsize=6, color=META_GREY,
    )
    for i, m in enumerate(models):
        per = res["accuracies"][m]["per_fold"]
        ax.scatter(np.full(len(per), i) + np.linspace(-0.045, 0.045, len(per)), per,
                   s=13, color=COL[m], zorder=3, lw=0)
        mu, sd = res["accuracies"][m]["mean"], res["accuracies"][m]["sd"]
        ax.hlines(mu, i - 0.14, i + 0.14, color=COL[m], lw=1.6, zorder=4)
        ax.annotate(f"{mu:.3f} ± {sd:.3f}", (i + 0.17, mu), fontsize=6.5,
                    va="center", ha="left", color=COL[m])
    ax.set_xticks(range(3), labels)
    ax.set_xlim(-0.45, 2.75)
    ax.set_ylim(0.28, 1.04)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("balanced accuracy\n(5 stratified folds)")
    ax.set_title("Every model separates the three conditions far above chance", pad=6)
    set_frame(ax, "open")
    panel_letter(ax, "a", dx=-0.085, dy=1.02)

    # (b) how many genes each model actually uses
    ax = ax_b
    used = [res["genes_with_nonzero_importance"][m] for m in models]
    pos = np.arange(3)
    ax.hlines(pos, 0, used, color=META_GREY, lw=0.5)
    ax.scatter(used, pos, s=30, color=[COL[m] for m in models], zorder=3)
    for pp, v in zip(pos, used):
        ax.annotate(f"{v}", (v, pp), textcoords="offset points", xytext=(6, -2), fontsize=6.5)
    ax.axvline(1011, color=META_GREY, lw=0.7, ls=(0, (3, 2)))
    ax.annotate("all 1,011\nfeatures", (990, -0.42), fontsize=6, color=META_GREY,
                ha="right", va="bottom")
    ax.set_yticks(pos, labels)
    ax.set_xlim(0, 1250)
    ax.set_xlabel("genes with non-zero importance")
    ax.set_title("Six boosting rounds leave most\ngenes unused", pad=6)
    ax.set_ylim(-0.6, 2.5)
    set_frame(ax, "open")
    panel_letter(ax, "b", dx=-0.30, dy=1.02)

    # (c) HVG-selection leakage, fold 0
    ax = ax_c
    pairs = [("boosting\n6 rounds", "xgb6", "cached_xgb6_balanced_accuracy",
              "fold_xgb6_balanced_accuracy"),
             ("neural\nnetwork", "mlp", "cached_mlp_balanced_accuracy",
              "fold_mlp_balanced_accuracy")]
    for i, (lab, m, k_cached, k_fold) in enumerate(pairs):
        ax.plot([i - 0.10, i + 0.10], [leak[k_cached], leak[k_fold]],
                color=COL[m], lw=0.7, zorder=2)
        ax.scatter([i - 0.10], [leak[k_cached]], s=28, color=COL[m], zorder=3)
        ax.scatter([i + 0.10], [leak[k_fold]], s=28, facecolors="white",
                   edgecolors=COL[m], linewidths=1.0, zorder=3)
        ax.annotate(f"{leak[k_cached]:.4f}", (i - 0.13, leak[k_cached]), fontsize=6,
                    ha="right", va="center", color=COL[m])
        ax.annotate(f"{leak[k_fold]:.4f}", (i + 0.13, leak[k_fold]), fontsize=6,
                    ha="left", va="center", color=COL[m])
    ax.set_xticks(range(2), [pr[0] for pr in pairs])
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylabel("balanced accuracy, fold 0")
    ax.set_title("Reselecting genes inside the fold\ndoes not lower accuracy", pad=6)
    ax.set_ylim(0.9715, 0.9895)
    handles = [
        Line2D([], [], marker="o", ls="", ms=4.4, color=META_GREY,
               label="selected on all cells"),
        Line2D([], [], marker="o", ls="", ms=4.4, mfc="white", mec=META_GREY,
               label="reselected within fold 0"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6, borderaxespad=0.1)
    set_frame(ax, "open")
    panel_letter(ax, "c", dx=-0.30, dy=1.02)

    out = FIG / "task1_fig4_accuracy_capacity.png"
    fig.savefig(out)
    return fig, out


def bbox_check(fig, name):
    """figure-style §9.1: no visible text may overlap other text or a foreign spine.

    ``savefig(bbox_inches="tight")`` renders at a different canvas geometry than the
    figure's own, and ``constrained_layout`` only settles during a draw -- so force a
    plain draw first, otherwise every extent read here belongs to the wrong pass.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes
              for s in ax.spines.values() if s.get_visible()]
    ticks = {ax: set(ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"))
             for ax in fig.axes}
    bad = [(a.get_text(), b.get_text())
           for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    bad += [(t.get_text(), f"spine:{s.axes.get_title()[:18]}") for t, bt in texts
            for s, bs in spines if bt.overlaps(bs) and t not in ticks[s.axes]]
    outside = [t.get_text() for t, bt in texts if not fig.bbox.overlaps(bt)]
    print(f"  {name}: {len(bad)} overlaps, {len(outside)} outside")
    for pair in bad[:12]:
        print("    overlap:", pair)
    return bad, outside


def main():
    apply_figure_style(frame="open", sizes=(8, 7, 6))
    imp, long, acc, res, leak = load()
    for builder, args in (
        (fig_importance, (imp, long, res)),
        (fig_concordance, (imp, res)),
        (fig_confusion, (res,)),
        (fig_accuracy, (acc, res, leak, imp)),
    ):
        fig, out = builder(*args)
        bbox_check(fig, out.name)
        plt.close(fig)
        print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    main()
