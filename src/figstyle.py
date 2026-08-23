"""Matplotlib mechanics for the figures shipped with this project.

Call ``apply_figure_style()`` once before any plotting.
"""

from __future__ import annotations

import warnings

import matplotlib as mpl

#: Neutral grey for reference marks, gridlines and de-emphasised series.
META_GREY = "#8A8F98"


def apply_figure_style(*, frame: str = "open", sizes=(8, 7, 6), grid: bool = False,
                       font: str | None = None) -> None:
    """Publication-grade rcParams.  ``frame`` is 'open' | 'boxed' | 'none'.
    """
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    base, secondary, tick = sizes
    boxed = frame == "boxed"
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": base,
            "axes.labelsize": base,
            "axes.titlesize": base,
            "legend.fontsize": secondary,
            "xtick.labelsize": tick,
            "ytick.labelsize": tick,
            "axes.linewidth": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.spines.top": boxed,
            "axes.spines.right": boxed,
            "axes.spines.left": frame != "none",
            "axes.spines.bottom": frame != "none",
            "axes.grid": bool(grid),
            "legend.frameon": False,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.labelweight": "normal",
            "lines.linewidth": 1.2,
            "patch.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    if font:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
        if font in available:
            mpl.rcParams["font.sans-serif"] = [font] + list(
                mpl.rcParams.get("font.sans-serif", [])
            )
        else:
            warnings.warn(
                f"font {font!r} not available to matplotlib; keeping default sans-serif",
                RuntimeWarning,
                stacklevel=2,
            )


def set_frame(ax, style: str = "open") -> None:
    """Spine visibility on an existing axes."""
    show = {
        "open": (False, False, True, True),
        "boxed": (True, True, True, True),
        "none": (False, False, False, False),
    }[style]
    for side, vis in zip(("top", "right", "bottom", "left"), show):
        ax.spines[side].set_visible(vis)
        if vis:
            ax.spines[side].set_linewidth(0.6)
    ax.tick_params(direction="out", length=0 if style == "none" else 3, width=0.6)


def panel_letter(ax, letter: str, dx: float = -0.18, dy: float = 1.02, fontsize=None,
                 case: str = "lower") -> None:
    """Bold panel letter just outside the top-left of the axes.

    ``case`` is 'lower' | 'upper' | 'asis' and normalises ``letter`` accordingly,
    matching the figure-style skill's contract.
    """
    if case not in ("lower", "upper", "asis"):
        raise ValueError(f"case must be 'lower'|'upper'|'asis', got {case!r}")
    if case == "lower":
        letter = letter.lower()
    elif case == "upper":
        letter = letter.upper()
    if fontsize is None:
        fontsize = mpl.rcParams.get("font.size", 8) + 1
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=fontsize, va="bottom", ha="left")
