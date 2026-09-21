"""Shared Matplotlib style for publication-quality project figures."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib as mpl
from cycler import cycler

JOURNAL_COLUMN_WIDTH_IN = 3.5
JOURNAL_TEXT_WIDTH_IN = 7.16
GOLDEN_RATIO = 0.618

COLOR_CYCLE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

LINE_STYLES = ["-", "--", "-.", ":"]

PUBLICATION_STYLE: dict[str, object] = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "0.15",
    "axes.labelcolor": "0.15",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(color=COLOR_CYCLE),
    "grid.color": "0.85",
    "grid.linestyle": "-",
    "grid.linewidth": 0.45,
    "grid.alpha": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.color": "0.15",
    "ytick.color": "0.15",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.7,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.borderaxespad": 0.4,
    "figure.dpi": 150,
    "figure.figsize": (JOURNAL_TEXT_WIDTH_IN, JOURNAL_TEXT_WIDTH_IN * GOLDEN_RATIO),
    "figure.constrained_layout.use": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def figure_size(
    width: str | float = "single",
    height_ratio: float = GOLDEN_RATIO,
) -> tuple[float, float]:
    """Return a journal-friendly figure size in inches."""
    if width == "single":
        width_in = JOURNAL_COLUMN_WIDTH_IN
    elif width == "double":
        width_in = JOURNAL_TEXT_WIDTH_IN
    else:
        width_in = float(width)

    return width_in, width_in * height_ratio


def apply_publication_style(
    overrides: Mapping[str, object] | None = None,
) -> None:
    """Apply the shared publication style to Matplotlib."""
    style = PUBLICATION_STYLE.copy()
    if overrides:
        style.update(overrides)
    mpl.rcParams.update(style)
