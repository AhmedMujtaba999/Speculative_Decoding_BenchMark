"""One visual style for every chart in the repo (report + README images)."""
from __future__ import annotations

import matplotlib
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Color follows the method, never its rank (fixed slots, validated for color-vision deficiency).
METHOD_COLORS = {
    "baseline": MUTED,
    "ngram": "#2a78d6",   # blue
    "draft": "#eb6834",   # orange
    "suffix": "#1baf7a",  # aqua
}
METHOD_NAMES = {"baseline": "baseline", "ngram": "n-gram", "draft": "draft model", "suffix": "suffix"}

# Diverging: red (slower) <- neutral gray at 1.0x -> blue (faster)
SPEEDUP_CMAP = LinearSegmentedColormap.from_list(
    "speedup", ["#b8312f", "#e34948", "#f0efec", "#86b6ef", "#2a78d6", "#184f95"])


def apply() -> None:
    """Set matplotlib defaults: quiet axes, hairline solid grid, readable type."""
    family = "DejaVu Sans"
    for name in ("Inter", "Helvetica Neue", "Arial"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            family = name
            break
    matplotlib.rcParams.update({
        "font.family": family,
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })
