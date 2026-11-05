"""Shared figure style: recessive axes, fixed categorical order, text in ink colours."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def save(fig, path_stem):
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_stem}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
