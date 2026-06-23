import matplotlib.pyplot as plt
from matplotlib import cycler


COLORS = [
    "#CAE7B9",  # Tea Green
    "#F3DE8A",  # Light Gold
    "#EB9486",  # Sweet Salmon
    "#7E7F9A",  # Lavender Grey
    "#97A7B3",  # Cool Steel
]


def set_thesis_style():
    plt.rcParams.update({

        # Figure
        "figure.figsize": (8, 5),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # Fonts
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # Colours
        "axes.prop_cycle": cycler(color=COLORS),

        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,

        # Grid
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,

        # Legend
        "legend.frameon": False,

        # Lines
        "lines.linewidth": 2.5,
        "lines.markersize": 6,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",

    })
    