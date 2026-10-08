from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


# ============================================================
# Configuration
# ============================================================

INPUT = Path("data/analysis/exp2/run_metrics.csv")
OUTPUT_DIR = Path("data/analysis/exp2/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_ORDER = [
    "unframed",
    "self_interested",
    "cooperative",
    "altruistic",
]

RULE_ORDER = [
    "self_interested",
    "cooperative",
    "altruistic",
]

PROMPT_LABELS = {
    "unframed": "Unframed",
    "self_interested": "Self-interested",
    "cooperative": "Cooperative",
    "altruistic": "Altruistic",
}

RULE_LABELS = {
    "self_interested": "Self-interested",
    "cooperative": "Cooperative",
    "altruistic": "Altruistic",
}

COMPOSITION_LABELS = {
    1: "1:3 LLM-to-rule",
    2: "2:2 LLM-to-rule",
}


# ============================================================
# Load and validate data
# ============================================================

runs = pd.read_csv(INPUT)

mixed = runs.loc[
    runs["num_llm_agents"].isin([1, 2])
].copy()

required_columns = {
    "seed",
    "num_llm_agents",
    "prompt",
    "rule_policy",
    "llm_rule_wellbeing_gap",
}

missing = required_columns - set(mixed.columns)

if missing:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(sorted(missing))
    )

condition_counts = (
    mixed
    .groupby(
        ["num_llm_agents", "prompt", "rule_policy"]
    )["seed"]
    .nunique()
)

if not (condition_counts == 10).all():
    raise ValueError(
        "Expected 10 seeds per mixed condition.\n"
        + condition_counts.to_string()
    )


# ============================================================
# Condition means
# ============================================================

summary = (
    mixed
    .groupby(
        ["num_llm_agents", "prompt", "rule_policy"],
        as_index=False,
    )
    .agg(
        mean_gap=("llm_rule_wellbeing_gap", "mean")
    )
)


# ============================================================
# Build heatmap matrices
# ============================================================

matrices = {}

for num_llm_agents in [1, 2]:

    matrix = np.empty(
        (len(PROMPT_ORDER), len(RULE_ORDER))
    )

    for i, prompt in enumerate(PROMPT_ORDER):
        for j, rule_policy in enumerate(RULE_ORDER):

            row = summary.loc[
                (summary["num_llm_agents"] == num_llm_agents)
                & (summary["prompt"] == prompt)
                & (summary["rule_policy"] == rule_policy)
            ]

            if len(row) != 1:
                raise ValueError(
                    "Expected exactly one value for "
                    f"composition={num_llm_agents}, "
                    f"prompt={prompt}, "
                    f"rule_policy={rule_policy}"
                )

            matrix[i, j] = row.iloc[0]["mean_gap"]

    matrices[num_llm_agents] = matrix


# ============================================================
# Shared colour scale
# ============================================================
#
# Use a rounded symmetric limit so that:
#
#   -70 and +70 have equal visual intensity
#    0 is exactly neutral
#
# A fixed rounded scale also makes the figure easier to compare
# across regenerated versions.
# ============================================================

observed_max = max(
    np.abs(matrix).max()
    for matrix in matrices.values()
)

COLOUR_LIMIT = np.ceil(observed_max / 10) * 10

norm = TwoSlopeNorm(
    vmin=-COLOUR_LIMIT,
    vcenter=0,
    vmax=COLOUR_LIMIT,
)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(8, 4.5),
    sharey=True,
)

for ax, num_llm_agents in zip(axes, [1, 2]):

    matrix = matrices[num_llm_agents]

    image = ax.imshow(
        matrix,
        cmap="RdBu_r",
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )

    # --------------------------------------------------------
    # Disable interactive cursor formatting
    #
    # Avoids Matplotlib OverflowError triggered by cursor
    # formatting with TwoSlopeNorm in some versions.
    # --------------------------------------------------------

    image.format_cursor_data = lambda data: ""

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(len(RULE_ORDER))
    )

    ax.set_xticklabels(
        [RULE_LABELS[r] for r in RULE_ORDER],
        rotation=20,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_yticks(
        np.arange(len(PROMPT_ORDER))
    )

    ax.set_yticklabels(
        [PROMPT_LABELS[p] for p in PROMPT_ORDER]
    )

    ax.set_title(
        COMPOSITION_LABELS[num_llm_agents],
        fontsize=11,
        pad=8,
    )

    ax.set_xlabel(
        "Rule-based policy"
    )

    # --------------------------------------------------------
    # Cell values
    # --------------------------------------------------------

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            value = matrix[i, j]

            # White text only on strongly coloured cells.
            text_colour = (
                "white"
                if abs(value) >= 0.45 * COLOUR_LIMIT
                else "black"
            )

            ax.text(
                j,
                i,
                f"{value:+.1f}",
                ha="center",
                va="center",
                color=text_colour,
                fontsize=9,
                fontweight="semibold",
            )


axes[0].set_ylabel(
    "LLM ethical framing"
)


# ============================================================
# Colourbar
# ============================================================

# Dedicated colourbar axis below both panels.
cbar_ax = fig.add_axes([
    0.24,   # left
    0.075,  # bottom
    0.58,   # width
    0.032,  # height
])

cbar = fig.colorbar(
    image,
    cax=cbar_ax,
    orientation="horizontal",
)

cbar.set_ticks(
    [-80, -40, 0, 40, 80]
)

cbar.set_label(
    "Mean final wellbeing difference (LLM − rule)",
    fontsize=9,
    labelpad=5,
)


# ============================================================
# Direction labels
# ============================================================
#
# Put these ABOVE the colourbar rather than underneath it.
# This keeps them separate from the quantitative axis label.
# ============================================================


# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.16,
    right=0.97,
    top=0.91,
    bottom=0.30,
    wspace=0.08,
)


# ============================================================
# Save
# ============================================================

pdf_path = (
    OUTPUT_DIR
    / "llm_rule_wellbeing_heatmap.pdf"
)

png_path = (
    OUTPUT_DIR
    / "llm_rule_wellbeing_heatmap.png"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

plt.show()