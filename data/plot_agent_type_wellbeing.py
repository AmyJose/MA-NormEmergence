from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    "self_interested": "Self-interested rules",
    "cooperative": "Cooperative rules",
    "altruistic": "Altruistic rules",
}

COMPOSITION_LABELS = {
    1: "(1:3) population",
    2: "(2:2) population",
}


# ============================================================
# Colours
# ============================================================
#
# Match the framing colours used in the existing figures:
#
#   Unframed        = blue
#   Self-interested = orange
#   Cooperative     = green
#   Altruistic      = red
#

default_colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

PROMPT_COLOURS = {
    prompt: default_colours[i]
    for i, prompt in enumerate(PROMPT_ORDER)
}


# ============================================================
# Load data
# ============================================================

runs = pd.read_csv(INPUT)

mixed = runs.loc[
    runs["num_llm_agents"].isin([1, 2])
].copy()


# ============================================================
# Validate
# ============================================================

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
        "Missing required columns from run_metrics.csv: "
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
# Summarise gap across seeds
# ============================================================
#
# Positive:
#   LLM-based agents have higher mean final wellbeing.
#
# Negative:
#   Rule-based agents have higher mean final wellbeing.
#

summary = (
    mixed
    .groupby(
        ["num_llm_agents", "prompt", "rule_policy"],
        as_index=False,
    )
    .agg(
        mean_gap=(
            "llm_rule_wellbeing_gap",
            "mean",
        ),
        sd_gap=(
            "llm_rule_wellbeing_gap",
            "std",
        ),
    )
)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 7),
    sharex=True,
    sharey=True,
)

# Positions:
#
# Each rule policy gets four rows, with a small gap
# between policy groups.

group_gap = 1.2
group_height = len(PROMPT_ORDER) + group_gap

y_positions = {}
group_centres = {}

for rule_index, rule_policy in enumerate(RULE_ORDER):

    start = rule_index * group_height

    positions = (
        start + np.arange(len(PROMPT_ORDER))
    )

    group_centres[rule_policy] = positions.mean()

    for prompt, y in zip(
        PROMPT_ORDER,
        positions,
    ):
        y_positions[(rule_policy, prompt)] = y


# ============================================================
# Draw each population
# ============================================================

for ax, num_llm_agents in zip(axes, [1, 2]):

    data = summary.loc[
        summary["num_llm_agents"] == num_llm_agents
    ]

    # --------------------------------------------------------
    # Equality reference
    # --------------------------------------------------------

    ax.axvline(
        0,
        color="0.25",
        linewidth=1.2,
        zorder=0,
    )

    # --------------------------------------------------------
    # Conditions
    # --------------------------------------------------------

    for rule_policy in RULE_ORDER:

        for prompt in PROMPT_ORDER:

            row = data.loc[
                (data["prompt"] == prompt)
                & (data["rule_policy"] == rule_policy)
            ]

            if len(row) != 1:
                raise ValueError(
                    "Expected exactly one condition for "
                    f"composition={num_llm_agents}, "
                    f"prompt={prompt}, "
                    f"rule_policy={rule_policy}"
                )

            row = row.iloc[0]

            y = y_positions[
                (rule_policy, prompt)
            ]

            ax.errorbar(
                row["mean_gap"],
                y,
                xerr=row["sd_gap"],
                fmt="o",
                color=PROMPT_COLOURS[prompt],
                markersize=7,
                capsize=3,
                elinewidth=1.3,
                markeredgewidth=1,
                zorder=3,
            )

    # --------------------------------------------------------
    # Separators between rule-policy groups
    # --------------------------------------------------------

    for rule_index in range(len(RULE_ORDER) - 1):

        separator = (
            (rule_index + 1) * group_height
            - group_gap / 2
        )

        ax.axhline(
            separator,
            color="0.8",
            linestyle=":",
            linewidth=0.8,
            zorder=0,
        )

    # --------------------------------------------------------
    # Styling
    # --------------------------------------------------------

    ax.set_title(
        COMPOSITION_LABELS[num_llm_agents],
        fontsize=12,
    )

    ax.grid(
        axis="x",
        alpha=0.15,
        zorder=0,
    )

    ax.set_xlabel(
        "LLM − rule mean final wellbeing"
    )


# ============================================================
# Y-axis labels
# ============================================================

tick_positions = []
tick_labels = []

for rule_policy in RULE_ORDER:

    for prompt in PROMPT_ORDER:

        tick_positions.append(
            y_positions[(rule_policy, prompt)]
        )

        tick_labels.append(
            PROMPT_LABELS[prompt]
        )

axes[0].set_yticks(tick_positions)
axes[0].set_yticklabels(tick_labels)


# Put first condition at the top.
axes[0].invert_yaxis()


# ============================================================
# Rule-policy group labels
# ============================================================
#
# Put these to the left of the prompt labels on the first
# panel only.
# ============================================================

for rule_policy in RULE_ORDER:

    axes[0].text(
        -0.34,
        group_centres[rule_policy],
        RULE_LABELS[rule_policy],
        transform=axes[0].get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=9,
        fontweight="bold",
    )


# ============================================================
# Direction labels
# ============================================================

for ax in axes:

    ax.text(
        0.02,
        1.02,
        "← Rule-based agents better off",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )

    ax.text(
        0.98,
        1.02,
        "LLM-based agents better off →",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )


# ============================================================
# Framing legend
# ============================================================

from matplotlib.lines import Line2D

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        color=PROMPT_COLOURS[prompt],
        markersize=7,
        label=PROMPT_LABELS[prompt],
    )
    for prompt in PROMPT_ORDER
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=False,
    title="LLM ethical framing",
)


# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.25,
    right=0.98,
    bottom=0.10,
    top=0.86,
    wspace=0.08,
)


# ============================================================
# Save
# ============================================================

pdf_path = (
    OUTPUT_DIR
    / "llm_rule_wellbeing_difference.pdf"
)

png_path = (
    OUTPUT_DIR
    / "llm_rule_wellbeing_difference.png"
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