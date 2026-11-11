from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("data/analysis/exp2_time_averaged")
OUTPUT = ROOT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

runs = pd.read_csv(ROOT / "run_metrics.csv")
mixed = runs.loc[
    runs["num_llm_agents"].isin([1, 2, 3])
    & (runs["num_rule_agents"] > 0)
].copy()

POLICIES = [
    ("self_interested", "Self-interested rules"),
    ("cooperative", "Cooperative rules"),
    ("altruistic", "Altruistic rules"),
]

FRAMINGS = [
    ("unframed", "Unframed", "#0072B2", "o"),
    ("self_interested", "Self-interested", "#D55E00", "s"),
    ("cooperative", "Cooperative", "#009E73", "^"),
    ("altruistic", "Altruistic", "#CC79A7", "D"),
]

METRICS = [
    (
        "time_averaged_gini_wellbeing",
        "Time-averaged Gini\n(lower is more equal)",
    ),
    (
        "time_averaged_minimum_wellbeing",
        "Time-averaged minimum wellbeing\n(higher is better for the worst-off agent)",
    ),
]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    2, 3,
    figsize=(11, 6),
    sharex=True,
    sharey="row",
)

summary_rows = []
upper_limits = [0.0, 0.0]

# Small offsets keep the four sets of error bars distinguishable.
offsets = [-0.09, -0.03, 0.03, 0.09]

for column, (policy, title) in enumerate(POLICIES):
    axes[0, column].set_title(title)

    for row, (metric, ylabel) in enumerate(METRICS):
        ax = axes[row, column]

        for (framing, label, colour, marker), offset in zip(
            FRAMINGS, offsets
        ):
            means = []
            sds = []

            for population in [1, 2, 3]:
                condition = mixed.loc[
                    (mixed["rule_policy"] == policy)
                    & (mixed["prompt"] == framing)
                    & (mixed["num_llm_agents"] == population)
                ]

                if (
                    len(condition) != 10
                    or set(condition["seed"]) != set(range(1, 11))
                ):
                    raise ValueError(
                        f"Expected seeds 1–10: "
                        f"{policy}, {framing}, {population} LLMs"
                    )

                values = condition[metric].to_numpy(dtype=float)

                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f"Invalid values for {metric}")

                if "gini" in metric and (values > 1).any():
                    raise ValueError("Gini must be between zero and one")

                mean = values.mean()
                sd = values.std(ddof=1)

                means.append(mean)
                sds.append(sd)

                summary_rows.append({
                    "rule_policy": policy,
                    "prompt": framing,
                    "num_llm_agents": population,
                    "metric": metric,
                    "count": len(values),
                    "mean": mean,
                    "std": sd,
                })

            upper_limits[row] = max(
                upper_limits[row],
                np.max(np.array(means) + np.array(sds)),
            )

            ax.errorbar(
                np.array([1, 2, 3]) + offset,
                means,
                yerr=sds,
                label=label,
                color=colour,
                marker=marker,
                markersize=5,
                linewidth=1.5,
                elinewidth=0.9,
                capsize=3,
            )

        ax.set_xticks([1, 2, 3])
        ax.set_xlim(0.75, 3.25)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if column == 0:
            ax.set_ylabel(ylabel)

        if row == 1:
            ax.set_xlabel("Number of LLM agents")

# Common scales within each row, including the SD error bars.
axes[0, 0].set_ylim(0, min(1.0, upper_limits[0] * 1.12))
axes[1, 0].set_ylim(0, upper_limits[1] * 1.12)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.01),
)

fig.tight_layout(rect=(0, 0.09, 1, 1))

for extension in ["pdf", "png"]:
    path = OUTPUT / f"fairness_mixed_populations.{extension}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

pd.DataFrame(summary_rows).to_csv(
    OUTPUT / "fairness_plot_summary.csv",
    index=False,
)

plt.close(fig)
print("Fairness figure checks passed")