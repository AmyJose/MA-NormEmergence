import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FRAMINGS = {
    "unframed": ("Unframed", "#666666"),
    "self_interested": ("Self-interested", "#D55E00"),
    "cooperative": ("Cooperative", "#0072B2"),
    "altruistic": ("Altruistic", "#009E73"),
}
POLICIES = ("self_interested", "cooperative", "altruistic")

METRICS = {
    "total_final_wellbeing": "Total final wellbeing",
    "minimum_final_wellbeing": "Minimum final wellbeing",
    "gini_final_wellbeing": "Gini of final wellbeing",
    "survivors": "Surviving agents",
    "llm_mean_berries_consumed": "Berries consumed per LLM agent",
    "rule_mean_berries_consumed": "Berries consumed per rule agent",
}


def plot_metric(runs, metric, output_dir):
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4), sharex=True, sharey=True
    )
    eligible = runs.loc[runs["num_llm_agents"] < 4]
    if metric.startswith("llm_"):
        eligible = eligible.loc[eligible["num_llm_agents"] > 0]

    populations = sorted(eligible["num_llm_agents"].unique())

    for ax, policy in zip(axes, POLICIES):
        subset = runs.loc[runs["rule_policy"] == policy]

        # Plot the rule-only control once, without assigning a framing.
        control = subset.loc[subset["num_llm_agents"] == 0, metric].dropna()
        if not control.empty:
            jitter = np.linspace(-0.04, 0.04, len(control))
            ax.scatter(
                jitter, control, color="black", alpha=0.25, s=14
            )
            ax.errorbar(
                0, control.mean(), yerr=control.std(ddof=1),
                fmt="s", color="black", capsize=4,
                label="Rule-only",
            )

        for index, (framing, (label, colour)) in enumerate(FRAMINGS.items()):
            framed = subset.loc[
                (subset["prompt"] == framing)
                & (subset["num_llm_agents"] > 0)
                & (subset["num_llm_agents"] < 4)
            ]
            offset = (index - 1.5) * 0.08

            for population, group in framed.groupby("num_llm_agents"):
                values = group.sort_values("seed")[metric].dropna()
                if values.empty:
                    continue

                x = population + offset
                jitter = np.linspace(-0.025, 0.025, len(values))
                ax.scatter(
                    x + jitter, values,
                    color=colour, alpha=0.3, s=14,
                )
                ax.errorbar(
                    x, values.mean(), yerr=values.std(ddof=1),
                    fmt="o", color=colour, capsize=4,
                )

            # Legend entries even when a framing is absent from this panel.
            ax.plot([], [], "o", color=colour, label=label)

        ax.set_title(f"{policy.replace('_', ' ').capitalize()} rules")
        ax.set_xticks(populations)
        ax.set_xlabel("Number of LLM agents")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ylabel = METRICS[metric]
    if metric == "llm_mean_berries_consumed":
        ylabel = "Berries consumed\nper LLM agent"
    elif metric == "rule_mean_berries_consumed":
        ylabel = "Berries consumed\nper rule agent"
    axes[0].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center",
        ncol=5, frameon=False,
    )
    fig.suptitle(METRICS[metric])
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))

    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{metric}.{extension}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/analysis/exp2/run_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/analysis/exp2/figures"),
    )
    parser.add_argument(
        "--metric", choices=METRICS,
        default="total_final_wellbeing",
    )
    args = parser.parse_args()

    runs = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(runs, args.metric, args.output_dir)
    print(f"Saved {args.metric}.png and .pdf to {args.output_dir}")


if __name__ == "__main__":
    main()