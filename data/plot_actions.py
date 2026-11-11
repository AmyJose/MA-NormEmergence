import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


FRAMINGS = (
    "unframed", "self_interested", "cooperative", "altruistic"
)
LABELS = ("Unframed", "Self-interested", "Cooperative", "Altruistic")
POLICIES = ("self_interested", "cooperative", "altruistic")

ACTIONS = {
    "move": ("Movement", "#0072B2"),
    "eat": ("Consumption", "#E69F00"),
    "throw": ("Successful sharing", "#009E73"),
    "unsuccessful": ("Unsuccessful action", "#999999"),
}


def plot_population(runs, population, output_dir):
    subset = runs.loc[runs["num_llm_agents"] == population]
    policies = ("not_applicable",) if population == 4 else POLICIES

    fig, axes = plt.subplots(
        1, len(policies),
        figsize=(4 * len(policies), 4.5),
        sharey=True,
        squeeze=False,
    )

    columns = [
        f"llm_action_{action}_proportion" for action in ACTIONS
    ]

    for ax, policy in zip(axes[0], policies):
        panel = subset.loc[subset["rule_policy"] == policy]
        available = [p for p in FRAMINGS if p in panel["prompt"].values]
        x = np.arange(len(available))
        bottom = np.zeros(len(available))

        for action, (label, colour) in ACTIONS.items():
            means = []
            for framing in available:
                group = panel.loc[panel["prompt"] == framing]
                proportions = group[columns]

                if proportions.isna().any().any():
                    raise ValueError("Missing action proportions")
                if not np.allclose(proportions.sum(axis=1), 1):
                    raise ValueError("Action proportions do not sum to one")

                means.append(
                    group[f"llm_action_{action}_proportion"].mean()
                )

            means = np.asarray(means)
            ax.bar(
                x, means, bottom=bottom,
                color=colour, label=label, width=0.7,
            )
            bottom += means

        ax.set_xticks(x)
        ax.set_xticklabels(
            [LABELS[FRAMINGS.index(p)] for p in available],
            rotation=25, ha="right",
        )
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(
            "All LLM agents" if population == 4
            else f"{policy.replace('_', ' ').capitalize()} rules"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0, 0].set_ylabel("Mean proportion of LLM activations")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center",
        ncol=2, frameon=False,
    )
    fig.suptitle(f"{population} LLM + {4 - population} rule agents")
    fig.tight_layout(rect=(0, 0.18, 1, 0.94))

    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"llm_actions_{population}_llm.{extension}",
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
    args = parser.parse_args()

    runs = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for population in sorted(runs["num_llm_agents"].unique()):
        if population > 0:
            plot_population(runs, population, args.output_dir)
            print(f"Saved action plot for {population} LLM agents")


if __name__ == "__main__":
    main()