import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


FRAMINGS = (
    "unframed",
    "self_interested",
    "cooperative",
    "altruistic",
)

FRAMING_LABELS = (
    "Unframed",
    "Self-interested",
    "Cooperative",
    "Altruistic",
)

RULE_POLICIES = (
    ("self_interested", "Self-interested rules"),
    ("cooperative", "Cooperative rules"),
    ("altruistic", "Altruistic rules"),
)

POPULATIONS = (
    (1, "(1:3)", "#0072B2", "o", -0.24),
    (2, "(2:2)", "#D55E00", "s", 0.00),
    (3, "(3:1)", "#009E73", "^", 0.24),
)

METRIC = "llm_action_throw_proportion"


def plot_sharing(runs, output_dir):
    selected = runs.loc[
        runs["num_llm_agents"].isin((1, 2, 3))
    ].copy()

    if selected.empty:
        raise ValueError("No LLM runs found")

    fig, axes = plt.subplots(
        3, 1,
        figsize=(3.5, 6.2),
        sharey=True,
        sharex=True,
    )
    x = np.arange(len(FRAMINGS))

    summary_rows = []

    for ax, (policy, title) in zip(axes[:3], RULE_POLICIES):
        for population, label, colour, marker, offset in POPULATIONS:
            means = []
            standard_deviations = []

            for framing in FRAMINGS:
                condition = selected.loc[
                    (selected["rule_policy"] == policy)
                    & (selected["prompt"] == framing)
                    & (selected["num_llm_agents"] == population)
                ]

                if condition["seed"].duplicated().any():
                    raise ValueError(
                        f"Duplicate seeds: {policy}, {framing}, "
                        f"{population} LLM"
                    )

                values = pd.to_numeric(condition[METRIC], errors="raise")

                if len(values) != 10:
                    raise ValueError(
                        f"Expected 10 runs: {policy}, {framing}, "
                        f"{population} LLM; found {len(values)}"
                    )

                if (
                    values.isna().any()
                    or not np.isfinite(values.to_numpy()).all()
                    or not values.between(0, 1).all()
                ):
                    raise ValueError(
                        f"Invalid sharing proportions: {policy}, "
                        f"{framing}, {population} LLM"
                    )

                mean = values.mean()
                sd = values.std(ddof=1)

                means.append(mean)
                standard_deviations.append(sd)

                summary_rows.append({
                    "rule_policy": policy,
                    "prompt": framing,
                    "num_llm_agents": population,
                    "num_rule_agents": 4 - population,
                    "n": len(values),
                    "mean_sharing_proportion": mean,
                    "sample_sd": sd,
                    "mean_percent": 100 * mean,
                    "sd_percentage_points": 100 * sd,
                })

            bars = ax.bar(
                x + offset,
                means,
                width=0.22,
                yerr=standard_deviations,
                color=colour,
                label=label,
                zorder=3,
                error_kw={
                    "ecolor": "#333333",
                    "elinewidth": 1,
                    "capsize": 3,
                    "capthick": 1,
                },
            )
            for bar, mean, sd in zip(bars, means, standard_deviations):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean + sd + 0.015,
                    f"{100 * mean:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                    color="#333333",
                )

        ax.set_title(title, fontsize=9, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ("Unframed", "Self-interested", "Cooperative", "Altruistic"),
            fontsize=8,
            rotation=30,
            ha="right",
            rotation_mode="anchor",
        )
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlim(-0.5, len(FRAMINGS) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.supylabel(
        "Successful sharing (% of LLM activations)",
        fontsize=9,
        x=0.02,
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        title="Population composition (LLM:rule)",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )

    fig.subplots_adjust(
        left=0.22,
        right=0.98,
        top=0.95,
        bottom=0.22,
        hspace=0.35,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"llm_sharing_comparison.{extension}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "llm_sharing_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nFigures and sharing summary saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/analysis/exp2_time_averaged/run_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/exp2_time_averaged/figures"),
    )
    args = parser.parse_args()

    runs = pd.read_csv(args.input)
    plot_sharing(runs, args.output_dir)


if __name__ == "__main__":
    main()