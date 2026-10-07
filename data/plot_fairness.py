import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FRAMINGS = (
    "unframed",
    "self_interested",
    "cooperative",
    "altruistic",
)

FRAMING_LABELS = (
    "Unframed",
    "Self-\ninterested",
    "Cooperative",
    "Altruistic",
)

RULE_POLICIES = (
    ("self_interested", "Self-interested rules"),
    ("cooperative", "Cooperative rules"),
    ("altruistic", "Altruistic rules"),
)

POPULATIONS = (
    (1, "1 LLM + 3 rule", "#0072B2", "o", -0.10),
    (2, "2 LLM + 2 rule", "#D55E00", "s", 0.10),
)

METRICS = (
    (
        "gini_final_wellbeing",
        "Gini of final wellbeing\n(lower is more equal)",
    ),
    (
        "minimum_final_wellbeing",
        "Minimum final wellbeing\n(higher is better)",
    ),
)


def condition_values(runs, policy, population, metric, framing=None):
    mask = (
        (runs["rule_policy"] == policy)
        & (runs["num_llm_agents"] == population)
    )

    if framing is not None:
        mask &= runs["prompt"] == framing

    condition = runs.loc[mask]

    description = f"{policy}, {population} LLM, {framing or 'rule-only'}"

    if len(condition) != 10:
        raise ValueError(
            f"Expected 10 runs for {description}; found {len(condition)}"
        )

    if condition["seed"].duplicated().any():
        raise ValueError(f"Duplicate seeds for {description}")

    values = pd.to_numeric(condition[metric], errors="raise")

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"Invalid {metric} values for {description}")

    if (values < 0).any():
        raise ValueError(f"Negative {metric} values for {description}")

    if metric == "gini_final_wellbeing" and (values > 1).any():
        raise ValueError(f"Gini above 1 for {description}")

    return values


def plot_fairness(runs, output_dir):
    fig, axes = plt.subplots(
        2, 3,
        figsize=(11, 6),
        sharex=True,
        sharey="row",
    )

    x = np.arange(len(FRAMINGS))
    summary_rows = []
    minimum_upper = 0.0

    for row, (metric, ylabel) in enumerate(METRICS):
        for column, (policy, title) in enumerate(RULE_POLICIES):
            ax = axes[row, column]

            baseline = condition_values(runs, policy, 0, metric)
            baseline_mean = baseline.mean()

            ax.axhline(
                baseline_mean,
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                zorder=1,
            )

            # Keep the label clear of the framing points.
            ax.annotate(
                "Rule-only mean",
                xy=(0.98, baseline_mean),
                xycoords=ax.get_yaxis_transform(),
                xytext=(0, 4),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=8,
                color="#555555",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 1,
                },
            )

            summary_rows.append({
                "metric": metric,
                "rule_policy": policy,
                "prompt": "not_applicable",
                "num_llm_agents": 0,
                "n": len(baseline),
                "mean": baseline_mean,
                "sample_sd": baseline.std(ddof=1),
            })

            if metric == "minimum_final_wellbeing":
                minimum_upper = max(minimum_upper, baseline_mean)

            for population, label, colour, marker, offset in POPULATIONS:
                means = []
                standard_deviations = []

                for framing in FRAMINGS:
                    values = condition_values(
                        runs, policy, population, metric, framing
                    )
                    mean = values.mean()
                    sd = values.std(ddof=1)

                    means.append(mean)
                    standard_deviations.append(sd)

                    summary_rows.append({
                        "metric": metric,
                        "rule_policy": policy,
                        "prompt": framing,
                        "num_llm_agents": population,
                        "n": len(values),
                        "mean": mean,
                        "sample_sd": sd,
                    })

                    if metric == "minimum_final_wellbeing":
                        minimum_upper = max(minimum_upper, mean + sd)

                ax.errorbar(
                    x + offset,
                    means,
                    yerr=standard_deviations,
                    fmt=marker,
                    linestyle="none",
                    color=colour,
                    markersize=5,
                    capsize=3,
                    elinewidth=1.2,
                    zorder=3,
                )

            if row == 0:
                ax.set_title(title, fontsize=11)

            ax.set_xticks(x)
            ax.set_xticklabels(FRAMING_LABELS, fontsize=9)
            ax.set_xlim(-0.5, len(FRAMINGS) - 0.5)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[row, 0].set_ylabel(ylabel, fontsize=9)

    # Shared limits within each row, including the full SD intervals.
    gini_rows = [
        row for row in summary_rows
        if row["metric"] == "gini_final_wellbeing"
    ]
    gini_lower = min(
        0.0,
        min(row["mean"] - row["sample_sd"] for row in gini_rows),
    )
    gini_upper = max(
        1.0,
        max(row["mean"] + row["sample_sd"] for row in gini_rows),
    )
    axes[0, 0].set_ylim(gini_lower - 0.02, gini_upper + 0.02)
    axes[0, 0].set_yticks(np.linspace(0, 1, 6))

    minimum_rows = [
        row for row in summary_rows
        if row["metric"] == "minimum_final_wellbeing"
    ]
    minimum_lower = min(
        0.0,
        min(row["mean"] - row["sample_sd"] for row in minimum_rows),
    )
    padding = 0.05 * max(minimum_upper - minimum_lower, 1.0)
    axes[1, 0].set_ylim(
        minimum_lower - padding,
        minimum_upper + padding,
    )

    legend_handles = [
        Line2D(
            [], [],
            color=colour,
            marker=marker,
            linestyle="none",
            markersize=6,
            label=label,
        )
        for _, label, colour, marker, _ in POPULATIONS
    ]
    legend_handles.append(
        Line2D(
            [], [],
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            label="Rule-only mean",
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=1.5, w_pad=1.2)

    output_dir.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"fairness_comparison.{extension}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "fairness_summary.csv",
        index=False,
    )
    print(f"Fairness figures and summary saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/analysis/exp2/run_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/exp2/figures"),
    )
    args = parser.parse_args()

    runs = pd.read_csv(args.input)
    plot_fairness(runs, args.output_dir)


if __name__ == "__main__":
    main()