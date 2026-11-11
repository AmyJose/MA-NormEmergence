import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FRAMINGS = [
    ("unframed", "Unframed", "#0072B2", "o"),
    ("self_interested", "Self-interested", "#D55E00", "s"),
    ("cooperative", "Cooperative", "#009E73", "^"),
    ("altruistic", "Altruistic", "#CC79A7", "D"),
]

POLICIES = [
    ("self_interested", "Self-interested rules"),
    ("cooperative", "Cooperative rules"),
    ("altruistic", "Altruistic rules"),
]

METRIC = "total_final_wellbeing"


def condition_values(condition):
    if (
        len(condition) != 10
        or set(condition["seed"]) != set(range(1, 11))
    ):
        raise ValueError("Expected exactly one run for each seed 1–10")

    values = condition[METRIC].to_numpy(dtype=float)

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Wellbeing must be finite and nonnegative")

    return values


def plot_welfare(runs, output_dir):
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        1, 4,
        figsize=(13, 3.8),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 0.9]},
    )

    offsets = [-0.09, -0.03, 0.03, 0.09]
    upper_limit = 0.0
    summary_rows = []

    for ax, (policy, title) in zip(axes[:3], POLICIES):
        control = runs.loc[
            (runs["num_llm_agents"] == 0)
            & (runs["num_rule_agents"] == 4)
            & (runs["rule_policy"] == policy)
        ]
        control_values = condition_values(control)
        control_mean = control_values.mean()

        ax.axhline(
            control_mean,
            color="#555555",
            linestyle="--",
            linewidth=1.2,
        )
        ax.annotate(
            f"Rule-only mean: {control_mean:.1f}",
            xy=(0.03, control_mean),
            xycoords=ax.get_yaxis_transform(),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=8,
            color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )

        summary_rows.append({
            "num_llm_agents": 0,
            "prompt": "not_applicable",
            "rule_policy": policy,
            "mean": control_mean,
            "std": control_values.std(ddof=1),
        })
        upper_limit = max(upper_limit, control_mean)

        for (framing, label, colour, marker), offset in zip(
            FRAMINGS, offsets
        ):
            means = []
            sds = []

            for population in [1, 2, 3]:
                condition = runs.loc[
                    (runs["num_llm_agents"] == population)
                    & (runs["num_rule_agents"] == 4 - population)
                    & (runs["rule_policy"] == policy)
                    & (runs["prompt"] == framing)
                ]
                values = condition_values(condition)
                mean = values.mean()
                sd = values.std(ddof=1)

                means.append(mean)
                sds.append(sd)
                summary_rows.append({
                    "num_llm_agents": population,
                    "prompt": framing,
                    "rule_policy": policy,
                    "mean": mean,
                    "std": sd,
                })

            ax.errorbar(
                np.array([1, 2, 3]) + offset,
                means,
                yerr=sds,
                color=colour,
                marker=marker,
                markersize=5,
                linewidth=1.5,
                elinewidth=0.9,
                capsize=3,
            )
            upper_limit = max(
                upper_limit,
                np.max(np.array(means) + np.array(sds)),
            )

        ax.set_title(title)
        ax.set_xticks([1, 2, 3])
        ax.set_xlim(0.75, 3.25)
        ax.set_xlabel("Number of LLM agents")

    # All-LLM conditions have no rule-policy factor.
    ax = axes[3]
    for position, (framing, label, colour, marker) in enumerate(FRAMINGS):
        condition = runs.loc[
            (runs["num_llm_agents"] == 4)
            & (runs["num_rule_agents"] == 0)
            & (runs["prompt"] == framing)
        ]
        values = condition_values(condition)
        mean = values.mean()
        sd = values.std(ddof=1)

        ax.errorbar(
            position,
            mean,
            yerr=sd,
            fmt=marker,
            color=colour,
            markersize=6,
            elinewidth=0.9,
            capsize=3,
        )
        upper_limit = max(upper_limit, mean + sd)
        summary_rows.append({
            "num_llm_agents": 4,
            "prompt": framing,
            "rule_policy": "not_applicable",
            "mean": mean,
            "std": sd,
        })

    ax.set_title("All-LLM population (4:0)")
    ax.set_xticks(range(4))
    ax.set_xticklabels(
        ["Unframed", "Self-interested", "Cooperative", "Altruistic"],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_xlim(-0.4, 3.4)
    ax.set_xlabel("LLM framing")

    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Total final wellbeing")
    axes[0].set_ylim(0, upper_limit * 1.12)

    handles = [
        Line2D(
            [0], [0],
            color=colour,
            marker=marker,
            linewidth=1.5,
            label=label,
        )
        for _, label, colour, marker in FRAMINGS
    ]
    handles.append(
        Line2D(
            [0], [0],
            color="#555555",
            linestyle="--",
            label="Rule-only mean",
        )
    )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0, 0.15, 1, 1))

    for extension in ["pdf", "png"]:
        path = output_dir / f"total_final_wellbeing.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "total_wellbeing_plot_summary.csv",
        index=False,
    )
    plt.close(fig)


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_welfare(runs, args.output_dir)
    print("Total wellbeing figure checks passed")


if __name__ == "__main__":
    main()