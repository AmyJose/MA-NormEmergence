import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotting_style

# =================================================
# Helpers
# =================================================

def parse_run_dir(run_dir: Path):
    """
    Extract metadata from folder name:
    e.g. selfish_selfless_seed_7
    """
    parts = run_dir.name.split("_")
    prompt_type = parts[0]
    rule_type = parts[1]
    seed = int(parts[-1])
    return prompt_type, rule_type, seed


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def gini(values):
    values = sorted(values)
    total = sum(values)

    if total == 0:
        return 0

    n = len(values)
    weighted_sum = sum((i + 1) * v for i, v in enumerate(values))

    return (2 * weighted_sum) / (n * total) - (n + 1) / n

# =================================================
# LOAD DATA
# =================================================

def load_all_runs(exp_dir: Path):
    runs = []

    for run_dir in exp_dir.iterdir():
        #print("Checking:", run_dir)

        #print("  has metadata:", (run_dir / "metadata.json").exists())
        #print("  has model:", (run_dir / "model_episode_reports.csv").exists())
        #print("  has agent:", (run_dir / "agent_reports.csv").exists())
        if not run_dir.is_dir():
            continue
        if "plots" in run_dir.name:
            continue
        if not any(f.name == "metadata.json" for f in run_dir.iterdir()):
            continue

        prompt, rule, seed = parse_run_dir(run_dir)

        meta_path = run_dir / "metadata.json"
        model_report = run_dir / "model_episode_reports.csv"
        agent_reports = run_dir / "agent_reports.csv"

        if not model_report.exists():
            continue

        model_df = pd.read_csv(model_report)
        agent_df = pd.read_csv(agent_reports)

        runs.append({
            "run_dir": run_dir,
            "prompt": prompt,
            "rule": rule,
            "seed": seed,
            "model_df": model_df,
            "agent_df": agent_df
        })

    return runs

# =================================================
# SUMMARY BUILDING
# =================================================

def build_outcome_summary(runs):
    rows = []

    for r in runs:
        df = r["model_df"]

        rows.append({
            "prompt": r["prompt"],
            "rule": r["rule"],
            "seed": r["seed"],
            "mean_wellbeing": df["mean_health"].mean(),
            "deaths": df["deceased"].sum(),
            "gini": df["gini_berries_consumed"].mean(),
            "mean_health": df["mean_health"].mean(),
            "emerged_norms": df["num_emerged_norms"].mean()
        })

    return pd.DataFrame(rows)


def build_action_summary(runs):
    rows = []

    for r in runs:
        df = r["agent_df"]

        llm = df[df["agent_id"] == 0]

        total = len(llm)
        move = (llm["action"] == "move").sum()
        eat = (llm["action"] == "eat").sum()
        throw = llm["action"].str.startswith("throw").sum()

        rows.append({
            "prompt": r["prompt"],
            "rule": r["rule"],
            "seed": r["seed"],
            "move": move,
            "eat": eat,
            "throw": throw,
            "move_prop": move / total,
            "eat_prop": eat / total,
            "throw_prop": throw / total,
        })

    return pd.DataFrame(rows)

def build_full_summary(runs):
    rows = []

    for r in runs:
        agents = r["agent_df"]
        model = r["model_df"]

        if model.empty:
            print(f"EMPTY MODEL REPORT: {r['run_dir']}")
            continue

        final_step = agents["step"].max()
        final_agents = agents[agents["step"] == final_step]

        wellbeing = final_agents["wellbeing"].tolist()
        berries = final_agents["berries_consumed"].tolist()

        rows.append({
            "prompt": r["prompt"],
            "rule": r["rule"],
            "seed": r["seed"],

            # outcomes
            "episode_length": model["end_step"].iloc[-1],
            "total_wellbeing": sum(wellbeing),
            "mean_wellbeing": np.mean(wellbeing),
            "min_wellbeing": min(wellbeing),
            "gini_wellbeing": gini(wellbeing),

            "total_berries": sum(berries),
            "mean_berries": np.mean(berries),
            "min_berries": min(berries),
            "gini_berries": gini(berries),

            "deceased": model["deceased"].iloc[-1],
            "num_emerged_norms": model["num_emerged_norms"].iloc[-1]
        })

    return pd.DataFrame(rows)

def build_results_table(df):
    return (
        df.groupby(["prompt", "rule"])
        .agg({
            "total_wellbeing": ["mean", "std"],
            "gini_wellbeing": ["mean", "std"],
            "episode_length": ["mean", "std"],
            "num_emerged_norms": ["mean", "std"],
        })
    )

# =================================================
# PLOTTING GENERICS
# =================================================

def plot_box(df, column, title, outpath, groupby=("prompt", "rule")):
    ensure_dir(Path(outpath).parent)

    fig, ax = plt.subplots(figsize=(10, 6))

    box = df.boxplot(
        column=column,
        by=list(groupby),
        ax=ax,
        patch_artist=True,  # allows box filling
        return_type="dict"
    )

    # Colour boxes
    colors = plotting_style.COLORS

    for i, patch in enumerate(box[column]["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    # Median lines
    for median in box[column]["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # Whiskers
    for whisker in box[column]["whiskers"]:
        whisker.set_color("#555555")
        whisker.set_linewidth(1.2)

    # Caps
    for cap in box[column]["caps"]:
        cap.set_color("#555555")
        cap.set_linewidth(1.2)

    ax.set_title(title)
    ax.set_xlabel("")
    plt.suptitle("")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_heatmap(df, value, title, outpath, agg="mean"):
    heat = df.pivot_table(
        values=value,
        index="prompt",
        columns="rule",
        aggfunc=agg
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(heat, annot=True, fmt=".1f")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# =================================================
# OUTCOMES
# =================================================

def plot_outcomes(df, outdir):
    ensure_dir(outdir)

    metrics = ["mean_wellbeing", "deaths", "gini"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, metric in zip(axes, metrics):
        pivot = df.groupby(["prompt", "rule"])[metric].mean().unstack()

        pivot.plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("Prompt type")
        ax.set_ylabel(metric)

    plt.tight_layout()
    plt.savefig(outdir / "outcomes_summary.png", dpi=300)
    plt.close()


# =================================================
# ACTIONS COMPOSITIONS
# =================================================

def plot_actions(df, outdir):
    ensure_dir(outdir)

    pivot = df.set_index(["prompt", "rule"])[
        ["move_prop", "eat_prop", "throw_prop"]
    ]

    pivot.plot(kind="bar", stacked=True, figsize=(10, 5))

    plt.ylabel("Proportion of actions")
    plt.title("Action Composition (LLM agent)")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(outdir / "action_composition.png", dpi=300)
    plt.close()

# =================================================
# BOX PLOTS
# =================================================

def plot_all(summary_df, outdir):

    # --- welfare ---
    plot_box(
        summary_df,
        "total_wellbeing",
        "Total Social Wellbeing",
        outdir / "social_welfare.png"
    )

    # --- inequality ---
    plot_box(
        summary_df,
        "gini_wellbeing",
        "Wellbeing Inequality",
        outdir / "gini_wellbeing.png"
    )

    plot_box(
        summary_df,
        "gini_berries",
        "Berry Consumption Inequality",
        outdir / "gini_berries.png"
    )

    # --- worst-off ---
    plot_box(
        summary_df,
        "min_wellbeing",
        "Worst-Off Agent Wellbeing",
        outdir / "min_wellbeing.png"
    )

    # --- norms ---
    plot_box(
        summary_df,
        "num_emerged_norms",
        "Emerging Norms",
        outdir / "norms.png"
    )

    # --- heatmap ---
    plot_heatmap(
        summary_df,
        "total_wellbeing",
        "Average Social Wellbeing",
        outdir / "welfare_heatmap.png"
    )

# =================================================
# MAIN
# =================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")

    args = parser.parse_args()

    base = Path(".")
    exp_dir = base / args.experiment
    plot_dir = base / "plots" / args.experiment

    ensure_dir(plot_dir)

    print("Loading runs...")
    runs = load_all_runs(exp_dir)

    print("Building summaries...")
    outcome_df = build_outcome_summary(runs)
    action_df = build_action_summary(runs)
    summary_df = build_full_summary(runs)

    # save CSVs
    outcome_df.to_csv(plot_dir / "outcomes_summary.csv", index=False)
    action_df.to_csv(plot_dir / "actions_summary.csv", index=False)
    summary_df.to_csv(plot_dir / "summary.csv", index=False)

    plotting_style.set_thesis_style()

    plot_outcomes(outcome_df, plot_dir)
    plot_actions(action_df, plot_dir)
    plot_all(summary_df, plot_dir)

    results = build_results_table(summary_df)
    results.to_csv(plot_dir / "results_table.csv")
    
    print(f"Done. Saved to {plot_dir}")


if __name__ == "__main__":
    main()