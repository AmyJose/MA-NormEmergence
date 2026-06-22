#Build a single summary dataframe from all runs
import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_style import set_thesis_style

ROOT = Path("saved_runs/exp1")
PLOT_DIR = ROOT / "plots"

def gini(values):
    values = sorted(values)
    total = sum(values)

    if total == 0:
        return 0

    n = len(values)
    weighted_sum = sum((i+1)* v for i, v in enumerate(values))

    return (2 * weighted_sum)/ (n *total) - (n+1)/n

set_thesis_style()

rows = []
for run_dir in ROOT.iterdir():
    if not run_dir.is_dir():
        continue

    match = re.match(r"(.*?)_(.*?)_seed_(\d+)", run_dir.name)

    if not match:
        continue

    prompt = match.group(1)
    rule = match.group(2)
    seed = int(match.group(3))

    agent_file = run_dir/"agent_reports.csv"
    model_file = run_dir/"model_episode_reports.csv"

    if not agent_file.exists():
        continue

    agents = pd.read_csv(agent_file)
    model = pd.read_csv(model_file)

    #get the final step of each agent
    final_step = agents["step"].max()

    final_agents = agents[agents["step"] == final_step]

    wellbeing = final_agents["wellbeing"].tolist()
    berries = final_agents["berries_consumed"].tolist()

    rows.append({
        "prompt": prompt,
        "rule": rule,
        "seed": seed,

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

summary = pd.DataFrame(rows)

summary.to_csv(ROOT/"summary.csv", index=False)

print(summary.head())

# plotting
PLOT_DIR.mkdir(exist_ok=True)

#Social welfare
fig, ax = plt.subplots(figsize=(10,6))
summary.boxplot(
    column="total_wellbeing",
    by=["prompt", "rule"],
    ax=ax
)
plt.title("Total Social Wellbeing")
plt.suptitle("")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / "social_welfare.png")

#Fairness (gini wellbeing)
fig, ax = plt.subplots(figsize=(10,6))
summary.boxplot(
    column="gini_wellbeing",
    by=["prompt", "rule"],
    ax=ax
)
plt.title("Wellbeing Inequality")
plt.suptitle("")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / "gini_wellbeing.png")

#fairness (gini berries)
fig, ax = plt.subplots(figsize=(10,6))
summary.boxplot(
    column="gini_berries",
    by=["prompt", "rule"],
    ax=ax
)
plt.title("Berry Consumption Inequality")
plt.suptitle("")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / "gini_berries.png")

#minimum wellbeing
fig, ax = plt.subplots(figsize=(10,6))
summary.boxplot(
    column="min_wellbeing",
    by=["prompt", "rule"],
    ax=ax
)
plt.title("Worst-Off Agent Wellbeing")
plt.suptitle("")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / "min_wellbeing.png")

#robustness
fig, ax = plt.subplots(figsize=(10,6))
summary.boxplot(
    column="episode_length",
    by=["prompt", "rule"],
    ax=ax
)
plt.title("Episode Length")
plt.suptitle("")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / "episode_length.png")

#norm emergence
fig, ax = plt.subplots(figsize=(10,6))
summary.boxplot(
    column="num_emerged_norms",
    by=["prompt", "rule"],
    ax=ax
)
plt.title("Emerged Norms")
plt.suptitle("")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / "norms.png")

#Action heatmap
heat = summary.pivot_table(
    values="total_wellbeing",
    index="prompt",
    columns="rule",
    aggfunc="mean"
)
plt.figure(figsize=(8,6))
sns.heatmap(heat, annot=True, fmt=".1f")
plt.title("Average Social Wellbeing")
plt.tight_layout()
plt.savefig(PLOT_DIR / "welfare_heatmap.png")

#numbers table
results = (
    summary
    .groupby(["prompt", "rule"])
    .agg({
        "total_wellbeing":["mean", "std"],
        "gini_wellbeing":["mean", "std"],
        "episode_length":["mean", "std"],
        "num_emerged_norms":["mean", "std"],
    })
)
results.to_csv(ROOT/"results_table.csv")
print(results)