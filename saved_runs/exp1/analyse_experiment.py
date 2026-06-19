#Build a single summary dataframe from all runs
import pandas as pd
import numpy as numpy
from pathlib import Path
import re

ROOT = Path("saved_runs/exp1")

def gini(values):
    values = sorted(values)
    total = sum(values)

    if total == 0:
        return 0

    n = len(values)
    weighted_sum = sum((i+1)* v for i, v in enumerate(values))

    return (2 * weighted_sum)/ (n *total) - (n+1)/n

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