from pathlib import Path

import pandas as pd

root = Path("data/analysis/exp2")
summary = pd.read_csv(root / "condition_summary.csv")

metrics = [
    "total_final_wellbeing",
    "gini_final_wellbeing",
    "minimum_final_wellbeing",
    "survivors",
]

selected = summary.loc[summary["metric"].isin(metrics)].copy()
selected["mean_sd"] = selected.apply(
    lambda row: (
        f"{row['mean']:.2f} ± {row['std']:.2f}"
        if pd.notna(row["std"])
        else f"{row['mean']:.2f} ± NA"
    ),
    axis=1,
)

table = selected.pivot(
    index=[
        "num_llm_agents", "num_rule_agents",
        "model", "prompt", "rule_policy",
    ],
    columns="metric",
    values="mean_sd",
).reindex(columns=metrics)

table.to_csv(root / "paper_results.csv")
print(table.to_string())
print("\nSaved: data/analysis/exp2/paper_results.csv")