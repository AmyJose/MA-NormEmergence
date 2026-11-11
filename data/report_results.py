from pathlib import Path

import pandas as pd

root = Path("data/analysis/exp2_time_averaged")
summary = pd.read_csv(root / "condition_summary.csv")

metrics = [
    "total_final_wellbeing",
    "time_averaged_gini_wellbeing",
    "time_averaged_minimum_wellbeing",
    "survivors",
]

selected = summary.loc[summary["metric"].isin(metrics)].copy()

# Every condition should contain ten episodes for each metric.
assert selected["count"].eq(10).all(), "Expected ten runs per condition"
assert selected[["mean", "std"]].notna().all().all(), "Missing mean or SD"

selected["mean_sd"] = selected.apply(
    lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}",
    axis=1,
)

table = selected.pivot(
    index=[
        "num_llm_agents",
        "num_rule_agents",
        "model",
        "prompt",
        "rule_policy",
    ],
    columns="metric",
    values="mean_sd",
).reindex(columns=metrics).sort_index()

assert len(table) == 43, "Expected 43 experimental conditions"
assert table.notna().all().all(), "Missing metrics in results table"

output = root / "paper_results.csv"
table.to_csv(output)

print(table.to_string())
print(f"\nSaved: {output}")
print("Results table checks passed: 43 conditions, all five populations")