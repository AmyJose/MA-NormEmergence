from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


# =====================================================================
# Configuration
# =====================================================================

DATA = Path("data/analysis/exp2/run_metrics.csv")

OUTCOME = "minimum_final_wellbeing"


# =====================================================================
# Load data
# =====================================================================

runs = pd.read_csv(DATA)

mixed = runs.loc[
    (runs["num_llm_agents"] > 0)
    & (runs["num_rule_agents"] > 0)
].copy()


# =====================================================================
# 1. Overall distribution
# =====================================================================

print("=" * 80)
print("MINIMUM FINAL WELLBEING")
print("=" * 80)

print("\nOVERALL DISTRIBUTION")

print(
    mixed[OUTCOME]
    .describe()
    .to_string()
)

zero_count = int(
    np.isclose(
        mixed[OUTCOME],
        0,
    ).sum()
)

print(
    "\nExact zeros:",
    zero_count,
    "/",
    len(mixed),
)

print(
    "Proportion zero:",
    zero_count / len(mixed),
)


# =====================================================================
# 2. By composition
# =====================================================================

print("\n" + "=" * 80)
print("BY COMPOSITION")
print("=" * 80)

by_composition = (
    mixed.groupby(
        "num_llm_agents"
    )[OUTCOME]
    .agg(
        n="count",
        mean="mean",
        sd="std",
        median="median",
        minimum="min",
        maximum="max",
        zeros=lambda x:
            np.isclose(x, 0).sum(),
        zero_proportion=lambda x:
            np.isclose(x, 0).mean(),
    )
)

print(
    by_composition.to_string(
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# =====================================================================
# 3. By experimental condition
# =====================================================================

print("\n" + "=" * 80)
print("BY EXPERIMENTAL CONDITION")
print("=" * 80)

by_condition = (
    mixed.groupby(
        [
            "num_llm_agents",
            "prompt",
            "rule_policy",
        ]
    )[OUTCOME]
    .agg(
        n="count",
        mean="mean",
        sd="std",
        median="median",
        minimum="min",
        maximum="max",
        zeros=lambda x:
            np.isclose(x, 0).sum(),
        zero_proportion=lambda x:
            np.isclose(x, 0).mean(),
    )
    .reset_index()
)

print(
    by_condition.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =====================================================================
# 4. Unique values
# =====================================================================

print("\n" + "=" * 80)
print("UNIQUE VALUES")
print("=" * 80)

unique_values = np.sort(
    mixed[OUTCOME]
    .dropna()
    .unique()
)

print(
    "Number of unique values:",
    len(unique_values),
)

print(
    "First 20:",
    unique_values[:20],
)

print(
    "Last 20:",
    unique_values[-20:],
)


# =====================================================================
# 5. Relationship with survival
# =====================================================================

print("\n" + "=" * 80)
print("MINIMUM WELLBEING BY NUMBER OF SURVIVORS")
print("=" * 80)

by_survivors = (
    mixed.groupby(
        "survivors"
    )[OUTCOME]
    .agg(
        n="count",
        mean="mean",
        sd="std",
        median="median",
        minimum="min",
        maximum="max",
        zeros=lambda x:
            np.isclose(x, 0).sum(),
    )
    .reset_index()
)

print(
    by_survivors.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =====================================================================
# 6. Verify relationship between death and zero minimum wellbeing
# =====================================================================

print("\n" + "=" * 80)
print("DEATH / ZERO-MINIMUM RELATIONSHIP")
print("=" * 80)

mixed["any_death"] = (
    mixed["survivors"] < 4
)

mixed["zero_minimum"] = np.isclose(
    mixed[OUTCOME],
    0,
)

cross_tab = pd.crosstab(
    mixed["any_death"],
    mixed["zero_minimum"],
    rownames=["Any death"],
    colnames=["Minimum wellbeing = 0"],
)

print(cross_tab)


death_implies_zero = (
    mixed.loc[
        mixed["any_death"],
        "zero_minimum",
    ].all()
)

zero_implies_death = (
    mixed.loc[
        mixed["zero_minimum"],
        "any_death",
    ].all()
)

print(
    "\nEvery episode with a death has minimum wellbeing = 0:",
    death_implies_zero,
)

print(
    "Every episode with minimum wellbeing = 0 has a death:",
    zero_implies_death,
)


# =====================================================================
# 7. Distribution conditional on all agents surviving
# =====================================================================

print("\n" + "=" * 80)
print("ALL-SURVIVOR EPISODES ONLY")
print("=" * 80)

all_survive = mixed.loc[
    mixed["survivors"] == 4
].copy()

print(
    f"\nEpisodes: {len(all_survive)} / {len(mixed)}"
)

print(
    all_survive[OUTCOME]
    .describe()
    .to_string()
)


print(
    "\nBY CONDITION — ALL SURVIVE"
)

survivor_condition = (
    all_survive.groupby(
        [
            "num_llm_agents",
            "prompt",
            "rule_policy",
        ]
    )[OUTCOME]
    .agg(
        n="count",
        mean="mean",
        sd="std",
        median="median",
        minimum="min",
        maximum="max",
    )
    .reset_index()
)

print(
    survivor_condition.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =====================================================================
# 8. Number of all-survivor episodes per condition
# =====================================================================

print("\n" + "=" * 80)
print("ALL-SURVIVOR COUNTS BY CONDITION")
print("=" * 80)

survival_counts = (
    mixed.groupby(
        [
            "num_llm_agents",
            "prompt",
            "rule_policy",
        ]
    )
    .agg(
        episodes=("seed", "size"),
        all_survive=(
            "survivors",
            lambda x: (x == 4).sum(),
        ),
    )
    .reset_index()
)

survival_counts[
    "all_survive_proportion"
] = (
    survival_counts["all_survive"]
    / survival_counts["episodes"]
)

print(
    survival_counts.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =====================================================================
# 9. OLS diagnostics — full minimum-wellbeing outcome
# =====================================================================
#
# This is diagnostic only. We are NOT yet deciding that ANOVA is
# appropriate.
#

print("\n" + "=" * 80)
print("FACTORIAL OLS DIAGNOSTICS — FULL OUTCOME")
print("=" * 80)

model = smf.ols(
    (
        f"{OUTCOME} ~ "
        "C(prompt) * "
        "C(rule_policy) * "
        "C(num_llm_agents) + "
        "C(seed)"
    ),
    data=mixed,
).fit()

residuals = model.resid

shapiro = stats.shapiro(
    residuals
)

print(
    "\nShapiro-Wilk residual normality:"
)

print(
    f"W = {shapiro.statistic:.6f}"
)

print(
    f"p = {shapiro.pvalue:.6f}"
)


groups = []

for _, group in mixed.groupby(
    [
        "num_llm_agents",
        "prompt",
        "rule_policy",
    ]
):
    groups.append(
        group[OUTCOME].to_numpy()
    )

brown_forsythe = stats.levene(
    *groups,
    center="median",
)

print(
    "\nBrown-Forsythe homogeneity of variance:"
)

print(
    f"W = {brown_forsythe.statistic:.6f}"
)

print(
    f"p = {brown_forsythe.pvalue:.6f}"
)


# =====================================================================
# 10. Residual summary
# =====================================================================

print("\n" + "=" * 80)
print("RESIDUAL SUMMARY")
print("=" * 80)

print(
    pd.Series(
        residuals,
        name="residual",
    )
    .describe()
    .to_string()
)


# =====================================================================
# 11. Positive minimum-wellbeing distribution
# =====================================================================

print("\n" + "=" * 80)
print("POSITIVE MINIMUM WELLBEING ONLY")
print("=" * 80)

positive = mixed.loc[
    mixed[OUTCOME] > 0
].copy()

print(
    f"\nEpisodes: {len(positive)} / {len(mixed)}"
)

print(
    positive[OUTCOME]
    .describe()
    .to_string()
)


print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)