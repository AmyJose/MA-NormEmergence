from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# Configuration
# =====================================================================

DATA = Path("data/analysis/exp2/run_metrics.csv")

EXPECTED_SEEDS_PER_CONDITION = 10


# =====================================================================
# Load data
# =====================================================================

df = pd.read_csv(DATA)

print("=" * 80)
print("RESULTS DATASET AUDIT")
print("=" * 80)

print(f"\nFile: {DATA}")
print(f"Runs: {len(df)}")
print(f"Columns: {len(df.columns)}")


# =====================================================================
# 1. Population counts
# =====================================================================

print("\n" + "=" * 80)
print("POPULATION COUNTS")
print("=" * 80)

population_counts = (
    df.groupby(
        [
            "num_llm_agents",
            "num_rule_agents",
        ]
    )
    .size()
    .reset_index(name="runs")
)

print(
    population_counts.to_string(
        index=False
    )
)


# =====================================================================
# 2. Condition / seed completeness
# =====================================================================

print("\n" + "=" * 80)
print("CONDITION COMPLETENESS")
print("=" * 80)

mixed = df.loc[
    (df["num_llm_agents"] > 0)
    & (df["num_rule_agents"] > 0)
].copy()

mixed_condition_counts = (
    mixed.groupby(
        [
            "num_llm_agents",
            "num_rule_agents",
            "prompt",
            "rule_policy",
        ]
    )
    .agg(
        runs=("seed", "size"),
        unique_seeds=("seed", "nunique"),
    )
    .reset_index()
)

bad_mixed_conditions = (
    mixed_condition_counts.loc[
        (
            mixed_condition_counts["runs"]
            != EXPECTED_SEEDS_PER_CONDITION
        )
        | (
            mixed_condition_counts["unique_seeds"]
            != EXPECTED_SEEDS_PER_CONDITION
        )
    ]
)

print(
    f"Mixed conditions: "
    f"{len(mixed_condition_counts)}"
)

print(
    "Complete mixed conditions:",
    len(mixed_condition_counts)
    - len(bad_mixed_conditions),
    "/",
    len(mixed_condition_counts),
)

if not bad_mixed_conditions.empty:
    print("\nIncomplete mixed conditions:")
    print(
        bad_mixed_conditions.to_string(
            index=False
        )
    )


# Rule-only conditions do not have a meaningful LLM prompt,
# so audit them separately by rule policy.
rule_only = df.loc[
    df["num_llm_agents"] == 0
].copy()

if not rule_only.empty:

    rule_condition_counts = (
        rule_only.groupby(
            [
                "num_llm_agents",
                "num_rule_agents",
                "rule_policy",
            ]
        )
        .agg(
            runs=("seed", "size"),
            unique_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    bad_rule_conditions = (
        rule_condition_counts.loc[
            (
                rule_condition_counts["runs"]
                != EXPECTED_SEEDS_PER_CONDITION
            )
            | (
                rule_condition_counts["unique_seeds"]
                != EXPECTED_SEEDS_PER_CONDITION
            )
        ]
    )

    print(
        "\nRule-only conditions:",
        len(rule_condition_counts),
    )

    print(
        "Complete rule-only conditions:",
        len(rule_condition_counts)
        - len(bad_rule_conditions),
        "/",
        len(rule_condition_counts),
    )

    if not bad_rule_conditions.empty:
        print("\nIncomplete rule-only conditions:")
        print(
            bad_rule_conditions.to_string(
                index=False
            )
        )


# =====================================================================
# 3. Duplicate experimental runs
# =====================================================================

print("\n" + "=" * 80)
print("DUPLICATE RUN CHECK")
print("=" * 80)

mixed_duplicate_columns = [
    "num_llm_agents",
    "num_rule_agents",
    "model",
    "prompt",
    "rule_policy",
    "seed",
]

mixed_duplicates = mixed.duplicated(
    subset=mixed_duplicate_columns,
    keep=False,
)

print(
    "Duplicate mixed runs:",
    int(mixed_duplicates.sum()),
)

if mixed_duplicates.any():
    print(
        mixed.loc[
            mixed_duplicates,
            mixed_duplicate_columns,
        ]
        .sort_values(
            mixed_duplicate_columns
        )
        .to_string(index=False)
    )


# =====================================================================
# 4. Welfare identities
# =====================================================================

print("\n" + "=" * 80)
print("WELFARE IDENTITIES")
print("=" * 80)

welfare_rows = mixed[
    [
        "llm_total_final_wellbeing",
        "rule_total_final_wellbeing",
        "total_final_wellbeing",
    ]
].dropna()

welfare_match = np.isclose(
    (
        welfare_rows[
            "llm_total_final_wellbeing"
        ]
        + welfare_rows[
            "rule_total_final_wellbeing"
        ]
    ),
    welfare_rows[
        "total_final_wellbeing"
    ],
)

print(
    "LLM total + rule total = society total:",
    int(welfare_match.sum()),
    "/",
    len(welfare_match),
)


# =====================================================================
# 5. Survival identities
# =====================================================================

print("\n" + "=" * 80)
print("SURVIVAL IDENTITIES")
print("=" * 80)

survival_rows = mixed[
    [
        "llm_survivors",
        "rule_survivors",
        "survivors",
    ]
].dropna()

survival_match = np.isclose(
    (
        survival_rows["llm_survivors"]
        + survival_rows["rule_survivors"]
    ),
    survival_rows["survivors"],
)

print(
    "LLM survivors + rule survivors = total survivors:",
    int(survival_match.sum()),
    "/",
    len(survival_match),
)


# =====================================================================
# 6. Executed action proportions sum to one
# =====================================================================

print("\n" + "=" * 80)
print("EXECUTED ACTION PROPORTIONS")
print("=" * 80)

for subgroup in ["llm", "rule"]:

    columns = [
        f"{subgroup}_executed_move_proportion",
        f"{subgroup}_executed_eat_proportion",
        f"{subgroup}_executed_throw_proportion",
        f"{subgroup}_executed_unsuccessful_proportion",
    ]

    valid = mixed[columns].notna().all(
        axis=1
    )

    sums = (
        mixed.loc[
            valid,
            columns,
        ]
        .sum(axis=1)
    )

    matches = np.isclose(
        sums,
        1.0,
    )

    print(
        f"{subgroup.upper()} action proportions sum to 1:",
        int(matches.sum()),
        "/",
        len(matches),
    )


# =====================================================================
# 7. LLM reasoning-log vs executed-action consistency
# =====================================================================

print("\n" + "=" * 80)
print("LLM LOGGING CONSISTENCY")
print("=" * 80)

logging_pairs = {
    "move": (
        "llm_action_move_proportion",
        "llm_executed_move_proportion",
    ),
    "eat": (
        "llm_action_eat_proportion",
        "llm_executed_eat_proportion",
    ),
    "throw": (
        "llm_action_throw_proportion",
        "llm_executed_throw_proportion",
    ),
    "unsuccessful": (
        "llm_action_unsuccessful_proportion",
        "llm_executed_unsuccessful_proportion",
    ),
}

for action, (
    logged_column,
    executed_column,
) in logging_pairs.items():

    valid = mixed[
        [
            logged_column,
            executed_column,
        ]
    ].notna().all(axis=1)

    matches = np.isclose(
        mixed.loc[
            valid,
            logged_column,
        ],
        mixed.loc[
            valid,
            executed_column,
        ],
    )

    print(
        f"{action}:",
        int(matches.sum()),
        "/",
        len(matches),
    )


# =====================================================================
# 8. Successful-sharing count / proportion consistency
# =====================================================================

print("\n" + "=" * 80)
print("SUCCESSFUL-SHARING CONSISTENCY")
print("=" * 80)

for subgroup in ["llm", "rule"]:

    count_column = (
        f"{subgroup}_successful_shares"
    )

    actions_column = (
        f"{subgroup}_executed_actions"
    )

    proportion_column = (
        f"{subgroup}_executed_throw_proportion"
    )

    required = [
        count_column,
        actions_column,
        proportion_column,
    ]

    valid = (
        mixed[required]
        .notna()
        .all(axis=1)
        & (
            mixed[actions_column] > 0
        )
    )

    calculated_proportion = (
        mixed.loc[
            valid,
            count_column,
        ]
        / mixed.loc[
            valid,
            actions_column,
        ]
    )

    matches = np.isclose(
        calculated_proportion,
        mixed.loc[
            valid,
            proportion_column,
        ],
    )

    print(
        f"{subgroup.upper()} successful shares / "
        f"executed actions = throw proportion:",
        int(matches.sum()),
        "/",
        len(matches),
    )

    # Successful-share count must also be a valid count
    # between zero and the number of activations.
    valid_counts = (
        (
            mixed.loc[
                valid,
                count_column,
            ] >= 0
        )
        & (
            mixed.loc[
                valid,
                count_column,
            ]
            <= mixed.loc[
                valid,
                actions_column,
            ]
        )
    )

    print(
        f"{subgroup.upper()} successful-share counts "
        f"within valid range:",
        int(valid_counts.sum()),
        "/",
        len(valid_counts),
    )


# =====================================================================
# 9. Welfare-gap identities
# =====================================================================

print("\n" + "=" * 80)
print("SUBGROUP GAP IDENTITIES")
print("=" * 80)

wellbeing_gap_rows = mixed[
    [
        "llm_mean_final_wellbeing",
        "rule_mean_final_wellbeing",
        "llm_rule_wellbeing_gap",
    ]
].dropna()

calculated_wellbeing_gap = (
    wellbeing_gap_rows[
        "llm_mean_final_wellbeing"
    ]
    - wellbeing_gap_rows[
        "rule_mean_final_wellbeing"
    ]
)

wellbeing_gap_match = np.isclose(
    calculated_wellbeing_gap,
    wellbeing_gap_rows[
        "llm_rule_wellbeing_gap"
    ],
)

print(
    "LLM mean wellbeing - rule mean wellbeing = gap:",
    int(wellbeing_gap_match.sum()),
    "/",
    len(wellbeing_gap_match),
)


survival_gap_rows = mixed[
    [
        "llm_survival_rate",
        "rule_survival_rate",
        "llm_rule_survival_gap",
    ]
].dropna()

calculated_survival_gap = (
    survival_gap_rows[
        "llm_survival_rate"
    ]
    - survival_gap_rows[
        "rule_survival_rate"
    ]
)

survival_gap_match = np.isclose(
    calculated_survival_gap,
    survival_gap_rows[
        "llm_rule_survival_gap"
    ],
)

print(
    "LLM survival rate - rule survival rate = gap:",
    int(survival_gap_match.sum()),
    "/",
    len(survival_gap_match),
)


# =====================================================================
# 10. Missing values in mixed-population analysis variables
# =====================================================================

print("\n" + "=" * 80)
print("MISSING VALUES — MIXED POPULATIONS")
print("=" * 80)

analysis_columns = [
    "total_final_wellbeing",
    "minimum_final_wellbeing",
    "gini_final_wellbeing",
    "survival_rate",

    "llm_mean_final_wellbeing",
    "rule_mean_final_wellbeing",

    "llm_survival_rate",
    "rule_survival_rate",

    "llm_rule_wellbeing_gap",
    "llm_rule_survival_gap",

    "llm_executed_throw_proportion",
    "rule_executed_throw_proportion",

    "llm_successful_shares",
    "rule_successful_shares",

    "llm_executed_actions",
    "rule_executed_actions",

    "total_berries_thrown",
]

for column in analysis_columns:
    missing = int(
        mixed[column].isna().sum()
    )

    print(
        f"{column}: {missing}"
    )


# =====================================================================
# 11. Outcome ranges
# =====================================================================

print("\n" + "=" * 80)
print("OUTCOME RANGES — MIXED POPULATIONS")
print("=" * 80)

range_columns = [
    "total_final_wellbeing",
    "minimum_final_wellbeing",
    "gini_final_wellbeing",
    "survival_rate",

    "llm_mean_final_wellbeing",
    "rule_mean_final_wellbeing",

    "llm_survival_rate",
    "rule_survival_rate",

    "llm_rule_wellbeing_gap",
    "llm_rule_survival_gap",

    "llm_executed_throw_proportion",
    "rule_executed_throw_proportion",

    "llm_successful_shares",
    "rule_successful_shares",

    "llm_executed_actions",
    "rule_executed_actions",

    "total_berries_thrown",
]

print(
    mixed[range_columns]
    .describe()
    .T
    .to_string()
)


# =====================================================================
# 12. Final audit status
# =====================================================================

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print(
    "Review any result above that does not match the "
    "expected number of valid runs."
)