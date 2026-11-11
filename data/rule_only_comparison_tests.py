from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# Configuration
# =====================================================================

DATA = Path("data/analysis/exp2/run_metrics.csv")

OUTCOME = "total_final_wellbeing"


# =====================================================================
# Load
# =====================================================================

runs = pd.read_csv(DATA)

rule_only = runs.loc[
    runs["num_llm_agents"] == 0
].copy()

mixed = runs.loc[
    runs["num_llm_agents"].isin([1, 2])
].copy()


print("=" * 80)
print("MIXED VS RULE-ONLY CONTROL")
print("=" * 80)


# =====================================================================
# 1. Inspect rule-only baseline
# =====================================================================

print("\n" + "=" * 80)
print("RULE-ONLY BASELINE")
print("=" * 80)

baseline_summary = (
    rule_only.groupby(
        "rule_policy"
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
    baseline_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# =====================================================================
# 2. Check baseline completeness
# =====================================================================

print("\n" + "=" * 80)
print("BASELINE COMPLETENESS")
print("=" * 80)

baseline_counts = (
    rule_only.groupby(
        "rule_policy"
    )["seed"]
    .agg(
        n="count",
        unique_seeds="nunique",
    )
    .reset_index()
)

print(
    baseline_counts.to_string(
        index=False
    )
)


duplicates = (
    rule_only.groupby(
        [
            "rule_policy",
            "seed",
        ]
    )
    .size()
)

print(
    "\nDuplicate rule-policy × seed rows:",
    int(
        (duplicates > 1).sum()
    ),
)


# =====================================================================
# 3. Construct paired mixed-minus-control dataset
# =====================================================================

control = (
    rule_only[
        [
            "rule_policy",
            "seed",
            OUTCOME,
        ]
    ]
    .rename(
        columns={
            OUTCOME:
                "rule_only_total_wellbeing"
        }
    )
)


paired = mixed.merge(
    control,
    on=[
        "rule_policy",
        "seed",
    ],
    how="left",
    validate="many_to_one",
)


paired[
    "mixed_minus_rule_only"
] = (
    paired[OUTCOME]
    - paired[
        "rule_only_total_wellbeing"
    ]
)


print("\n" + "=" * 80)
print("PAIRING CHECK")
print("=" * 80)

print(
    "Mixed runs:",
    len(mixed),
)

print(
    "Paired rows:",
    len(paired),
)

print(
    "Missing control matches:",
    int(
        paired[
            "rule_only_total_wellbeing"
        ]
        .isna()
        .sum()
    ),
)


# =====================================================================
# 4. Descriptive paired differences
# =====================================================================

print("\n" + "=" * 80)
print("MIXED MINUS RULE-ONLY DIFFERENCES")
print("=" * 80)

difference_summary = (
    paired.groupby(
        [
            "num_llm_agents",
            "prompt",
            "rule_policy",
        ]
    )
    .agg(
        n=("seed", "size"),

        mixed_mean=(
            OUTCOME,
            "mean",
        ),

        control_mean=(
            "rule_only_total_wellbeing",
            "mean",
        ),

        mean_difference=(
            "mixed_minus_rule_only",
            "mean",
        ),

        sd_difference=(
            "mixed_minus_rule_only",
            "std",
        ),

        median_difference=(
            "mixed_minus_rule_only",
            "median",
        ),

        min_difference=(
            "mixed_minus_rule_only",
            "min",
        ),

        max_difference=(
            "mixed_minus_rule_only",
            "max",
        ),

        improved_seeds=(
            "mixed_minus_rule_only",
            lambda x: (x > 0).sum(),
        ),

        worsened_seeds=(
            "mixed_minus_rule_only",
            lambda x: (x < 0).sum(),
        ),

        tied_seeds=(
            "mixed_minus_rule_only",
            lambda x: np.isclose(
                x,
                0,
            ).sum(),
        ),
    )
    .reset_index()
)

print(
    difference_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =====================================================================
# 5. Inspect difference distribution
# =====================================================================

print("\n" + "=" * 80)
print("ALL PAIRED DIFFERENCES")
print("=" * 80)

print(
    paired[
        "mixed_minus_rule_only"
    ]
    .describe()
    .to_string()
)


print("\n" + "=" * 80)
print("DIFFERENCES BY COMPOSITION")
print("=" * 80)

print(
    paired.groupby(
        "num_llm_agents"
    )[
        "mixed_minus_rule_only"
    ]
    .describe()
    .to_string()
)


# =====================================================================
# 6. Sign direction by condition
# =====================================================================

print("\n" + "=" * 80)
print("DIRECTION SUMMARY")
print("=" * 80)

direction = (
    difference_summary[
        [
            "num_llm_agents",
            "prompt",
            "rule_policy",
            "mean_difference",
            "improved_seeds",
            "worsened_seeds",
            "tied_seeds",
        ]
    ]
    .sort_values(
        [
            "num_llm_agents",
            "rule_policy",
            "mean_difference",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )
)

print(
    direction.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =====================================================================
# 7. Verify every condition has 10 paired seeds
# =====================================================================

print("\n" + "=" * 80)
print("PAIR COMPLETENESS BY CONDITION")
print("=" * 80)

pair_counts = (
    paired.groupby(
        [
            "num_llm_agents",
            "prompt",
            "rule_policy",
        ]
    )
    .agg(
        n=("seed", "size"),
        unique_seeds=(
            "seed",
            "nunique",
        ),
        control_missing=(
            "rule_only_total_wellbeing",
            lambda x:
                x.isna().sum(),
        ),
    )
    .reset_index()
)

print(
    pair_counts.to_string(
        index=False
    )
)


assert len(paired) == 240

assert (
    paired[
        "rule_only_total_wellbeing"
    ]
    .notna()
    .all()
)

assert (
    pair_counts["n"] == 10
).all()

assert (
    pair_counts[
        "unique_seeds"
    ] == 10
).all()


print(
    "\nAll pairing checks passed."
)


print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

# =====================================================================
# Inferential analysis
# =====================================================================

from statsmodels.stats.multitest import multipletests


OUTPUT_DIR = Path(
    "data/analysis/exp2/statistics/rule_only"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# Helpers
# =====================================================================

def cohen_dz(differences):

    differences = np.asarray(
        differences,
        dtype=float,
    )

    sd = differences.std(ddof=1)

    if np.isclose(sd, 0):

        if np.isclose(
            differences.mean(),
            0,
        ):
            return 0.0

        return (
            np.inf
            if differences.mean() > 0
            else -np.inf
        )

    return (
        differences.mean()
        / sd
    )


def exact_sign_flip_test(
    differences,
):
    """
    Exact two-sided paired sign-flip permutation test.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    n = len(differences)

    if n == 0:
        return np.nan

    if np.allclose(
        differences,
        0,
    ):
        return 1.0

    observed = abs(
        differences.mean()
    )

    permutation_statistics = []

    for pattern in range(
        2 ** n
    ):

        signs = np.array([
            1
            if (pattern >> i) & 1
            else -1
            for i in range(n)
        ])

        statistic = abs(
            np.mean(
                differences
                * signs
            )
        )

        permutation_statistics.append(
            statistic
        )

    permutation_statistics = np.asarray(
        permutation_statistics
    )

    return float(
        np.mean(
            permutation_statistics
            >= observed - 1e-12
        )
    )


# =====================================================================
# Test each mixed condition against corresponding rule-only baseline
# =====================================================================

print("\n" + "=" * 80)
print("EXACT PAIRED MIXED VS RULE-ONLY TESTS")
print("=" * 80)


test_rows = []


for (
    num_llm,
    prompt,
    rule_policy,
), condition in paired.groupby(
    [
        "num_llm_agents",
        "prompt",
        "rule_policy",
    ]
):

    condition = condition.sort_values(
        "seed"
    )

    differences = condition[
        "mixed_minus_rule_only"
    ].to_numpy()

    test_rows.append({
        "num_llm_agents":
            num_llm,

        "num_rule_agents":
            4 - num_llm,

        "prompt":
            prompt,

        "rule_policy":
            rule_policy,

        "n":
            len(condition),

        "mixed_mean":
            condition[
                OUTCOME
            ].mean(),

        "control_mean":
            condition[
                "rule_only_total_wellbeing"
            ].mean(),

        "mean_difference":
            differences.mean(),

        "median_difference":
            np.median(
                differences
            ),

        "cohen_dz":
            cohen_dz(
                differences
            ),

        "improved_seeds":
            int(
                (differences > 0).sum()
            ),

        "worsened_seeds":
            int(
                (differences < 0).sum()
            ),

        "tied_seeds":
            int(
                np.isclose(
                    differences,
                    0,
                ).sum()
            ),

        "permutation_p_raw":
            exact_sign_flip_test(
                differences
            ),
    })


tests = pd.DataFrame(
    test_rows
)


# =====================================================================
# Holm correction across all 24 planned comparisons
# =====================================================================

tests[
    "permutation_p_holm"
] = multipletests(
    tests[
        "permutation_p_raw"
    ],
    alpha=0.05,
    method="holm",
)[1]


tests[
    "significant_holm"
] = (
    tests[
        "permutation_p_holm"
    ] < 0.05
)


# =====================================================================
# Save full results
# =====================================================================

tests = tests.sort_values(
    [
        "num_llm_agents",
        "rule_policy",
        "prompt",
    ]
)


tests.to_csv(
    OUTPUT_DIR
    / "total_wellbeing_vs_rule_only.csv",
    index=False,
)


# =====================================================================
# Print all tests
# =====================================================================

print("\nALL COMPARISONS")

print(
    tests[
        [
            "num_llm_agents",
            "prompt",
            "rule_policy",
            "mixed_mean",
            "control_mean",
            "mean_difference",
            "cohen_dz",
            "improved_seeds",
            "worsened_seeds",
            "tied_seeds",
            "permutation_p_raw",
            "permutation_p_holm",
            "significant_holm",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# =====================================================================
# Print significant tests
# =====================================================================

print("\n" + "=" * 80)
print("SIGNIFICANT MIXED VS RULE-ONLY COMPARISONS")
print("=" * 80)

significant = tests.loc[
    tests[
        "significant_holm"
    ]
]


if significant.empty:

    print("None")

else:

    print(
        significant[
            [
                "num_llm_agents",
                "prompt",
                "rule_policy",
                "mixed_mean",
                "control_mean",
                "mean_difference",
                "cohen_dz",
                "improved_seeds",
                "worsened_seeds",
                "permutation_p_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


print(
    "\nSaved results to:",
    OUTPUT_DIR
    / "total_wellbeing_vs_rule_only.csv"
)