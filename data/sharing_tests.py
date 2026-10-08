from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# =====================================================================
# Configuration
# =====================================================================

DATA = Path("data/analysis/exp2/run_metrics.csv")

OUTPUT_DIR = Path(
    "data/analysis/exp2/statistics/sharing"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTCOME = "llm_executed_throw_proportion"

PROMPT_ORDER = [
    "unframed",
    "self_interested",
    "cooperative",
    "altruistic",
]

RULE_ORDER = [
    "self_interested",
    "cooperative",
    "altruistic",
]


# =====================================================================
# Load data
# =====================================================================

runs = pd.read_csv(DATA)

mixed = runs.loc[
    (runs["num_llm_agents"] > 0)
    & (runs["num_rule_agents"] > 0)
].copy()


# =====================================================================
# Helpers
# =====================================================================

def cohen_dz(differences):
    """
    Cohen's dz for paired observations.

    Used here as a standardized descriptive effect size.
    """

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


def exact_sign_flip_test(differences):
    """
    Exact two-sided paired permutation test.

    The experimental unit is the episode/seed.

    Under the null hypothesis, the signs of the paired
    seed-level differences are exchangeable.

    With 10 seeds, all 2^10 = 1024 sign assignments
    can be enumerated exactly.
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

    observed = abs(
        differences.mean()
    )

    permutation_statistics = []

    for pattern in range(2 ** n):

        signs = np.array([
            1 if (pattern >> i) & 1
            else -1
            for i in range(n)
        ])

        statistic = abs(
            np.mean(
                differences * signs
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


def paired_prompt_comparison(
    population,
    rule_policy,
    prompt_a,
    prompt_b,
):
    """
    Compare two LLM prompts within the same composition
    and rule-policy context.

    Episodes are paired by seed.
    """

    subset = population.loc[
        population["rule_policy"]
        == rule_policy
    ]

    a = (
        subset.loc[
            subset["prompt"] == prompt_a,
            ["seed", OUTCOME],
        ]
        .rename(
            columns={
                OUTCOME: "value_a"
            }
        )
    )

    b = (
        subset.loc[
            subset["prompt"] == prompt_b,
            ["seed", OUTCOME],
        ]
        .rename(
            columns={
                OUTCOME: "value_b"
            }
        )
    )

    paired = a.merge(
        b,
        on="seed",
        how="inner",
        validate="one_to_one",
    )

    differences = (
        paired["value_a"]
        - paired["value_b"]
    )

    return {
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "n": len(paired),

        "mean_a":
            paired["value_a"].mean(),

        "mean_b":
            paired["value_b"].mean(),

        "mean_difference":
            differences.mean(),

        "median_difference":
            differences.median(),

        "cohen_dz":
            cohen_dz(differences),

        "permutation_p_raw":
            exact_sign_flip_test(
                differences
            ),

        "any_sharing_a":
            (
                paired["value_a"] > 0
            ).mean(),

        "any_sharing_b":
            (
                paired["value_b"] > 0
            ).mean(),
    }


def paired_composition_comparison(
    data,
    prompt,
    rule_policy,
):
    """
    Compare 1:3 against 2:2 for an exact prompt ×
    rule-policy context.

    Episodes are paired by seed.

    Difference is defined as:

        1:3 - 2:2

    so a negative value means successful sharing was
    higher in the 2:2 population.
    """

    subset = data.loc[
        (
            data["prompt"] == prompt
        )
        & (
            data["rule_policy"]
            == rule_policy
        )
    ]

    one_llm = (
        subset.loc[
            subset["num_llm_agents"] == 1,
            ["seed", OUTCOME],
        ]
        .rename(
            columns={
                OUTCOME: "value_1_3"
            }
        )
    )

    two_llm = (
        subset.loc[
            subset["num_llm_agents"] == 2,
            ["seed", OUTCOME],
        ]
        .rename(
            columns={
                OUTCOME: "value_2_2"
            }
        )
    )

    paired = one_llm.merge(
        two_llm,
        on="seed",
        how="inner",
        validate="one_to_one",
    )

    differences = (
        paired["value_1_3"]
        - paired["value_2_2"]
    )

    return {
        "prompt": prompt,
        "rule_policy": rule_policy,
        "n": len(paired),

        "mean_1_3":
            paired["value_1_3"].mean(),

        "mean_2_2":
            paired["value_2_2"].mean(),

        "mean_difference_1_3_minus_2_2":
            differences.mean(),

        "median_difference_1_3_minus_2_2":
            differences.median(),

        "cohen_dz":
            cohen_dz(differences),

        "permutation_p_raw":
            exact_sign_flip_test(
                differences
            ),

        "any_sharing_1_3":
            (
                paired["value_1_3"] > 0
            ).mean(),

        "any_sharing_2_2":
            (
                paired["value_2_2"] > 0
            ).mean(),
    }


# =====================================================================
# 1. Descriptive statistics
# =====================================================================

print("=" * 80)
print("LLM SUCCESSFUL SHARING")
print("=" * 80)

descriptive = (
    mixed.groupby(
        [
            "num_llm_agents",
            "num_rule_agents",
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
        zero_count=lambda x:
            (x == 0).sum(),
        any_sharing_proportion=lambda x:
            (x > 0).mean(),
    )
    .reset_index()
)

descriptive.to_csv(
    OUTPUT_DIR / "descriptive.csv",
    index=False,
)


# =====================================================================
# 2. Prompt comparisons
# =====================================================================

prompt_pairs = list(
    combinations(
        PROMPT_ORDER,
        2,
    )
)

prompt_rows = []

for num_llm in [1, 2]:

    population = mixed.loc[
        mixed["num_llm_agents"]
        == num_llm
    ].copy()

    for rule_policy in RULE_ORDER:

        family = []

        for prompt_a, prompt_b in (
            prompt_pairs
        ):

            result = (
                paired_prompt_comparison(
                    population,
                    rule_policy,
                    prompt_a,
                    prompt_b,
                )
            )

            result.update({
                "num_llm_agents":
                    num_llm,

                "num_rule_agents":
                    4 - num_llm,

                "rule_policy":
                    rule_policy,
            })

            family.append(
                result
            )

        # Holm correction across the six prompt
        # comparisons in this composition × rule context.
        raw_p = [
            row["permutation_p_raw"]
            for row in family
        ]

        reject, corrected_p, _, _ = (
            multipletests(
                raw_p,
                alpha=0.05,
                method="holm",
            )
        )

        for (
            row,
            significant,
            p_holm,
        ) in zip(
            family,
            reject,
            corrected_p,
        ):
            row[
                "permutation_p_holm"
            ] = p_holm

            row[
                "significant_holm"
            ] = bool(significant)

            prompt_rows.append(
                row
            )


prompt_posthocs = pd.DataFrame(
    prompt_rows
)

prompt_posthocs.to_csv(
    OUTPUT_DIR
    / "prompt_posthocs.csv",
    index=False,
)


# =====================================================================
# 3. Composition comparisons
# =====================================================================

composition_rows = []

for rule_policy in RULE_ORDER:

    for prompt in PROMPT_ORDER:

        result = (
            paired_composition_comparison(
                mixed,
                prompt,
                rule_policy,
            )
        )

        composition_rows.append(
            result
        )


# Treat all 12 planned composition contrasts as one
# multiplicity family.
raw_p = [
    row["permutation_p_raw"]
    for row in composition_rows
]

reject, corrected_p, _, _ = (
    multipletests(
        raw_p,
        alpha=0.05,
        method="holm",
    )
)

for (
    row,
    significant,
    p_holm,
) in zip(
    composition_rows,
    reject,
    corrected_p,
):
    row[
        "permutation_p_holm"
    ] = p_holm

    row[
        "significant_holm"
    ] = bool(significant)


composition_posthocs = pd.DataFrame(
    composition_rows
)

composition_posthocs.to_csv(
    OUTPUT_DIR
    / "composition_posthocs.csv",
    index=False,
)


# =====================================================================
# 4. Significant prompt results
# =====================================================================

print("\n" + "=" * 80)
print("SIGNIFICANT PROMPT COMPARISONS")
print("=" * 80)

significant_prompt = (
    prompt_posthocs.loc[
        prompt_posthocs[
            "significant_holm"
        ]
    ]
)

if significant_prompt.empty:
    print("None")
else:
    print(
        significant_prompt[
            [
                "num_llm_agents",
                "rule_policy",
                "prompt_a",
                "prompt_b",
                "mean_a",
                "mean_b",
                "mean_difference",
                "cohen_dz",
                "permutation_p_raw",
                "permutation_p_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


# =====================================================================
# 5. Significant composition results
# =====================================================================

print("\n" + "=" * 80)
print("SIGNIFICANT COMPOSITION COMPARISONS")
print("=" * 80)

significant_composition = (
    composition_posthocs.loc[
        composition_posthocs[
            "significant_holm"
        ]
    ]
)

if significant_composition.empty:
    print("None")
else:
    print(
        significant_composition[
            [
                "prompt",
                "rule_policy",
                "mean_1_3",
                "mean_2_2",
                "mean_difference_1_3_minus_2_2",
                "cohen_dz",
                "permutation_p_raw",
                "permutation_p_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


# =====================================================================
# 6. Output locations
# =====================================================================

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

for filename in [
    "descriptive.csv",
    "prompt_posthocs.csv",
    "composition_posthocs.csv",
]:
    print(
        OUTPUT_DIR / filename
    )