from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# Configuration
# =====================================================================

DATA = Path("data/analysis/exp2/run_metrics.csv")


# =====================================================================
# Load data
# =====================================================================

runs = pd.read_csv(DATA)

mixed = runs.loc[
    (runs["num_llm_agents"] > 0)
    & (runs["num_rule_agents"] > 0)
].copy()


# =====================================================================
# 1. Overall society-level survival
# =====================================================================

print("=" * 80)
print("SOCIETY-LEVEL SURVIVAL")
print("=" * 80)

print("\nSURVIVORS DISTRIBUTION")

survivor_counts = (
    mixed["survivors"]
    .value_counts()
    .sort_index()
)

for survivors, count in survivor_counts.items():

    print(
        f"{int(survivors)} survivors: "
        f"{count:3d} / {len(mixed)} "
        f"({count / len(mixed):.3f})"
    )


print("\nSURVIVAL RATE")

print(
    mixed["survival_rate"]
    .describe()
    .to_string()
)


# =====================================================================
# 2. Society survival by composition
# =====================================================================

print("\n" + "=" * 80)
print("SURVIVAL BY COMPOSITION")
print("=" * 80)

composition_summary = (
    mixed.groupby(
        "num_llm_agents"
    )
    .agg(
        n=("seed", "size"),

        mean_survivors=(
            "survivors",
            "mean",
        ),

        sd_survivors=(
            "survivors",
            "std",
        ),

        mean_survival_rate=(
            "survival_rate",
            "mean",
        ),

        all_survive_proportion=(
            "survivors",
            lambda x: (x == 4).mean(),
        ),

        any_death_proportion=(
            "survivors",
            lambda x: (x < 4).mean(),
        ),
    )
    .reset_index()
)

print(
    composition_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =====================================================================
# 3. Society survival by experimental condition
# =====================================================================

print("\n" + "=" * 80)
print("SURVIVAL BY EXPERIMENTAL CONDITION")
print("=" * 80)

condition_summary = (
    mixed.groupby(
        [
            "num_llm_agents",
            "prompt",
            "rule_policy",
        ]
    )
    .agg(
        n=("seed", "size"),

        mean_survivors=(
            "survivors",
            "mean",
        ),

        sd_survivors=(
            "survivors",
            "std",
        ),

        mean_survival_rate=(
            "survival_rate",
            "mean",
        ),

        all_survive=(
            "survivors",
            lambda x: (x == 4).sum(),
        ),

        all_survive_proportion=(
            "survivors",
            lambda x: (x == 4).mean(),
        ),

        zero_survivors=(
            "survivors",
            lambda x: (x == 0).sum(),
        ),
    )
    .reset_index()
)

print(
    condition_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =====================================================================
# 4. Full survivor-count distributions by condition
# =====================================================================

print("\n" + "=" * 80)
print("SURVIVOR COUNT DISTRIBUTIONS BY CONDITION")
print("=" * 80)

distribution = (
    mixed.groupby(
        [
            "num_llm_agents",
            "prompt",
            "rule_policy",
            "survivors",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

for count in range(5):

    if count not in distribution.columns:
        distribution[count] = 0

distribution = distribution[
    [0, 1, 2, 3, 4]
]

distribution.columns = [
    "survive_0",
    "survive_1",
    "survive_2",
    "survive_3",
    "survive_4",
]

print(
    distribution
    .reset_index()
    .to_string(
        index=False
    )
)


# =====================================================================
# 5. Agent-type survival
# =====================================================================

print("\n" + "=" * 80)
print("AGENT-TYPE SURVIVAL")
print("=" * 80)

print(
    "\nOverall mixed-population subgroup means:"
)

print(
    mixed[
        [
            "llm_survival_rate",
            "rule_survival_rate",
            "llm_rule_survival_gap",
        ]
    ]
    .describe()
    .to_string()
)


# =====================================================================
# 6. Agent-type survival by condition
# =====================================================================

print("\n" + "=" * 80)
print("AGENT-TYPE SURVIVAL BY CONDITION")
print("=" * 80)

subgroup_summary = (
    mixed.groupby(
        [
            "num_llm_agents",
            "prompt",
            "rule_policy",
        ]
    )
    .agg(
        n=("seed", "size"),

        llm_survival=(
            "llm_survival_rate",
            "mean",
        ),

        rule_survival=(
            "rule_survival_rate",
            "mean",
        ),

        survival_gap=(
            "llm_rule_survival_gap",
            "mean",
        ),

        survival_gap_sd=(
            "llm_rule_survival_gap",
            "std",
        ),
    )
    .reset_index()
)

print(
    subgroup_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =====================================================================
# 7. Unique values
# =====================================================================

print("\n" + "=" * 80)
print("UNIQUE SURVIVAL VALUES")
print("=" * 80)

print(
    "\nSociety survival rate:",
    np.sort(
        mixed[
            "survival_rate"
        ].unique()
    ),
)

print(
    "LLM survival rate:",
    np.sort(
        mixed[
            "llm_survival_rate"
        ].unique()
    ),
)

print(
    "Rule survival rate:",
    np.sort(
        mixed[
            "rule_survival_rate"
        ].unique()
    ),
)

print(
    "LLM-rule survival gap:",
    np.sort(
        mixed[
            "llm_rule_survival_gap"
        ].unique()
    ),
)


# =====================================================================
# 8. Relationship with minimum wellbeing
# =====================================================================

print("\n" + "=" * 80)
print("SURVIVAL / MINIMUM-WELLBEING CHECK")
print("=" * 80)

death = (
    mixed["survivors"] < 4
)

zero_minimum = np.isclose(
    mixed["minimum_final_wellbeing"],
    0,
)

print(
    "Episodes with any death:",
    int(death.sum()),
)

print(
    "Episodes with minimum wellbeing = 0:",
    int(zero_minimum.sum()),
)

print(
    "Exact agreement:",
    bool(
        np.array_equal(
            death.to_numpy(),
            zero_minimum,
        )
    ),
)


print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

# =====================================================================
# Inferential analysis
# =====================================================================

from itertools import combinations
from scipy import stats
from statsmodels.stats.multitest import multipletests


OUTPUT_DIR = Path(
    "data/analysis/exp2/statistics/survival"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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
    Exact two-sided sign-flip permutation test.

    Zero differences are retained. With 10 seeds,
    all 2^10 sign assignments are enumerated.
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
                differences * signs
            )
        )

        permutation_statistics.append(
            statistic
        )

    permutation_statistics = (
        np.asarray(
            permutation_statistics
        )
    )

    return float(
        np.mean(
            permutation_statistics
            >= observed - 1e-12
        )
    )


def paired_result(
    a,
    b,
):

    differences = (
        a.to_numpy()
        - b.to_numpy()
    )

    return {
        "n":
            len(differences),

        "mean_a":
            a.mean(),

        "mean_b":
            b.mean(),

        "mean_difference":
            differences.mean(),

        "cohen_dz":
            cohen_dz(
                differences
            ),

        "permutation_p_raw":
            exact_sign_flip_test(
                differences
            ),
    }


def holm_correct(
    rows,
    p_column="permutation_p_raw",
):

    corrected = multipletests(
        [
            row[p_column]
            for row in rows
        ],
        alpha=0.05,
        method="holm",
    )[1]

    for row, p_holm in zip(
        rows,
        corrected,
    ):
        row[
            "permutation_p_holm"
        ] = p_holm

        row[
            "significant_holm"
        ] = (
            p_holm < 0.05
        )


# =====================================================================
# PART A
# Society-level survival
# =====================================================================

print("\n" + "=" * 80)
print("SOCIETY SURVIVAL — PROMPT COMPARISONS")
print("=" * 80)

society_prompt_rows = []

prompt_pairs = list(
    combinations(
        PROMPT_ORDER,
        2,
    )
)


for num_llm in [1, 2]:

    for rule_policy in RULE_ORDER:

        context = mixed.loc[
            (
                mixed["num_llm_agents"]
                == num_llm
            )
            & (
                mixed["rule_policy"]
                == rule_policy
            )
        ]

        family = []

        for prompt_a, prompt_b in prompt_pairs:

            a = (
                context.loc[
                    context["prompt"]
                    == prompt_a,
                    [
                        "seed",
                        "survival_rate",
                    ],
                ]
                .rename(
                    columns={
                        "survival_rate":
                            "a"
                    }
                )
            )

            b = (
                context.loc[
                    context["prompt"]
                    == prompt_b,
                    [
                        "seed",
                        "survival_rate",
                    ],
                ]
                .rename(
                    columns={
                        "survival_rate":
                            "b"
                    }
                )
            )

            paired = a.merge(
                b,
                on="seed",
                validate="one_to_one",
            )

            result = paired_result(
                paired["a"],
                paired["b"],
            )

            result.update({
                "num_llm_agents":
                    num_llm,

                "num_rule_agents":
                    4 - num_llm,

                "rule_policy":
                    rule_policy,

                "prompt_a":
                    prompt_a,

                "prompt_b":
                    prompt_b,
            })

            family.append(
                result
            )

        # Six prompt comparisons form one family
        # within this composition × rule context.
        holm_correct(
            family
        )

        society_prompt_rows.extend(
            family
        )


society_prompt = pd.DataFrame(
    society_prompt_rows
)

society_prompt.to_csv(
    OUTPUT_DIR
    / "society_prompt_posthocs.csv",
    index=False,
)


# =====================================================================
# Society survival — composition comparisons
# =====================================================================

print("\n" + "=" * 80)
print("SOCIETY SURVIVAL — COMPOSITION COMPARISONS")
print("=" * 80)

society_composition_rows = []

for rule_policy in RULE_ORDER:

    for prompt in PROMPT_ORDER:

        context = mixed.loc[
            (
                mixed["rule_policy"]
                == rule_policy
            )
            & (
                mixed["prompt"]
                == prompt
            )
        ]

        one = (
            context.loc[
                context[
                    "num_llm_agents"
                ] == 1,
                [
                    "seed",
                    "survival_rate",
                ],
            ]
            .rename(
                columns={
                    "survival_rate":
                        "one"
                }
            )
        )

        two = (
            context.loc[
                context[
                    "num_llm_agents"
                ] == 2,
                [
                    "seed",
                    "survival_rate",
                ],
            ]
            .rename(
                columns={
                    "survival_rate":
                        "two"
                }
            )
        )

        paired = one.merge(
            two,
            on="seed",
            validate="one_to_one",
        )

        result = paired_result(
            paired["one"],
            paired["two"],
        )

        result.update({
            "prompt":
                prompt,

            "rule_policy":
                rule_policy,
        })

        society_composition_rows.append(
            result
        )


# Twelve planned composition contrasts = one family.
holm_correct(
    society_composition_rows
)

society_composition = pd.DataFrame(
    society_composition_rows
)

society_composition.to_csv(
    OUTPUT_DIR
    / "society_composition_posthocs.csv",
    index=False,
)


# =====================================================================
# PART B
# LLM-rule survival gap
# =====================================================================

print("\n" + "=" * 80)
print("SURVIVAL GAP — PROMPT COMPARISONS")
print("=" * 80)

gap_prompt_rows = []


for num_llm in [1, 2]:

    for rule_policy in RULE_ORDER:

        context = mixed.loc[
            (
                mixed["num_llm_agents"]
                == num_llm
            )
            & (
                mixed["rule_policy"]
                == rule_policy
            )
        ]

        family = []

        for prompt_a, prompt_b in prompt_pairs:

            a = (
                context.loc[
                    context["prompt"]
                    == prompt_a,
                    [
                        "seed",
                        "llm_rule_survival_gap",
                    ],
                ]
                .rename(
                    columns={
                        "llm_rule_survival_gap":
                            "a"
                    }
                )
            )

            b = (
                context.loc[
                    context["prompt"]
                    == prompt_b,
                    [
                        "seed",
                        "llm_rule_survival_gap",
                    ],
                ]
                .rename(
                    columns={
                        "llm_rule_survival_gap":
                            "b"
                    }
                )
            )

            paired = a.merge(
                b,
                on="seed",
                validate="one_to_one",
            )

            result = paired_result(
                paired["a"],
                paired["b"],
            )

            result.update({
                "num_llm_agents":
                    num_llm,

                "num_rule_agents":
                    4 - num_llm,

                "rule_policy":
                    rule_policy,

                "prompt_a":
                    prompt_a,

                "prompt_b":
                    prompt_b,
            })

            family.append(
                result
            )

        holm_correct(
            family
        )

        gap_prompt_rows.extend(
            family
        )


gap_prompt = pd.DataFrame(
    gap_prompt_rows
)

gap_prompt.to_csv(
    OUTPUT_DIR
    / "survival_gap_prompt_posthocs.csv",
    index=False,
)


# =====================================================================
# Survival gap against zero
# =====================================================================

print("\n" + "=" * 80)
print("SURVIVAL GAP — TESTS AGAINST ZERO")
print("=" * 80)

zero_gap_rows = []


for num_llm in [1, 2]:

    family = []

    composition = mixed.loc[
        mixed["num_llm_agents"]
        == num_llm
    ]

    for rule_policy in RULE_ORDER:

        for prompt in PROMPT_ORDER:

            condition = composition.loc[
                (
                    composition["rule_policy"]
                    == rule_policy
                )
                & (
                    composition["prompt"]
                    == prompt
                )
            ].sort_values(
                "seed"
            )

            gaps = condition[
                "llm_rule_survival_gap"
            ].to_numpy()

            row = {
                "num_llm_agents":
                    num_llm,

                "num_rule_agents":
                    4 - num_llm,

                "prompt":
                    prompt,

                "rule_policy":
                    rule_policy,

                "n":
                    len(gaps),

                "mean_gap":
                    gaps.mean(),

                "cohen_dz":
                    cohen_dz(
                        gaps
                    ),

                "permutation_p_raw":
                    exact_sign_flip_test(
                        gaps
                    ),
            }

            family.append(
                row
            )

    # Twelve condition-vs-zero tests within
    # each population composition.
    holm_correct(
        family
    )

    zero_gap_rows.extend(
        family
    )


zero_gap = pd.DataFrame(
    zero_gap_rows
)

zero_gap.to_csv(
    OUTPUT_DIR
    / "survival_gap_zero_tests.csv",
    index=False,
)


# =====================================================================
# Print significant results
# =====================================================================

def print_significant(
    title,
    dataframe,
    columns,
):

    print(
        "\n"
        + "=" * 80
    )

    print(title)

    print(
        "=" * 80
    )

    significant = dataframe.loc[
        dataframe[
            "significant_holm"
        ]
    ]

    if significant.empty:

        print("None")

        return

    print(
        significant[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


print_significant(
    "SIGNIFICANT SOCIETY SURVIVAL PROMPT COMPARISONS",
    society_prompt,
    [
        "num_llm_agents",
        "rule_policy",
        "prompt_a",
        "prompt_b",
        "mean_a",
        "mean_b",
        "mean_difference",
        "cohen_dz",
        "permutation_p_holm",
    ],
)


print_significant(
    "SIGNIFICANT SOCIETY SURVIVAL COMPOSITION COMPARISONS",
    society_composition,
    [
        "prompt",
        "rule_policy",
        "mean_a",
        "mean_b",
        "mean_difference",
        "cohen_dz",
        "permutation_p_holm",
    ],
)


print_significant(
    "SIGNIFICANT SURVIVAL-GAP PROMPT COMPARISONS",
    gap_prompt,
    [
        "num_llm_agents",
        "rule_policy",
        "prompt_a",
        "prompt_b",
        "mean_a",
        "mean_b",
        "mean_difference",
        "cohen_dz",
        "permutation_p_holm",
    ],
)


print_significant(
    "SIGNIFICANT SURVIVAL GAPS AGAINST ZERO",
    zero_gap,
    [
        "num_llm_agents",
        "prompt",
        "rule_policy",
        "mean_gap",
        "cohen_dz",
        "permutation_p_holm",
    ],
)


print(
    "\nSaved survival analyses to:",
    OUTPUT_DIR
)
