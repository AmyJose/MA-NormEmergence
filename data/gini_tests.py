from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# =====================================================================
# Configuration
# =====================================================================

DATA = Path("data/analysis/exp2/run_metrics.csv")

OUTCOME = "gini_final_wellbeing"


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
print("GINI OF FINAL WELLBEING")
print("=" * 80)

print("\nOVERALL DISTRIBUTION")

print(
    mixed[OUTCOME]
    .describe()
    .to_string()
)

print(
    "\nExact zeros:",
    int((mixed[OUTCOME] == 0).sum()),
    "/",
    len(mixed),
)

print(
    "Proportion zero:",
    (mixed[OUTCOME] == 0).mean(),
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
        zeros=lambda x: (x == 0).sum(),
    )
)

print(
    by_composition.to_string(
        float_format=lambda x: f"{x:.4f}"
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
        zeros=lambda x: (x == 0).sum(),
    )
    .reset_index()
)

print(
    by_condition.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
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
# 5. Factorial OLS diagnostic model
# =====================================================================
#
# This does NOT yet mean that OLS/ANOVA is the final test.
# We fit the model here only so that we can inspect the
# residual distribution and variance structure.
#
# Same model structure used for total final wellbeing:
#
# outcome ~ prompt * rule_policy * composition + seed
#

print("\n" + "=" * 80)
print("FACTORIAL MODEL DIAGNOSTICS")
print("=" * 80)

import statsmodels.formula.api as smf

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


# ---------------------------------------------------------------------
# Shapiro-Wilk test of residuals
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Brown-Forsythe test
# ---------------------------------------------------------------------
#
# Brown-Forsythe is Levene's test centred on the median.
# Each prompt × rule × composition combination is treated
# as a group.
#

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
# 6. Residual summary
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
# 7. Boundary observations
# =====================================================================

print("\n" + "=" * 80)
print("BOUNDARY / EXTREME OBSERVATIONS")
print("=" * 80)

print(
    "Gini = 0:",
    int(
        np.isclose(
            mixed[OUTCOME],
            0,
        ).sum()
    ),
)

print(
    "Gini >= 0.70:",
    int(
        (
            mixed[OUTCOME]
            >= 0.70
        ).sum()
    ),
)

print(
    "Gini >= 0.75:",
    int(
        (
            mixed[OUTCOME]
            >= 0.75
        ).sum()
    ),
)


# =====================================================================
# 8. Relationship with survival
# =====================================================================
#
# This is descriptive only.
#
# Death sets final wellbeing to zero, so it is useful to see how
# strongly Gini is associated with the number of survivors.
#

print("\n" + "=" * 80)
print("GINI BY NUMBER OF SURVIVORS")
print("=" * 80)

gini_by_survivors = (
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
    )
    .reset_index()
)

print(
    gini_by_survivors.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

# =====================================================================
# 9. Factorial ANOVA
# =====================================================================

from statsmodels.stats.anova import anova_lm


print("\n" + "=" * 80)
print("BLOCKED FACTORIAL ANOVA")
print("=" * 80)


def partial_eta_squared(anova_table):
    """
    Add partial eta squared to an ANOVA table.

    eta_p^2 = SS_effect / (SS_effect + SS_residual)
    """

    result = anova_table.copy()

    residual_ss = result.loc[
        "Residual",
        "sum_sq",
    ]

    result["partial_eta_squared"] = np.nan

    for index in result.index:

        if index == "Residual":
            continue

        effect_ss = result.loc[
            index,
            "sum_sq",
        ]

        result.loc[
            index,
            "partial_eta_squared",
        ] = (
            effect_ss
            / (
                effect_ss
                + residual_ss
            )
        )

    return result


# ---------------------------------------------------------------------
# 9a. 1:3 population
# ---------------------------------------------------------------------

one_llm = mixed.loc[
    mixed["num_llm_agents"] == 1
].copy()

model_1_3 = smf.ols(
    (
        f"{OUTCOME} ~ "
        "C(prompt) * C(rule_policy) "
        "+ C(seed)"
    ),
    data=one_llm,
).fit()

anova_1_3 = anova_lm(
    model_1_3,
    typ=2,
)

anova_1_3 = partial_eta_squared(
    anova_1_3
)

print("\n1:3 POPULATION")
print(
    anova_1_3.to_string()
)


# ---------------------------------------------------------------------
# 9b. 2:2 population
# ---------------------------------------------------------------------

two_llm = mixed.loc[
    mixed["num_llm_agents"] == 2
].copy()

model_2_2 = smf.ols(
    (
        f"{OUTCOME} ~ "
        "C(prompt) * C(rule_policy) "
        "+ C(seed)"
    ),
    data=two_llm,
).fit()

anova_2_2 = anova_lm(
    model_2_2,
    typ=2,
)

anova_2_2 = partial_eta_squared(
    anova_2_2
)

print("\n2:2 POPULATION")
print(
    anova_2_2.to_string()
)


# ---------------------------------------------------------------------
# 9c. Combined population model
# ---------------------------------------------------------------------

model_combined = smf.ols(
    (
        f"{OUTCOME} ~ "
        "C(prompt) * "
        "C(rule_policy) * "
        "C(num_llm_agents) "
        "+ C(seed)"
    ),
    data=mixed,
).fit()

anova_combined = anova_lm(
    model_combined,
    typ=2,
)

anova_combined = partial_eta_squared(
    anova_combined
)

print("\nCOMBINED POPULATIONS")
print(
    anova_combined.to_string()
)


# =====================================================================
# 10. Save omnibus results
# =====================================================================

OUTPUT_DIR = Path(
    "data/analysis/exp2/statistics/gini"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_anova(
    table,
    filename,
    analysis,
):
    output = (
        table
        .reset_index()
        .rename(
            columns={
                "index": "effect"
            }
        )
    )

    output.insert(
        0,
        "analysis",
        analysis,
    )

    output.to_csv(
        OUTPUT_DIR / filename,
        index=False,
    )


save_anova(
    anova_1_3,
    "omnibus_1_3.csv",
    "1:3",
)

save_anova(
    anova_2_2,
    "omnibus_2_2.csv",
    "2:2",
)

save_anova(
    anova_combined,
    "omnibus_combined.csv",
    "combined",
)

# =====================================================================
# 11. Post-hoc helpers
# =====================================================================

from itertools import combinations
from statsmodels.stats.multitest import multipletests


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


def cohen_dz(differences):
    """
    Cohen's dz for paired observations.
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

    return differences.mean() / sd


def exact_sign_flip_test(differences):
    """
    Exact two-sided paired sign-flip permutation test.

    Seed is the paired experimental unit.
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


def paired_statistics(
    a,
    b,
):
    """
    Calculate paired statistics for two seed-aligned series.

    Difference is defined as A - B.
    """

    differences = a - b

    t_result = stats.ttest_rel(
        a,
        b,
    )

    return {
        "n": len(differences),

        "mean_a": a.mean(),
        "mean_b": b.mean(),

        "mean_difference":
            differences.mean(),

        "median_difference":
            differences.median(),

        "t_statistic":
            t_result.statistic,

        "t_p_raw":
            t_result.pvalue,

        "cohen_dz":
            cohen_dz(
                differences
            ),

        "permutation_p_raw":
            exact_sign_flip_test(
                differences
            ),
    }


# =====================================================================
# 12. Prompt post-hocs
# =====================================================================

print("\n" + "=" * 80)
print("PROMPT POST-HOC COMPARISONS")
print("=" * 80)

prompt_pairs = list(
    combinations(
        PROMPT_ORDER,
        2,
    )
)

prompt_rows = []

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
                    ["seed", OUTCOME],
                ]
                .rename(
                    columns={
                        OUTCOME: "a"
                    }
                )
            )

            b = (
                context.loc[
                    context["prompt"]
                    == prompt_b,
                    ["seed", OUTCOME],
                ]
                .rename(
                    columns={
                        OUTCOME: "b"
                    }
                )
            )

            paired = a.merge(
                b,
                on="seed",
                how="inner",
                validate="one_to_one",
            )

            result = paired_statistics(
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

        # Holm correction separately for the six
        # planned prompt comparisons in this context.
        _, t_holm, _, _ = multipletests(
            [
                row["t_p_raw"]
                for row in family
            ],
            alpha=0.05,
            method="holm",
        )

        _, permutation_holm, _, _ = (
            multipletests(
                [
                    row[
                        "permutation_p_raw"
                    ]
                    for row in family
                ],
                alpha=0.05,
                method="holm",
            )
        )

        for (
            row,
            t_corrected,
            permutation_corrected,
        ) in zip(
            family,
            t_holm,
            permutation_holm,
        ):
            row["t_p_holm"] = (
                t_corrected
            )

            row[
                "permutation_p_holm"
            ] = permutation_corrected

            row[
                "t_significant_holm"
            ] = (
                t_corrected < 0.05
            )

            row[
                "permutation_significant_holm"
            ] = (
                permutation_corrected
                < 0.05
            )

            prompt_rows.append(
                row
            )


prompt_posthocs = pd.DataFrame(
    prompt_rows
)

prompt_posthocs.to_csv(
    OUTPUT_DIR / "prompt_posthocs.csv",
    index=False,
)


# =====================================================================
# 13. Composition post-hocs
# =====================================================================

print("\n" + "=" * 80)
print("COMPOSITION POST-HOC COMPARISONS")
print("=" * 80)

composition_rows = []

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

        one_llm = (
            context.loc[
                context["num_llm_agents"]
                == 1,
                ["seed", OUTCOME],
            ]
            .rename(
                columns={
                    OUTCOME: "gini_1_3"
                }
            )
        )

        two_llm = (
            context.loc[
                context["num_llm_agents"]
                == 2,
                ["seed", OUTCOME],
            ]
            .rename(
                columns={
                    OUTCOME: "gini_2_2"
                }
            )
        )

        paired = one_llm.merge(
            two_llm,
            on="seed",
            how="inner",
            validate="one_to_one",
        )

        result = paired_statistics(
            paired["gini_1_3"],
            paired["gini_2_2"],
        )

        result.update({
            "prompt":
                prompt,

            "rule_policy":
                rule_policy,

            "mean_1_3":
                result.pop(
                    "mean_a"
                ),

            "mean_2_2":
                result.pop(
                    "mean_b"
                ),
        })

        # Rename difference so its direction is explicit.
        result[
            "mean_difference_1_3_minus_2_2"
        ] = result.pop(
            "mean_difference"
        )

        result[
            "median_difference_1_3_minus_2_2"
        ] = result.pop(
            "median_difference"
        )

        composition_rows.append(
            result
        )


# All 12 planned composition comparisons form
# one multiplicity family.
_, t_holm, _, _ = multipletests(
    [
        row["t_p_raw"]
        for row in composition_rows
    ],
    alpha=0.05,
    method="holm",
)

_, permutation_holm, _, _ = (
    multipletests(
        [
            row[
                "permutation_p_raw"
            ]
            for row in composition_rows
        ],
        alpha=0.05,
        method="holm",
    )
)

for (
    row,
    t_corrected,
    permutation_corrected,
) in zip(
    composition_rows,
    t_holm,
    permutation_holm,
):
    row[
        "t_p_holm"
    ] = t_corrected

    row[
        "permutation_p_holm"
    ] = permutation_corrected

    row[
        "t_significant_holm"
    ] = (
        t_corrected < 0.05
    )

    row[
        "permutation_significant_holm"
    ] = (
        permutation_corrected
        < 0.05
    )


composition_posthocs = pd.DataFrame(
    composition_rows
)

composition_posthocs.to_csv(
    OUTPUT_DIR
    / "composition_posthocs.csv",
    index=False,
)


# =====================================================================
# 14. Print significant prompt comparisons
# =====================================================================

print("\n" + "=" * 80)
print("SIGNIFICANT PROMPT COMPARISONS — T TEST")
print("=" * 80)

sig_prompt_t = prompt_posthocs.loc[
    prompt_posthocs[
        "t_significant_holm"
    ]
]

if sig_prompt_t.empty:
    print("None")

else:
    print(
        sig_prompt_t[
            [
                "num_llm_agents",
                "rule_policy",
                "prompt_a",
                "prompt_b",
                "mean_a",
                "mean_b",
                "mean_difference",
                "cohen_dz",
                "t_p_holm",
                "permutation_p_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


# =====================================================================
# 15. Robustness disagreements
# =====================================================================

print("\n" + "=" * 80)
print("PROMPT ROBUSTNESS DISAGREEMENTS")
print("=" * 80)

prompt_disagreement = (
    prompt_posthocs.loc[
        prompt_posthocs[
            "t_significant_holm"
        ]
        != prompt_posthocs[
            "permutation_significant_holm"
        ]
    ]
)

if prompt_disagreement.empty:
    print("None")

else:
    print(
        prompt_disagreement[
            [
                "num_llm_agents",
                "rule_policy",
                "prompt_a",
                "prompt_b",
                "mean_difference",
                "cohen_dz",
                "t_p_holm",
                "permutation_p_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


print("\n" + "=" * 80)
print("SIGNIFICANT COMPOSITION COMPARISONS — T TEST")
print("=" * 80)

sig_composition_t = (
    composition_posthocs.loc[
        composition_posthocs[
            "t_significant_holm"
        ]
    ]
)

if sig_composition_t.empty:
    print("None")

else:
    print(
        sig_composition_t[
            [
                "prompt",
                "rule_policy",
                "mean_1_3",
                "mean_2_2",
                "mean_difference_1_3_minus_2_2",
                "cohen_dz",
                "t_p_holm",
                "permutation_p_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


print("\n" + "=" * 80)
print("COMPOSITION ROBUSTNESS DISAGREEMENTS")
print("=" * 80)

composition_disagreement = (
    composition_posthocs.loc[
        composition_posthocs[
            "t_significant_holm"
        ]
        != composition_posthocs[
            "permutation_significant_holm"
        ]
    ]
)

if composition_disagreement.empty:
    print("None")

else:
    print(
        composition_disagreement[
            [
                "prompt",
                "rule_policy",
                "mean_1_3",
                "mean_2_2",
                "cohen_dz",
                "t_p_holm",
                "permutation_p_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


print("\nSaved:")
print(
    OUTPUT_DIR / "prompt_posthocs.csv"
)
print(
    OUTPUT_DIR
    / "composition_posthocs.csv"
)


print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(
    OUTPUT_DIR / "omnibus_1_3.csv"
)

print(
    OUTPUT_DIR / "omnibus_2_2.csv"
)

print(
    OUTPUT_DIR / "omnibus_combined.csv"
)