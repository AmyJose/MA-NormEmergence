from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests


# =====================================================================
# Configuration
# =====================================================================

DATA = Path("data/analysis/exp2/run_metrics.csv")

OUTPUT_DIR = Path(
    "data/analysis/exp2/statistics/subgroup_wellbeing"
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

def partial_eta_squared(anova):
    residual_ss = anova.loc[
        "Residual",
        "sum_sq",
    ]

    values = []

    for index, row in anova.iterrows():
        if index == "Residual":
            values.append(np.nan)
        else:
            values.append(
                row["sum_sq"]
                / (
                    row["sum_sq"]
                    + residual_ss
                )
            )

    return values


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

    return (
        differences.mean()
        / sd
    )


def exact_sign_flip_test(differences):
    """
    Exact two-sided paired permutation test.

    Under the null hypothesis, the sign of each paired
    difference is exchangeable.

    With 10 seeds there are only 2^10 = 1024 possible
    sign assignments, so the exact distribution can be
    enumerated.
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
            1 if (pattern >> i) & 1 else -1
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
    Compare two prompts within the same rule policy and
    composition, pairing observations by seed.
    """

    subset = population.loc[
        population["rule_policy"]
        == rule_policy
    ]

    a = (
        subset.loc[
            subset["prompt"] == prompt_a,
            ["seed", "llm_rule_wellbeing_gap"],
        ]
        .rename(
            columns={
                "llm_rule_wellbeing_gap":
                    "value_a"
            }
        )
    )

    b = (
        subset.loc[
            subset["prompt"] == prompt_b,
            ["seed", "llm_rule_wellbeing_gap"],
        ]
        .rename(
            columns={
                "llm_rule_wellbeing_gap":
                    "value_b"
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

    test = stats.ttest_rel(
        paired["value_a"],
        paired["value_b"],
    )

    return {
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "n": len(paired),

        "mean_a": paired[
            "value_a"
        ].mean(),

        "mean_b": paired[
            "value_b"
        ].mean(),

        "mean_difference": (
            differences.mean()
        ),

        "t": test.statistic,
        "p_raw": test.pvalue,

        "cohen_dz": cohen_dz(
            differences
        ),

        "permutation_p_raw":
            exact_sign_flip_test(
                differences
            ),
    }


# =====================================================================
# 1. Composition-specific omnibus models
# =====================================================================

omnibus_rows = []
diagnostic_rows = []

for num_llm in [1, 2]:

    num_rule = 4 - num_llm

    population = mixed.loc[
        mixed["num_llm_agents"]
        == num_llm
    ].copy()

    print("\n" + "=" * 80)
    print(
        f"LLM–RULE WELLBEING GAP: "
        f"{num_llm}:{num_rule}"
    )
    print("=" * 80)

    model = ols(
        "llm_rule_wellbeing_gap ~ "
        "C(prompt) * C(rule_policy) "
        "+ C(seed)",
        data=population,
    ).fit()

    anova = sm.stats.anova_lm(
        model,
        typ=2,
    )

    anova[
        "partial_eta_sq"
    ] = partial_eta_squared(
        anova
    )

    print("\nANOVA")
    print(
        anova.to_string()
    )

    for effect, row in anova.iterrows():
        omnibus_rows.append({
            "model": (
                f"{num_llm}:{num_rule}"
            ),
            "effect": effect,
            "sum_sq": row["sum_sq"],
            "df": row["df"],
            "F": row["F"],
            "p": row["PR(>F)"],
            "partial_eta_sq":
                row["partial_eta_sq"],
        })

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------

    residuals = model.resid

    shapiro = stats.shapiro(
        residuals
    )

    groups = [
        group[
            "llm_rule_wellbeing_gap"
        ].values
        for _, group
        in population.groupby(
            ["prompt", "rule_policy"]
        )
    ]

    brown_forsythe = stats.levene(
        *groups,
        center="median",
    )

    diagnostic_rows.append({
        "model":
            f"{num_llm}:{num_rule}",
        "shapiro_W":
            shapiro.statistic,
        "shapiro_p":
            shapiro.pvalue,
        "brown_forsythe_W":
            brown_forsythe.statistic,
        "brown_forsythe_p":
            brown_forsythe.pvalue,
    })

    print("\nDIAGNOSTICS")
    print(
        "Shapiro-Wilk: "
        f"W={shapiro.statistic:.4f}, "
        f"p={shapiro.pvalue:.6f}"
    )

    print(
        "Brown-Forsythe: "
        f"W={brown_forsythe.statistic:.4f}, "
        f"p={brown_forsythe.pvalue:.6f}"
    )


# =====================================================================
# 2. Combined composition model
# =====================================================================

print("\n" + "=" * 80)
print("COMBINED COMPOSITION MODEL")
print("=" * 80)

combined_model = ols(
    "llm_rule_wellbeing_gap ~ "
    "C(prompt) * C(rule_policy) "
    "* C(num_llm_agents) "
    "+ C(seed)",
    data=mixed,
).fit()

combined_anova = sm.stats.anova_lm(
    combined_model,
    typ=2,
)

combined_anova[
    "partial_eta_sq"
] = partial_eta_squared(
    combined_anova
)

print(
    combined_anova.to_string()
)

for effect, row in (
    combined_anova.iterrows()
):
    omnibus_rows.append({
        "model": "combined",
        "effect": effect,
        "sum_sq": row["sum_sq"],
        "df": row["df"],
        "F": row["F"],
        "p": row["PR(>F)"],
        "partial_eta_sq":
            row["partial_eta_sq"],
    })


# =====================================================================
# 3. Prompt post-hoc comparisons
# =====================================================================
#
# Six pairwise prompt comparisons are performed within each
# composition × rule-policy context.
#
# Holm correction is applied separately within each six-test
# family.
# =====================================================================

print("\n" + "=" * 80)
print("PROMPT POST-HOC COMPARISONS")
print("=" * 80)

prompt_posthoc_rows = []

prompt_pairs = list(
    combinations(
        PROMPT_ORDER,
        2,
    )
)

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

        # Holm correction for paired t-tests
        raw_p = [
            row["p_raw"]
            for row in family
        ]

        _, corrected_p, _, _ = (
            multipletests(
                raw_p,
                alpha=0.05,
                method="holm",
            )
        )

        # Holm correction for exact permutation tests
        permutation_p = [
            row[
                "permutation_p_raw"
            ]
            for row in family
        ]

        _, corrected_permutation_p, _, _ = (
            multipletests(
                permutation_p,
                alpha=0.05,
                method="holm",
            )
        )

        for (
            row,
            p_holm,
            permutation_p_holm,
        ) in zip(
            family,
            corrected_p,
            corrected_permutation_p,
        ):
            row["p_holm"] = p_holm

            row[
                "significant_holm"
            ] = p_holm < 0.05

            row[
                "permutation_p_holm"
            ] = permutation_p_holm

            row[
                "significant_permutation_holm"
            ] = (
                permutation_p_holm
                < 0.05
            )

            prompt_posthoc_rows.append(
                row
            )


prompt_posthocs = pd.DataFrame(
    prompt_posthoc_rows
)

for num_llm in [1, 2]:
    for rule_policy in RULE_ORDER:

        subset = prompt_posthocs.loc[
            (
                prompt_posthocs[
                    "num_llm_agents"
                ]
                == num_llm
            )
            & (
                prompt_posthocs[
                    "rule_policy"
                ]
                == rule_policy
            )
        ]

        print(
            "\n"
            f"{num_llm}:{4-num_llm} | "
            f"Rule policy: {rule_policy}"
        )

        print(
            subset[
                [
                    "prompt_a",
                    "prompt_b",
                    "mean_a",
                    "mean_b",
                    "mean_difference",
                    "t",
                    "p_raw",
                    "p_holm",
                    "cohen_dz",
                    "permutation_p_holm",
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.6f}",
            )
        )


# =====================================================================
# 4. Within-condition zero-gap tests
# =====================================================================
#
# H0:
#     mean LLM wellbeing = mean rule-agent wellbeing
#
# Equivalently:
#     mean llm_rule_wellbeing_gap = 0
#
# Since the gap is already calculated within each episode,
# this is a one-sample test of the seed-level gaps against zero.
#
# All 12 conditions within each composition form one Holm family.
# =====================================================================

print("\n" + "=" * 80)
print("WITHIN-CONDITION ZERO-GAP TESTS")
print("=" * 80)

zero_gap_rows = []

for num_llm in [1, 2]:

    family = []

    population = mixed.loc[
        mixed["num_llm_agents"]
        == num_llm
    ].copy()

    for rule_policy in RULE_ORDER:

        for prompt in PROMPT_ORDER:

            condition = population.loc[
                (
                    population["prompt"]
                    == prompt
                )
                & (
                    population[
                        "rule_policy"
                    ]
                    == rule_policy
                ),
                "llm_rule_wellbeing_gap",
            ].dropna()

            values = condition.to_numpy(
                dtype=float
            )

            test = stats.ttest_1samp(
                values,
                popmean=0,
            )

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
                    len(values),
                "mean_gap":
                    values.mean(),
                "sd_gap":
                    values.std(ddof=1),
                "t":
                    test.statistic,
                "p_raw":
                    test.pvalue,
                "cohen_dz":
                    cohen_dz(values),
                "permutation_p_raw":
                    exact_sign_flip_test(
                        values
                    ),
            }

            family.append(
                row
            )

    # -------------------------------------------------------------
    # Holm across all 12 conditions in this composition
    # -------------------------------------------------------------

    raw_p = [
        row["p_raw"]
        for row in family
    ]

    _, corrected_p, _, _ = (
        multipletests(
            raw_p,
            alpha=0.05,
            method="holm",
        )
    )

    permutation_p = [
        row["permutation_p_raw"]
        for row in family
    ]

    _, corrected_permutation_p, _, _ = (
        multipletests(
            permutation_p,
            alpha=0.05,
            method="holm",
        )
    )

    for (
        row,
        p_holm,
        permutation_p_holm,
    ) in zip(
        family,
        corrected_p,
        corrected_permutation_p,
    ):
        row["p_holm"] = p_holm

        row[
            "significant_holm"
        ] = p_holm < 0.05

        row[
            "permutation_p_holm"
        ] = permutation_p_holm

        row[
            "significant_permutation_holm"
        ] = (
            permutation_p_holm
            < 0.05
        )

        zero_gap_rows.append(
            row
        )


zero_gap_tests = pd.DataFrame(
    zero_gap_rows
)

for num_llm in [1, 2]:

    subset = zero_gap_tests.loc[
        zero_gap_tests[
            "num_llm_agents"
        ]
        == num_llm
    ]

    print(
        "\n"
        f"{num_llm}:{4-num_llm}"
    )

    print(
        subset[
            [
                "prompt",
                "rule_policy",
                "mean_gap",
                "sd_gap",
                "t",
                "p_raw",
                "p_holm",
                "cohen_dz",
                "permutation_p_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


# =====================================================================
# 5. Save Appendix-ready outputs
# =====================================================================

omnibus = pd.DataFrame(
    omnibus_rows
)

diagnostics = pd.DataFrame(
    diagnostic_rows
)

omnibus.to_csv(
    OUTPUT_DIR / "omnibus.csv",
    index=False,
)

diagnostics.to_csv(
    OUTPUT_DIR / "diagnostics.csv",
    index=False,
)

prompt_posthocs.to_csv(
    OUTPUT_DIR
    / "prompt_posthocs.csv",
    index=False,
)

zero_gap_tests.to_csv(
    OUTPUT_DIR
    / "zero_gap_tests.csv",
    index=False,
)


# =====================================================================
# 6. Compact significant-results summary
# =====================================================================

print("\n" + "=" * 80)
print("SIGNIFICANT PROMPT POST-HOCS — HOLM")
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
                "mean_difference",
                "p_holm",
                "cohen_dz",
                "permutation_p_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


print("\n" + "=" * 80)
print("SIGNIFICANT ZERO-GAP TESTS — HOLM")
print("=" * 80)

significant_zero = (
    zero_gap_tests.loc[
        zero_gap_tests[
            "significant_holm"
        ]
    ]
)

if significant_zero.empty:
    print("None")
else:
    print(
        significant_zero[
            [
                "num_llm_agents",
                "prompt",
                "rule_policy",
                "mean_gap",
                "p_holm",
                "cohen_dz",
                "permutation_p_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

for filename in [
    "omnibus.csv",
    "diagnostics.csv",
    "prompt_posthocs.csv",
    "zero_gap_tests.csv",
]:
    print(
        OUTPUT_DIR / filename
    )