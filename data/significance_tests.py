"""
Statistical significance testing for Experiment 2.

Input:
    data/analysis/exp2/run_metrics.csv

Analyses:
    1. Factorial ANOVA within each mixed population composition
       - LLM ethical framing
       - rule-based policy
       - framing x rule-policy interaction
       - random seed included as a blocking factor

    2. Factorial ANOVA across mixed population compositions
       - LLM ethical framing
       - rule-based policy
       - population composition
       - all two-way and three-way interactions
       - random seed included as a blocking factor

    3. Post-hoc paired comparisons between LLM framings
       within each composition and rule-policy context
       - paired by random seed
       - paired t-test
       - Holm correction
       - Cohen's dz effect size
       - exact paired permutation robustness test
       - Shapiro-Wilk test of paired differences

    4. Post-hoc paired comparisons between population compositions
       within each prompt x rule-policy condition
       - paired by random seed
       - paired t-test
       - Holm correction
       - Cohen's dz effect size
       - exact paired permutation robustness test
       - Shapiro-Wilk test of paired differences

    5. ANOVA assumption diagnostics
       - Shapiro-Wilk test of model residuals
       - Brown-Forsythe test of homogeneity of variance
       - residual Q-Q plots
       - residual-vs-fitted plots

Outputs:
    data/analysis/exp2/significance_anova.csv
    data/analysis/exp2/significance_anova_within_composition.csv
    data/analysis/exp2/posthoc_prompt_comparisons.csv
    data/analysis/exp2/posthoc_composition_comparisons.csv
    data/analysis/exp2/anova_assumption_checks.csv
    data/analysis/exp2/assumptions_*.png
"""

from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import (
    ttest_rel,
    shapiro,
    levene,
    probplot,
)

from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests


# =====================================================================
# Configuration
# =====================================================================

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--metric",
    choices=(
        "total_final_wellbeing",
        "time_averaged_minimum_wellbeing",
        "time_averaged_gini_wellbeing",
        "llm_action_throw_proportion",
    ),
    default="total_final_wellbeing",
)
args = parser.parse_args()

RESULTS_DIR = Path("data/analysis/exp2_time_averaged")
INPUT_FILE = RESULTS_DIR / "run_metrics.csv"
METRIC = args.metric

# Separate outputs for each metric.
RESULTS_DIR = RESULTS_DIR / "significance" / METRIC
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05

PROMPT_LEVELS = [
    "unframed",
    "self_interested",
    "cooperative",
    "altruistic",
]


# =====================================================================
# Helper functions
# =====================================================================

def partial_eta_squared(anova_table):
    """
    Calculate partial eta-squared for every effect in an ANOVA table.

    partial eta^2 =
        SS_effect / (SS_effect + SS_error)

    where SS = sum of squares.
    """

    ss_error = anova_table.loc["Residual", "sum_sq"]

    result = anova_table.copy()

    result["partial_eta_sq"] = (
        result["sum_sq"]
        / (result["sum_sq"] + ss_error)
    )

    # Effect size is not meaningful for the residual row.
    result.loc["Residual", "partial_eta_sq"] = np.nan

    return result


def print_anova(title, table):
    """
    Print an ANOVA table in a cleaner format.
    """

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    display = table[
        [
            "sum_sq",
            "df",
            "F",
            "PR(>F)",
            "partial_eta_sq",
        ]
    ].copy()

    display = display.rename(
        columns={
            "sum_sq": "SS",
            "PR(>F)": "p",
        }
    )

    print(
        display.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )


def cohens_dz(x, y):
    """
    Calculate Cohen's dz for paired samples.

    dz =
        mean(paired differences)
        ------------------------
        SD(paired differences)

    Positive dz:
        x tends to be larger than y.

    Negative dz:
        x tends to be smaller than y.
    """

    differences = np.asarray(x) - np.asarray(y)

    sd_difference = np.std(
        differences,
        ddof=1,
    )

    if sd_difference == 0:
        return np.nan

    return np.mean(differences) / sd_difference


def exact_paired_permutation_test(x, y):
    """
    Exact two-sided paired permutation test.

    Under the null hypothesis, the signs of the paired differences
    are exchangeable.

    For n paired observations there are 2^n possible sign patterns.

    With 10 seeds:
        2^10 = 1024

    so the exact distribution can be enumerated directly.

    Test statistic:
        absolute mean paired difference

    Returns:
        exact two-sided permutation p-value
    """

    differences = (
        np.asarray(x, dtype=float)
        - np.asarray(y, dtype=float)
    )

    n = len(differences)

    observed = abs(
        np.mean(differences)
    )

    permutation_statistics = []

    # Enumerate every possible combination of sign flips.
    for mask in range(2 ** n):

        signs = np.array(
            [
                1 if (mask >> i) & 1 else -1
                for i in range(n)
            ]
        )

        permuted_differences = (
            differences * signs
        )

        statistic = abs(
            np.mean(permuted_differences)
        )

        permutation_statistics.append(
            statistic
        )

    permutation_statistics = np.asarray(
        permutation_statistics
    )

    # Exact proportion of permutation statistics that are at
    # least as extreme as the observed statistic.
    p_value = np.mean(
        permutation_statistics
        >= observed - 1e-12
    )

    return p_value


def paired_comparison(
    data,
    group_col,
    group_a,
    group_b,
    metric,
):
    """
    Perform paired statistical comparisons between two conditions.

    Runs are paired using random seed.

    Includes:
        - paired t-test
        - exact paired permutation test
        - Shapiro-Wilk test of paired differences
        - Cohen's dz

    Mean difference is always:

        group_a - group_b

    Therefore:
        positive -> group A is higher
        negative -> group B is higher
    """

    a = (
        data.loc[
            data[group_col] == group_a,
            ["seed", metric],
        ]
        .rename(columns={metric: "a"})
    )

    b = (
        data.loc[
            data[group_col] == group_b,
            ["seed", metric],
        ]
        .rename(columns={metric: "b"})
    )

    # Keep only seeds that occur in BOTH conditions.
    #
    # validate="one_to_one" ensures that each seed contributes
    # exactly one observation per condition.
    paired = a.merge(
        b,
        on="seed",
        how="inner",
        validate="one_to_one",
    )

    if len(paired) < 2:
        raise ValueError(
            f"Not enough paired observations for "
            f"{group_a} vs {group_b}. "
            f"Found {len(paired)} matching seeds."
        )

    differences = (
        paired["a"] - paired["b"]
    )

    # -------------------------------------------------------------
    # Paired t-test
    # -------------------------------------------------------------

    t_test = ttest_rel(
        paired["a"],
        paired["b"],
    )

    # -------------------------------------------------------------
    # Exact paired permutation test
    # -------------------------------------------------------------

    permutation_p = exact_paired_permutation_test(
        paired["a"].to_numpy(),
        paired["b"].to_numpy(),
    )

    # -------------------------------------------------------------
    # Normality of paired differences
    #
    # This is the relevant normality assumption for the paired
    # t-test.
    # -------------------------------------------------------------

    shapiro_result = shapiro(
        differences
    )

    return {
        "group_a": group_a,
        "group_b": group_b,

        "n_pairs": len(paired),

        "mean_a": paired["a"].mean(),
        "mean_b": paired["b"].mean(),

        "mean_difference": (
            differences.mean()
        ),

        "t": t_test.statistic,
        "p_raw": t_test.pvalue,

        "p_permutation_raw": (
            permutation_p
        ),

        "difference_shapiro_W": (
            shapiro_result.statistic
        ),

        "difference_shapiro_p": (
            shapiro_result.pvalue
        ),

        "cohens_dz": cohens_dz(
            paired["a"],
            paired["b"],
        ),
    }


def add_holm_correction(results):
    """
    Apply Holm multiple-comparison correction to a family of tests.

    Corrects both:
        - paired t-test p-values
        - exact permutation p-values

    Adds:
        p_holm
        significant_holm

        p_permutation_holm
        significant_permutation_holm
    """

    if len(results) == 0:
        return results

    # -------------------------------------------------------------
    # Paired t-test Holm correction
    # -------------------------------------------------------------

    raw_p_values = [
        result["p_raw"]
        for result in results
    ]

    (
        reject,
        corrected_p_values,
        _,
        _,
    ) = multipletests(
        raw_p_values,
        alpha=ALPHA,
        method="holm",
    )

    for (
        result,
        p_corrected,
        significant,
    ) in zip(
        results,
        corrected_p_values,
        reject,
    ):

        result["p_holm"] = (
            p_corrected
        )

        result["significant_holm"] = (
            bool(significant)
        )

    # -------------------------------------------------------------
    # Exact permutation Holm correction
    # -------------------------------------------------------------

    permutation_p_values = [
        result["p_permutation_raw"]
        for result in results
    ]

    (
        permutation_reject,
        permutation_corrected,
        _,
        _,
    ) = multipletests(
        permutation_p_values,
        alpha=ALPHA,
        method="holm",
    )

    for (
        result,
        p_corrected,
        significant,
    ) in zip(
        results,
        permutation_corrected,
        permutation_reject,
    ):

        result["p_permutation_holm"] = (
            p_corrected
        )

        result[
            "significant_permutation_holm"
        ] = bool(significant)

    return results


def analyse_anova_assumptions(
    model,
    data,
    title,
    output_prefix,
    group_columns,
):
    """
    Run assumption diagnostics for a fitted ANOVA model.

    Checks:
        1. Shapiro-Wilk normality test on model residuals
        2. Brown-Forsythe test for equal variance between cells
        3. Q-Q plot of model residuals
        4. residual-vs-fitted plot

    Brown-Forsythe is implemented using scipy's Levene test
    centred on the median.
    """

    residuals = model.resid
    fitted = model.fittedvalues

    print()
    print("-" * 80)
    print(f"ASSUMPTION CHECKS: {title}")
    print("-" * 80)

    # -------------------------------------------------------------
    # 1. Residual normality
    # -------------------------------------------------------------

    shapiro_result = shapiro(
        residuals
    )

    print(
        "\nResidual normality "
        "(Shapiro-Wilk):"
    )

    print(
        f"    W = "
        f"{shapiro_result.statistic:.6f}"
    )

    print(
        f"    p = "
        f"{shapiro_result.pvalue:.6f}"
    )

    if shapiro_result.pvalue < ALPHA:

        print(
            "    -> Evidence of departure "
            "from normality."
        )

    else:

        print(
            "    -> No significant evidence "
            "of departure from normality."
        )

    # -------------------------------------------------------------
    # 2. Homogeneity of variance
    #
    # Brown-Forsythe = Levene's test centred on the median.
    # -------------------------------------------------------------

    variance_groups = []

    grouped = data.groupby(
        group_columns,
        observed=True,
    )

    for _, group in grouped:

        variance_groups.append(
            group[METRIC].to_numpy()
        )

    bf_result = levene(
        *variance_groups,
        center="median",
    )

    print(
        "\nHomogeneity of variance "
        "(Brown-Forsythe):"
    )

    print(
        f"    statistic = "
        f"{bf_result.statistic:.6f}"
    )

    print(
        f"    p = "
        f"{bf_result.pvalue:.6f}"
    )

    if bf_result.pvalue < ALPHA:

        print(
            "    -> Evidence that variance "
            "differs between experimental cells."
        )

    else:

        print(
            "    -> No significant evidence "
            "of unequal variance between "
            "experimental cells."
        )

    # -------------------------------------------------------------
    # 3. Q-Q plot
    # -------------------------------------------------------------

    plt.figure(
        figsize=(6, 6)
    )

    probplot(
        residuals,
        dist="norm",
        plot=plt,
    )

    plt.title(
        f"Residual Q-Q Plot\n{title}"
    )

    plt.tight_layout()

    qq_file = (
        RESULTS_DIR
        / f"{output_prefix}_qq.png"
    )

    plt.savefig(
        qq_file,
        dpi=300,
    )

    plt.close()

    # -------------------------------------------------------------
    # 4. Residual-vs-fitted plot
    # -------------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        fitted,
        residuals,
        alpha=0.7,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(f"Fitted {METRIC}")

    plt.ylabel(
        "Residual"
    )

    plt.title(
        f"Residuals vs Fitted\n{title}"
    )

    plt.tight_layout()

    residual_file = (
        RESULTS_DIR
        / (
            f"{output_prefix}"
            "_residuals_vs_fitted.png"
        )
    )

    plt.savefig(
        residual_file,
        dpi=300,
    )

    plt.close()

    print(
        f"\nQ-Q plot saved to: "
        f"{qq_file}"
    )

    print(
        "Residual-vs-fitted plot saved to: "
        f"{residual_file}"
    )

    return {
        "shapiro_W": (
            shapiro_result.statistic
        ),

        "shapiro_p": (
            shapiro_result.pvalue
        ),

        "brown_forsythe_statistic": (
            bf_result.statistic
        ),

        "brown_forsythe_p": (
            bf_result.pvalue
        ),
    }


# =====================================================================
# Load data
# =====================================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"{INPUT_FILE} does not exist.\n"
        "Run data/summarise_results.py first."
    )


runs = pd.read_csv(
    INPUT_FILE
)


print(
    f"Loaded {len(runs)} completed runs "
    f"from {INPUT_FILE}"
)


# =====================================================================
# Keep mixed populations
# =====================================================================

# Rule-only populations have no LLM framing.
#
# All-LLM populations have no rule policy.
#
# Therefore neither boundary population belongs in the factorial
# framing x rule-policy analyses below.

mixed = runs.loc[
    (runs["num_llm_agents"] > 0)
    &
    (runs["num_rule_agents"] > 0)
].copy()


# Make experimental variables explicitly categorical.

mixed["prompt"] = (
    mixed["prompt"].astype("category")
)

mixed["rule_policy"] = (
    mixed["rule_policy"].astype("category")
)

mixed["seed"] = (
    mixed["seed"].astype("category")
)

mixed["num_llm_agents"] = (
    mixed["num_llm_agents"].astype("category")
)


# =====================================================================
# Basic sanity checks
# =====================================================================

print(
    "\nMixed-population runs:"
)

print(
    mixed.groupby(
        [
            "num_llm_agents",
            "num_rule_agents",
        ],
        observed=True,
    ).size()
)


print(
    "\nRuns per experimental condition:"
)

print(
    mixed.groupby(
        [
            "num_llm_agents",
            "prompt",
            "rule_policy",
        ],
        observed=True,
    ).size()
)


# =====================================================================
# PART 1
# Factorial ANOVA within each population composition
# =====================================================================

print("\n\n")
print("#" * 80)
print("WITHIN-COMPOSITION ANALYSES")
print("#" * 80)


within_anova_results = []
assumption_results = []


for num_llm in sorted(
    mixed["num_llm_agents"].cat.categories
):

    num_llm = int(num_llm)
    num_rule = 4 - num_llm

    if num_llm == 0 or num_rule == 0:
        continue

    population = mixed.loc[
        mixed["num_llm_agents"]
        == num_llm
    ].copy()

    print(
        f"\nPopulation: "
        f"{num_llm} LLM + "
        f"{num_rule} rule"
    )

    print(
        f"Number of episodes: "
        f"{len(population)}"
    )

    # -------------------------------------------------------------
    # Factorial model
    #
    # C(prompt) * C(rule_policy)
    #
    # expands to:
    #
    #     C(prompt)
    #     C(rule_policy)
    #     C(prompt):C(rule_policy)
    #
    # C(seed) is included as a blocking factor because the same
    # random seeds are used across experimental conditions.
    # -------------------------------------------------------------

    model = ols(
        f"{METRIC} ~ "
        "C(prompt) * C(rule_policy) "
        "+ C(seed)",
        data=population,
    ).fit()

    anova = sm.stats.anova_lm(
        model,
        typ=2,
    )

    anova = partial_eta_squared(
        anova
    )

    print_anova(
        (
            f"{METRIC}: "
            f"{num_llm} LLM + "
            f"{num_rule} rule"
        ),
        anova,
    )

    # Save ANOVA results.

    stored = (
        anova
        .reset_index()
        .rename(
            columns={
                "index": "effect"
            }
        )
    )

    stored[
        "num_llm_agents"
    ] = num_llm

    stored[
        "num_rule_agents"
    ] = num_rule

    within_anova_results.append(
        stored
    )

    # -------------------------------------------------------------
    # Assumption checks for this composition
    # -------------------------------------------------------------

    assumptions = (
        analyse_anova_assumptions(
            model=model,
            data=population,
            title=(
                f"{num_llm} LLM + "
                f"{num_rule} rule"
            ),
            output_prefix=(
                f"assumptions_"
                f"{num_llm}_{num_rule}"
            ),
            group_columns=[
                "prompt",
                "rule_policy",
            ],
        )
    )

    assumptions[
        "population"
    ] = f"{num_llm}:{num_rule}"

    assumptions[
        "num_llm_agents"
    ] = num_llm

    assumptions[
        "num_rule_agents"
    ] = num_rule

    assumption_results.append(
        assumptions
    )


# =====================================================================
# PART 2
# Factorial ANOVA across population compositions
# =====================================================================

print("\n\n")
print("#" * 80)
print("BETWEEN-COMPOSITION ANALYSIS")
print("#" * 80)


# This model tests:
#
# MAIN EFFECTS:
#     prompt
#     rule policy
#     population composition
#
# TWO-WAY INTERACTIONS:
#     prompt x rule
#     prompt x composition
#     rule x composition
#
# THREE-WAY INTERACTION:
#     prompt x rule x composition
#
# Seed is again included as a blocking factor.

composition_model = ols(
    f"{METRIC} ~ "
    "C(prompt) * "
    "C(rule_policy) * "
    "C(num_llm_agents) "
    "+ C(seed)",
    data=mixed,
).fit()


composition_anova = (
    sm.stats.anova_lm(
        composition_model,
        typ=2,
    )
)


composition_anova = (
    partial_eta_squared(
        composition_anova
    )
)


print_anova(
    (
        f"{METRIC}: "
        "all mixed population compositions"
    ),
    composition_anova,
)


# ---------------------------------------------------------------------
# Assumption checks for combined composition model
# ---------------------------------------------------------------------

combined_assumptions = (
    analyse_anova_assumptions(
        model=composition_model,
        data=mixed,
        title=(
            "All mixed population "
            "compositions"
        ),
        output_prefix=(
            "assumptions_all_mixed"
        ),
        group_columns=[
            "prompt",
            "rule_policy",
            "num_llm_agents",
        ],
    )
)


combined_assumptions[
    "population"
] = "all_mixed"

combined_assumptions[
    "num_llm_agents"
] = "all"

combined_assumptions[
    "num_rule_agents"
] = "mixed"

assumption_results.append(
    combined_assumptions
)


# =====================================================================
# Save ANOVA results
# =====================================================================

composition_anova_output = (
    RESULTS_DIR
    / "significance_anova.csv"
)

composition_anova.to_csv(
    composition_anova_output
)


if within_anova_results:

    within_anova_df = pd.concat(
        within_anova_results,
        ignore_index=True,
    )

    within_anova_output = (
        RESULTS_DIR
        / (
            "significance_anova_"
            "within_composition.csv"
        )
    )

    within_anova_df.to_csv(
        within_anova_output,
        index=False,
    )

else:

    within_anova_output = None


print(
    f"\nCombined ANOVA saved to: "
    f"{composition_anova_output}"
)

if within_anova_output is not None:

    print(
        "Within-composition ANOVAs "
        f"saved to: {within_anova_output}"
    )


# =====================================================================
# Save assumption checks
# =====================================================================

assumption_df = pd.DataFrame(
    assumption_results
)


assumption_output = (
    RESULTS_DIR
    / "anova_assumption_checks.csv"
)


assumption_df.to_csv(
    assumption_output,
    index=False,
)


print(
    "ANOVA assumption checks saved to: "
    f"{assumption_output}"
)


# =====================================================================
# PART 3
# Post-hoc comparisons between LLM framings
# =====================================================================

print("\n\n")
print("#" * 80)
print(
    "POST-HOC: "
    "LLM FRAMING COMPARISONS"
)
print("#" * 80)


prompt_results = []


for num_llm in sorted(
    mixed["num_llm_agents"].cat.categories
):

    num_llm = int(num_llm)
    num_rule = 4 - num_llm

    if num_llm == 0 or num_rule == 0:
        continue

    population = mixed.loc[
        mixed["num_llm_agents"]
        == num_llm
    ].copy()

    for rule_policy in sorted(
        population[
            "rule_policy"
        ].dropna().unique()
    ):

        context = population.loc[
            population["rule_policy"]
            == rule_policy
        ].copy()

        # Four prompt levels produce six unique comparisons.

        comparisons = list(
            combinations(
                PROMPT_LEVELS,
                2,
            )
        )

        family_results = []

        for (
            prompt_a,
            prompt_b,
        ) in comparisons:

            result = paired_comparison(
                data=context,
                group_col="prompt",
                group_a=prompt_a,
                group_b=prompt_b,
                metric=METRIC,
            )

            result[
                "num_llm_agents"
            ] = num_llm

            result[
                "num_rule_agents"
            ] = num_rule

            result[
                "rule_policy"
            ] = rule_policy

            family_results.append(
                result
            )

        # Holm correction across the six prompt comparisons
        # within this composition x rule-policy context.

        family_results = (
            add_holm_correction(
                family_results
            )
        )

        prompt_results.extend(
            family_results
        )


prompt_posthoc = pd.DataFrame(
    prompt_results
)


prompt_posthoc = prompt_posthoc[
    [
        "num_llm_agents",
        "num_rule_agents",
        "rule_policy",

        "group_a",
        "group_b",

        "n_pairs",

        "mean_a",
        "mean_b",
        "mean_difference",

        "t",

        "p_raw",
        "p_holm",
        "significant_holm",

        "p_permutation_raw",
        "p_permutation_holm",
        "significant_permutation_holm",

        "difference_shapiro_W",
        "difference_shapiro_p",

        "cohens_dz",
    ]
]


print(
    prompt_posthoc.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


prompt_output = (
    RESULTS_DIR
    / "posthoc_prompt_comparisons.csv"
)


prompt_posthoc.to_csv(
    prompt_output,
    index=False,
)


print(
    "\nPrompt post-hoc results "
    f"saved to: {prompt_output}"
)


# =====================================================================
# PART 4
# Post-hoc comparisons between population compositions
# =====================================================================

print("\n\n")
print("#" * 80)
print(
    "POST-HOC: "
    "POPULATION COMPOSITION COMPARISONS"
)
print("#" * 80)


composition_results = []


# At present:
#
#     group A = 1 LLM + 3 rule
#     group B = 2 LLM + 2 rule
#
# Therefore:
#
#     mean_difference = 1:3 - 2:2
#
# Negative:
#     welfare is higher in 2:2
#
# Positive:
#     welfare is higher in 1:3


composition_pairs = list(combinations((1, 2, 3), 2))

for prompt in PROMPT_LEVELS:
    for rule_policy in sorted(mixed["rule_policy"].dropna().unique()):
        context = mixed.loc[
            (mixed["prompt"] == prompt)
            & (mixed["rule_policy"] == rule_policy)
        ].copy()

        for composition_a, composition_b in composition_pairs:
            result = paired_comparison(
                data=context,
                group_col="num_llm_agents",
                group_a=composition_a,
                group_b=composition_b,
                metric=METRIC,
            )

            result["prompt"] = prompt
            result["rule_policy"] = rule_policy
            composition_results.append(result)

# 4 framings × 3 rule policies × 3 composition pairs = 36 tests.
# Apply Holm correction across this family for the current metric.

composition_results = (
    add_holm_correction(
        composition_results
    )
)


composition_posthoc = pd.DataFrame(
    composition_results
)


composition_posthoc = composition_posthoc[
    [
        "prompt",
        "rule_policy",

        "group_a",
        "group_b",

        "n_pairs",

        "mean_a",
        "mean_b",
        "mean_difference",

        "t",

        "p_raw",
        "p_holm",
        "significant_holm",

        "p_permutation_raw",
        "p_permutation_holm",
        "significant_permutation_holm",

        "difference_shapiro_W",
        "difference_shapiro_p",

        "cohens_dz",
    ]
]


print(
    composition_posthoc.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


composition_output = (
    RESULTS_DIR
    / (
        "posthoc_composition_"
        "comparisons.csv"
    )
)


composition_posthoc.to_csv(
    composition_output,
    index=False,
)


print(
    "\nComposition post-hoc results "
    f"saved to: {composition_output}"
)


# =====================================================================
# Final summary
# =====================================================================

print("\n\n")
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)


print(
    f"""
Metric analysed:
    {METRIC}

Significance threshold:
    alpha = {ALPHA}

ANOVA output:
    {composition_anova_output}
    {within_anova_output}

Post-hoc output:
    {prompt_output}
    {composition_output}

Assumption checks:
    {assumption_output}

Diagnostic plots are saved in:
    {RESULTS_DIR}

Post-hoc interpretation:

    p_raw
        Uncorrected paired t-test p-value.

    p_holm
        Holm-corrected paired t-test p-value.

    p_permutation_raw
        Exact paired permutation p-value.

    p_permutation_holm
        Holm-corrected exact permutation p-value.

    difference_shapiro_p
        Shapiro-Wilk normality test of the paired
        differences used by the paired t-test.

    cohens_dz
        Paired-sample effect size.

Mean differences are calculated as:

    group_a - group_b

For population-composition comparisons, group_a and group_b
give the number of LLM agents. Differences are group_a minus
group_b. Comparisons cover 1 vs 2, 1 vs 3, and 2 vs 3.
"""
)