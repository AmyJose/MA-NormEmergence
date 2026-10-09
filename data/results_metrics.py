import numpy as np
import pandas as pd


def gini(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("Cannot calculate Gini for an empty population")

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Gini requires finite, nonnegative values")

    if values.sum() == 0:
        return 0.0

    differences = np.abs(
        values[:, None] - values[None, :]
    )

    return float(
        differences.sum()
        / (2 * len(values) * values.sum())
    )

def time_averaged_fairness(agent_reports):
    step_minima = []
    step_ginis = []

    for step, snapshot in agent_reports.groupby("step", sort=True):
        if (
            len(snapshot) != 4
            or snapshot["agent_id"].nunique() != 4
        ):
            raise ValueError(
                f"Expected four unique agents at step {step}"
            )

        dead = (
            snapshot["dead"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

        wellbeing = snapshot["wellbeing"].astype(float).copy()
        wellbeing.loc[dead] = 0.0

        # gini also validates that values are finite and nonnegative.
        step_ginis.append(gini(wellbeing))
        step_minima.append(float(wellbeing.min()))

    if not step_minima:
        raise ValueError("Cannot calculate fairness without agent snapshots")

    return {
        "time_averaged_minimum_wellbeing": float(np.mean(step_minima)),
        "time_averaged_gini_wellbeing": float(np.mean(step_ginis)),
    }


def final_metrics(agent_reports, llm_agent_ids):
    """
    Calculate episode-level outcomes from the final agent
    snapshot, plus time-averaged fairness across recorded steps.

    Dead agents are assigned wellbeing of zero.

    For mixed populations, subgroup metrics are calculated
    separately for LLM-based and rule-based agents. The
    LLM-rule gaps represent differences in per-agent outcomes:

        wellbeing gap = LLM mean wellbeing - rule mean wellbeing
        survival gap  = LLM survival rate - rule survival rate

    Positive gaps therefore indicate higher outcomes for LLM
    agents, while negative gaps indicate higher outcomes for
    rule-based agents.
    """

    final_step = agent_reports["step"].max()

    final = agent_reports.loc[
        agent_reports["step"] == final_step
    ].copy()

    if (
        len(final) != 4
        or final["agent_id"].nunique() != 4
    ):
        raise ValueError(
            "Expected four unique agents in final snapshot"
        )

    # ---------------------------------------------------------
    # Final agent state
    # ---------------------------------------------------------

    dead = (
        final["dead"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    wellbeing = (
        final["wellbeing"]
        .astype(float)
        .copy()
    )

    # Dead agents contribute zero final wellbeing.
    wellbeing.loc[dead] = 0.0

    consumption = (
        final["berries_consumed"]
        .astype(float)
    )

    llm_mask = final["agent_id"].isin(
        llm_agent_ids
    )

    rule_mask = ~llm_mask

    # ---------------------------------------------------------
    # Society-level metrics
    # ---------------------------------------------------------

    metrics = {
        "episode_steps": int(final_step) + 1,

        # Survival
        "survivors": int((~dead).sum()),
        "survival_rate": float(
            (~dead).mean()
        ),

        # Final wellbeing
        "total_final_wellbeing": float(
            wellbeing.sum()
        ),
        "mean_final_wellbeing": float(
            wellbeing.mean()
        ),
        "minimum_final_wellbeing": float(
            wellbeing.min()
        ),
        "gini_final_wellbeing": gini(
            wellbeing
        ),

        # Berry consumption
        "total_berries_consumed": float(
            consumption.sum()
        ),
        "minimum_berries_consumed": float(
            consumption.min()
        ),
        "gini_berries_consumed": gini(
            consumption
        ),

        # Resource sharing
        "total_berries_thrown": float(
            final["berries_thrown"].sum()
        ),
    }

    # ---------------------------------------------------------
    # Agent-type metrics
    # ---------------------------------------------------------

    for name, mask in (
        ("llm", llm_mask),
        ("rule", rule_mask),
    ):
        population_size = int(mask.sum())

        if population_size > 0:
            subgroup_wellbeing = wellbeing.loc[
                mask
            ]

            subgroup_consumption = (
                consumption.loc[mask]
            )

            subgroup_survivors = int(
                (mask & ~dead).sum()
            )

            metrics[
                f"{name}_total_final_wellbeing"
            ] = float(
                subgroup_wellbeing.sum()
            )

            metrics[
                f"{name}_mean_final_wellbeing"
            ] = float(
                subgroup_wellbeing.mean()
            )

            metrics[
                f"{name}_mean_berries_consumed"
            ] = float(
                subgroup_consumption.mean()
            )

            metrics[
                f"{name}_survivors"
            ] = subgroup_survivors

            metrics[
                f"{name}_survival_rate"
            ] = float(
                subgroup_survivors
                / population_size
            )

        else:
            metrics[
                f"{name}_total_final_wellbeing"
            ] = np.nan

            metrics[
                f"{name}_mean_final_wellbeing"
            ] = np.nan

            metrics[
                f"{name}_mean_berries_consumed"
            ] = np.nan

            metrics[
                f"{name}_survivors"
            ] = np.nan

            metrics[
                f"{name}_survival_rate"
            ] = np.nan

    # ---------------------------------------------------------
    # Within-society LLM vs rule-agent outcome gaps
    # ---------------------------------------------------------
    #
    # These are only meaningful for mixed populations.
    #
    # Positive wellbeing gap:
    #     LLM agents have higher mean final wellbeing.
    #
    # Negative wellbeing gap:
    #     rule-based agents have higher mean final wellbeing.
    #
    # Positive survival gap:
    #     LLM agents have a higher survival rate.
    #
    # Negative survival gap:
    #     rule-based agents have a higher survival rate.
    # ---------------------------------------------------------

    if llm_mask.any() and rule_mask.any():
        metrics[
            "llm_rule_wellbeing_gap"
        ] = float(
            metrics[
                "llm_mean_final_wellbeing"
            ]
            - metrics[
                "rule_mean_final_wellbeing"
            ]
        )

        metrics[
            "llm_rule_survival_gap"
        ] = float(
            metrics[
                "llm_survival_rate"
            ]
            - metrics[
                "rule_survival_rate"
            ]
        )

    else:
        metrics[
            "llm_rule_wellbeing_gap"
        ] = np.nan

        metrics[
            "llm_rule_survival_gap"
        ] = np.nan

    metrics.update(time_averaged_fairness(agent_reports))

    return metrics