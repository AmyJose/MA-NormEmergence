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

    differences = np.abs(values[:, None] - values[None, :])
    return float(
        differences.sum() / (2 * len(values) * values.sum())
    )


def final_metrics(agent_reports, llm_agent_ids):
    final_step = agent_reports["step"].max()
    final = agent_reports.loc[
        agent_reports["step"] == final_step
    ].copy()

    if len(final) != 4 or final["agent_id"].nunique() != 4:
        raise ValueError("Expected four unique agents in final snapshot")

    dead = final["dead"].astype(str).str.lower().eq("true")
    wellbeing = final["wellbeing"].astype(float).copy()
    wellbeing.loc[dead] = 0.0

    consumption = final["berries_consumed"].astype(float)
    llm_mask = final["agent_id"].isin(llm_agent_ids)

    metrics = {
        "episode_steps": int(final_step) + 1,
        "survivors": int((~dead).sum()),
        "total_final_wellbeing": float(wellbeing.sum()),
        "minimum_final_wellbeing": float(wellbeing.min()),
        "gini_final_wellbeing": gini(wellbeing),
        "total_berries_consumed": float(consumption.sum()),
        "minimum_berries_consumed": float(consumption.min()),
        "gini_berries_consumed": gini(consumption),
        "total_berries_thrown": float(final["berries_thrown"].sum()),
    }

    for name, mask in (("llm", llm_mask), ("rule", ~llm_mask)):
        metrics[f"{name}_mean_final_wellbeing"] = (
            float(wellbeing.loc[mask].mean()) if mask.any() else np.nan
        )
        metrics[f"{name}_mean_berries_consumed"] = (
            float(consumption.loc[mask].mean()) if mask.any() else np.nan
        )
        metrics[f"{name}_survivors"] = int((mask & ~dead).sum())

    return metrics