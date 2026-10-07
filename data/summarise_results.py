import argparse
import json
from pathlib import Path

import pandas as pd

from results_metrics import final_metrics


GROUP_COLUMNS = [
    "num_llm_agents",
    "num_rule_agents",
    "model",
    "prompt",
    "rule_policy",
]


def read_run(run):
    metadata = json.loads((run / "metadata.json").read_text())
    agents = pd.read_csv(run / "agent_reports.csv")
    episodes = pd.read_csv(run / "model_episode_reports.csv")

    if len(episodes) != 1:
        raise ValueError("Expected one completed episode")

    end_step = int(episodes.iloc[0]["end_step"])
    if not 0 <= end_step < metadata["max_steps"]:
        raise ValueError("Invalid episode end step")
    if set(agents["step"]) != set(range(end_step + 1)):
        raise ValueError("Missing steps or extra unfinished steps")
    if agents.duplicated(["step", "agent_id"]).any():
        raise ValueError("Duplicate agent snapshots")
    if not agents.groupby("step")["agent_id"].apply(
        lambda ids: set(ids) == {0, 1, 2, 3}
    ).all():
        raise ValueError("Missing agent snapshots")
    if not agents[["health", "wellbeing"]].notna().all().all():
        raise ValueError("Missing health or wellbeing")

    final = agents.loc[agents["step"] == end_step]
    if end_step < metadata["max_steps"] - 1:
        if not final["dead"].astype(str).str.lower().eq("true").all():
            raise ValueError("Early finish without all agents dead")

    ids = metadata["llm_agent_ids"]
    num_llm = len(ids)
    if metadata["num_llm_agents"] != num_llm:
        raise ValueError("LLM population metadata mismatch")
    if metadata["num_rule_agents"] != 4 - num_llm:
        raise ValueError("Rule population metadata mismatch")

    model_names = {
        metadata["agents"][str(agent_id)]["model"]
        for agent_id in ids
    }
    if len(model_names) > 1:
        raise ValueError("Multiple models need separate grouping")

    metrics = final_metrics(agents, ids)

    action_counts = {
        "move": 0,
        "eat": 0,
        "throw": 0,
        "unsuccessful": 0,
    }

    # Decision logs contain real LLM activations, including death turns.
    expected = set()
    for agent_id in ids:
        alive = True
        snapshots = agents.loc[
            agents["agent_id"] == agent_id
        ].sort_values("step")
        for _, snapshot in snapshots.iterrows():
            if alive:
                expected.add((int(snapshot["step"]), agent_id))
            alive = alive and str(snapshot["dead"]).lower() != "true"

    seen = set()
    fallback_count = 0
    with (run / "llm_reasoning.jsonl").open() as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["step"], record["agent_id"])
            if key in seen:
                raise ValueError("Duplicate LLM decision")
            seen.add(key)

            snapshot = agents.loc[
                (agents["step"] == record["step"])
                & (agents["agent_id"] == record["agent_id"])
            ]
            if len(snapshot) != 1:
                raise ValueError("Decision has no unique agent snapshot")

            action = str(snapshot.iloc[0]["action"])
            if action == "move":
                category = "move"
            elif action == "eat":
                category = "eat"
            elif action == "throw":
                category = "throw"
            elif action == "wait":
                category = "unsuccessful"
            else:
                raise ValueError(f"Unknown executed action: {action}")

            action_counts[category] += 1

            if not isinstance(record["fallback_used"], bool):
                raise ValueError("Invalid fallback flag")
            fallback_count += record["fallback_used"]

    if seen != expected:
        raise ValueError("Missing or extra LLM decisions")

    metrics.update({
        "llm_decisions": len(seen),
        "fallback_count": fallback_count,
        "fallback_rate": (
            fallback_count / len(seen) if seen else float("nan")
        ),
    })

    for category, count in action_counts.items():
        metrics[f"llm_action_{category}_proportion"] = (
            count / len(seen) if seen else float("nan")
        )

    return {
        "run_dir": str(run),
        "seed": metadata["rng"],
        "num_llm_agents": num_llm,
        "num_rule_agents": 4 - num_llm,
        "model": next(iter(model_names), "not_applicable"),
        "prompt": metadata["prompt_type"] if num_llm else "not_applicable",
        "rule_policy": (
            metadata["rule_policy"] if num_llm < 4 else "not_applicable"
        ),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/analysis/exp2")
    )
    args = parser.parse_args()

    for root in args.roots:
        if not root.is_dir():
            parser.error(f"Results folder does not exist: {root}")

    rows = []
    skipped = []

    # Resolve paths to avoid reading overlapping roots twice.
    metadata_paths = sorted({
        path.resolve()
        for root in args.roots
        for path in root.rglob("metadata.json")
    })

    for path in metadata_paths:
        try:
            rows.append(read_run(path.parent))
        except Exception as error:
            skipped.append({
                "run_dir": str(path.parent),
                "reason": f"{type(error).__name__}: {error}",
            })

    if not rows:
        raise SystemExit("No completed valid runs found")

    runs = pd.DataFrame(rows)
    keys = GROUP_COLUMNS + ["seed"]
    if runs.duplicated(keys).any():
        duplicates = runs.loc[runs.duplicated(keys, keep=False), keys]
        raise ValueError(
            "Duplicate experimental conditions/seeds:\n"
            + duplicates.to_string(index=False)
        )

    metric_columns = [
        column for column in runs.columns
        if column not in GROUP_COLUMNS + ["run_dir", "seed"]
    ]
    summary = (
        runs.melt(
            id_vars=GROUP_COLUMNS + ["seed"],
            value_vars=metric_columns,
            var_name="metric",
            value_name="value",
        )
        .groupby(GROUP_COLUMNS + ["metric"], dropna=False)["value"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "run_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "condition_summary.csv", index=False)
    pd.DataFrame(
        skipped, columns=["run_dir", "reason"]
    ).to_csv(args.output_dir / "skipped_runs.csv", index=False)

    print(f"Completed valid runs: {len(runs)}")
    print(f"Skipped runs: {len(skipped)}")
    print("\nRuns by population:")
    print(runs.groupby(
        ["num_llm_agents", "num_rule_agents"]
    ).size().to_string())
    print(f"\nTables saved to: {args.output_dir}")


if __name__ == "__main__":
    main()