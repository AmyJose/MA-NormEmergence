import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_metric(df, metric, output_dir, title=None):
    plt.figure(figsize=(10, 5))

    for agent_id, group in df.groupby("agent_id"):
        group = group.sort_values("step")

        plt.plot(
            group["step"],
            group[metric],
            label=f"Agent {agent_id}"
        )

    plt.title(title or metric.replace("_", " ").title())
    plt.xlabel("Step")
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / f"{metric}.png")
    plt.close()

def plot_action_heatmap(df, output_dir):
    action_counts = (
        df.groupby(["agent_id", "action"])
        .size()
        .unstack(fill_value=0)
    )

    plt.figure(figsize=(8, 4))

    plt.imshow(action_counts, aspect="auto")

    plt.xticks(
        range(len(action_counts.columns)),
        action_counts.columns,
        rotation=45
    )

    plt.yticks(
        range(len(action_counts.index)),
        [f"Agent {i}" for i in action_counts.index]
    )

    plt.colorbar(label="Count")
    plt.title("Action Frequency Heatmap")
    plt.tight_layout()

    plt.savefig(output_dir / "action_heatmap.png")
    plt.close()


def plot_throw_events(df, output_dir):
    plt.figure(figsize=(10, 5))

    for agent_id, group in df.groupby("agent_id"):
        group = group.sort_values("step")

        plt.plot(
            group["step"],
            group["health"],
            label=f"Agent {agent_id}"
        )

        throws = group[group["action"] == "throw"]

        plt.scatter(
            throws["step"],
            throws["health"],
            s=50
        )

    plt.title("Health with Throw Events")
    plt.xlabel("Step")
    plt.ylabel("Health")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "health_with_throw_events.png")
    plt.close()


def main():

    if len(sys.argv) != 2:
        print(
            "Usage: python agent_reports_plot.py <run_name>"
        )
        sys.exit(1)

    run_name = sys.argv[1]

    run_dir = (
        Path("results")
        / "saved_runs"
        / run_name
    )

    csv_path = run_dir / "agent_reports_.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find: {csv_path}"
        )

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)

    metrics = [
        "health",
        "wellbeing",
        "berries",
        "berries_foraged",
        "berries_consumed",
        "berries_thrown",
        "num_behaviours",
    ]

    for metric in metrics:
        plot_metric(df, metric, plots_dir)

    plot_action_heatmap(df, plots_dir)
    plot_throw_events(df, plots_dir)

    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()