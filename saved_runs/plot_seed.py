import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import plotting_style

#USE SEED 5

def plot_metric(df, metric, output_dir, run_name, title=None):
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

    plt.savefig(output_dir / f"{run_name}_{metric}.png")
    plt.close()

def plot_health_matrix(exp_dir, seed, output_file):
    rows = ["None", "selfish", "cooperative", "baseline"]
    cols = ["selfish", "selfless", "utilitarian"]

    fig, axes = plt.subplots(
        len(rows),
        len(cols),
        figsize=(15, 14),
        sharex=True,
        sharey=True
    )

    colours = {
        0: "#CAE7B9",
        1: "#F3DE8A",
        2: "#EB9486",
        3: "#7E7F9A",
    }

    for r, prompt in enumerate(rows):
        for c, rule in enumerate(cols):

            run_name = f"{prompt}_{rule}_seed_{seed}"
            run_dir = exp_dir / run_name

            csv_path = run_dir / "agent_reports.csv"

            ax = axes[r, c]

            if not csv_path.exists():
                ax.set_title("Missing")
                continue

            df = pd.read_csv(csv_path)

            for agent_id, group in df.groupby("agent_id"):
                group = group.sort_values("step")

                ax.plot(
                    group["step"],
                    group["health"],
                    label=f"Agent {agent_id}",
                    color=colours.get(agent_id)
                )

            if r == 0:
                ax.set_title(rule.title())

            if c == 0:
                ax.set_ylabel(prompt.title())

    handles, labels = axes[0,0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4
    )

    fig.supxlabel("Step")
    fig.supylabel("Health")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file, dpi=300)
    plt.close()

def main():
    run_name = "baseline_selfless_seed_5"

    run_dir = Path("exp1") / run_name

    csv_path = run_dir / "agent_reports.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find: {csv_path}")
    
    plots_dir = Path(".")/ "plots"/ "exp1" / "single_run_stats"
    plots_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)

    metrics = [
        "health",
        "wellbeing",
    ]

    plotting_style.set_thesis_style()

    for metric in metrics:
        plot_metric(df, metric, plots_dir, run_name)

    plot_health_matrix(Path("exp1"), 5, plots_dir / "health_matrix")


if __name__ == "__main__":
    main()