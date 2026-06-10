import json
import shutil
import sys
from pathlib import Path


def save_run(run_name):
    source = Path("data/results/current_run")
    destination = Path(f"data/results/saved_runs/{run_name}")

    if destination.exists():
        raise ValueError(
            f"Run '{run_name}' already exists."
        )

    shutil.copytree(source, destination)

    metadata = {
        "run_name": run_name,
    }

    with open(destination / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved run to {destination}")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        run_name = sys.argv[1]
    else:
        run_name = input("Run name: ")

    save_run(run_name)
