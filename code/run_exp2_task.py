import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("mixed", "rule_only"), required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("saved_runs/exp2"),
    )
    parser.add_argument("--llm-agent-ids", type=int, nargs="+", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    manifest = (
        root / "experiment_records" / "exp2" / f"{args.kind}_runs.csv"
    )

    with manifest.open(newline="") as file:
        rows = list(csv.DictReader(file))

    matches = [
        row for row in rows
        if int(row["task_id"]) == args.task_id
    ]
    if len(matches) != 1:
        parser.error(f"Expected one row for task {args.task_id}")

    row = matches[0]
    command = [
        sys.executable,
        "-u",
        str(root / "code" / "run_condition.py"),
        "--rule-policy", row["rule_policy"],
        "--seed", row["seed"],
        "--max-steps", "75",
        "--output-root", str(root / args.output_root),
    ]

    if args.kind == "mixed":
        command.extend(["--model", row["model"], "--prompt", row["prompt"]])
        if args.llm_agent_ids is not None:
            command.extend([
                "--llm-agent-ids",
                *map(str, args.llm_agent_ids),
            ])
    else:
        if args.llm_agent_ids is not None:
            parser.error("--llm-agent-ids only applies to mixed runs")
        command.append("--rule-only")

    print(f"Selected condition: {row}", flush=True)
    print("Command:", " ".join(command), flush=True)

    if not args.dry_run:
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()