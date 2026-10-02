import csv
from itertools import product
from pathlib import Path

FRAMINGS = (
    "unframed",
    "self_interested",
    "cooperative",
    "altruistic",
)
POLICIES = ("self_interested", "cooperative", "altruistic")
SEEDS = range(1, 11)

output = Path("experiment_records/exp2")
output.mkdir(parents=True, exist_ok=True)

mixed = [
    {
        "task_id": task_id,
        "model": "qwen3_8b",
        "prompt": prompt,
        "rule_policy": policy,
        "seed": seed,
    }
    for task_id, (prompt, policy, seed) in enumerate(
        product(FRAMINGS, POLICIES, SEEDS)
    )
]

controls = [
    {
        "task_id": task_id,
        "rule_policy": policy,
        "seed": seed,
    }
    for task_id, (policy, seed) in enumerate(product(POLICIES, SEEDS))
]

for filename, rows in (
    ("mixed_runs.csv", mixed),
    ("rule_only_runs.csv", controls),
):
    path = output / filename
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path}: {len(rows)} runs")

assert len(mixed) == 120
assert len(controls) == 30
print("Manifest checks passed")