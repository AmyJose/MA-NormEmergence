import json
import pandas as pd
from pathlib import Path

rows = []

base_dir = Path("saved_runs/exp1")

files = list(base_dir.glob("*/llm_reasoning.jsonl"))

print(f"Found {len(files)} reasoning files")

for reasoning_file in files:

    run_name = reasoning_file.parent.name

    # Expected format:
    # prompt_rule_seed_X

    parts = run_name.split("_")

    try:
        seed = int(parts[-1])
        rule = parts[-3]
        prompt = "_".join(parts[:-3])

    except Exception:
        print(f"Could not parse run name: {run_name}")
        continue

    print(f"Loading {run_name}")

    with open(reasoning_file, encoding="utf-8") as f:

        # Skip CSV-style header
        next(f)

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

                rows.append({
                    "prompt": prompt,
                    "rule": rule,
                    "seed": seed,
                    **record
                })

            except json.JSONDecodeError:
                print(f"Failed to parse line in {reasoning_file}")
                print(line[:200])
                continue

df = pd.DataFrame(rows)

# Create unique identifier for later coding
df["trace_id"] = range(len(df))

print("\n===== SUMMARY =====")
print(f"Rows loaded: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\nPrompts:")
print(df["prompt"].value_counts())

print("\nRules:")
print(df["rule"].value_counts())

print("\nSample:")
print(df.head())

# Save outputs
try:
    df.to_parquet(
        "saved_runs/exp1/all_reasoning.parquet",
        index=False
    )
    print("\nSaved parquet file")
except Exception as e:
    print(f"\nCould not save parquet: {e}")
