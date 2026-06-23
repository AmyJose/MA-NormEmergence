import pandas as pd
from pathlib import Path

SAMPLE_SIZE = 200
RANDOM_SEED = 42

DATA_FILE = "saved_runs/exp1/all_reasoning.parquet"
LABEL_FILE = "saved_runs/exp1/manual_labels.csv"

# --------------------------------------------------
# Load reasoning data
# --------------------------------------------------

df = pd.read_parquet(DATA_FILE)
print(df.columns.tolist())

# Stratified sample by rule
n_rules = df["rule"].nunique()
per_rule = SAMPLE_SIZE // n_rules

sample = pd.concat(
    [
        group.sample(
            n=min(per_rule, len(group)),
            random_state=RANDOM_SEED
        )
        for _, group in df.groupby("rule")
    ]
).sort_values("trace_id")

# --------------------------------------------------
# Load existing labels if resuming
# --------------------------------------------------

if Path(LABEL_FILE).exists():
    labels_df = pd.read_csv(LABEL_FILE)
    completed = set(labels_df["trace_id"])
    labels = labels_df.to_dict("records")
else:
    completed = set()
    labels = []

remaining = sample[~sample["trace_id"].isin(completed)]

print(f"Completed: {len(completed)}")
print(f"Remaining: {len(remaining)}")

# --------------------------------------------------
# Helper
# --------------------------------------------------

def binary_input(prompt):

    while True:

        value = input(prompt).strip()

        if value in ["0", "1"]:
            return int(value)

        print("Enter 0 or 1")

# --------------------------------------------------
# Coding loop
# --------------------------------------------------

for _, row in remaining.iterrows():

    print("\n" + "=" * 120)

    print(
        f"""
Trace ID : {row['trace_id']}
Prompt   : {row['prompt']}
Rule     : {row['rule']}
Seed     : {row['seed']}
Action   : {row['action']}
Health   : {row['health']}
Berries  : {row['berries']}
"""
    )

    print("\nREASONING:\n")
    print(row["reasoning"])

    print(
        """

Coding guide:

state_interpretation
  Mentions health, berries, distances, wellbeing etc.

constraint_checking
  Discusses valid actions or action requirements.

option_generation
  Considers multiple possible actions.

self_evaluation
  Evaluates impact on own health/resources.

social_evaluation
  Evaluates impact on another agent.

future_reasoning
  Mentions future outcomes or consequences.

uncertainty
  Shows confusion, doubt, speculation, or self-correction.

action_justification
  Explicitly explains final choice.

reasoning_error
  Makes a factual mistake about state or rules.

0 = absent
1 = present
"""
    )

    try:

        coding = {

    "trace_id": row["trace_id"],
    "prompt": row["prompt"],
    "rule": row["rule"],
    "action": row["action"],

    # --------------------------
    # State references
    # --------------------------

    "mentions_health":
        binary_input("Mentions health? (0/1): "),

    "mentions_berries":
        binary_input("Mentions berries? (0/1): "),

    "mentions_wellbeing":
        binary_input("Mentions wellbeing? (0/1): "),

    "mentions_distance":
        binary_input("Mentions berry distance? (0/1): "),

    # --------------------------
    # Reasoning process
    # --------------------------

    "checks_constraints":
        binary_input(
            "Checks action requirements or validity? (0/1): "
        ),

    "considers_multiple_actions":
        binary_input(
            "Considers multiple possible actions? (0/1): "
        ),

    "mentions_future_consequences":
        binary_input(
            "Discusses future outcomes? (0/1): "
        ),

    "expresses_uncertainty":
        binary_input(
            "Shows uncertainty/confusion? (0/1): "
        ),

    # --------------------------
    # Social reasoning
    # --------------------------

    "mentions_other_agents":
        binary_input(
            "Mentions another agent? (0/1): "
        ),

    "evaluates_other_agents":
        binary_input(
            "Evaluates another agent's state? (0/1): "
        ),

    "considers_helping_others":
        binary_input(
            "Considers helping another agent? (0/1): "
        ),

    # --------------------------
    # Decision making
    # --------------------------

    "justifies_final_action":
        binary_input(
            "Explains why final action was chosen? (0/1): "
        ),

    "contains_reasoning_error":
        binary_input(
            "Contains factual error? (0/1): "
        ),

    "notes":
        input("Notes (optional): ")
}

    except KeyboardInterrupt:
        print("\nStopping...")
        break

    labels.append(coding)

    pd.DataFrame(labels).to_csv(
        LABEL_FILE,
        index=False
    )

    print("Saved.")