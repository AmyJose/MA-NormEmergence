import json
import os
from pathlib import Path

from harvest_model import HarvestModel
from llm_client import IsambardClient

model_path = Path(os.environ["SCRATCHDIR"]) / "models/qwen3-8b"
run_dir = Path("saved_runs/smoke_two_llms_qwen")

client = IsambardClient(model_path=str(model_path))
model = HarvestModel(
    rng=42,
    llm_client=client,
    llm_agent_ids=(0, 1),
    prompt_type="baseline",
    rule_policy="selfish",
    run_dir=run_dir,
)

for _ in range(3):
    model.step()

records = [
    json.loads(line)
    for line in (run_dir / "llm_reasoning.jsonl").read_text().splitlines()
]

assert len(records) == 6, f"Expected 6 LLM records, got {len(records)}"
assert {record["agent_id"] for record in records} == {0, 1}

print("Qwen two-LLM smoke test passed")
print("Reasoning records:", len(records))
print("Actions:", [(r["agent_id"], r["action"]) for r in records])