import json
import os
from pathlib import Path

from harvest_model import HarvestModel
from llm_client import IsambardClient
from model_config import get_model_config

model_name = "qwen3_8b"
config = get_model_config(model_name)
model_path = Path(os.environ["SCRATCHDIR"]) / "models"/ config["local_dir"]

run_dir = Path("saved_runs/smoke_two_llms_qwen_raw")

client = IsambardClient(
    model_path=str(model_path),
    model_id=config["hf_id"],
    max_new_tokens=config["max_new_tokens"],
    )
model = HarvestModel(
    rng=42,
    llm_client=client,
    llm_agent_ids=(0, 1),
    prompt_type="unframed",
    rule_policy="self_interested",
    run_dir=run_dir,
)

#for _ in range(3):
#    model.step()

model.step()

records = [
    json.loads(line)
    for line in (run_dir / "llm_reasoning.jsonl").read_text().splitlines()
]

#assert len(records) == 6, f"Expected 6 LLM records, got {len(records)}"
assert len(records) ==2, f"Expected 2 LLM records, got {len(records)}"
assert {record["agent_id"] for record in records} == {0, 1}

print("Qwen two-LLM smoke test passed")
print("Reasoning records:", len(records))
print("Actions:", [(r["agent_id"], r["action"]) for r in records])
