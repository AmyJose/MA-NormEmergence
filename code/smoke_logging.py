import csv
import json
import tempfile
from pathlib import Path

from harvest_model import HarvestModel


class FakeClient:
    model_name = "logging-test"
    temperature = 0.0

    def chat(self, messages):
        return {
            "content": "THROW_2",
            "thinking": "",
            "raw_response": "THROW_2",
        }


with tempfile.TemporaryDirectory() as directory:
    model = HarvestModel(
        rng=42,
        llm_client=FakeClient(),
        llm_agent_ids=(0, 1),
        prompt_type="unframed",
        rule_policy="self_interested",
        run_dir=directory,
    )

    # Agent 0 cannot throw at this health and dies after its action.
    agent = model.harvest_agents[0]
    agent.health = 0.01
    agent.berries = 1

    model.step()
    assert agent.dead
    assert agent.get_wellbeing() == 0.0
    model.step()

    run = Path(directory)
    records = [
        json.loads(line)
        for line in (run / "llm_reasoning.jsonl").read_text().splitlines()
    ]
    agent_0_records = [r for r in records if r["agent_id"] == 0]

    # Its final decision is logged once, not repeated after death.
    assert len(agent_0_records) == 1
    record = agent_0_records[0]

    assert record["requested_action"] == "throw_2"
    assert record["throw_target"] == 2
    assert record["action"] == "wait"
    assert record["observation"]["health"] == 0.01
    assert record["observation"]["berries"] == 1
    assert record["raw_response"] == "THROW_2"

    with (run / "agent_reports.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))

    first_row = next(
        r for r in rows if r["agent_id"] == "0" and r["step"] == "0"
    )
    assert first_row["requested_action"] == "throw_2"
    assert float(first_row["throw_target"]) == 2

print("Logging and death checks passed")