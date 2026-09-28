import json
import tempfile
from pathlib import Path

from harvest_model import HarvestModel


class FakeLLMClient:
    model_name = "fake-test-model"
    temperature = 0.0

    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return {"content": "MOVE", "thinking": "smoke test"}


client = FakeLLMClient()

with tempfile.TemporaryDirectory() as run_dir:
    model = HarvestModel(
        rng=42,
        llm_client=client,
        llm_agent_ids=(0, 1),
        rule_policy="selfish",
        run_dir=run_dir,
    )

    agent_0, agent_1 = model.harvest_agents[:2]
    module_0 = agent_0.decision_module
    module_1 = agent_1.decision_module

    assert module_0.llm_client is module_1.llm_client is client
    assert module_0.messages is not module_1.messages
    assert "You are agent 0." in module_0.prompt_text
    assert "You are agent 1." in module_1.prompt_text
    assert "THROW_1" not in module_1.valid_actions

    agent_0.dead = True
    observation = agent_1.observe()
    message = module_1.observation_to_message(observation)

    assert 0 not in observation["society_wellbeing"]
    assert 1 in observation["society_wellbeing"]
    assert "agent 1 wellbeing:" in message
    assert "agent 0 wellbeing:" not in message

    agent_0.dead = False

    for _ in range(3):
        model.step()

    records = [
        json.loads(line)
        for line in (Path(run_dir) / "llm_reasoning.jsonl").read_text().splitlines()
    ]

    assert client.calls == 6
    assert len(records) == 6
    assert {record["agent_id"] for record in records} == {0, 1}

    model.reset_episode()
    assert len(module_0.messages) == 1
    assert len(module_1.messages) == 1

print("Two-LLM smoke test passed")