from harvest_model import HarvestModel
from llm_client import LLMClient

llm = LLMClient()
action = llm.generate(
    "Choose one action: MOVE, EAT, THROW. Return only the action"
)
print(action)

model = HarvestModel(seed=None)

for _ in range(550):
    model.step()
