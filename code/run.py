from harvest_model import HarvestModel
from llm_client import LLMClient

def llm_test():
    llm = LLMClient()
    action = llm.generate(
        "You are a harvest agent in an allotment. You are agent 0" \
        "At each step, you must choose one of the following options: " \
        "move towards nearest berry (return MOVE), " \
        "eat a berry from your bag (return EAT) or " \
        "throw a berry to another agent (return THROW_<agent_id>). " \
        "To help you with your decision, here is a current observation of the society: " \
        "your current health: 1.03, number of berries in your bag: 0, " \
        "distance to nearest berry: 4, " \
        "society wellbeing : [0: 34.3, 1: 34.3, 2 : 14.3, 3 : 21.0]. " \
        "Using this observation, choose ONE action. Dont give any explanation, just" \
        "return MOVE, EAT, THROW_0, THROW_2 or THROW_3"
    )
    print(action)

model = HarvestModel(seed=None)

for _ in range(400):
    model.step()

#llm_test()
