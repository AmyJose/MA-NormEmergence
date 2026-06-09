from harvest_model import HarvestModel
from llm_client import LLMClient

def llm_test():
    llm = LLMClient()
    action = llm.generate(
        "You are a harvest agent in an allotment. You are agent 0. " \
        "You must choose one of the following options: " \
        "move towards the nearest berry (return MOVE), " \
        "eat a berry from your bag (return EAT) or " \
        "throw a berry to another agent (return THROW_<agent_id>). " \
        "Here is an observation of the current state of society: " \
        "your current health: 1.03, number of berries in your bag: 2, " \
        "distance to nearest berry: 4, " \
        "society wellbeing : [0: 34.3, 1: 34.3, 2 : 14.3, 3 : 21.0]. " \
        "Everyone's health decays by 0.03 each timestep. " \
        "To throw to another agent, you must have at least 0.6 health. " \
        "Wellbeing is represent by the following function:" \
        "wellbeing = (health + (berries * 0.2))/ 0.03." \
        "Using the observation and information provided, " \
        "choose ONE action. Do not give any explanation, just " \
        "return MOVE, EAT, THROW_1, THROW_2 or THROW_3"
    )
    print(action)

model = HarvestModel(seed=None)

for _ in range(400):
    model.step()

#llm_test()
