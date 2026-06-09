from harvest_model import HarvestModel
from llm_client import LLMClient

def llm_test():
    llm = LLMClient()
    action = llm.generate(
        "You are a harvest agent in an allotment. " \
        "At each step, you must choose one of the following options: " \
        "move towards nearest berry (return MOVE), " \
        "eat a berry from your bag (return EAT) or " \
        "throw a berry to another agent (return THROW_<agent_id>). " \
        "To help you with your decision, here is a current observation of the society: " \
        "your current health: 4.94, number of berries in your bag: 2, " \
        "distance to nearest berry: 3, " \
        "society wellbeing : [0: 100, 1 (you): 504, 2 : 503, 3 : 503]. " \
        "Using this observation, choose ONE action. Dont give any explanation, just" \
        "return MOVE, EAT, THROW_0, THROW_2 or THROW_3"
    )
    print(action)

#model = HarvestModel(seed=None)

#for _ in range(500):
    #model.step()

llm_test()
