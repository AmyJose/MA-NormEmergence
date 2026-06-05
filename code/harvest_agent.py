from mesa.discrete_space import CellAgent
from modules.decisions import DecisionModule
from modules.norms import NormsModule

class HarvestAgent(CellAgent):
    """Agent in the model environment"""

    def __init__(self, model, id):
        super().__init__(model)

        self.id = id

        self.health = 10
        self.berries = 0
        self.actions = self._generate_actions()

        self.norms_module = NormsModule(id)
        self.decision_module = DecisionModule(id)

    def step(self):
        action = self.decision_module.choose_action()

        self.perform_action(action)
    
    def perform_action(self, action):
        pre = self.norms_module.get_pre(self.berries, self.health)
        self.norms_module.update_behaviour_base(pre, action)

    def _generate_actions(self):
        actions = ["move", "eat"]
        return actions
    
