from mesa.discrete_space import CellAgent
from modules.decisions import DecisionModule
from modules.norms import NormsModule
from modules.moving import MovingModule

class HarvestAgent(CellAgent):
    """Agent in the model environment"""

    def __init__(self, model, id):
        super().__init__(model)

        self.id = id

        self.health = 0.8
        self.berries = 0
        self.actions = self._generate_actions()

        self.health_decay = 0.1

        self.norms_module = NormsModule(id)
        self.decision_module = DecisionModule(self, self.actions)
        self.moving_module = MovingModule(self)

    def step(self):
        action = self.decision_module.choose_action()

        self.perform_transition(action)
    
    def perform_transition(self, action):
        pre = self.norms_module.get_pre(self.berries, self.health)

        self._perform_action(action)

        self._update_attributes()
        self.norms_module.update_behaviour_base(pre, action)

    def _generate_actions(self):
        actions = ["move", "eat"]
        return actions
    
    def _perform_action(self, action):
        if action == "move":
            self._move()
    
    def _update_attributes(self):
        self.health -= self.health_decay

    def _move(self):
        berry_found, new_cell = self.moving_module.move_towards_nearest_berry()
        
        if berry_found:
            self.berries += 1

        if new_cell != self.cell:
            self.move_to(new_cell)
        
