from mesa.discrete_space import CellAgent
from modules.decisions import DecisionModule
from modules.norms import NormsModule
from modules.moving import MovingModule

class HarvestAgent(CellAgent):
    """Agent in the model environment"""

    def __init__(self, model, id):
        super().__init__(model)

        self.id = id
        self.dead = False

        self.health = 0.8
        self.berries = 0
        self.actions = self._generate_actions()

        self.health_decay = 0.1
        self.berry_health_payoff = 0.6

        self.norms_module = NormsModule(self)
        self.decision_module = DecisionModule(self)
        self.moving_module = MovingModule(self)

    def step(self):
        if self.dead:
            return
        action = self.decision_module.choose_action()

        self.perform_transition(action)
    
    def perform_transition(self, action):
        pre = self.norms_module.get_pre(self.berries, self.health)

        self._perform_action(action)

        self._update_attributes()

        norm_action = "throw" if action.startswith("throw_") else action
        self.norms_module.update_behaviour_base(pre, norm_action)

    def _generate_actions(self):
        actions = ["move", "eat"]

        for other_agents in self.model.harvest_agents:
            if other_agents.id != self.id:
                actions.append(f"throw_{other_agents.id}")

        return actions
    
    def _perform_action(self, action):
        if action == "move":
            self._move()
        elif action == "eat":
            self._eat()
        elif action.startswith("throw_"):
            target_id = int(action.split("_")[1])
            self._throw(target_id)
    
    def _update_attributes(self):
        self.health -= self.health_decay
        if self.health < 0:
            self.health = 0
            self.dead = True

    def _move(self):
        berry_found, new_cell = self.moving_module.move_towards_nearest_berry()
        
        if berry_found:
            self.berries += 1

        if new_cell != self.cell:
            self.move_to(new_cell)

    def _eat(self):
        if self.berries > 0:
            self.health += self.berry_health_payoff
            self.berries -= 1

    def _throw(self, target_id):
        if self.berries <= 0:
            return False
        
        target = self.model.get_agent_by_id(target_id)

        if target is None or target.dead:
            return False
        
        self.berries -= 1
        target.health += self.berry_health_payoff
        return True

