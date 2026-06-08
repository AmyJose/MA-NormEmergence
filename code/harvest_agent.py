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

        self.health = 5.0
        self.berries = 0
        self.actions = self._generate_actions()

        self.health_decay = 0.01
        self.berry_health_payoff = 0.1
        self.throw_berry_threshold = 0.6

        self.norms_module = NormsModule(self)
        self.decision_module = DecisionModule(self)
        self.moving_module = MovingModule(self)

    def step(self):
        if self.dead:
            return
        
        observation = self.observe
        action = self.decision_module.choose_action(observation)

        self.perform_transition(action)
    
    def perform_transition(self, action):
        observation = self.observe()

        pre = self.norms_module.get_pre(observation)

        self._perform_action(action)
        self._forage()
        self._update_attributes()

        norm_action = "throw" if action.startswith("throw_") else action
        self.norms_module.update_behaviour_base(pre, norm_action)

    def get_wellbeing(self):
        return (self.health + (self.berries * self.berry_health_payoff)) / self.health_decay
    
    def observe(self):
        return {
            "health" : self.health,
            "berries" : self.berries,
            "distance_to_nearest_berry" : self.moving_module.distance_to_berry(),
            "society_wellbeing" : self.model.get_society_wellbeing()
        }

    def _generate_actions(self):
        actions = ["north","south", "east", "west", "eat"]

        for agent_id in range(self.model.num_agents):
            if agent_id != self.id:
                actions.append(f"throw_{agent_id}")

        return actions
    
    def _perform_action(self, action):
        if action in ["north", "south", "east", "west"]:
            self.moving_module.move(action)
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

    def _forage(self):
        if self.cell in self.model.berries:
            self.model.berries.remove(self.cell)
            self.berries += 1

    def _eat(self):
        if self.berries > 0:
            self.health += self.berry_health_payoff
            self.berries -= 1
            self.model.spawn_one_berry()

    def _throw(self, target_id):
        if self.berries <= 0:
            return False
        if self.health < self.throw_berry_threshold:
            return False

        target = self.model.get_agent_by_id(target_id)

        if target is None or target.dead:
            return False
        
        self.berries -= 1
        target.berries += 1
        return True

