from mesa import Agent

class HarvestAgent(Agent):
    """Agent in the model environment"""

    def __init__(self, model, id):
        super().__init__(model)

        self.id = id

        self.health = 10
        self.berries = 0

    def step(self):
        self.health -= 1

        action = self.choose_action()

        self.perform_action(action)

    def choose_action(self):
        return
    
    def perform_action(self):
        None