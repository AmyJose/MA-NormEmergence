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
        #Rule based for now!
        if self.health < 3 and self.berries > 0:
            return "eat"
        
        if self.on_berry():
            return "collect"
        
        if self.can_help_someone():
            return "give"
        
        return "move"

    
    def perform_action(self):
        None

    def on_berry(self):
        return self.model.get_berry_from_coord(self.grid.cell)
    
    def can_help_someone():
        return