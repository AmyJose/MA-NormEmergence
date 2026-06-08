class DecisionModule:
    def __init__(self, agent, actions):
        self.agent= agent
        self.actions = actions

        self.eat_threshold = 0.5
        

    def choose_action(self):
        can_eat = self.agent.berries > 0
        hungry = self.agent.health < self.eat_threshold

        if hungry and can_eat:
            return "eat"

        return "move"
