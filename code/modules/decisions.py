class DecisionModule:
    def __init__(self, agent, actions):
        self.agent= agent
        self.actions = actions

    def choose_action(self):
        # keep as move for now!
        return self.actions[0]
