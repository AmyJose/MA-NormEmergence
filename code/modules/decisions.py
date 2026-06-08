class DecisionModule:
    def __init__(self, agent):
        self.agent = agent

    def choose_action(self):
        if self.agent.berries > 0:
            worst_off_agent = self._get_worst_off_other_agent()

            if worst_off_agent is not None:
                if worst_off_agent.health < self.agent.health:
                    return f"throw_{worst_off_agent.id}"

        if self.agent.health < 0.5 and self.agent.berries > 0:
            return "eat"

        return "move"

    def _get_worst_off_other_agent(self):
        others = [
            agent
            for agent in self.agent.model.harvest_agents
            if agent.id != self.agent.id and agent.health > 0
        ]

        if not others:
            return None

        return min(others, key=lambda agent: agent.health)