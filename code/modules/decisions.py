class DecisionModule:
    def __init__(self, agent):
        self.agent = agent
        self.epsilon = 0.1

    def choose_action(self, observation):
        """
        Choose one action from:
        north, south, east, west, eat, throw_<agent_id>
        """

        #add randomness
        if self.agent.random.random() < self.epsilon:
            return self.agent.random.choice(self.agent.actions)

        health = observation["health"]
        berries = observation["berries"]

        #if carrying berries and health is low, eat
        #CHECK THIS THRESHOLD
        if berries > 0 and health < 2.5:
            return "eat"
        
        # if carrying berries and someone else is worse off, throw
        if berries > 0 and health >= self.agent.throw_berry_threshold:
            worst_off_agent = self._get_worst_off_other_agent()

            if worst_off_agent is not None:
                if worst_off_agent.get_wellbeing() < self.agent.get_wellbeing():
                    return f"throw_{worst_off_agent.id}"

        #otherwise, move towards nearest berry
        return self._move_towards_nearest_berry()
    
    def _get_worst_off_other_agent(self):
        living_others = [
            agent
            for agent in self.agent.model.harvest_agents
            if agent.id != self.agent.id and not agent.dead
        ]
        
        if not living_others:
            return None
        
        return min(
            living_others,
            key=lambda agent: agent.get_wellbeing()
        )
    def _move_towards_nearest_berry(self):
        return self.agent.moving_module.direction_towards_nearest_berry()
