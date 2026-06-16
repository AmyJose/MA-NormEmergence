class RuleBasedDecisionModule:
    def __init__(self, agent, policy_type="utilitarian"):
        self.agent = agent
        self.policy_type = policy_type

    def decide(self, observation:dict) -> str:
        if self.policy_type == "selfish":
            return self._selfish(observation)

        if self.policy_type == "utilitarian":
            return self._utilitarian(observation)

        if self.policy_type == "selfless":
            return self._selfless(observation)

        raise ValueError(f"Unknown policy: {self.policy_type}")
    
    #selfish policy
    def _selfish(self, obs):
        if obs["berries"] > 0:
            return "eat"
        
        return self._move_towards_nearest_berry()

    #utiliarianism policy
    def _utilitarian(self, obs):
        if obs["berries"] > 0 and obs["health"] < 0.6:
            return "eat"

        if obs["berries"] > 0 and obs["health"] >= 0.6:
            worst = self._get_worst_off_other_agent()
            if (
                worst is not None
                and worst.get_wellbeing() < self.agent.get_wellbeing()
            ):
                return f"throw_{worst.id}"

        return self._move_towards_nearest_berry()
    
    #selfless policy : others matter more
    def _selfless(self, obs):
        if obs["berries"] > 0 and obs["health"] >= 0.6:
            worst = self._get_worst_off_other_agent()

            if worst is not None:
                return f"throw_{worst.id}"

        if obs["berries"] > 0:
            return "eat"

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

