class RuleBasedDecisionModule:
    def __init__(self, agent, policy_type="cooperative"):
        self.agent = agent
        self.policy_type = policy_type

    def decide(self, observation:dict) -> str:
        if self.policy_type == "self_interested":
            return self._self_interested(observation)

        if self.policy_type == "cooperative":
            return self._cooperative(observation)

        if self.policy_type == "altruistic":
            return self._altruistic(observation)

        raise ValueError(f"Unknown policy: {self.policy_type}")
    
    #self-interested policy
    def _self_interested(self, obs):
        if obs["berries"] > 0:
            return "eat"
        
        return self._move_towards_nearest_berry()

    #cooperative policy
    def _cooperative(self, obs):
        #if i have a berry and cant throw it, just eat
        if obs["berries"] > 0 and obs["health"] < self.agent.throw_berry_threshold:
            return "eat"

        # if i have a berry and CAN throw, decide who to throw it to
        if obs["berries"] > 0 and obs["health"] >= self.agent.throw_berry_threshold:
            worst = self._get_worst_off_other_agent()
            if (
                worst is not None
                and worst.get_wellbeing() < self.agent.get_wellbeing()
            ):
                return f"throw_{worst.id}"

        #otherwise, just move
        return self._move_towards_nearest_berry()
    
    #altruistic policy : others matter more
    def _altruistic(self, obs):
        if obs["berries"] > 0 and obs["health"] >= self.agent.throw_berry_threshold:
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

