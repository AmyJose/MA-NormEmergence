class LLMDecisionModule:
    def __init__(self, llm_client, agent):
        self.llm_client = llm_client
        self.agent = agent

        valid_actions = ["MOVE", "EAT"]
        for agent_id in range(self.agent.model.num_agents):
            if agent_id != self.agent.id:
                valid_actions.append(f"THROW_{agent_id}")
        
        self.valid_actions = valid_actions

    def decide(self, observation: dict) -> str:
        prompt = self.build_prompt(observation)
        response = self.llm_client.ask(prompt)

        action = self.parse_action(response)

        if action is None:
            print(f"[LLM WARNING] Invalid response: {response}")
            return self.fallback_action(observation)
        
        return action
    
    def build_prompt(self, obs: dict) -> str:
        return f"""
You are a harvest agent in an allotment. You are agent {self.agent.id}. 

You must choose one of the following options: 
    move towards the nearest berry (return MOVE), 
    eat a berry from your bag (return EAT), or 
    throw a berry to another agent (return THROW_<agent_id>). 

Here is an observation of the current state of society: 
    your current health: {obs["health"]},
    number of berries in your bag: {obs["berries"]}, 
    distance to nearest berry: {obs["distance_to_nearest_berry"]}, 
    society wellbeing : {obs["society_wellbeing"]}. 
Everyone's health decays by {self.agent.health_decay} each timestep. 
To throw to another agent, you must have at least {self.agent.throw_berry_threshold} health. 
Wellbeing is represented by the following function: 
    wellbeing = (health + (berries * {self.agent.berry_health_payoff}))/{self.agent.health_decay}. 
Using the observation and information provided, 
choose ONE action. Do not give any explanation, 
just return {", ".join(self.valid_actions)}

"""
    
    def parse_action(self, response: str):
        if response == "MOVE":
            return self.agent.moving_module.direction_towards_nearest_berry()
        if response == "EAT":
            return "eat"
        if response.startswith("THROW_"):
            return response.lower()
        return None
    
    def fallback_action(self, obs) -> str:
        return self.agent.moving_module.direction_towards_nearest_berry()