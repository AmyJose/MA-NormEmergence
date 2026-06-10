import re

class LLMDecisionModule:
    def __init__(self, llm_client, agent):
        self.llm_client = llm_client
        self.agent = agent

        valid_actions = ["MOVE", "EAT"]
        for agent_id in range(self.agent.model.num_agents):
            if agent_id != self.agent.id:
                valid_actions.append(f"THROW_{agent_id}")
        
        self.valid_actions = valid_actions

        self.messages = [
            {
                "role":"system",
                "content": f"""
You are a harvest agent in an allotment. You are agent {self.agent.id}. 
You are in the allotment with agent 1, agent 2 and agent 3.

Each turn you must choose one of the following options: 
    move towards the nearest berry (return MOVE), 
    eat a berry from your bag (return EAT), or 
    throw a berry to another agent (return THROW_<agent_id>). 

Everyone's health decays by {self.agent.health_decay} each timestep.
When an agent's health reaches 0, they die and no longer exist in the allotment.
To throw to another agent, you must have at least {self.agent.throw_berry_threshold} health. 
Wellbeing is represented by the following function: 
    wellbeing = (health + (berries * {self.agent.berry_health_payoff}))/{self.agent.health_decay}. 

Your aim is to survive as long as possible.

You will recieve an observation each turn to help you with your decision.
Using the observation and information provided, choose ONE action. 

You must return exactly one action token.

Valid actions:
{", ".join(self.valid_actions)}

Return one of these strings and nothing else.

Valid responses:
{chr(10).join(self.valid_actions)}

Invalid responses:
"The best action is MOVE"
"Answer: MOVE"
"$$\boxed{{MOVE}}$$"

If you include explanation, punctuation, 
markdown or extra text, your response is invalid.
"""
            }
        ]

    def decide(self, observation: dict) -> str:
        self.messages.append({
            "role":"user",
            "content": self.observation_to_message(observation)
        })

        self.trim_history()
        response = self.llm_client.chat(self.messages)

        self.messages.append({
            "role": "assistant",
            "content": response,
        })

        action = self.parse_action(response)

        if action is None:
            return self.get_user_action(response)
        
        return action
    
    def observation_to_message(self, obs: dict) -> str:
        return f"""
Step: {self.agent.model.steps}

Here is an observation of the current state of society: 
    your current health: {obs["health"]},
    number of berries in your bag: {obs["berries"]}, 
    distance to nearest berry: {obs["distance_to_nearest_berry"]}, 
    society wellbeing : {obs["society_wellbeing"]}.

Valid actions:
{", ".join(self.valid_actions)}

Return one of these strings and nothing else.

"""
    
    def parse_action(self, response: str):
        response = response.strip().upper()

        match = re.search(r"\b(MOVE|EAT|THROW_\d+)\b", response)

        if not match:
            return None

        action = match.group(1)

        if action not in self.valid_actions:
            return None

        return self.convert_action_token(action)
    
    def trim_history(self, max_turns=20):
        system_message = self.messages[0]

        non_system_messages = self.messages[1:]
        recent_messages = non_system_messages[-max_turns * 2:]

        self.messages = [system_message] + recent_messages

    def get_user_action(self, response: str) -> str:
        print("\n[LLM WARNING] Invalid response received:")
        print("-" * 60)
        print(response)
        print("-" * 60)

        print("Valid actions:")
        for action in self.valid_actions:
            print(f"- {action}")

        while True:
            user_input = input("Enter action to use: ").strip().upper()

            if user_input in self.valid_actions:
                return self.convert_action_token(user_input)

            print(f"Invalid action: {user_input}")

    def convert_action_token(self, action: str):
        if action == "MOVE":
            return self.agent.moving_module.direction_towards_nearest_berry()

        if action == "EAT":
            return "eat"

        if action.startswith("THROW_"):
            return action.lower()

        return None
