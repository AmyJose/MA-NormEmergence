import re
from pathlib import Path

class LLMDecisionModule:
    def __init__(self, llm_client, agent, prompt_name):
        self.llm_client = llm_client
        self.agent = agent

        valid_actions = ["MOVE", "EAT"]
        for agent_id in range(self.agent.model.num_agents):
            if agent_id != self.agent.id:
                valid_actions.append(f"THROW_{agent_id}")
        
        self.valid_actions = valid_actions
        self.prompt_name = prompt_name

        self.prompt_text = self.load_prompt(self.prompt_name)

        self.messages = [
            {
                "role":"system",
                "content": self.prompt_text,
            }
        ]

    def load_prompt(self, name):
        path = Path("code/prompts") / f"{name}.txt"

        with open(path, "r", encoding="utf-8") as f:
            return f.read()


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

Here is an observation of the current state: 
    your current health: {obs["health"]},
    number of berries in your bag: {obs["berries"]}, 
    your distance to nearest berry: {obs["distance_to_nearest_berry"]}, 
    your wellbeing: {obs['society_wellbeing'][0]},
    agent 1 wellbeing: {obs['society_wellbeing'][1]},
    agent 2 wellbeing: {obs['society_wellbeing'][2]},
    agent 3 wellbeiong : {obs['society_wellbeing'][3]}

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
    
    def trim_history(self, max_turns=8):
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
