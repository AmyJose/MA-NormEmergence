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
            text = f.read()
            other_agents = ", ".join(
                f"agent {i}"
                for i in range(self.agent.model.num_agents)
                if i != self.agent.id
            )
            return (
                text.replace("{agent_id}", str(self.agent.id))
                    .replace("{other_agents}", other_agents)
                    .replace("{valid_actions}", ", ".join(self.valid_actions))
                    .replace("{valid_examples}", "\n".join(self.valid_actions))
            )


    def reset(self):
        self.messages = [
            {
                "role": "system",
                "content": self.prompt_text,
            }
        ]

    def decide(self, observation: dict) -> str:
        self.messages.append({
            "role":"user",
            "content": self.observation_to_message(observation)
        })

        self.trim_history()
        response = self.llm_client.chat(self.messages)
        self.agent.last_raw_response = response.get("raw_response", response["content"])

        raw_text = response["content"]
        think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)

        if think_match:
            reasoning = think_match.group(1).strip()
            action_text = re.sub(
                r"<think>.*?</think>", "", raw_text, flags=re.DOTALL
            ).strip()
        elif "<think>" in raw_text:
            # The model ran out of tokens before finishing its reasoning.
            reasoning = raw_text.split("<think>", 1)[1].strip()
            action_text = ""
        else:
            reasoning = response.get("thinking", "")
            action_text = raw_text.strip()

        self.messages.append({
            "role": "assistant",
            "content": action_text,
        })
        self.agent.last_reasoning = reasoning

        action = self.parse_action(action_text)

        fallback_used = False

        if action is None:
            fallback_used = True
            #return self.get_user_action(reasoning)
            action = self.default_action(observation)

        self.agent.last_reasoning = reasoning
        self.agent.last_fallback_used = fallback_used
        
        return action

    def default_action(self, observation: dict):
        """
        Safe fallback when LLM output is invalid or incomplete.
        """
        return self.convert_action_token("MOVE")

    def observation_to_message(self, obs: dict) -> str:
        wellbeing_lines = "\n".join(
            f"agent {i} wellbeing: {w}"
            for i, w in obs["society_wellbeing"].items()
        )
        return f"""
Step: {self.agent.model.episode_step}

Here is an observation of the current state: 
    your current health: {obs["health"]},
    number of berries in your bag: {obs["berries"]}, 
    your distance to nearest berry: {obs["distance_to_nearest_berry"]}, 
    your wellbeing: {self.agent.get_wellbeing()},
    {wellbeing_lines}

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
    
    def trim_history(self, max_turns=5):
        system_message = self.messages[0]

        non_system_messages = self.messages[1:]
        recent_messages = non_system_messages[-max_turns * 2:]

        self.messages = [system_message] + recent_messages

    def get_user_action(self, response: str) -> str:
        print("\n[LLM WARNING] Invalid response received. Thinking is:")
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
