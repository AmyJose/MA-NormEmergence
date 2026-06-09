VALID_ACTIONS = ["MOVE", "EAT", "THROW_1", "THROW_2", "THROW_3"]

class LLMDecisionModule:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    def decide(self, observation: dict) -> str:
        prompt = self.build_prompt(observation)
        response = self.llm_client.ask(prompt)

        action = self.parse_action(response)

        if action is None:
            print(f"[LLM WARNING] Invalid response: {response}")
            return self.fallback_action(observation)
        
        return action
    
    def build_prompt(self, obs: dict) -> str:
        return ""
    
    def parse_action(self, response: str):
        return
    
    def fallback_action(self, obs) -> str:
        return ""