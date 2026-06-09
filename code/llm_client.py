import os
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

class HuggingFaceClient:
    def __init__(self):
        self.client = InferenceClient(
            api_key=os.getenv("HF_TOKEN")
        )
        self.model = "Qwen/Qwen2.5-7B-Instruct"
    
    def generate(self, prompt : str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.0
        )
        
        return response.choices[0].message.content.strip()
    
class OllamaClient:
    def __init__(self, model="qwen3:8b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"
    
    def generate(self, prompt : str) -> str:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        return response.json()["response"].strip()