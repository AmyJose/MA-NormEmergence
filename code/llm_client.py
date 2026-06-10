import os
import requests
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

class HuggingFaceClient:
    def __init__(self):
        self.client = InferenceClient(
            api_key=os.getenv("HF_TOKEN")
        )
        self.model = "Qwen/Qwen2.5-7B-Instruct"
    
    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=20,
            temperature=0.0
        )
        
        return response.choices[0].message.content.strip()
    
class OllamaClient:
    def __init__(self, model="qwen3:8b", temp = 0.0):
        self.model = model
        self.url = "http://localhost:11434/api/chat"
        self.temperature = temp
    
    def chat(self, messages, max_retries = 3):
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.url,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature" : self.temperature
                        },
                    },
                    timeout=600
                )
                response.raise_for_status()
                return response.json()["message"]["content"].strip()
            except(
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
            ) as e:
                print(
                    f"[OLLAMA WARNING] Attempt "
                    f"{attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt<max_retries -1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            "Failed to get response from Ollama after retries"
        )