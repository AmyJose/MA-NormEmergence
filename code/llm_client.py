import os
import requests
import time
import torch
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

class HuggingFaceClient:
    def __init__(self):
        self.client = InferenceClient(
            api_key=os.getenv("HF_TOKEN")
        )
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.model = self.model_name
    
    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=20,
            temperature=0.0
        )
        
        return {
            "content":
                response.choices[0].message.content.strip(),
            "thinking": ""
        }
    
class OllamaClient:
    def __init__(self, model="qwen3:8b", temp = 0.0):
        self.model_name = model
        self.model=self.model_name
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
                            "temperature" : self.temperature,
                            "num_predict": 5000,
                        },
                    },
                    timeout=600
                )
                #print(response.json())
                response.raise_for_status()
                data = response.json()
                return {
                    "content": data["message"]["content"].strip(),
                    "thinking": data["message"].get("thinking", "").strip()
                }

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

class IsambardClient():
    def __init__(self, model_path, temperature=0.0, max_new_tokens=3000):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.model_name= model_path

        #load tokeniser from model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        #load model from model_path
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype=torch.bfloat16)

        self.device = next(self.model.parameters()).device

    def chat(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )

        generated_ids = outputs[
            0,
            inputs["input_ids"].shape[1]:,
        ]

        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        parsed_response = self.parse_qwen_output(response)

        return parsed_response

    def parse_qwen_output(self, text:str):
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""

        #removing the thinking block
        cleaned = re.sub(r"<think> .*?</think>", "", text, flags=re.DOTALL).strip()

        #final action = last non-empty line
        action = cleaned.splitlines()[-1].strip()

        return{
            "content": action,
            "thinking": thinking
        }
