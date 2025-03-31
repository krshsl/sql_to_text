'''
author: chris, williams
'''
from enum import Enum
import requests
import json
import os

fireworks_url = "https://api.fireworks.ai/inference/v1/chat/completions"
fireworks_headers = {
  "Accept": "application/json",
  "Content-Type": "application/json",
  "Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"
}

class PAYLOADS(Enum):
    LLAMA_8B = 1
    FINE_TUNE_8B = 2

class LLMS:
    payload = {
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6
    }

    def __init__(self, payload):
        self.init_payload(payload)

    def init_payload(self, payload):
        if payload == PAYLOADS.LLAMA_8B:
            self.payload["model"] = "accounts/fireworks/models/llama-v3p1-8b-instruct"
            self.payload["max_tokens"] = 16384
        elif payload == PAYLOADS.FINE_TUNE_8B:
            self.payload["model"] = "accounts/ks2025-c3e5c8/models/al1"
            self.payload["max_tokens"] = 4000
        else:
            exit("Invalid payload")

    def request_model(self, messages):
        self.payload["messages"] = messages
        res = requests.request("POST", fireworks_url, headers=fireworks_headers, data=json.dumps(self.payload))
        if res.status_code == 200:
            response_json = res.json()

            if "choices" in response_json:
                model_reply = response_json["choices"][0]["message"]["content"]
                return model_reply
            else:
                print("Unexpected response format:", response_json)
        else:
            print(f"Error {res.status_code}: {res.text}")
            return None

    def infer_messages(self, message):
        message = [{"role": "user","content": message}]
        return self.request_model(message)
