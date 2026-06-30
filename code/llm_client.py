"""
LLM Client Module
For interacting with LLMs such as Qwen, DeepSeek, Claude, etc.
"""

import openai
from openai import OpenAI
from anthropic import Anthropic
import requests


# Model configurations
MODEL_CONFIGS = {
    "qwen": {
        "base_url": "",
        "models": ["qwen-max-latest", "qwen3-max", "qwen3-max-2025-09-23", "qwen3-max-preview"]
    },
    "deepseek": {
        "base_url": "",
        "models": ["deepseek-chat", "deepseek-reasoner"]
    },
    "claude": {
        "base_url": "",
        "models": ["claude-opus-4-5-20251101", "claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022"]
    },
    "Gemini":{
        "base_url": "",
        "models": ["gemini-3-pro-preview"],
    },
    "gpt":{
        "base_url": "",
        "models": ["gpt-5.2"]
    },
    "ollama":{
        "base_url": "",
        "models": ["local-qwen3:32b","local-qwen3:14b","local-qwen3:8b"]
    }
}


def get_provider(model_type):

    model_lower = model_type.lower()
    if model_lower.startswith("deepseek"):
        return "deepseek"
    elif model_lower.startswith("qwen"):
        return "qwen"
    elif model_lower.startswith("claude"):
        return "claude"
    elif model_lower.startswith("gemini"):
        return "Gemini"
    elif model_lower.startswith("gpt"):
        return "gpt"
    elif model_lower.startswith("local"):
        return "ollama"
    else:
        return "qwen"


class LLMClient:


    def __init__(self, model_type, api_key):

        self.model_type = model_type
        self.provider = get_provider(model_type)
        self.api_key = api_key


        base_url = MODEL_CONFIGS[self.provider]["base_url"]
        if self.provider == "claude":
            self.client = Anthropic(
                api_key=api_key,
                base_url=base_url
            )
        elif self.provider in ("gpt", "Gemini"):

            self.client = None
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )

    def query(self, prompt, temperature=0.1, system_prompt="You are a helpful AI assistant."):
        
        if self.provider == "claude":
            response = self.client.messages.create(
                model=self.model_type,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.content[0].text if response.content else ""
        elif self.provider == "qwen":
            completion = self.client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                extra_body={
                    "enable_search": True,
                    "search_options": {
                        "search_strategy": "agent"
                    }
                }
            )
        elif self.provider == "deepseek":
            completion = self.client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                stream=False
            )
        elif self.provider in ("gpt", "Gemini"):
            base_url = MODEL_CONFIGS[self.provider]["base_url"]
            url = f"{base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model_type,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature
            }
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"] if data.get("choices") else ""
        elif self.provider == "ollama":
            completion = self.client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature
            )
        else:
            completion = self.client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature
            )

        return completion.choices[0].message.content if completion.choices else ""
