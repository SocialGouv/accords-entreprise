import json
import logging
import logging.config
import random
from abc import ABC, abstractmethod
from typing import Optional

import httpx
import requests
from openai import OpenAI
from pydantic_core import Url

from tca.custom_types import (
    LLMType,
)

logging.config.fileConfig("logging.conf")


class LLMAPIClient(ABC):
    def __init__(self, type: LLMType):
        self.type = type
        self.name = f"{type}_" + "".join(
            random.choices("abcdefghijklmnopqrstuvwxyz", k=5)  # nosec
        )

    @abstractmethod
    def generate_text(self, prompt):
        pass


class OllamaAPIClient(LLMAPIClient):
    def __init__(
        self,
        model_name: str,
        endpoint: Url,
        timeout: int = 10,
    ):
        super().__init__(type="ollama")
        self.model_name = model_name
        self.endpoint = endpoint
        self.timeout = timeout

    def generate_text(self, prompt: str) -> str:
        logging.info(f"{self.name} is generating text")

        headers = {"Content-Type": "application/json"}

        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }

        response = requests.post(
            str(self.endpoint), json=data, headers=headers, timeout=self.timeout
        )
        if response.status_code == 200:
            return json.loads(response.text)["response"].strip()
        else:
            return f"Failed to get a response. Status code: {response.status_code}"


class OpenAIAPIClient(LLMAPIClient):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[httpx.URL] = None,
    ):
        super().__init__(type="openai")
        self.model_name = model_name
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else httpx.URL("https://api.openai.com/v1"),
        )

    def generate_text(self, prompt: str) -> str:
        logging.info(f"{self.name} is generating a summary")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": f"{prompt}"}],
            temperature=0,
        )
        return response.choices[0].message.content or ""
