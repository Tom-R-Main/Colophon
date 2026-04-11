"""OpenAI provider (Chat Completions API)."""

from __future__ import annotations

import httpx


class OpenAIProvider:
    name: str = "openai"

    def __init__(self, *, api_key: str, model: str = "gpt-4o", base_url: str = "https://api.openai.com"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, system: str, user: str) -> str:
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 4096,
                "temperature": 0.7,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
