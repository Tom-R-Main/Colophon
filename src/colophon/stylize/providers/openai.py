"""OpenAI provider (Responses API)."""

from __future__ import annotations

import httpx


class OpenAIProvider:
    name: str = "openai"

    def __init__(self, *, api_key: str, model: str = "gpt-5.6-sol", base_url: str = "https://api.openai.com"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, system: str, user: str) -> str:
        response = httpx.post(
            f"{self.base_url}/v1/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "instructions": system,
                "input": user,
                "max_output_tokens": 4096,
                "reasoning": {"effort": "none"},
                "store": False,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        output_text = [
            content["text"]
            for item in data.get("output", [])
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text" and content.get("text")
        ]
        if not output_text:
            raise ValueError("OpenAI response did not contain output text")
        return "\n".join(output_text)
