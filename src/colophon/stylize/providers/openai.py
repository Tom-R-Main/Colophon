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
        details = data.get("incomplete_details") or {}
        if details.get("reason") == "max_output_tokens":
            raise RuntimeError("The model output was truncated; no complete rewrite was returned")
        if data.get("status") != "completed":
            raise RuntimeError("The provider returned an incomplete or unsuccessful response")
        if any(
            item.get("type") == "message" and item.get("status") != "completed"
            for item in data.get("output", [])
        ):
            raise RuntimeError("The provider returned an incomplete output message")
        output_text = [
            content["text"]
            for item in data.get("output", [])
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text" and isinstance(content.get("text"), str)
        ]
        text = "".join(output_text)
        if not text.strip():
            raise RuntimeError("The provider returned no text")
        return text
