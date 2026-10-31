"""OpenRouter provider (OpenAI-compatible Chat Completions API)."""

from __future__ import annotations

import httpx

from colophon.stylize.providers.openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    name: str = "openrouter"

    def __init__(self, *, api_key: str, model: str = "anthropic/claude-sonnet-5"):
        super().__init__(api_key=api_key, model=model, base_url="https://openrouter.ai/api")

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
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError("The model output was truncated; no complete rewrite was returned")
        text = choice["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("The provider returned no text")
        return text
