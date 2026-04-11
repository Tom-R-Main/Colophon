"""OpenRouter provider (OpenAI-compatible API)."""

from __future__ import annotations

from colophon.stylize.providers.openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    name: str = "openrouter"

    def __init__(self, *, api_key: str, model: str = "anthropic/claude-sonnet-4"):
        super().__init__(api_key=api_key, model=model, base_url="https://openrouter.ai/api")
