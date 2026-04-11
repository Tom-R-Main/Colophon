"""Anthropic Claude provider."""

from __future__ import annotations


class AnthropicProvider:
    name: str = "anthropic"

    def __init__(self, *, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model

    def generate(self, system: str, user: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text
