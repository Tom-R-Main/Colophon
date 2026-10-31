"""Anthropic Claude provider."""

from __future__ import annotations


class AnthropicProvider:
    name: str = "anthropic"

    def __init__(self, *, api_key: str, model: str = "claude-sonnet-5"):
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
        if message.stop_reason == "max_tokens":
            raise RuntimeError("The model output was truncated; no complete rewrite was returned")
        # Match the SDK's get_final_text(): preserve text, including its own separators.
        text = "".join(block.text for block in message.content if block.type == "text")
        if not text.strip():
            raise RuntimeError("The provider returned no text")
        return text
