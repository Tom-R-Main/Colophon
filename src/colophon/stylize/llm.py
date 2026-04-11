"""LLM client for style transfer using Anthropic Claude."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def stylize_text(system_prompt: str, article_text: str, *, model: str = "claude-sonnet-4-20250514") -> str:
    """Send an article to Claude for style transfer, guided by the system prompt."""
    import anthropic

    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env or set the environment variable.\n"
            "  You can retrieve it with: exf vault read <vault-id>"
        )

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    "Rewrite the following article in the target author's style. "
                    "Preserve all facts, quotes, and reporting. Change only the voice, "
                    "rhythm, and rhetorical approach.\n\n"
                    "---\n\n"
                    f"{article_text}"
                ),
            }
        ],
    )

    return message.content[0].text
