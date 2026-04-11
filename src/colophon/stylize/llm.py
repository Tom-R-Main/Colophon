"""LLM client for style transfer — provider-agnostic."""

from __future__ import annotations


def stylize_text(
    system_prompt: str,
    article_text: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Send an article to an LLM for style transfer, guided by the system prompt."""
    from colophon.stylize.providers import get_provider

    llm = get_provider(provider, model=model)

    user_message = (
        "Rewrite the following article in the target author's style. "
        "Preserve all facts, quotes, and reporting. Change only the voice, "
        "rhythm, and rhetorical approach.\n\n"
        "---\n\n"
        f"{article_text}"
    )

    return llm.generate(system_prompt, user_message)
