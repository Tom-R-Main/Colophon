from types import SimpleNamespace

import pytest

from colophon.stylize.llm import stylize_text
from colophon.stylize.providers.anthropic import AnthropicProvider
from colophon.stylize.providers.gemini import GeminiProvider
from colophon.stylize.providers.openai import OpenAIProvider


def test_generate_task_is_not_wrapped_as_a_rewrite(monkeypatch):
    from colophon.stylize import providers

    observed = []
    client = SimpleNamespace(generate=lambda system, user: observed.append(user) or "output")
    monkeypatch.setattr(providers, "get_provider", lambda *args, **kwargs: client)
    stylize_text("system", "Write about gardens using this brief.", mode="generate")
    assert observed == ["Write about gardens using this brief."]
    stylize_text("system", "A supplied article.")
    assert "Rewrite the following article" in observed[1]


def test_anthropic_reads_only_text_and_rejects_truncation(monkeypatch):
    import anthropic

    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking"),
            SimpleNamespace(type="text", text="A "),
            SimpleNamespace(type="text", text="complete answer."),
        ],
    )
    monkeypatch.setattr(
        anthropic,
        "Anthropic",
        lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response)),
    )
    client = AnthropicProvider(api_key="test-placeholder")
    assert client.generate("system", "input") == "A complete answer."
    response.stop_reason = "max_tokens"
    with pytest.raises(RuntimeError, match="truncated"):
        client.generate("system", "input")


@pytest.mark.parametrize(
    "provider,payload",
    [
        (OpenAIProvider, {"choices": [{"finish_reason": "length", "message": {"content": "incomplete"}}]}),
        (GeminiProvider, {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "partial"}]}}]}),
    ],
)
def test_http_provider_truncation_cannot_look_complete(monkeypatch, provider, payload):
    import httpx

    monkeypatch.setattr(
        httpx, "post", lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    )
    with pytest.raises(RuntimeError, match="truncated"):
        provider(api_key="test-placeholder").generate("system", "input")
