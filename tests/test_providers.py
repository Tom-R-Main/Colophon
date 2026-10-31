from __future__ import annotations

from typing import Any

import pytest

from colophon.stylize.providers import PROVIDERS
from colophon.stylize.providers.anthropic import AnthropicProvider
from colophon.stylize.providers.gemini import GeminiProvider
from colophon.stylize.providers.openai import OpenAIProvider
from colophon.stylize.providers.openrouter import OpenRouterProvider


class FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self.payload


def test_provider_registry_uses_current_models() -> None:
    assert PROVIDERS["ANTHROPIC_API_KEY"][2] == "claude-sonnet-5"
    assert PROVIDERS["OPENAI_API_KEY"][2] == "gpt-5.6-sol"
    assert PROVIDERS["OPENROUTER_API_KEY"][2] == "anthropic/claude-sonnet-5"
    assert PROVIDERS["GEMINI_API_KEY"][2] == "gemini-3.6-flash"


def test_anthropic_preserves_current_model_and_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import anthropic

    request: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> SimpleNamespace:
        request.update(kwargs)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="styled")])

    monkeypatch.setattr(
        anthropic, "Anthropic", lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    )
    assert AnthropicProvider(api_key="test-key").generate("system", "user") == "styled"
    assert request == {
        "model": "claude-sonnet-5",
        "max_tokens": 4096,
        "system": "system",
        "messages": [{"role": "user", "content": "user"}],
    }


def test_openai_uses_responses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    request: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        request.update(url=url, **kwargs)
        return FakeResponse(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "first\n"},
                            {"type": "output_text", "text": "second"},
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr("colophon.stylize.providers.openai.httpx.post", fake_post)

    result = OpenAIProvider(api_key="test-key").generate("system", "user")

    assert result == "first\nsecond"
    assert request["url"] == "https://api.openai.com/v1/responses"
    assert request["json"] == {
        "model": "gpt-5.6-sol",
        "instructions": "system",
        "input": "user",
        "max_output_tokens": 4096,
        "reasoning": {"effort": "none"},
        "store": False,
    }


def test_openai_rejects_response_without_output_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "colophon.stylize.providers.openai.httpx.post",
        lambda *args, **kwargs: FakeResponse({"status": "completed", "output": []}),
    )

    with pytest.raises(RuntimeError, match="no text"):
        OpenAIProvider(api_key="test-key").generate("system", "user")


def test_openrouter_keeps_chat_completions_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    request: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        request.update(url=url, **kwargs)
        return FakeResponse({"choices": [{"finish_reason": "stop", "message": {"content": "styled"}}]})

    monkeypatch.setattr("colophon.stylize.providers.openrouter.httpx.post", fake_post)

    result = OpenRouterProvider(api_key="test-key").generate("system", "user")

    assert result == "styled"
    assert request["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert request["json"]["model"] == "anthropic/claude-sonnet-5"
    assert request["json"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert request["json"]["max_tokens"] == 4096
    assert "temperature" not in request["json"]


def test_gemini_uses_current_model_without_deprecated_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        request.update(url=url, **kwargs)
        return FakeResponse({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "styled"}]}}]})

    monkeypatch.setattr("colophon.stylize.providers.gemini.httpx.post", fake_post)

    result = GeminiProvider(api_key="test-key").generate("system", "user")

    assert result == "styled"
    assert "/models/gemini-3.6-flash:generateContent" in request["url"]
    assert request["json"]["generationConfig"] == {"maxOutputTokens": 4096}
