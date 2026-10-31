"""Provider protocol tests use synthetic responses and never call live providers."""

from types import SimpleNamespace

import pytest

from colophon.stylize.llm import stylize_text
from colophon.stylize.providers.anthropic import AnthropicProvider
from colophon.stylize.providers.gemini import GeminiProvider
from colophon.stylize.providers.openai import OpenAIProvider
from colophon.stylize.providers.openrouter import OpenRouterProvider


def test_generate_task_is_not_wrapped_as_a_rewrite(monkeypatch):
    from colophon.stylize import providers

    observed = []
    client = SimpleNamespace(generate=lambda system, user: observed.append(user) or "output")
    monkeypatch.setattr(providers, "get_provider", lambda *args, **kwargs: client)
    stylize_text("system", "Write about gardens using this brief.", mode="generate")
    assert observed == ["Write about gardens using this brief."]
    stylize_text("system", "A supplied article.")
    assert "Rewrite the following article" in observed[1]


@pytest.mark.parametrize(
    "fragments,expected",
    [
        (["A ", "complete answer."], "A complete answer."),
        (["First paragraph.\n\n", "Second paragraph.\n"], "First paragraph.\n\nSecond paragraph.\n"),
    ],
)
def test_anthropic_reads_only_text_and_preserves_authored_boundaries(monkeypatch, fragments, expected):
    import anthropic

    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking"),
            *(SimpleNamespace(type="text", text=text) for text in fragments),
        ],
    )
    monkeypatch.setattr(
        anthropic,
        "Anthropic",
        lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response)),
    )
    client = AnthropicProvider(api_key="test-placeholder")
    # These are final content blocks, not streaming deltas; the SDK concatenates them.
    assert client.generate("system", "input") == expected
    response.stop_reason = "max_tokens"
    with pytest.raises(RuntimeError, match="truncated"):
        client.generate("system", "input")


@pytest.mark.parametrize(
    "provider,payload",
    [
        (
            OpenAIProvider,
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [{"type": "message", "status": "incomplete", "content": [
                    {"type": "output_text", "text": "partial"},
                ]}],
            },
        ),
        (
            OpenRouterProvider,
            {"choices": [{"finish_reason": "length", "message": {"content": "incomplete"}}]},
        ),
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


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled", "queued", "in_progress", None])
def test_openai_requires_completed_status_even_with_output_text(monkeypatch, status):
    import httpx

    payload = {
        "status": status,
        "incomplete_details": {"reason": "content_filter"} if status == "incomplete" else None,
        "output": [{"type": "message", "status": "completed", "content": [
            {"type": "output_text", "text": "apparently complete text"},
        ]}],
    }
    monkeypatch.setattr(
        httpx, "post", lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    )
    with pytest.raises(RuntimeError, match="incomplete or unsuccessful"):
        OpenAIProvider(api_key="test-placeholder").generate("system", "input")


def test_openai_rejects_incomplete_output_message(monkeypatch):
    import httpx

    payload = {
        "status": "completed",
        "output": [{"type": "message", "status": "incomplete", "content": [
            {"type": "output_text", "text": "partial"},
        ]}],
    }
    monkeypatch.setattr(
        httpx, "post", lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    )
    with pytest.raises(RuntimeError, match="incomplete output message"):
        OpenAIProvider(api_key="test-placeholder").generate("system", "input")


def test_openai_preserves_text_across_items_and_ignores_nontext(monkeypatch):
    import httpx

    payload = {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "status": "completed", "content": [
                {"type": "output_text", "text": " A "},
                {"type": "output_text", "text": "complete answer.\n\n"},
            ]},
            {"type": "message", "status": "completed", "content": [
                {"type": "refusal", "refusal": "not output text"},
                {"type": "output_text", "text": "Second paragraph.\n"},
            ]},
        ],
    }
    monkeypatch.setattr(
        httpx, "post", lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    )
    assert OpenAIProvider(api_key="test-placeholder").generate("system", "input") == (
        " A complete answer.\n\nSecond paragraph.\n"
    )


@pytest.mark.parametrize("text", ["", " \n\t", None])
@pytest.mark.parametrize("provider", [OpenAIProvider, OpenRouterProvider, GeminiProvider])
def test_http_providers_reject_empty_text(monkeypatch, provider, text):
    import httpx

    payload = {
        OpenAIProvider: {"status": "completed", "output": [
            {"type": "message", "status": "completed", "content": [{"type": "output_text", "text": text}]},
        ]},
        OpenRouterProvider: {"choices": [{"finish_reason": "stop", "message": {"content": text}}]},
        GeminiProvider: {"candidates": [{"finishReason": "STOP", "content": {"parts": [] if text is None else [
            {"text": text},
        ]}}]},
    }[provider]
    monkeypatch.setattr(
        httpx, "post", lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    )
    with pytest.raises(RuntimeError, match="no text"):
        provider(api_key="test-placeholder").generate("system", "input")


@pytest.mark.parametrize("fragments", [[], [""], [" \n", "\t"]])
def test_anthropic_rejects_empty_text(monkeypatch, fragments):
    import anthropic

    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="thinking"), *(SimpleNamespace(type="text", text=text) for text in fragments)],
    )
    monkeypatch.setattr(
        anthropic,
        "Anthropic",
        lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response)),
    )
    with pytest.raises(RuntimeError, match="no text"):
        AnthropicProvider(api_key="test-placeholder").generate("system", "input")


def test_gemini_preserves_text_parts_and_whitespace(monkeypatch):
    import httpx

    payload = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": " A "}, {"inlineData": {}}, {"text": "complete answer.\n"},
    ]}}]}
    monkeypatch.setattr(
        httpx, "post", lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    )
    assert GeminiProvider(api_key="test-placeholder").generate("system", "input") == " A complete answer.\n"


@pytest.mark.parametrize("target_exists", [False, True])
def test_incomplete_openai_output_is_never_saved_by_cli(monkeypatch, tmp_path, target_exists):
    import httpx
    from typer.testing import CliRunner

    from colophon.cli import app
    from colophon.models.features import StyleProfile
    from colophon.stylize import providers

    article = tmp_path / "article.txt"
    article.write_text("Synthetic article with supplied facts.")
    style = tmp_path / "style.analysis.json"
    style.write_text(StyleProfile(document_id="synthetic", document_title="Synthetic style").model_dump_json())
    target = tmp_path / "rewrite.md"
    if target_exists:
        target.write_bytes(b"Existing rewrite must survive.")
    payload = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "message", "status": "incomplete", "content": [
            {"type": "output_text", "text": "Do not save this partial rewrite."},
        ]}],
    }
    monkeypatch.setattr(
        httpx, "post", lambda *args, **kwargs: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    )
    monkeypatch.setattr(providers, "get_provider", lambda *args, **kwargs: OpenAIProvider(api_key="test-placeholder"))
    result = CliRunner().invoke(app, [
        "stylize", str(article), str(style), "--no-examples", "--no-diff", "--output", str(target),
    ])
    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "truncated" in str(result.exception)
    assert "saved to" not in result.output
    if target_exists:
        assert target.read_bytes() == b"Existing rewrite must survive."
    else:
        assert not target.exists()
