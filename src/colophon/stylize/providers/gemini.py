"""Google Gemini provider."""

from __future__ import annotations

import httpx


class GeminiProvider:
    name: str = "gemini"

    def __init__(self, *, api_key: str, model: str = "gemini-3.6-flash"):
        self.api_key = api_key
        self.model = model

    def generate(self, system: str, user: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
            f":generateContent?key={self.api_key}"
        )
        response = httpx.post(
            url,
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": user}]}],
                "generationConfig": {"maxOutputTokens": 4096},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        candidate = data["candidates"][0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise RuntimeError("The model output was truncated; no complete rewrite was returned")
        text = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", []))
        if not text.strip():
            raise RuntimeError("The provider returned no text")
        return text
