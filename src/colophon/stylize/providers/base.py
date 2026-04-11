"""Base provider protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    """Protocol for LLM providers."""

    name: str
    model: str

    def generate(self, system: str, user: str) -> str:
        """Send a system prompt + user message and return the response text."""
        ...
