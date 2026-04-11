"""LLM provider abstraction — auto-detects from .env or explicit selection."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from colophon.stylize.providers.base import Provider

# Provider registry: env var -> (module, class, default model)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "ANTHROPIC_API_KEY": ("colophon.stylize.providers.anthropic", "AnthropicProvider", "claude-sonnet-4-20250514"),
    "OPENAI_API_KEY": ("colophon.stylize.providers.openai", "OpenAIProvider", "gpt-4o"),
    "OPENROUTER_API_KEY": ("colophon.stylize.providers.openrouter", "OpenRouterProvider", "anthropic/claude-sonnet-4"),
    "GEMINI_API_KEY": ("colophon.stylize.providers.gemini", "GeminiProvider", "gemini-2.5-flash"),
}


def auto_detect() -> Provider:
    """Detect which API key is set and return the corresponding provider."""
    load_dotenv()

    for env_var, (module_path, class_name, default_model) in PROVIDERS.items():
        key = os.environ.get(env_var)
        if key:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            return cls(api_key=key, model=default_model)

    raise RuntimeError(
        "No LLM API key found. Set one of these in .env:\n"
        "  ANTHROPIC_API_KEY=sk-ant-...\n"
        "  OPENAI_API_KEY=sk-...\n"
        "  OPENROUTER_API_KEY=sk-or-...\n"
        "  GEMINI_API_KEY=AI..."
    )


def get_provider(name: str | None = None, *, model: str | None = None) -> Provider:
    """Get a provider by name, or auto-detect."""
    load_dotenv()

    if name is None:
        provider = auto_detect()
        if model:
            provider.model = model
        return provider

    name_to_env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }

    env_var = name_to_env.get(name.lower())
    if not env_var:
        raise ValueError(f"Unknown provider: {name}. Available: {', '.join(name_to_env.keys())}")

    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(f"{env_var} not set. Add it to .env or set the environment variable.")

    module_path, class_name, default_model = PROVIDERS[env_var]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(api_key=key, model=model or default_model)


__all__ = ["Provider", "auto_detect", "get_provider"]
