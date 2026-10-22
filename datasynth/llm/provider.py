"""LLM provider factory.

Returns a LangChain `BaseChatModel` so downstream agent/graph code uses one API
(`.bind_tools`, `.stream`, structured output) regardless of provider. The only
provider-specific concern is how reasoning/thinking surfaces in the stream; the
trace renderer (built later) handles that, not callers here.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from datasynth.config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider
    if provider == "anthropic":
        return _anthropic(settings)
    if provider == "openai":
        return _openai(settings)
    raise ValueError(f"Unknown llm_provider: {provider!r}")


def _anthropic(settings: Settings) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    if not settings.anthropic_api_key:
        raise ValueError("anthropic_api_key is not set (ANTHROPIC_API_KEY).")

    # Extended thinking requires temperature=1; budget is a slice of max_tokens.
    thinking = {"thinking": {"type": "enabled", "budget_tokens": 4096}} if settings.llm_reasoning else {}
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=1.0 if settings.llm_reasoning else settings.llm_temperature,
        max_tokens=8192,
        **thinking,
    )


def _openai(settings: Settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise ValueError("openai_api_key is not set (OPENAI_API_KEY).")

    reasoning = {"reasoning_effort": "medium"} if settings.llm_reasoning else {}
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
        **reasoning,
    )
