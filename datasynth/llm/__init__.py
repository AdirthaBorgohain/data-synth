"""Provider-neutral LLM access. Swap OpenAI/Anthropic by config; agent code stays the same."""

from datasynth.llm.provider import get_chat_model

__all__ = ["get_chat_model"]
