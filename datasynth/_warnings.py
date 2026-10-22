"""Central warning filters for CLI and graph runner entrypoints."""

from __future__ import annotations

import warnings

_CONFIGURED = False


def configure_warnings() -> None:
    """Suppress harmless Pydantic serializer warnings from LangChain structured output.

    with_structured_output stores the parsed schema instance on the OpenAI SDK
    message.parsed field, which is typed as None. When astream_events serializes
    chat model events, Pydantic emits UserWarning even though behavior is correct.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    warnings.filterwarnings(
        "ignore",
        message=r"Pydantic serializer warnings",
        category=UserWarning,
        module=r"pydantic.*",
    )
    _CONFIGURED = True
