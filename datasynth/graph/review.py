"""Reusable human-in-the-loop review gate.

A "gate" is the pattern: pause the graph, surface a payload to the user, collect
a free-text comment, and — if the comment is non-blank — let an LLM decide
whether to revise the phase's working state and how. Empty comment = proceed.

The mechanics here are phase-independent. Each phase supplies four hooks:
  get_payload(state)   → what the user sees at the interrupt
  schema_model         → a ReviewDecision subclass holding the revisable content
  build_context(state) → text appended after the comment for the LLM
  apply(state, result) → validate the model's result into a state update

so the same factories drive table-selection review today and clean / feature-gen
/ train review later, each with its own redo target.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from datasynth.graph.state import AgentState


class ReviewDecision(BaseModel):
    """Base structured output for a revise node.

    Phases subclass this to add the content they may change (e.g. a table list).
    """

    reasoning: str = Field(description="Why you are proceeding or what you changed")
    # Subclasses may widen the allowed values (e.g. add "clean"); the per-phase
    # prompt documents the choices. "proceed" is special: it means no change.
    action: str = Field(
        default="proceed",
        description="proceed: keep current state unchanged. Any other value applies changes.",
    )


GetPayload = Callable[[AgentState], dict]
BuildContext = Callable[[AgentState], str]
Apply = Callable[[AgentState, ReviewDecision], dict]
Node = Callable[[AgentState, RunnableConfig], Awaitable[dict]]


def make_review_node(get_payload: GetPayload) -> Node:
    """Return an async node that pauses on interrupt() and stores the comment.

    The interrupt halts the graph; resuming with Command(resume=<text>) makes
    interrupt() return that text. A blank comment means "proceed".
    """

    async def review(state: AgentState, config: RunnableConfig) -> dict:
        comment = interrupt(get_payload(state))
        return {"user_comment": (comment or "").strip()}

    return review


def route_after_review(state: AgentState) -> str:
    """Blank comment → 'proceed'; any comment → 'revise'. Caller maps the keys."""
    return "revise" if state.get("user_comment", "").strip() else "proceed"


def make_revise_node(
    model: BaseChatModel,
    *,
    system_prompt: str,
    schema_model: type[ReviewDecision],
    build_context: BuildContext,
    apply: Apply,
) -> Node:
    """Return an async revise node: LLM interprets the comment and decides.

    Streams via astream so on_chat_model_stream (thinking blocks) propagate to
    the renderer, matching the connect node.
    """
    selector = model.with_structured_output(schema_model)

    async def revise(state: AgentState, config: RunnableConfig) -> dict:
        comment = state.get("user_comment", "")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"User comment:\n{comment}\n\n{build_context(state)}"
            ),
        ]

        result: ReviewDecision | None = None
        async for partial in selector.astream(messages, config=config):
            result = partial  # type: ignore[assignment]

        if result is None:
            raise RuntimeError("LLM returned no output for review revision.")

        # "proceed" means no change; any other action applies the phase's edit.
        if result.action == "proceed":
            return {"revise_action": "proceed", "user_comment": ""}

        return {**apply(state, result), "revise_action": result.action, "user_comment": ""}

    return revise


def make_route_after_revise(route_map: dict):
    """Route on revise_action via route_map. Values are node names or END.

    Example: {"proceed": END, "revise": "profile", "clean": "clean_propose"}.
    """

    def route(state: AgentState):
        return route_map[state.get("revise_action", "proceed")]

    return route


__all__ = [
    "ReviewDecision",
    "make_review_node",
    "route_after_review",
    "make_revise_node",
    "make_route_after_revise",
]
