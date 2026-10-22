"""Data-cleaning phase nodes.

Flow: clean_propose (agentic) → clean_review (interrupt) → clean_revise (agentic,
optional) → clean_apply (deterministic) → back to profile.

The propose/revise nodes decide *what* to clean (LLM); clean_apply executes the
plan deterministically via datasynth.clean.ops on frames held in the FrameStore.
clean_review / clean_revise reuse the generic gate in datasynth.graph.review.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from datasynth.clean.ops import OP_VOCABULARY, apply_ops
from datasynth.clean.store import FrameStore
from datasynth.graph.review import ReviewDecision, make_review_node, make_revise_node
from datasynth.graph.state import AgentState

_OPS_DOC = "\n".join(f"  - {name}: {desc}" for name, desc in OP_VOCABULARY.items())

_PROPOSE_SYSTEM = f"""\
You are a data-cleaning agent. Profiling flagged quality issues in sampled table
data. Propose a concrete cleaning plan using ONLY these operations:
{_OPS_DOC}

Each op targets one table (qualified schema.table) and, where relevant, one column.
Put operation-specific settings in `params` (e.g. dtype, strategy, value, lower,
upper). Give a short, specific `reason` per op. Propose the minimum that fixes the
flagged issues; do not invent problems. If the user left a request, prioritize it.
"""

_REVISE_SYSTEM = f"""\
You are adjusting a proposed data-cleaning plan based on the user's comment.
Operations available:
{_OPS_DOC}

Choose an action:
  - "proceed": the user approved as-is or said to go ahead — keep the current plan.
  - "revise": the user wants changes — return the COMPLETE revised list of ops.
Each op: table (schema.table), op, column (if relevant), params, reason.
"""


class CleanOp(BaseModel):
    table: str = Field(description="Qualified table name (schema.table)")
    op: str = Field(description=f"One of: {', '.join(OP_VOCABULARY)}")
    column: str | None = Field(default=None, description="Target column, if the op needs one")
    # JSON object string (e.g. '{\"dtype\": \"Float64\"}'). A string keeps the
    # schema closed so OpenAI strict structured output accepts it.
    params_json: str = Field(
        default="{}",
        description='Operation settings as a JSON object string, e.g. {"strategy":"mean"} or {"dtype":"Float64"} or {"lower":0,"upper":100}',
    )
    reason: str = Field(description="Why this op is needed")


class CleanProposal(BaseModel):
    reasoning: str = Field(description="Overall rationale for the cleaning plan")
    ops: list[CleanOp] = Field(default_factory=list)


class CleanRevision(ReviewDecision):
    action: Literal["proceed", "revise"] = Field(default="proceed")
    ops: list[CleanOp] = Field(
        default_factory=list, description="Complete revised op list (used when action='revise')"
    )


def _validate_ops(state: AgentState, ops: list[CleanOp]) -> list[dict]:
    """Drop ops referencing unknown tables/ops/columns."""
    selected = set(state.get("selected_tables", []))
    profiles = state.get("profiles", {})
    valid: list[dict] = []
    for o in ops:
        if o.table not in selected or o.op not in OP_VOCABULARY:
            continue
        cols = set(profiles.get(o.table, {}).get("columns", {}))
        if o.column is not None and o.column not in cols:
            continue
        try:
            params = json.loads(o.params_json or "{}")
        except (ValueError, TypeError):
            params = {}
        if not isinstance(params, dict):
            params = {}
        valid.append(
            {"table": o.table, "op": o.op, "column": o.column,
             "params": params, "reason": o.reason}
        )
    return valid


def _profiles_brief(profiles: dict) -> str:
    lines: list[str] = []
    for table, p in profiles.items():
        if "error" in p:
            lines.append(f"{table}: ERROR {p['error']}")
            continue
        cols = p.get("columns", {})
        bits = [
            f"{c} ({s.get('dtype')}, null {s.get('null_rate', 0):.0%})"
            for c, s in cols.items()
        ]
        lines.append(f"{table}: " + ", ".join(bits))
    return "\n".join(lines)


def _propose_context(state: AgentState) -> str:
    request = state.get("clean_request", "")
    return (
        (f"User request: {request}\n\n" if request else "")
        + f"Flagged issues:\n{state.get('profile_issues', [])}\n\n"
        f"Profiles:\n{_profiles_brief(state.get('profiles', {}))}"
    )


def make_clean_propose_node(model: BaseChatModel):
    selector = model.with_structured_output(CleanProposal)

    async def clean_propose(state: AgentState, config: RunnableConfig) -> dict:
        messages = [
            SystemMessage(content=_PROPOSE_SYSTEM),
            HumanMessage(content=_propose_context(state)),
        ]
        result: CleanProposal | None = None
        async for partial in selector.astream(messages, config=config):
            result = partial  # type: ignore[assignment]
        if result is None:
            raise RuntimeError("LLM returned no output for cleaning proposal.")
        return {
            "clean_plan": _validate_ops(state, result.ops),
            "clean_reasoning": result.reasoning,
        }

    return clean_propose


# ── gate nodes (reuse the generic review gate) ──────────────────────────────

clean_review_node = make_review_node(
    lambda state: {
        "kind": "clean",
        "clean_plan": state.get("clean_plan", []),
        "clean_reasoning": state.get("clean_reasoning", ""),
    }
)


def _revise_context(state: AgentState) -> str:
    return (
        f"Current plan:\n{state.get('clean_plan', [])}\n\n"
        f"Profiles:\n{_profiles_brief(state.get('profiles', {}))}"
    )


def _revise_apply(state: AgentState, result: CleanRevision) -> dict:
    return {"clean_plan": _validate_ops(state, result.ops)}


def make_clean_revise_node(model: BaseChatModel):
    return make_revise_node(
        model,
        system_prompt=_REVISE_SYSTEM,
        schema_model=CleanRevision,
        build_context=_revise_context,
        apply=_revise_apply,
    )


# ── deterministic apply ─────────────────────────────────────────────────────

def clean_apply_node(state: AgentState, config: RunnableConfig) -> dict:
    store: FrameStore = config["configurable"]["frame_store"]
    plan = state.get("clean_plan", [])

    by_table: dict[str, list[dict]] = {}
    for op in plan:
        by_table.setdefault(op["table"], []).append(op)

    applied: list[dict] = []
    for table, ops in by_table.items():
        if not store.has(table):
            continue
        df, log = apply_ops(store.get(table), ops)
        store.put(table, df)
        applied.extend(log)

    return {
        "cleaning_log": state.get("cleaning_log", []) + applied,
        "clean_attempted": True,
        "clean_plan": [],
        "clean_request": "",
        "phase": "profile",
    }
