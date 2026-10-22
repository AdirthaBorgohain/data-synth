"""Profiling-phase instance of the reusable review gate (datasynth/graph/review.py).

Configures the generic gate for table *selection* review: after profiling, the
user comments, and the LLM decides whether to add/remove tables. The only
phase-specific pieces are the four hooks and the TableRevision schema; all
mechanics live in review.py.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field

from datasynth.graph.review import (
    ReviewDecision,
    make_review_node,
    make_revise_node,
)
from datasynth.graph.state import AgentState

_SYSTEM = """\
You are a data analyst refining the set of tables chosen for a machine learning
question. The user has reviewed the profiled tables and left a comment.

Choose one action:
  - "proceed": the comment needs no change (e.g. a question, an approval, or a
    remark unrelated to the tables or their data).
  - "revise": the comment asks to add and/or remove tables. Return the COMPLETE
    desired list of qualified table names (schema.table), not a diff.
  - "clean": the comment asks to clean, fix, or transform the *data* of a table
    or column (e.g. fill nulls, cast a type, drop outliers). Routing will hand
    off to the cleaning agent.

Only use table names that exist in the provided schema.
"""


class TableRevision(ReviewDecision):
    action: Literal["proceed", "revise", "clean"] = Field(
        default="proceed",
        description="proceed: no change. revise: change table list. clean: clean table data.",
    )
    tables: list[str] = Field(
        default_factory=list,
        description="Complete revised list of qualified table names (used when action='revise')",
    )


def _format_schema(schema_tables: list[dict]) -> str:
    lines: list[str] = []
    for t in schema_tables:
        lines.append(t["qualified_name"])
        for c in t.get("columns", []):
            tag = " [PK]" if c.get("is_primary_key") else ""
            tag += " nullable" if c.get("nullable") else ""
            lines.append(f"    {c['name']} ({c['dtype']}){tag}")
        for fk in t.get("foreign_keys", []):
            lines.append(
                f"    FK: {fk['column']} → {fk['ref_table']}.{fk['ref_column']}"
            )
        lines.append("")
    return "\n".join(lines)


def _format_profiles(profiles: dict[str, dict]) -> str:
    lines: list[str] = []
    for table, profile in profiles.items():
        if "error" in profile:
            lines.append(f"{table}: ERROR {profile['error']}")
            continue
        n = profile.get("row_count_sample", "?")
        cols = profile.get("columns", {})
        col_bits = [
            f"{name} ({stats.get('dtype', '?')}, null {stats.get('null_rate', 0):.0%})"
            for name, stats in cols.items()
        ]
        lines.append(f"{table} ({n} sample rows): " + ", ".join(col_bits))
    return "\n".join(lines)


def _select_payload(state: AgentState) -> dict:
    return {"kind": "select", "selected_tables": state.get("selected_tables", [])}


def _select_context(state: AgentState) -> str:
    return (
        f"Currently selected tables: {state.get('selected_tables', [])}\n\n"
        f"Full database schema:\n{_format_schema(state.get('schema_tables', []))}\n\n"
        f"Profiles of selected tables:\n{_format_profiles(state.get('profiles', {}))}"
    )


def _select_apply(state: AgentState, result: TableRevision) -> dict:
    # Comment-driven cleaning: hand the comment to the clean agent and re-arm
    # the reactive clean guard so the cycle can run again.
    if result.action == "clean":
        return {
            "clean_request": state.get("user_comment", ""),
            "clean_attempted": False,
            "selection_reasoning": result.reasoning,
        }

    schema_tables = state.get("schema_tables", [])
    valid = {t["qualified_name"] for t in schema_tables} | {
        t["name"] for t in schema_tables
    }
    tables = [t for t in result.tables if t in valid]
    return {"selected_tables": tables, "selection_reasoning": result.reasoning}


select_review_node = make_review_node(_select_payload)


def make_select_revise_node(model: BaseChatModel):
    return make_revise_node(
        model,
        system_prompt=_SYSTEM,
        schema_model=TableRevision,
        build_context=_select_context,
        apply=_select_apply,
    )
