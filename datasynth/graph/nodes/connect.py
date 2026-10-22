"""Connect phase: introspect schema, LLM selects relevant tables.

Uses with_structured_output(TableSelection) so the response is a typed Pydantic
object. Node is async and calls selector.astream(config=config) so that
on_chat_model_stream events (Anthropic thinking blocks) propagate through
astream_events to the CLI renderer.

DataSource is read from config["configurable"]["datasource"] — not stored in state.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from datasynth.datasource.base import DataSource, SchemaInfo
from datasynth.graph.state import AgentState

_SYSTEM = """\
You are a data analyst helping build a machine learning model.
Given a user's ML question and a database schema, select the tables needed
to answer it. Include tables that contribute features, the target variable,
or essential joins. Exclude unrelated tables.
"""


class TableSelection(BaseModel):
    reasoning: str = Field(description="Why these tables are relevant to the question")
    tables: list[str] = Field(
        description="Qualified table names (schema.table) to analyze"
    )


def _format_schema(schema: SchemaInfo) -> str:
    lines: list[str] = []
    for t in schema.tables:
        col_parts = []
        for c in t.columns:
            tag = " [PK]" if c.is_primary_key else ""
            tag += " nullable" if c.nullable else ""
            col_parts.append(f"    {c.name} ({c.dtype}){tag}")
        fk_parts = [
            f"    FK: {fk.column} → {fk.references_table}.{fk.references_column}"
            for fk in t.foreign_keys
        ]
        lines.append(t.qualified_name)
        lines.extend(col_parts)
        lines.extend(fk_parts)
        lines.append("")
    return "\n".join(lines)


def _schema_to_dict(schema: SchemaInfo) -> list[dict]:
    return [
        {
            "qualified_name": t.qualified_name,
            "schema": t.schema,
            "name": t.name,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "nullable": c.nullable,
                    "is_primary_key": c.is_primary_key,
                }
                for c in t.columns
            ],
            "foreign_keys": [
                {
                    "column": fk.column,
                    "ref_table": fk.references_table,
                    "ref_column": fk.references_column,
                }
                for fk in t.foreign_keys
            ],
        }
        for t in schema.tables
    ]


def make_connect_node(model: BaseChatModel):
    selector = model.with_structured_output(TableSelection)

    async def connect(state: AgentState, config: RunnableConfig) -> dict:
        ds: DataSource = config["configurable"]["datasource"]
        schema = ds.introspect_schema()

        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(
                content=(
                    f"Question: {state['ask']}\n\n"
                    f"Database schema:\n{_format_schema(schema)}"
                )
            ),
        ]

        # astream propagates on_chat_model_stream events (thinking blocks) through
        # astream_events. The last yielded chunk is the complete TableSelection.
        result: TableSelection | None = None
        async for partial in selector.astream(messages, config=config):
            result = partial  # type: ignore[assignment]

        if result is None:
            raise RuntimeError("LLM returned no output for table selection.")

        # Drop any names the LLM hallucinated
        valid = {t.qualified_name for t in schema.tables} | {t.name for t in schema.tables}
        tables = [t for t in result.tables if t in valid]

        return {
            "schema_tables": _schema_to_dict(schema),
            "selected_tables": tables,
            "selection_reasoning": result.reasoning,
            "phase": "profile",
        }

    return connect
