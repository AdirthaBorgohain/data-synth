"""Target-definition phase nodes.

Flow: target_propose (agentic) → target_review (interrupt) → target_revise
      (agentic, optional) → target_apply (deterministic) → eda_propose.

The agent decides which mode to use:
  - "existing"     — pick a column already in profiles.
  - "derived"      — write a column-level SQL expression that becomes the label
                     (e.g. CASE WHEN last_order_at < now() - interval '30 days'
                                 THEN 1 ELSE 0 END).
  - "unsupervised" — declare no target.

For "derived", target_apply runs EXPLAIN to validate the expression, then
materializes the column on the entity table's cached frame in FrameStore and
re-profiles that table so EDA + baseline scoring see a real column.
"""

from __future__ import annotations

from typing import Literal

import polars as pl
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from datasynth.clean.store import FrameStore
from datasynth.datasource.base import DataSource
from datasynth.features import sql_build as sb
from datasynth.graph.nodes._helpers import profiles_brief as _profiles_brief
from datasynth.graph.nodes.profile import profile_single_table
from datasynth.graph.review import ReviewDecision, make_review_node, make_revise_node
from datasynth.graph.state import AgentState

_PROPOSE_SYSTEM = """\
You are defining the prediction target for a machine-learning task. The data has
been profiled. Decide one of three modes:

  - "existing": a column already in the schema cleanly answers the ask
    (reasonable null rate, sensible cardinality, semantically a label).
  - "derived": no column is the label, but one can be computed from existing
    columns. Provide a column-level SQL expression that returns the label for
    each row of the entity table. Use CASE/WHEN, arithmetic, NULL checks,
    interval math, or a correlated subquery against child tables joined on the
    entity primary key. Do NOT write a full SELECT — only the expression.
    Examples:
      CASE WHEN last_login_at < now() - interval '30 days' THEN 1 ELSE 0 END
      (SELECT COUNT(*) FROM orders WHERE orders.customer_id = customers.id) > 0
  - "unsupervised": the ask is segmentation, clustering, or anomaly detection
    and no labelable outcome exists.

Pick exactly one target_table from the selected tables — the entity grain the
prediction operates over (one row per prediction subject).

Set needs_business_rule=true when the rule involves a policy choice the user
should confirm (windows like 30/60/90 days, dollar thresholds, severity cutoffs).
Set it false when the choice is data-driven and unambiguous.

target_column is the final column name to use in the entity frame. For mode
"existing" it must be an existing column on target_table. For mode "derived"
pick a clear new name (e.g. "is_churned", "high_value_flag"). For mode
"unsupervised" leave blank.
"""

_REVISE_SYSTEM = """\
You are revising a target proposal based on the user's comment. The user may:
  - Approve as-is → action="proceed".
  - Provide a business rule → translate it into a column-level SQL expression
    and return mode="derived" with the new derivation_sql.
  - Ask to use an existing column → mode="existing".
  - Say the task is unsupervised → mode="unsupervised".

If the user comment is exactly "unsupervised" (case-insensitive), set
mode="unsupervised" and clear target_column / derivation_sql.

When revising, return the COMPLETE updated target spec — every field as if you
were proposing fresh.
"""


class TargetProposal(BaseModel):
    mode: Literal["existing", "derived", "unsupervised"]
    target_table: str | None = Field(
        default=None,
        description="Qualified table (schema.table) holding the entity; None for unsupervised",
    )
    target_column: str = Field(
        default="",
        description="Final column name (existing for 'existing' mode, new name for 'derived'). Empty for unsupervised.",
    )
    derivation_sql: str = Field(
        default="",
        description="Column-level SQL expression for mode='derived'. Empty otherwise.",
    )
    reasoning: str = Field(description="Why this mode/column/expression fits the ask")
    needs_business_rule: bool = Field(
        default=False,
        description="True when the expression encodes a policy choice (window, threshold) the user should confirm",
    )


class TargetRevision(ReviewDecision):
    action: Literal["proceed", "revise"] = Field(default="proceed")
    mode: Literal["existing", "derived", "unsupervised"] | None = Field(default=None)
    target_table: str | None = Field(default=None)
    target_column: str = Field(default="")
    derivation_sql: str = Field(default="")
    needs_business_rule: bool = Field(default=False)


# ── helpers ─────────────────────────────────────────────────────────────────

def _validate_proposal(state: AgentState, p: TargetProposal) -> dict:
    """Normalize a proposal to a state dict. Coerces obviously bad inputs."""
    selected = set(state.get("selected_tables", []))
    profiles = state.get("profiles", {})

    if p.mode == "unsupervised":
        return {
            "target_mode": "unsupervised",
            "target_table": "",
            "target_column": "",
            "target_derivation_sql": "",
            "target_reasoning": p.reasoning,
        }

    tt = p.target_table if p.target_table in selected else ""
    if not tt:
        # Fall back: pick the first selected table as a best-effort entity.
        tt = next(iter(selected), "")

    if p.mode == "existing":
        cols = set(profiles.get(tt, {}).get("columns", {}))
        if p.target_column not in cols:
            # Can't validate further here; downstream apply will refuse.
            return {
                "target_mode": "existing",
                "target_table": tt,
                "target_column": p.target_column,
                "target_derivation_sql": "",
                "target_reasoning": p.reasoning + " [warning: column not in profiles]",
            }
        return {
            "target_mode": "existing",
            "target_table": tt,
            "target_column": p.target_column,
            "target_derivation_sql": "",
            "target_reasoning": p.reasoning,
        }

    # derived
    return {
        "target_mode": "derived",
        "target_table": tt,
        "target_column": p.target_column or "target_label",
        "target_derivation_sql": p.derivation_sql,
        "target_reasoning": p.reasoning,
    }


# ── propose ─────────────────────────────────────────────────────────────────

def make_target_propose_node(model: BaseChatModel):
    selector = model.with_structured_output(TargetProposal)

    async def target_propose(state: AgentState, config: RunnableConfig) -> dict:
        request = state.get("target_request", "")
        context = (
            (f"User business rule / request: {request}\n\n" if request else "")
            + f"User ask: {state.get('ask', '')}\n\n"
            f"Selected tables: {state.get('selected_tables', [])}\n\n"
            f"Profiles:\n{_profiles_brief(state.get('profiles', {}))}"
        )
        messages = [
            SystemMessage(content=_PROPOSE_SYSTEM),
            HumanMessage(content=context),
        ]
        result: TargetProposal | None = None
        async for partial in selector.astream(messages, config=config):
            result = partial  # type: ignore[assignment]
        if result is None:
            raise RuntimeError("LLM returned no output for target proposal.")

        update = _validate_proposal(state, result)
        update["needs_business_rule"] = result.needs_business_rule
        update["target_request"] = ""
        update["phase"] = "target"
        return update

    return target_propose


# ── review gate ─────────────────────────────────────────────────────────────

target_review_node = make_review_node(
    lambda state: {
        "kind": "target",
        "mode": state.get("target_mode", ""),
        "target_table": state.get("target_table", ""),
        "target_column": state.get("target_column", ""),
        "derivation_sql": state.get("target_derivation_sql", ""),
        "reasoning": state.get("target_reasoning", ""),
        "needs_business_rule": bool(state.get("needs_business_rule", False)),
    }
)


def _revise_context(state: AgentState) -> str:
    return (
        f"Original ask: {state.get('ask', '')}\n\n"
        f"Current proposal:\n"
        f"  mode = {state.get('target_mode', '')}\n"
        f"  target_table = {state.get('target_table', '')}\n"
        f"  target_column = {state.get('target_column', '')}\n"
        f"  derivation_sql = {state.get('target_derivation_sql', '')}\n\n"
        f"Selected tables: {state.get('selected_tables', [])}\n\n"
        f"Profiles:\n{_profiles_brief(state.get('profiles', {}))}"
    )


def _revise_apply(state: AgentState, result: TargetRevision) -> dict:
    # Treat the revised payload like a fresh proposal.
    proposal = TargetProposal(
        mode=result.mode or state.get("target_mode") or "existing",
        target_table=result.target_table or state.get("target_table") or None,
        target_column=result.target_column or state.get("target_column") or "",
        derivation_sql=result.derivation_sql or state.get("target_derivation_sql") or "",
        reasoning=result.reasoning,
        needs_business_rule=result.needs_business_rule,
    )
    update = _validate_proposal(state, proposal)
    update["needs_business_rule"] = result.needs_business_rule
    return update


def make_target_revise_node(model: BaseChatModel):
    return make_revise_node(
        model,
        system_prompt=_REVISE_SYSTEM,
        schema_model=TargetRevision,
        build_context=_revise_context,
        apply=_revise_apply,
    )


# ── deterministic apply ─────────────────────────────────────────────────────

def make_target_apply_node(cfg: dict):
    def _abandon(state: AgentState, reason: str) -> dict:
        return {
            "target_table": "",
            "target_column": "",
            "target_mode": "unsupervised",
            "target_reasoning": state.get("target_reasoning", "") + f" [derived target abandoned: {reason}]",
            "target_is_binary": False,
            "phase": "eda",
        }

    def target_apply(state: AgentState, config: RunnableConfig) -> dict:
        mode = state.get("target_mode", "")
        target_table = state.get("target_table", "")
        target_column = state.get("target_column", "")

        if mode == "unsupervised":
            return {
                "target_mode": "unsupervised",
                "target_table": "",
                "target_column": "",
                "target_is_binary": False,
                "phase": "eda",
            }

        ds: DataSource = config["configurable"]["datasource"]
        store: FrameStore = config["configurable"]["frame_store"]
        profiles = dict(state.get("profiles", {}))

        if mode == "existing":
            stats = profiles.get(target_table, {}).get("columns", {}).get(target_column, {})
            n_unique = stats.get("n_unique")
            return {
                "target_mode": "existing",
                "target_table": target_table,
                "target_column": target_column,
                "target_is_binary": isinstance(n_unique, int) and n_unique == 2,
                "phase": "eda",
            }

        # mode == "derived"
        derivation_sql = state.get("target_derivation_sql", "")
        if not target_table or not derivation_sql:
            return _abandon(state, "missing table or expression")

        schema = ds.introspect_schema()
        entity_key = sb.entity_key(schema, target_table)
        if not entity_key:
            return _abandon(state, f"no primary key on {target_table}")

        if not store.has(target_table):
            return _abandon(state, f"no cached frame for {target_table}")

        existing_cols = set(store.get(target_table).columns) - {entity_key}
        final_col = target_column
        if final_col in existing_cols:
            final_col = f"{target_column}_label"
            i = 2
            while final_col in existing_cols:
                final_col = f"{target_column}_label_{i}"
                i += 1

        # Schema-qualified table reference; quote both parts defensively.
        if "." in target_table:
            schema_name, tbl_name = target_table.split(".", 1)
            qualified = f'"{schema_name}"."{tbl_name}"'
        else:
            qualified = f'"{target_table}"'

        # 1. EXPLAIN dry-run on a LIMIT 0 query to catch syntax/column errors.
        validate_sql = (
            f'SELECT "{entity_key}", ({derivation_sql}) AS "{final_col}" '
            f'FROM {qualified} LIMIT 0'
        )
        try:
            ds.run_sql(f"EXPLAIN {validate_sql}", max_rows=1)
        except Exception as exc:  # noqa: BLE001
            return _abandon(state, f"EXPLAIN failed — {exc}")

        # 2. Materialize the expression for the sampled population only.
        sample_size = int(cfg.get("sample_size", 10000))
        fetch_sql = (
            f'SELECT "{entity_key}", ({derivation_sql}) AS "{final_col}" '
            f'FROM {qualified} LIMIT {sample_size}'
        )
        try:
            label_df = ds.run_sql(fetch_sql, max_rows=sample_size)
        except Exception as exc:  # noqa: BLE001
            return _abandon(state, f"fetch failed — {exc}")

        # 3. Polars-join onto the cached frame on entity_key.
        frame = store.get(target_table)
        if entity_key not in frame.columns:
            return _abandon(state, f"{entity_key} missing in cached frame")
        try:
            augmented = frame.join(label_df, on=entity_key, how="left")
        except Exception as exc:  # noqa: BLE001
            return _abandon(state, f"join failed — {exc}")
        store.put(target_table, augmented)

        # 4. Re-profile the entity table so EDA + baseline see the new column.
        new_profile, _ = profile_single_table(target_table, ds, store)
        profiles[target_table] = new_profile
        n_unique = new_profile.get("columns", {}).get(final_col, {}).get("n_unique")

        warning = ""
        if isinstance(n_unique, int) and n_unique <= 1:
            warning = f" [warning: derived target has only {n_unique} unique value(s) in sample]"

        return {
            "profiles": profiles,
            "target_mode": "derived",
            "target_table": target_table,
            "target_column": final_col,
            "target_is_binary": isinstance(n_unique, int) and n_unique == 2,
            "target_reasoning": state.get("target_reasoning", "") + warning,
            "phase": "eda",
        }

    return target_apply
