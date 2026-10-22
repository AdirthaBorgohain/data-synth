"""Agent state shared across all graph nodes.

Serialization constraint: DataSource and polars DataFrames are not
JSON-serializable. DataSource is injected via config["configurable"]["datasource"]
and never stored here. Profile data is stored as plain dicts.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    ask: str
    messages: Annotated[list[BaseMessage], add_messages]
    # Serializable schema snapshot: list of table dicts (name, columns, FKs)
    schema_tables: list[dict]
    # Qualified table names selected by the connect node
    selected_tables: list[str]
    selection_reasoning: str
    # table_name → {row_count_sample, columns: {col → {dtype, null_rate, n_unique, sample_values}}}
    profiles: dict[str, dict]
    # Tracks current pipeline phase for checkpointing / display
    phase: str
    # ── human-in-the-loop review gate (phase-agnostic, set per gate) ──
    # Free-text comment captured from the last review interrupt; blank = proceed.
    user_comment: str
    # Action chosen by a revise node (e.g. "proceed" | "revise" | "clean"),
    # read by revise routing.
    revise_action: str
    # ── data-cleaning phase ──────────────────────────────────────────
    # Quality flags from profiling that decide whether to enter clean.
    profile_issues: list[dict]
    # Proposed cleaning ops: {table, op, column, params, reason}.
    clean_plan: list[dict]
    # Agent's overall rationale for the proposed cleaning (rendered).
    clean_reasoning: str
    # Applied ops, accumulated across cycles (lineage + terminal display).
    cleaning_log: list[dict]
    # Loop guard: reactive clean fires at most once automatically.
    clean_attempted: bool
    # Carries a user's comment-driven clean request into the propose node.
    clean_request: str
    # ── target-definition phase ───────────────────────────────────────
    # Target/label the user wants to predict. May be None for unsupervised.
    target_table: str
    target_column: str
    # "existing" | "derived" | "unsupervised"
    target_mode: str
    # Column-level SQL expression (CASE/WHEN…) when target_mode == "derived".
    target_derivation_sql: str
    # Agent's rationale for the chosen mode/column (rendered).
    target_reasoning: str
    # True when the materialized target has exactly 2 distinct non-null values.
    target_is_binary: bool
    # Carries a user's revised business-rule comment back into target_propose.
    target_request: str
    # Surfaced at the target review gate to ask the user for a business rule.
    needs_business_rule: bool
    # ── EDA phase ─────────────────────────────────────────────────────
    # Analyses chosen by the propose node: {analysis, table, columns, params, reason}.
    eda_plan: list[dict]
    # Agent's rationale for the EDA plan (rendered).
    eda_reasoning: str
    # Deterministic analysis output: {table → {analysis → result}}.
    eda_results: dict
    # LLM-synthesized overview of the data's character.
    eda_summary: str
    # Conclusions for downstream phases: {topic, observation, downstream_implication}.
    eda_findings: list[dict]
    # Saved plot metadata: {title, path, kind, table, analysis}.
    eda_plots: list[dict]
    # ── feature generation phase ──────────────────────────────────────
    # Entity grain for the modeling matrix (target table + its primary key).
    entity_table: str
    entity_key: str
    # Current feature spec list (the exportable recipe): {kind, name, ...}.
    feature_plan: list[dict]
    # Agent's rationale for the latest plan/critique (rendered).
    feat_reasoning: str
    # Assembled matrix metadata: {n_rows, n_features, columns, dtypes, accepted_aggregates}.
    feature_matrix_meta: dict
    # Full scorecard from feat_evaluate: {task, metric_*, features, leakage, redundancy}.
    feature_scores: dict
    # Convenience copy of the baseline metric: {name, value}.
    baseline_metric: dict
    # One-line evaluation summary (rendered).
    feat_eval_summary: str
    # Per-iteration record: {iteration, metric, n_features, action}.
    feat_history: list[dict]
    # Spec diff from the last critique: {added, dropped, kept}.
    feat_changes: dict
    # Critique decision: "refine" | "done" (drives the loop router).
    feat_action: str
    # Loop counters / thresholds (seeded by feat_plan from settings).
    feat_iteration: int
    feat_max_iterations: int
    feat_metric_epsilon: float
    # Build skips + transform/encode logs (lineage + display).
    feat_build_log: list[dict]
