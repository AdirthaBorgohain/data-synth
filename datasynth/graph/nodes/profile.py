"""Profile phase: sample each selected table and compute descriptive stats.

Stats are stored as plain dicts (JSON-serializable) in state["profiles"].
DataSource is read from config["configurable"]["datasource"].
"""

from __future__ import annotations

import polars as pl
from langchain_core.runnables import RunnableConfig

from datasynth.clean.store import FrameStore
from datasynth.datasource.base import DataSource
from datasynth.graph.state import AgentState

# Report n_unique for columns below this threshold; above it we still count but
# skip the sample_values list (too noisy for the LLM context).
_CARDINALITY_SAMPLE_CUTOFF = 50

# Columns at or above this null rate are flagged as a cleaning candidate.
_HIGH_NULL = 0.40

# polars dtypes that signal a column failed to infer / carries mixed data.
_BAD_DTYPES = {"Null", "Object"}


def _profile_df(df: pl.DataFrame) -> dict:
    n = len(df)
    columns: dict[str, dict] = {}
    for col in df.columns:
        series = df[col]
        null_count = series.null_count()
        null_rate = round(null_count / n, 4) if n > 0 else 0.0

        n_unique: int | None = None
        sample_values: list = []
        try:
            unique = series.drop_nulls().unique()
            n_unique = len(unique)
            if n_unique <= _CARDINALITY_SAMPLE_CUTOFF:
                sample_values = sorted(
                    [v for v in unique.to_list() if v is not None],
                    key=str,
                )[:_CARDINALITY_SAMPLE_CUTOFF]
            else:
                sample_values = series.drop_nulls().head(5).to_list()
        except Exception:
            pass

        columns[col] = {
            "dtype": str(series.dtype),
            "null_rate": null_rate,
            "n_unique": n_unique,
            "sample_values": sample_values,
        }

    return {"row_count_sample": n, "columns": columns}


def _detect_issues(table: str, profile: dict) -> list[dict]:
    """Flag quality problems worth a cleaning pass. Decides *whether* to enter
    clean; the LLM decides the actual plan."""
    if "error" in profile:
        return [{"table": table, "kind": "load_error", "detail": profile["error"]}]

    issues: list[dict] = []
    for col, stats in profile.get("columns", {}).items():
        if stats.get("null_rate", 0.0) >= _HIGH_NULL:
            issues.append(
                {"table": table, "column": col, "kind": "high_null",
                 "detail": f"null_rate={stats['null_rate']}"}
            )
        if stats.get("dtype") in _BAD_DTYPES:
            issues.append(
                {"table": table, "column": col, "kind": "bad_dtype",
                 "detail": stats.get("dtype")}
            )
    return issues


def _coerce_objects(df: pl.DataFrame) -> pl.DataFrame:
    """Cast Object-dtype columns to String.

    Polars uses Object as a fallback for Python objects that don't map to a
    native dtype (e.g. PG UUID, unrecognised SQLAlchemy types). Object columns
    break profiling, Parquet serialisation, and most Polars expressions, so we
    convert them to String strings eagerly at sample time.
    """
    object_cols = [c for c, d in zip(df.columns, df.dtypes) if d == pl.Object]
    if not object_cols:
        return df
    return df.with_columns([
        pl.col(c).map_elements(lambda v: None if v is None else str(v), return_dtype=pl.String)
        for c in object_cols
    ])


def profile_single_table(
    table: str, ds: DataSource, store: FrameStore
) -> tuple[dict, list[dict]]:
    """Profile one table; returns (profile, issues). Used by profile_node and
    target_apply (to re-profile after derived label is materialized)."""
    try:
        if store.has(table):
            df = store.get(table)
        else:
            df = _coerce_objects(ds.sample(table))
            store.put(table, df)
        profile = _profile_df(df)
    except Exception as exc:
        profile = {"error": str(exc)}
    return profile, _detect_issues(table, profile)


def profile_node(state: AgentState, config: RunnableConfig) -> dict:
    ds: DataSource = config["configurable"]["datasource"]
    store: FrameStore = config["configurable"]["frame_store"]
    profiles: dict[str, dict] = {}
    issues: list[dict] = []

    for table in state.get("selected_tables", []):
        profile, table_issues = profile_single_table(table, ds, store)
        profiles[table] = profile
        issues.extend(table_issues)

    return {"profiles": profiles, "profile_issues": issues, "phase": "eda"}
