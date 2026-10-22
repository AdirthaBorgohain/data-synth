"""Deterministic cleaning operations.

The LLM proposes ops from this fixed vocabulary; apply_ops executes them on a
polars frame in order and returns a log of what actually happened. Anything the
executor can't run (unknown op, missing column, bad params) is recorded as
skipped rather than raising, so one bad op never aborts a cleaning cycle.
"""

from __future__ import annotations

import polars as pl

# op name → human description used in prompts and the terminal
OP_VOCABULARY: dict[str, str] = {
    "drop_column": "remove a column entirely",
    "cast": "change a column's dtype (params.dtype)",
    "fill_null": "fill nulls (params.strategy mean/median/zero/forward or params.value)",
    "drop_nulls": "drop rows where the column is null (all columns if none given)",
    "trim": "strip leading/trailing whitespace from a string column",
    "dedupe": "drop duplicate rows",
    "clip": "clip numeric outliers to params.lower / params.upper",
}

_DTYPES: dict[str, pl.DataType] = {
    "Int64": pl.Int64,
    "Float64": pl.Float64,
    "Utf8": pl.Utf8,
    "String": pl.Utf8,
    "Boolean": pl.Boolean,
    "Date": pl.Date,
    "Datetime": pl.Datetime,
}

_FILL_STRATEGIES = {"forward", "backward", "min", "max", "mean", "zero", "one"}


def _apply_one(df: pl.DataFrame, op: dict) -> pl.DataFrame:
    name = op.get("op")
    col = op.get("column")
    params = op.get("params") or {}

    if name == "drop_column":
        return df.drop(col)

    if name == "cast":
        dtype = _DTYPES.get(params.get("dtype", ""))
        if dtype is None:
            raise ValueError(f"unsupported dtype {params.get('dtype')!r}")
        # Object columns (e.g. UUID/Decimal returned by the DB driver) can't be
        # cast directly in polars — stringify the Python values first.
        if df.schema.get(col) == pl.Object:
            expr = pl.col(col).map_elements(
                lambda v: None if v is None else str(v), return_dtype=pl.String
            )
            if dtype != pl.Utf8:
                expr = expr.cast(dtype, strict=False)
            return df.with_columns(expr)
        return df.with_columns(pl.col(col).cast(dtype, strict=False))

    if name == "fill_null":
        if "value" in params and params["value"] is not None:
            return df.with_columns(pl.col(col).fill_null(params["value"]))
        strategy = params.get("strategy", "zero")
        if strategy == "median":
            return df.with_columns(pl.col(col).fill_null(pl.col(col).median()))
        if strategy in _FILL_STRATEGIES:
            return df.with_columns(pl.col(col).fill_null(strategy=strategy))
        raise ValueError(f"unsupported fill strategy {strategy!r}")

    if name == "drop_nulls":
        return df.drop_nulls(subset=[col] if col else None)

    if name == "trim":
        return df.with_columns(pl.col(col).str.strip_chars())

    if name == "dedupe":
        return df.unique(maintain_order=True)

    if name == "clip":
        return df.with_columns(
            pl.col(col).clip(params.get("lower"), params.get("upper"))
        )

    raise ValueError(f"unknown op {name!r}")


def apply_ops(df: pl.DataFrame, ops: list[dict]) -> tuple[pl.DataFrame, list[dict]]:
    """Apply ops to df in order. Returns (new_df, applied_log).

    Each log entry copies the op plus rows_before/rows_after and either
    applied=True or applied=False with an `error` string.
    """
    log: list[dict] = []
    for op in ops:
        before = len(df)
        try:
            df = _apply_one(df, op)
            log.append({**op, "rows_before": before, "rows_after": len(df), "applied": True})
        except Exception as exc:
            log.append({**op, "rows_before": before, "rows_after": before, "applied": False, "error": str(exc)})
    return df, log
