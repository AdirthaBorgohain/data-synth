"""Deterministic in-memory feature transforms + categorical encoders (polars).

Applied to the assembled entity-grain matrix after SQL. Each spec runs in order;
failures are recorded and skipped (never raised) so one bad spec can't abort a
build — mirroring datasynth.clean.ops.apply_ops.

transform spec:  {op, name, params:{input | input_a | input_b | k | part}}
encode spec:     {method, column, name?}
"""

from __future__ import annotations

import polars as pl

from datasynth.features.sql_build import slug_ident


def _apply_transform(df: pl.DataFrame, spec: dict) -> tuple[pl.DataFrame, str]:
    op = spec.get("op")
    p = spec.get("params") or {}
    name = slug_ident(spec.get("name") or f"{op}_feat")

    if op == "log":
        col = p["input"]
        return df.with_columns(
            pl.when(pl.col(col) >= 0).then((pl.col(col) + 1).log()).otherwise(None).alias(name)
        ), name
    if op == "ratio":
        a, b = p["input_a"], p["input_b"]
        return df.with_columns(
            pl.when(pl.col(b) != 0).then(pl.col(a) / pl.col(b)).otherwise(None).alias(name)
        ), name
    if op == "interaction":
        a, b = p["input_a"], p["input_b"]
        return df.with_columns((pl.col(a) * pl.col(b)).alias(name)), name
    if op == "bin":
        col, k = p["input"], int(p.get("k", 4))
        return df.with_columns(pl.col(col).qcut(k, allow_duplicates=True, labels=[str(i) for i in range(k)]).alias(name)), name
    if op == "datetime_part":
        col, part = p["input"], p.get("part", "year")
        expr = {"year": pl.col(col).dt.year(), "month": pl.col(col).dt.month(),
                "dow": pl.col(col).dt.weekday()}.get(part, pl.col(col).dt.year())
        return df.with_columns(expr.alias(name)), name
    if op == "tenure_days":
        col = p["input"]
        return df.with_columns(
            ((pl.lit(pl.datetime(2100, 1, 1)).cast(pl.Datetime) - pl.col(col).cast(pl.Datetime))
             .dt.total_days()).alias(name)
        ), name
    if op == "missing_indicator":
        col = p["input"]
        return df.with_columns(pl.col(col).is_null().cast(pl.Int8).alias(name)), name

    raise ValueError(f"unknown transform {op!r}")


def apply_transforms(df: pl.DataFrame, specs: list[dict]) -> tuple[pl.DataFrame, list[dict]]:
    log: list[dict] = []
    for spec in specs:
        try:
            inputs = [v for k, v in (spec.get("params") or {}).items()
                      if k.startswith("input")]
            missing = [c for c in inputs if c not in df.columns]
            if missing:
                raise ValueError(f"missing input columns {missing}")
            df, name = _apply_transform(df, spec)
            log.append({**spec, "output": name, "applied": True})
        except Exception as exc:  # noqa: BLE001
            log.append({**spec, "applied": False, "error": str(exc)})
    return df, log


def apply_encoders(
    df: pl.DataFrame, specs: list[dict], target: str | None
) -> tuple[pl.DataFrame, list[dict]]:
    log: list[dict] = []
    for spec in specs:
        method = spec.get("method")
        col = spec.get("column")
        try:
            if col not in df.columns:
                raise ValueError(f"missing column {col!r}")
            name = slug_ident(spec.get("name") or f"{col}_{method}")
            if method == "frequency":
                counts = df.group_by(col).len().rename({"len": name})
                df = df.join(counts, on=col, how="left")
            elif method == "target":
                if not target or target not in df.columns:
                    raise ValueError("target encoding needs a target column")
                means = (
                    df.select([col, pl.col(target).cast(pl.Float64)])
                    .group_by(col).agg(pl.col(target).mean().alias(name))
                )
                df = df.join(means, on=col, how="left")
            elif method == "one_hot":
                nu = df[col].n_unique()
                if nu > 20:
                    raise ValueError(f"cardinality {nu} too high for one_hot")
                dummies = df.select(col).to_dummies(col)
                dummies.columns = [slug_ident(c) for c in dummies.columns]
                df = pl.concat([df, dummies], how="horizontal")
                name = f"{col} → {dummies.width} cols"
            else:
                raise ValueError(f"unknown encoder {method!r}")
            log.append({**spec, "output": name, "applied": True})
        except Exception as exc:  # noqa: BLE001
            log.append({**spec, "applied": False, "error": str(exc)})
    return df, log
