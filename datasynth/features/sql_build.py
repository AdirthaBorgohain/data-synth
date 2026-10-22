"""Compile validated feature specs into one read-only entity-grain SQL query.

No raw LLM SQL is ever executed. Identifiers are checked against the introspected
SchemaInfo and re-validated by a strict regex before being double-quoted; aggregate
functions come from the AGG_FUNCS whitelist. The assembled query still passes through
assert_readonly + max_rows in run_sql as defense-in-depth.

Shape:
    WITH base AS (SELECT "pk", "c1", ... FROM "schema"."entity" LIMIT n),
         agg_<child> AS (SELECT c."fk" AS k, <aggs> FROM "schema"."child" c
                         JOIN base b ON c."fk" = b."pk" GROUP BY c."fk")
    SELECT b.*, agg_<child>."f1", ... FROM base b
    LEFT JOIN agg_<child> ON agg_<child>.k = b."pk"

Child aggregates are joined to the sampled base set, so scans stay bounded and the
result is exactly one row per sampled entity.
"""

from __future__ import annotations

import re

import polars as pl

from datasynth.datasource.base import DataSource, SchemaInfo, TableInfo

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_SQL_AGG = {
    "count": lambda col: "COUNT(*)",
    "count_distinct": lambda col: f'COUNT(DISTINCT c.{_q(col)})',
    "sum": lambda col: f'SUM(c.{_q(col)})',
    "avg": lambda col: f'AVG(c.{_q(col)})',
    "min": lambda col: f'MIN(c.{_q(col)})',
    "max": lambda col: f'MAX(c.{_q(col)})',
    "stddev": lambda col: f'STDDEV(c.{_q(col)})',
    "recency_days": lambda col: f'EXTRACT(EPOCH FROM (now() - MAX(c.{_q(col)}))) / 86400.0',
}


def _q(name: str) -> str:
    if not _IDENT.match(name or ""):
        raise ValueError(f"unsafe identifier {name!r}")
    return f'"{name}"'


def slug_ident(name: str) -> str:
    """Coerce a proposed feature name to a safe SQL identifier."""
    s = re.sub(r"[^A-Za-z0-9_]+", "_", name or "").strip("_").lower()
    if not s or s[0].isdigit():
        s = f"f_{s}"
    return s[:60]


def entity_key(schema: SchemaInfo, entity_table: str) -> str | None:
    """Primary-key column of the entity table (first PK if composite)."""
    t = schema.table(entity_table)
    if t is None:
        return None
    pks = [c.name for c in t.columns if c.is_primary_key]
    return pks[0] if pks else None


def child_links(
    schema: SchemaInfo, entity_table: str, selected_tables: list[str]
) -> dict[str, str]:
    """Map each selected child table → its FK column referencing the entity table."""
    ent = schema.table(entity_table)
    if ent is None:
        return {}
    ent_names = {ent.name, ent.qualified_name}
    links: dict[str, str] = {}
    for qt in selected_tables:
        if qt == entity_table:
            continue
        t = schema.table(qt)
        if t is None:
            continue
        for fk in t.foreign_keys:
            if fk.references_table in ent_names:
                links[t.qualified_name] = fk.column
                break
    return links


def _table_columns(schema: SchemaInfo, qualified: str) -> set[str]:
    t = schema.table(qualified)
    return {c.name for c in t.columns} if t else set()


def build_matrix_sql(
    schema: SchemaInfo,
    entity_table: str,
    key: str,
    base_columns: list[str],
    agg_specs: list[dict],
    links: dict[str, str],
    n: int,
) -> tuple[str, list[str], list[dict]]:
    """Return (sql, accepted_feature_names, skipped_log).

    base_columns: entity columns to pass through (key is always included).
    agg_specs: [{child_table, agg_fn, source_column, name}].
    """
    ent: TableInfo | None = schema.table(entity_table)
    if ent is None or not _IDENT.match(key):
        raise ValueError(f"invalid entity {entity_table!r} / key {key!r}")

    ent_cols = _table_columns(schema, entity_table)
    keep = [c for c in dict.fromkeys([key, *base_columns]) if c in ent_cols]
    base_sql = (
        f"base AS (SELECT {', '.join(_q(c) for c in keep)} "
        f"FROM {_q(ent.schema)}.{_q(ent.name)} LIMIT {int(n)})"
    )

    skipped: list[dict] = []
    accepted: list[str] = []
    used_aliases: set[str] = set(keep)
    by_child: dict[str, list[tuple[str, str]]] = {}  # child → [(alias, agg_sql)]

    for spec in agg_specs:
        child = spec.get("child_table")
        fn = spec.get("agg_fn")
        col = spec.get("source_column")
        if child not in links or fn not in _SQL_AGG:
            skipped.append({**spec, "skipped": "unknown child or agg_fn"})
            continue
        needs_col = fn != "count"
        if needs_col and (not col or col not in _table_columns(schema, child)):
            skipped.append({**spec, "skipped": "missing/invalid source_column"})
            continue
        alias = slug_ident(spec.get("name") or f"{schema.table(child).name}_{fn}_{col or 'all'}")
        while alias in used_aliases:
            alias += "_x"
        used_aliases.add(alias)
        try:
            agg_sql = _SQL_AGG[fn](col)
        except ValueError as exc:
            skipped.append({**spec, "skipped": str(exc)})
            continue
        by_child.setdefault(child, []).append((alias, agg_sql))
        accepted.append(alias)

    child_ctes: list[str] = []
    final_selects: list[str] = ["b.*"]
    joins: list[str] = []
    for i, (child, aggs) in enumerate(by_child.items()):
        ct = schema.table(child)
        fk = links[child]
        cte = f"agg_{i}"
        agg_cols = ", ".join(f"{sql} AS {_q(alias)}" for alias, sql in aggs)
        child_ctes.append(
            f"{cte} AS (SELECT c.{_q(fk)} AS k, {agg_cols} "
            f"FROM {_q(ct.schema)}.{_q(ct.name)} c "
            f"JOIN base b ON c.{_q(fk)} = b.{_q(key)} GROUP BY c.{_q(fk)})"
        )
        for alias, _ in aggs:
            final_selects.append(f"{cte}.{_q(alias)}")
        joins.append(f"LEFT JOIN {cte} ON {cte}.k = b.{_q(key)}")

    ctes = ", ".join([base_sql, *child_ctes])
    sql = (
        f"WITH {ctes} SELECT {', '.join(final_selects)} FROM base b "
        + " ".join(joins)
    )
    return sql, accepted, skipped


def assemble(ds: DataSource, sql: str, max_rows: int | None = None) -> pl.DataFrame:
    return ds.run_sql(sql, max_rows=max_rows)
