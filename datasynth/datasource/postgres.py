"""Postgres DataSource implementation (read-only).

Security model
--------------
The PRIMARY control is the database credential: point `database_url` at a role
with only SELECT/USAGE privileges. This class adds two further layers:

  1. `assert_readonly()` rejects non-SELECT/WITH statements before they run.
  2. Every query runs inside a `SET TRANSACTION READ ONLY` transaction, so the
     server itself refuses writes even if the guard is bypassed.

Do not rely on the application guard alone for untrusted input — keep the
read-only role.
"""

from __future__ import annotations

import re

import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from datasynth.datasource.base import (
    ColumnInfo,
    CostHint,
    DataSource,
    ForeignKey,
    SchemaInfo,
    TableInfo,
)
from datasynth.datasource.readonly import assert_readonly

# System schemas excluded from introspection.
_EXCLUDED_SCHEMAS = ("pg_catalog", "information_schema", "pg_toast")


class PostgresDataSource(DataSource):
    def __init__(
        self,
        database_url: str,
        *,
        default_sample_size: int = 10_000,
        max_result_rows: int = 10_000_000,
    ) -> None:
        if not database_url:
            raise ValueError("database_url is required.")
        self._engine: Engine = create_engine(database_url, pool_pre_ping=True)
        self._default_sample_size = default_sample_size
        self._max_result_rows = max_result_rows

    # --- internal helpers ---

    def _read_df(self, sql: str, max_rows: int | None) -> pl.DataFrame:
        """Run `sql` in a read-only transaction and return a polars frame."""
        with self._engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            result = conn.execute(text(sql))
            rows = result.fetchmany(max_rows or self._max_result_rows)
            cols = list(result.keys())
        # infer_schema_length=None scans all fetched rows for dtype inference —
        # the 100-row default mis-infers columns whose first rows are null/narrow
        # and then fails on a later string value.
        try:
            return pl.DataFrame(
                rows, schema=cols, orient="row", infer_schema_length=None
            )
        except Exception:
            # Genuine mixed types: force every column to string so the frame
            # always loads. The clean phase can cast columns afterward.
            return pl.DataFrame(
                rows,
                schema={c: pl.String for c in cols},
                orient="row",
                infer_schema_length=None,
            )

    def probe_connection(self) -> tuple[bool, str]:
        """Ping the database; return (connected, database_name)."""
        db_name = make_url(str(self._engine.url)).database or "unknown"
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, db_name
        except Exception:
            return False, db_name

    # --- DataSource interface ---

    def introspect_schema(self) -> SchemaInfo:
        excl = ", ".join(f"'{s}'" for s in _EXCLUDED_SCHEMAS)
        col_sql = f"""
            SELECT
                c.table_schema,
                c.table_name,
                c.column_name,
                c.data_type,
                (c.is_nullable = 'YES') AS nullable
            FROM information_schema.columns c
            WHERE c.table_schema NOT IN ({excl})
            ORDER BY c.table_schema, c.table_name, c.ordinal_position
        """
        pk_sql = f"""
            SELECT tc.table_schema, tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema NOT IN ({excl})
        """
        # pg_catalog: information_schema.constraint_column_usage is empty for
        # read-only roles that are not table owners.
        fk_sql = f"""
            SELECT
                n.nspname AS table_schema,
                cl.relname AS table_name,
                a.attname AS column_name,
                fcl.relname AS ref_table,
                af.attname AS ref_column
            FROM pg_constraint AS c
            JOIN pg_namespace AS n ON n.oid = c.connamespace
            JOIN pg_class AS cl ON cl.oid = c.conrelid
            JOIN pg_class AS fcl ON fcl.oid = c.confrelid
            JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS u (attnum, ord) ON TRUE
            JOIN LATERAL unnest(c.confkey) WITH ORDINALITY AS fu (attnum, ord)
              ON u.ord = fu.ord
            JOIN pg_attribute AS a ON a.attrelid = c.conrelid AND a.attnum = u.attnum
            JOIN pg_attribute AS af ON af.attrelid = c.confrelid AND af.attnum = fu.attnum
            WHERE c.contype = 'f'
              AND n.nspname NOT IN ({excl})
        """

        cols = self._read_df(col_sql, max_rows=None)
        pks = self._read_df(pk_sql, max_rows=None)
        fks = self._read_df(fk_sql, max_rows=None)

        pk_set = {
            (r["table_schema"], r["table_name"], r["column_name"])
            for r in pks.iter_rows(named=True)
        }
        fk_map: dict[tuple[str, str], list[ForeignKey]] = {}
        for r in fks.iter_rows(named=True):
            fk_map.setdefault((r["table_schema"], r["table_name"]), []).append(
                ForeignKey(
                    column=r["column_name"],
                    references_table=r["ref_table"],
                    references_column=r["ref_column"],
                )
            )

        grouped: dict[tuple[str, str], list[ColumnInfo]] = {}
        for r in cols.iter_rows(named=True):
            key = (r["table_schema"], r["table_name"])
            grouped.setdefault(key, []).append(
                ColumnInfo(
                    name=r["column_name"],
                    dtype=r["data_type"],
                    nullable=bool(r["nullable"]),
                    is_primary_key=(key[0], key[1], r["column_name"]) in pk_set,
                )
            )

        tables = tuple(
            TableInfo(
                schema=schema,
                name=name,
                columns=tuple(column_list),
                foreign_keys=tuple(fk_map.get((schema, name), ())),
            )
            for (schema, name), column_list in grouped.items()
        )
        return SchemaInfo(tables=tables)

    def sample(self, table: str, n: int | None = None) -> pl.DataFrame:
        n = n or self._default_sample_size
        # `table` comes from introspected schema, not user free-text; still quote it.
        if "." in table:
            schema, name = table.split(".", 1)
            ident = f'"{schema}"."{name}"'
        else:
            ident = f'"{table}"'
        sql = f"SELECT * FROM {ident} LIMIT {int(n)}"
        return self._read_df(sql, max_rows=n)

    def run_sql(self, query: str, max_rows: int | None = None) -> pl.DataFrame:
        return self._read_df(assert_readonly(query), max_rows=max_rows)

    def estimate_cost(self, query: str) -> CostHint:
        safe = assert_readonly(query)
        with self._engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            plan_rows = conn.execute(text(f"EXPLAIN {safe}")).fetchall()
        plan_text = "\n".join(r[0] for r in plan_rows)
        rows = cost = None
        m = re.search(r"rows=(\d+)", plan_text)
        if m:
            rows = float(m.group(1))
        m = re.search(r"cost=[\d.]+\.\.([\d.]+)", plan_text)
        if m:
            cost = float(m.group(1))
        return CostHint(
            estimated_rows=rows,
            estimated_cost=cost,
            notes=(plan_text.splitlines()[0],) if plan_rows else (),
        )

    def close(self) -> None:
        self._engine.dispose()
