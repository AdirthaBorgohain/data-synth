"""DataSource interface + schema value objects.

The agent reasons over `SchemaInfo` and small `sample()` frames, never raw tables.
Execution is read-only and logged. New backends (Snowflake, etc.) implement this ABC
and add their own cost/pushdown behavior without touching agent or graph code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import polars as pl


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    dtype: str  # source-reported type, normalized to a string label
    nullable: bool
    is_primary_key: bool = False


@dataclass(frozen=True)
class ForeignKey:
    column: str
    references_table: str
    references_column: str


@dataclass(frozen=True)
class TableInfo:
    schema: str
    name: str
    columns: tuple[ColumnInfo, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()
    approx_row_count: int | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"


@dataclass(frozen=True)
class SchemaInfo:
    tables: tuple[TableInfo, ...]

    def table(self, qualified_or_name: str) -> TableInfo | None:
        for t in self.tables:
            if qualified_or_name in (t.name, t.qualified_name):
                return t
        return None


@dataclass(frozen=True)
class CostHint:
    """Backend-neutral hint about running a query.

    Postgres fills `estimated_rows`/`estimated_cost` from EXPLAIN.
    Warehouses (Snowflake) can populate `estimated_bytes_scanned` for $ awareness.
    """

    estimated_rows: float | None = None
    estimated_cost: float | None = None
    estimated_bytes_scanned: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


class DataSource(ABC):
    """Read-only interface the agent and tools use to reach data."""

    @abstractmethod
    def introspect_schema(self) -> SchemaInfo:
        """Return tables, columns, primary keys, and foreign keys."""

    @abstractmethod
    def sample(self, table: str, n: int | None = None) -> pl.DataFrame:
        """Return up to `n` rows from `table` for profiling/EDA."""

    @abstractmethod
    def run_sql(self, query: str, max_rows: int | None = None) -> pl.DataFrame:
        """Run a read-only SELECT/WITH query. Must reject anything that writes."""

    @abstractmethod
    def estimate_cost(self, query: str) -> CostHint:
        """Estimate query cost without executing it (for approval gates)."""

    def close(self) -> None:  # optional override
        """Release connections/resources."""
