"""Data source abstraction. Postgres now; Snowflake and others plug in via the same ABC."""

from datasynth.datasource.base import (
    ColumnInfo,
    CostHint,
    DataSource,
    ForeignKey,
    SchemaInfo,
    TableInfo,
)
from datasynth.datasource.postgres import PostgresDataSource

__all__ = [
    "ColumnInfo",
    "CostHint",
    "DataSource",
    "ForeignKey",
    "SchemaInfo",
    "TableInfo",
    "PostgresDataSource",
]
