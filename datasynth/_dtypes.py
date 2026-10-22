"""Shared polars dtype sets used across analyses, render, transforms, evaluate."""

from __future__ import annotations

import polars as pl

NUMERIC = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}
TEMPORAL = {pl.Date, pl.Datetime}

# A target with at most this many distinct values is treated as classification.
CLASS_MAX_UNIQUE = 20
