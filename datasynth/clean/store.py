"""In-process cache of sample frames, keyed by qualified table name.

Polars frames are not JSON-serializable, so they cannot live in the checkpointed
AgentState. The FrameStore is injected via config["configurable"]["frame_store"]
(like the DataSource) and holds the working copy of each table's sample:
  - profile loads a table once and caches the raw frame here
  - clean_apply replaces it with the cleaned frame
  - re-profiling then reads the cleaned frame

When persist_dir is provided, every put() writes a Parquet snapshot so the frames
survive process restarts and can be reloaded for --resume runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def _safe_filename(table: str) -> str:
    return table.replace("/", "_sl_").replace("\\", "_bs_") + ".parquet"


class FrameStore:
    def __init__(self, persist_dir: Path | None = None) -> None:
        self._frames: dict[str, pl.DataFrame] = {}
        self._persist_dir = persist_dir

    def has(self, table: str) -> bool:
        return table in self._frames

    def get(self, table: str) -> pl.DataFrame:
        return self._frames[table]

    def put(self, table: str, df: pl.DataFrame) -> None:
        self._frames[table] = df
        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            # Parquet doesn't support Polars Object dtype (e.g. PG JSONB/Struct columns).
            # Cast Object → String for disk only; keep original dtype in _frames for nodes.
            # Use map_elements since .cast() is strict by default and fails on Object.
            object_cols = [c for c, d in zip(df.columns, df.dtypes) if d == pl.Object]
            to_write = df.with_columns([
                pl.col(c).map_elements(lambda v: None if v is None else str(v), return_dtype=pl.String)
                for c in object_cols
            ]) if object_cols else df
            to_write.write_parquet(self._persist_dir / _safe_filename(table))
            manifest = {_safe_filename(t): t for t in self._frames}
            (self._persist_dir / "manifest.json").write_text(json.dumps(manifest))

    @classmethod
    def load(cls, persist_dir: Path) -> "FrameStore":
        fs = cls(persist_dir=persist_dir)
        manifest_path = persist_dir / "manifest.json"
        if manifest_path.exists():
            manifest: dict[str, str] = json.loads(manifest_path.read_text())
            for filename, table in manifest.items():
                p = persist_dir / filename
                if p.exists():
                    fs._frames[table] = pl.read_parquet(p)
        return fs
