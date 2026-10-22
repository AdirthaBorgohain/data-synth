"""Run-scoped artifact directory.

Holds files a run produces (EDA plots now; model + lineage later). Injected via
config["configurable"]["artifacts"] like the FrameStore, so nodes write outputs
without each one re-deriving the path layout.

Layout: <base_dir>/<run_id>/<subdir>/<file>
"""

from __future__ import annotations

import re
from pathlib import Path

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify(text: str) -> str:
    """Filesystem-safe slug for plot filenames (column/table names → slugs)."""
    s = _SLUG_RE.sub("_", text).strip("_")
    return s or "unnamed"


class ArtifactStore:
    def __init__(self, base_dir: str, run_id: str) -> None:
        self._root = Path(base_dir) / run_id

    @property
    def root(self) -> Path:
        return self._root

    def dir(self, *parts: str) -> Path:
        """Return a subdirectory under the run root, creating it if needed."""
        d = self._root.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path(self, *parts: str) -> str:
        """Return a file path under the run root, ensuring its parent exists."""
        p = self._root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
