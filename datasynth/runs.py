"""Run registry and human-readable name generation.

Maintains artifacts/runs.json — a flat list of run records:
  {"name": str, "ask": str, "created_at": ISO8601, "status": "running"|"completed"|"interrupted"}
"""

from __future__ import annotations

import json
import random
from pathlib import Path

_ADJS = [
    "brave", "swift", "calm", "keen", "bright", "bold", "clear", "crisp",
    "deep", "fair", "fleet", "fresh", "glad", "grand", "hardy", "light",
    "lithe", "nimble", "plain", "prime", "quiet", "rapid", "sage", "sharp",
    "sleek", "slim", "solid", "stern", "true", "warm",
]

_NOUNS = [
    "acorn", "amber", "anvil", "birch", "bloom", "brook", "cedar", "cliff",
    "comet", "crane", "dune", "ember", "fern", "fjord", "flint", "glade",
    "grove", "haven", "heron", "inlet", "lark", "maple", "mesa", "peak",
    "pine", "ridge", "shore", "slate", "vale", "wave",
]


def load_registry(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def save_registry(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))


def append_run(path: Path, record: dict) -> None:
    records = load_registry(path)
    records.append(record)
    save_registry(path, records)


def update_status(path: Path, name: str, status: str) -> None:
    records = load_registry(path)
    for r in records:
        if r["name"] == name:
            r["status"] = status
            break
    save_registry(path, records)


def find_run(path: Path, name: str) -> dict | None:
    for r in load_registry(path):
        if r["name"] == name:
            return r
    return None


def new_name(registry_path: Path) -> str:
    existing = {r["name"] for r in load_registry(registry_path)}
    for _ in range(200):
        name = f"{random.choice(_ADJS)}-{random.choice(_NOUNS)}"
        if name not in existing:
            return name
    # Extremely unlikely fallback: append a number
    base = f"{random.choice(_ADJS)}-{random.choice(_NOUNS)}"
    i = 2
    while f"{base}-{i}" in existing:
        i += 1
    return f"{base}-{i}"
