"""Shared formatting helpers used across phase nodes."""

from __future__ import annotations


def profiles_brief(profiles: dict) -> str:
    """Per-table one-liner with dtype, null rate, and unique count per column."""
    lines: list[str] = []
    for table, p in profiles.items():
        if "error" in p:
            lines.append(f"{table}: ERROR {p['error']}")
            continue
        cols = p.get("columns", {})
        bits = [
            f"{c} ({s.get('dtype')}, null {s.get('null_rate', 0):.0%}, "
            f"uniq {s.get('n_unique')})"
            for c, s in cols.items()
        ]
        lines.append(f"{table}: " + ", ".join(bits))
    return "\n".join(lines)
