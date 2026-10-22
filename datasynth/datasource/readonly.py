"""Application-level read-only SQL guard.

This is *defense-in-depth*, not the primary control. The real boundary is a
read-only database role/connection (see PostgresDataSource docstring). This guard
catches obvious writes early and gives the agent a clean error before execution.
"""

from __future__ import annotations

import re


class WriteAttemptError(ValueError):
    """Raised when a query is not a single read-only statement."""


# Leading keywords that unambiguously read.
_READ_PREFIXES = ("select", "with", "show", "explain", "table", "values")

# Keywords that mutate data/schema, matched as whole words anywhere in the statement.
_WRITE_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "merge",
    "upsert",
    "truncate",
    "drop",
    "create",
    "alter",
    "grant",
    "revoke",
    "comment",
    "copy",
    "call",
    "do",
    "vacuum",
    "reindex",
    "refresh",
    "lock",
    "set",
)

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_comments(sql: str) -> str:
    sql = _BLOCK_COMMENT.sub(" ", sql)
    sql = _LINE_COMMENT.sub(" ", sql)
    return sql


def assert_readonly(query: str) -> str:
    """Validate `query` is a single read-only statement. Return it stripped.

    Raises WriteAttemptError otherwise.
    """
    cleaned = _strip_comments(query).strip()
    if not cleaned:
        raise WriteAttemptError("Empty query.")

    # Reject stacked statements. Allow a single optional trailing semicolon.
    body = cleaned.rstrip(";")
    if ";" in body:
        raise WriteAttemptError("Multiple statements are not allowed.")

    lowered = body.lstrip().lower()
    if not lowered.startswith(_READ_PREFIXES):
        raise WriteAttemptError(
            f"Query must start with one of {_READ_PREFIXES}; got: {body[:40]!r}"
        )

    for kw in _WRITE_KEYWORDS:
        if re.search(rf"\b{kw}\b", lowered):
            # `with`/`select` can legitimately contain none of these; a match means a write.
            raise WriteAttemptError(f"Disallowed keyword {kw!r} found in query.")

    return body
